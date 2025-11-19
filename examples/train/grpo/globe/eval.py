import os
import sys
import base64
import csv
import time
import re
import random
import json
import logging
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from typing import Dict, Tuple, List, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


PROMPT_COT = """You are participating in a geolocation challenge. Based on the provided image:
1. Carefully analyze the image for clues about its location (architecture, signage, vegetation, terrain, etc.)
2. Think step-by-step about what country, and city this is likely to be in and why
Your final answer include these two lines somewhere in your response:
country: [country name]
city: [city name]

You MUST output the thinking process in <think> </think> and give answer in <answer> </answer> tags."""

PROMPT_NO_COT = """Based on the provided image, please output its location (country and city) directly without any explanation.
Your final answer include these two lines in your response:
country: [country name]
city: [city name]"""



dataset_info = {
    "img2gps3k": {
        "prefix": "data/im2gps3ktest",
        "gt_file": "data/img2gps3k_gt.csv"
    },
    "mp16-reason-test-12k": {
        "prefix": "data/eval_images/",
        "gt_file": "data/mp16-reason-test-12k.csv"
    },
    "osv-test-mini-3k": {
        "prefix": "data/osv-test-mini-3k",
        "gt_file": "data/test_mini.csv"
    }
}


def infer_image(image_path, prompt, url="http://29.163.186.238:8081/v1", stream=False):
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return ""
    
    openai_api_key = "EMPTY"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=url,
    )
    
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
            encoded_image_text = encoded_image.decode("utf-8")
        base64_img = f"data:image;base64,{encoded_image_text}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

    start = time.time()
    try:
        chat_response = client.chat.completions.create(
            model="qwen2.5-vl",
            timeout=60,
            temperature=0.1,
            max_tokens=512,
            frequency_penalty=0.7,
            presence_penalty=0.7,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_img}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            stream=stream
        )
    except Exception as e:
        logger.error(f"API call failed for {image_path}: {e}")
        return ""
    
    i = 0
    buffer = ""
    if stream:
        for chunk in chat_response:
            content = chunk.choices[0].delta.content
            buffer += content
            if i == 0:
                end = time.time()
                dura = end - start
                logger.info(f"TTFT: {dura}s")
            i += 1
    
        final_end = time.time()
        dural = final_end - start
        logger.info(f"TOTAL: {dural}s")
        logger.info(f"TPOT: {(dural / i)}")
        return buffer
    else:
        final_end = time.time()
        dural = final_end - start
        logger.info(f"TOTAL: {dural}s")
        return chat_response.choices[0].message.content


def extract_location_from_content(data):
    if not data:
        return "", ""
        
    try:
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, data, re.DOTALL)
        pred = match.group(1).strip() if match else data

        country_match = re.search(r'country:\s*([^\n]+)', pred)
        city_match = re.search(r'city:\s*([^\n]+)', pred)
        country = country_match.group(1).strip().lower() if country_match else ""
        city = city_match.group(1).strip().lower() if city_match else ""
        return country, city
    except Exception as e:
        logger.error(f"Error extracting city: {e}")
        return "", ""


def extract_location_without_think(pred_content):
    if not pred_content:
        return "", ""
        
    country_match = re.search(r'country:\s*([^\n]+)', pred_content)
    city_match = re.search(r'city:\s*([^\n]+)', pred_content)
    country = country_match.group(1).strip().lower() if country_match else ""
    city = city_match.group(1).strip().lower() if city_match else ""
    return country, city


def match_location(pred, ground_truth, match_city_or_country_threshold=0.7):
    len1 = len(pred)
    len2 = len(ground_truth)
    pred = pred.lower()
    ground_truth = ground_truth.lower()
    if len1 == 0:
        return 0
    if (pred in ground_truth and len1 / len2 >= match_city_or_country_threshold) or (
        ground_truth in pred and len2 / len1 >= match_city_or_country_threshold
    ):
        return 1
    return 0


def process_row(row, prefix, prompt, url, use_cot = True):
    img_id = row["IMG_ID"]
    img_path = os.path.join(prefix, img_id)
    
    try:
        content = infer_image(img_path, prompt, url)
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        content = ""
    
    if use_cot:
        pred_country, pred_city = extract_location_from_content(content)
    else:
        pred_country, pred_city = extract_location_without_think(content)
    
    city = row.get("city", "")
    country = row.get("country", "")
    
    city_match = match_location(pred_city, city)
    country_match = match_location(pred_country, country)
    
    return city_match, country_match, pred_city, pred_country, content


def validate_dataset_config(ds: str) -> bool:
    if ds not in dataset_info:
        logger.error(f"Unknown dataset: {ds}")
        return False
    
    config = dataset_info[ds]
    if not os.path.exists(config["gt_file"]):
        logger.error(f"Ground truth file not found: {config['gt_file']}")
        return False
    
    if not os.path.exists(config["prefix"]):
        logger.warning(f"Image prefix directory not found: {config['prefix']}")
    
    return True


def infer_dataset(url, use_cot, dataset, output_file, parallel=16):
    if not validate_dataset_config(dataset):
        return
    
    city_cnt = 0.0
    country_cnt = 0.0

    input_file = dataset_info[dataset]["gt_file"]
    prefix = dataset_info[dataset]["prefix"]
    
    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames + ["prediction"]
        if "pred_city" not in fieldnames:
            fieldnames.append("pred_city")
        if "pred_country" not in fieldnames:
            fieldnames.append("pred_country")
        
        if "LAT" not in fieldnames:
            fieldnames.append("LAT")
        if "LON" not in fieldnames:
            fieldnames.append("LON")
        
        if "IMG_ID" not in fieldnames:
            fieldnames.append("IMG_ID")

        if "pred_LAT" not in fieldnames:
            fieldnames.append("pred_LAT")
        if "pred_LON" not in fieldnames:
            fieldnames.append("pred_LON")
        
        total_lines = []
        for row in reader:
            img_id = row.get("IMG_ID", None) or row.get("id", None)
            if not img_id:
                continue
            if not img_id.endswith(".jpg"):
                img_id = img_id + ".jpg"
            
            row["IMG_ID"] = img_id

            lat = row.get('LAT', None) or row.get('latitude', None)
            lon = row.get('LON', None) or row.get('longitude', None)
            if not lat or not lon:
                continue
            try:
                lat = float(lat)
                lon = float(lon)
                row["LAT"] = lat
                row["LON"] = lon
                total_lines.append(row)
            except ValueError:
                logger.warning(f"Invalid coordinates for {img_id}: lat={lat}, lon={lon}")
                continue
        
        logger.info(f"Processed lines: {len(total_lines)}")
        if not total_lines:
            logger.error("No valid data to process")
            return
        
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            prompt = PROMPT_COT if use_cot else PROMPT_NO_COT

            futures = {executor.submit(process_row, row, prefix, prompt, url, use_cot): row for row in total_lines}

            with open(output_file, "w", encoding="utf-8", newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for future in tqdm(as_completed(futures), total=len(total_lines), desc='Processing CSV'):
                    row = futures[future]
                    try:
                        city_match, country_match, p_city, p_country, content = future.result()
                        row["prediction"] = json.dumps({"data": content}, ensure_ascii=False)
                        row["pred_city"] = p_city
                        row["pred_country"] = p_country
                        writer.writerow(row)
                        city_cnt += city_match
                        country_cnt += country_match
                    except Exception as e:
                        logger.error(f"Error processing future result: {e}")
                        continue

        logger.info(f"City accuracy: {city_cnt / len(total_lines):.4f}")
        logger.info(f"Country accuracy: {country_cnt / len(total_lines):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GLOBE models.')
    parser.add_argument('--url', type=str, help='Base URL of the model API.')
    parser.add_argument('--output', type=str, help='Output file path.')
    parser.add_argument('--dataset', type=str, help='Dataset name.')
    parser.add_argument('--use_cot', action='store_true', help='Whether to use chain-of-thought prompting.')
    args = parser.parse_args()
    infer_dataset(args.url, use_cot=args.use_cot, dataset=args.dataset, output_file=args.output)


if __name__ == "__main__":
    main()