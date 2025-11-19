import asyncio
import re
from typing import List

import json

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()
"""
Step 1: Define a Reward Class
    Implement your custom reward calculation logic within the __call__ method.
    The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

Step 2: Register the Reward Class in orms
    For example:
    python orms['external_math_acc'] = MathAccuracy

Step 3: Configure the Arguments
    Use the following arguments when running the script:
    bash --plugin /path/to/plugin.py --reward_funcs external_math_acc
"""


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):
    def __init__(self):
        import importlib.util

        assert (
            importlib.util.find_spec("math_verify") is not None
        ), "The math_verify package is required but not installed. Please install it using 'pip install math_verify'."

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class GeoLocatablityORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        import math
        import base64
        from openai import OpenAI

        rewards = []
        rm_prompt = "Given input image and its location reasoning text, please determine if the image is visually locatable, then output “yes” or “no” directly.\n"
        
        # Locatablity reward model is servered remote by vLLM
        # Place your reward model url here
        reward_url = "http://29.163.178.251:8083/v1"
        openai_api_key = "EMPTY"

        def get_reward_from_remote_vllm(image_bytes, content, openai_api_base):
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            encoded_image = base64.b64encode(image_bytes)
            encoded_image_text = encoded_image.decode("utf-8")
            base64_qwen = f"data:image;base64,{encoded_image_text}"
            chat_response = client.chat.completions.create(
                model="qwen2.5-vl",
                timeout=300,
                temperature=0.6,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": rm_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": base64_qwen}},
                            {"type": "text", "text": content},
                        ],
                    },
                ],
                logprobs=True,
                top_logprobs=5
            )

            postive_tokens = ["yes", "Yes", "YES"] # "yes" token is 9693 in Qwen2.5
            negative_tokens = ["no", "No", "NO"] # "no" token is 2152 in Qwen2.5
            pos_prob, neg_prob = 0.0, 0.0
            for choice in chat_response.choices:
                for logprob in choice.logprobs.content[0].top_logprobs:
                    token_id = logprob.token
                    prob = logprob.logprob
                    prob = math.exp(prob)
                    if token_id in postive_tokens:
                        pos_prob += prob
                    elif token_id in negative_tokens:
                        neg_prob += prob
            reward = (pos_prob) / (pos_prob + neg_prob + 1e-19)    
            return reward

        image_bytes = kwargs["images"]

        for content, img_byte in zip(completions, image_bytes):
            try:
                reward = get_reward_from_remote_vllm(img_byte[0]["bytes"], content, reward_url)
                print(f"reward: {reward}")
            except Exception as e:
                logger.error(f"get reward failed {e}")
                reward = 0.0

            rewards.append(reward)
        return rewards


class GeoLocAccuracyV2ORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        match_city_or_country_threshhomd = 0.7

        def match_location(pred, ground_truth):
            len1 = len(pred)
            len2 = len(ground_truth)
            pred = pred.lower()
            ground_truth = ground_truth.lower()
            if len1 == 0:
                return False
            if (pred in ground_truth and len1 / len2 >= match_city_or_country_threshhomd) or (
                ground_truth in pred and len2 / len1 >= match_city_or_country_threshhomd
            ):
                return True
            return False

        for content, sol in zip(completions, solution):
            pattern = r'<answer>(.*?)</answer>'

            match = re.search(pattern, content, re.DOTALL)
            pred = match.group(1).strip() if match else ""

            gt_country, gt_city = sol.split("\t")
            
            country_match = re.search(r'country:\s*([^\n]+)', pred)
            city_match = re.search(r'city:\s*([^\n]+)', pred)

            country = country_match.group(1).strip().lower() if country_match else ""
            city = city_match.group(1).strip().lower() if city_match else ""


            if match_location(city, gt_city) and match_location(country, gt_country):
                reward = 1.0
            elif match_location(country, gt_country):
                reward = 0.2
            else:
                reward = 0.0

            rewards.append(reward)
        return rewards


class GeoVisalEntityMatching2ORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        batch_entities = kwargs["entities"]

        def calculate_entity_tag_reward(tags, string):
            string_lower = string.lower()
            tags_set = set(tag.lower() for tag in tags)
            matched_tags_count = sum(1 for tag in tags_set if tag in string_lower)
            if len(tags_set) == 0:
                return 0.0
            return matched_tags_count / len(tags_set)

        for content, entities in zip(completions, batch_entities):
            entity_tags = [ent["text"] for ent in entities]

            think_content = content.split("</think>")[0].replace("<think>", "")
            reward = calculate_entity_tag_reward(entity_tags, think_content)
            rewards.append(reward)
        return rewards


orms["external_math_acc"] = MathAccuracy
orms["external_math_format"] = MathFormat
orms["globe_locatablity"] = GeoLocatablityORM
orms["globe_accuracy"] = GeoLocAccuracyV2ORM
orms["globe_visual_match"] = GeoVisalEntityMatching2ORM
