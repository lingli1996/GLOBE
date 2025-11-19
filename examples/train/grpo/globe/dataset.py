import re
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset

prompt_en = """You are a geolocation expert. You are participating in a geolocation challenge. Based on the provided image:
1. Carefully analyze the image for clues about its location (architecture, signage, vegetation, terrain, etc.)
2. Think step-by-step about what country, and city this is likely to be in and why

Your final answer include these two lines somewhere in your response:
country: [country name]
city: [city name]

You MUST output the thinking process in <think> </think> and give answer in <answer> </answer> tags."""


class GeoLocRMmodelPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        ms_dataset_id='geo_loc_rm_en_55k',
        hf_dataset_id='geo_loc_rm_en_55k',
        preprocess_func=GeoLocRMmodelPreprocessor(),
        tags=['geo', 'vision', 'reward']))


class GeoLocEn2Preprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = prompt_en
        response = row.get('response', '')
        solution = row.get('__#solution', '')
        pattern2 = r'<city>(.*?)</city>'
        match2 = re.search(pattern2, solution)
        city = match2.group(1) if match2 else ""

        pattern1 = r'<country>(.*?)</country>'
        match1 = re.search(pattern1, solution)
        country = match1.group(1) if match1 else ""

        response = f"<think>{response}</think><answer>country: {country}\ncity: {city}</answer>"
        row.update({'query': query})
        row.update({'response': response})
        row.update({'solution': f"{country}\t{city}"})
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        ms_dataset_id='geo_loc_62k_en_entity',
        hf_dataset_id='geo_loc_62k_en_entity',
        preprocess_func=GeoLocEn2Preprocessor(),
        tags=['qa', 'geo', 'vision', 'grpo']))