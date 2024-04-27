import os
import requests
from typing import Dict, List

from transformers import BertModel, BertTokenizer

from mage_ai.settings.repo import get_repo_path

MODEL_PATH = os.path.join(get_repo_path(), 'models', 'bert-large-uncased')


class EmbeddingFetcher:
    @classmethod
    def fetch_embedding(_cls, texts: List[str]) -> Dict:
        url = 'https://ai.mage.ai/embeddings'
        payload = dict(texts=texts)
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch embeddings: {response.text}")


@factory
def model(*args, **kwargs) -> BertModel:
    # return BertModel.from_pretrained(MODEL_PATH)
    return EmbeddingFetcher.fetch_embedding


@factory
def tokenizer(*args, **kwargs) -> BertTokenizer:
    return BertTokenizer.from_pretrained(MODEL_PATH)
