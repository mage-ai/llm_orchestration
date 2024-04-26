import os

from transformers import BertModel, BertTokenizer

from mage_ai.settings.repo import get_repo_path

MODEL_PATH = os.path.join(get_repo_path(), 'models', 'bert-large-uncased')


@factory
def model(*args, **kwargs) -> BertModel:
    return BertModel.from_pretrained(MODEL_PATH)


@factory
def tokenizer(*args, **kwargs) -> BertTokenizer:
    return BertTokenizer.from_pretrained(MODEL_PATH)