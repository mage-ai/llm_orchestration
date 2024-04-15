from transformers import AutoModel, AutoTokenizer


def __get_model_name(**kwargs) -> str:
    return kwargs.get('transformer_model', 'sentence-transformers/all-MiniLM-L6-v2')


@factory
def model(*args, **kwargs) -> AutoModel:
    return AutoModel.from_pretrained( __get_model_name(**kwargs))


@factory
def tokenizer(*args, **kwargs) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained( __get_model_name(**kwargs))