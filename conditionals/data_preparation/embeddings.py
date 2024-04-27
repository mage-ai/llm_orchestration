from sentence_transformers import SentenceTransformer


@factory
def model(*args, **kwargs) -> bool:
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')
