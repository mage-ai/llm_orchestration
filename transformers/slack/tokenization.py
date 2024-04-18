from typing import Dict, List


def lemmatization(nlp, chunks: List[str]):
    tokenized_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
        doc = nlp(' '.join(tokens_without_stopwords))
        lemmatized_tokens = [token.lemma_ for token in doc]
        tokenized_chunks.append(lemmatized_tokens)
    return tokenized_chunks


@transformer
def transform(documents: List[List], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]
    
    arr = []

    for chunks, document_id, document in documents:
        tokenized_chunks = []
        
        arr.append([
            lemmatization(nlp, chunks),
            document_id,
            document,
        ])


    return [
        arr,
    ]