import os
from typing import List, Union

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def fit_transform_embedding(nlp, tokens: List[str]) -> List[float]:
    text = ' '.join(tokens)
    doc = nlp(text)
    embedding = doc.vector.tolist()

    return embedding


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp = factory_items_mapping['data_preparation/nlp'][0]
    
    print(f'documents: {len(documents)}')
    documents_more = []
    for document_id, document, chunk, tokens in documents:
        print(f'document_id: {document_id}')
        print(f'tokens: {len(tokens)}')

        embedding = fit_transform_embedding(nlp, tokens)
        print(f'embedding: {len(embedding)}')

        documents_more.append([
            document_id,
            document,
            chunk,
            tokens,
            embedding,
        ])

        print(f'{round(100 * len(documents_more) / len(documents))}% ({len(documents_more)}/{len(documents)})')
    
    return [
        documents_more,
    ]