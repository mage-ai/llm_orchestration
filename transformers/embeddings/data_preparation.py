from typing import List, Union

import numpy as np


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, _, client = factory_items_mapping['embeddings/clients']

    arr = []
    for document_id, document, metadata, chunk, tokens in documents:
        print(f'document_id: {document_id}')

        vector = client.post(tokens)
        
        arr.append([
            document_id,
            document,
            metadata,
            chunk,
            tokens,
            vector,
        ])

        print(f'{round(100 * len(arr) / len(documents))}% ({len(arr)}/{len(documents)})')
    
    return [
        arr,
    ]