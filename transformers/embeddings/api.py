from typing import List, Union

import numpy as np

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(documents: List[List[Union[str, List[str]]]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _1, _2, openai = factory_items_mapping['embeddings/clients']

    print(f'documents: {len(documents)}')

    arr = []
    for document_id, document, metadata, chunk, tokens in documents:
        print(f'document_id: {document_id}')

        vector = openai.post(tokens)
        
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