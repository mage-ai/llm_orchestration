from typing import List

import torch

from default_repo.llm_orchestration.utils.tokenization import named_entity_recognition_tokens


@transformer
def transform(documents: List[List[str]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _, tokenizer = factory_items_mapping['data_preparation/nlp']

    arr = []
    
    for document_id, document, metadata, chunk in documents[:10]:
        print(f'document_id: {document_id}')

        encoded = tokenizer.encode_plus(chunk, return_tensors='pt')
        token_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        arr.append([
            document_id,
            document,
            metadata,
            chunk,
            [
               token_ids.numpy(),
               attention_mask.numpy(),
            ],
        ])

        print(f'{round(100 * len(arr) / len(documents))}% ({len(arr)}/{len(documents)})')

    return [
        arr,
    ]
