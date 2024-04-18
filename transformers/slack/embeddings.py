from typing import Dict, List, Union


@transformer
def transform(documents: List[List[Union[List[str], str, Dict]]], *args, **kwargs):
    model = list(kwargs.get('factory_items_mapping').values())[0][0]

    arr = []

    for tokenized_chunks, document_id, document in documents:        
        arr.append([
            [model.encode(tokens)[0] for tokens in tokenized_chunks],
            document_id,
            document,
        ])


    return [
        arr,
    ]