from typing import Dict, List, Union


@transformer
def transform(documents: List[List[Union[List[str], str, Dict]]], *args, **kwargs):
    model = list(kwargs.get('factory_items_mapping').values())[0][0]

    arr = []

    for tokenized_chunks, document_id, document in documents:    
        vectors = []
        for tokens in tokenized_chunks:
            vector = model.encode(tokens)
            print('WTF0', len(vector))
            print('WTF1', vector)
            vectors.append(vector[0])

        arr.append([
            vectors,
            document_id,
            document,
        ])


    return [
        arr,
    ]