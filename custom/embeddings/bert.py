import json
from typing import Dict, List

import pandas as pd

from default_repo.llm_orchestration.utils.tokenization import (
    embeddings_concatenate, 
    embeddings_max_pooling, 
    embeddings_mean,
)


@custom
def transform_custom(data: List[Dict], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, model = factory_items_mapping['models/transformers']

    count = len(data)
    print(f'count: {count}')

    rows = []
    for index, row in enumerate(data):
        document_id = row['document_id']
        chunk = row['chunk'].strip()

        try:
            matrix = model([chunk])['embeddings'][0]
        except Exception as err:
            print(f'document_id: {document_id}')
            print(f'chunk:')
            print(json.dumps(chunk))
            print(err)
            raise err

        # vector = embeddings_concatenate([
        #     embeddings_mean(matrix),
        #     embeddings_max_pooling(matrix),
        # ])

        # row['vector'] = vector.tolist()

        row['vector'] = matrix

        rows.append(row)

        print(f'{index + 1}/{count}')

    return pd.DataFrame(rows)