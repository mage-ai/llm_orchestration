import json
from typing import Dict, List

import pandas as pd


@custom
def transform_custom(data: List[Dict], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    Transformers = factory_items_mapping['models/transformers'][0]

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

        row['vector'] = matrix
        rows.append(row)

        print(f'{index + 1}/{count}')

    return pd.DataFrame(rows)
