from typing import Dict, List

import pandas as pd


@data_loader
def load_data(data: List[Dict], *args, **kwargs):
    rows = []
    for row in data:
        document_id = row['document_id']
        with open(document_id, 'r') as f:
            rows.append([
                document_id,
                f.read(),
                {},
            ])

    df = pd.DataFrame(rows, columns=[
        'document_id',
        'document',
        'metadata',
    ])

    return df