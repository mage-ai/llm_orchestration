
from typing import List


@data_loader
def load_data(document_ids: List[str], *args, **kwargs):
    arr = []

    for document_id in document_ids:
        with open(document_id, 'r') as f:
            arr.append([
                document_id,
                f.read(),
                {},
            ])

    return [arr]