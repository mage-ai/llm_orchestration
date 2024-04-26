from typing import Dict, List, Union

import pandas as pd

from default_repo.llm_orchestration.utils.chunking import chunk_markdown
from default_repo.llm_orchestration.utils.grouping import bucket_items


@transformer
def transform(df: pd.DataFrame, *args, **kwargs):
    sample = int(kwargs.get('sample', 2))

    rows = []    
    for _index, row in df.iterrows():
        document_id, document, metadata = row
            
        chunks = chunk_markdown(document)
        for chunk in chunks:
            if sample >= 1 and len(rows) >= sample:
                break

            rows.append([
                document_id,
                document,
                metadata,
                chunk,
            ])

    print(f'rows: {len(rows)}')
    
    dfs = []
    buckets = bucket_items(
        rows, 
        max_num_buckets=1000, 
        max_items_per_bucket=100,
        override_on_max=True,
    )
    print(f'buckets: {len(buckets)}')

    for bucket in buckets:
        dfs.append(pd.DataFrame(bucket, columns=df.columns.to_list() + ['chunk']))

    return [
        dfs,
    ]