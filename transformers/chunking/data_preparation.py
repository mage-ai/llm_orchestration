from typing import Dict, List, Union

import pandas as pd

from default_repo.llm_orchestration.utils.chunking import chunk_sentences_with_overlap
from default_repo.llm_orchestration.utils.cleaner import clean_html_and_add_spaces
from default_repo.llm_orchestration.utils.grouping import bucket_items
from default_repo.llm_orchestration.utils.markdown import remove_markdown_keep_text, remove_markdown_metadata


@transformer
def transform(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _, _ = factory_items_mapping['data_preparation/nlp']
    sample = int(kwargs.get('sample', 2))

    print(f'df: {len(df)}')
    
    rows = []
    for _index, row in df.iterrows():
        if sample >= 1 and len(rows) >= sample:
            break

        document_id, document, metadata = row
        
        text = remove_markdown_metadata(document)
        text = clean_html_and_add_spaces(text)
        text = remove_markdown_keep_text(text)
        chunks = chunk_sentences_with_overlap(
            nlp,
            text,
            max_length=512,
            overlap=128,
        )

        for chunk in chunks:
            rows.append(dict(
                document_id=document_id,
                document=document,
                metadata=metadata,
                chunk=chunk,
            ))

            if sample >= 1:
                print(f'sample: {len(rows)}/{sample}')
                if len(rows) >= sample:
                    break

    print(f'rows: {len(rows)}')
    
    dfs = []
    buckets = bucket_items(
        rows, 
        max_num_buckets=10000, 
        max_items_per_bucket=20,
        override_on_max=True,
    )
    print(f'buckets: {len(buckets)}')

    for bucket in buckets:
        dfs.append(bucket)

    return [
        dfs,
    ]