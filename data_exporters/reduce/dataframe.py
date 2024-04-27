from typing import List

import pandas as pd

from default_repo.llm_orchestration.utils.grouping import bucket_dataframes


@data_exporter
def export_data(dfs: List[pd.DataFrame], *args, **kwargs):
    buckets = bucket_dataframes(dfs)

    print(f'buckets: {len(buckets)}')

    return buckets
    