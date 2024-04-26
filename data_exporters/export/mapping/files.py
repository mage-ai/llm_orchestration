from typing import List

import pandas as pd


@data_exporter
def export_data(dfs: List[pd.DataFrame], *args, **kwargs):
    return pd.concat(dfs)