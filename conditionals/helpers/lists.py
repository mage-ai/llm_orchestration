from typing import List, Union
import pandas as pd


@factory
def iterate(*args, **kwargs):
    def __reduce(df: Union[List[pd.DataFrame], pd.DataFrame], *args, **kwargs):
        df_list = []
        if isinstance(df, list):
            df_list = df
        else:
            df_list = [df]

        documents_mapping = {}
        for df_item in df_list:
            for index, row in df_item.iterrows():
                yield index, row

    return __reduce