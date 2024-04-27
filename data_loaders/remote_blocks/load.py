import numpy as np
import pandas as pd

from mage_ai.data_preparation.models.block.remote.models import RemoteBlock
from mage_ai.settings.repo import get_repo_path


@data_loader
def load_data(*args, **kwargs):
    sample = int(kwargs.get('sample', 2))
    dry_run = sample >= 1

    outputs = kwargs.get('remote_blocks')

    if not outputs:
        block_uuid = kwargs.get('remote_source_block_uuid')
        pipeline_uuid = kwargs.get('remote_source_pipeline_uuid')
        
        outputs = [
            RemoteBlock.load(
                block_uuid=block_uuid,
                pipeline_uuid=pipeline_uuid,
                repo_path=get_repo_path(),
            ).get_outputs(),
        ]

    df = pd.DataFrame()
    arr = []

    if dry_run:
        outputs = outputs[:sample]
    
    print(f'sample: {sample}')
    print(f'dry_run: {dry_run}')
    print(f'outputs: {len(outputs)}')
    
    for dfs in outputs:
        print(f'dfs: {len(dfs)}')

        if len(dfs) >= 1 and isinstance(dfs[0], list):
            for df_list in dfs:
                print(type(df_list), len(df_list), type(df_list[0]))
                
                if len(df_list) >= 1:
                    item = df_list[0]

                    if isinstance(item, list) and \
                            len(item) >= 1 and \
                            isinstance(item[0], dict):
                            
                        for item_list in df_list:
                            arr.append(pd.DataFrame(item_list))
                    elif isinstance(item, dict):
                        arr.append(pd.DataFrame(df_list))
                    else:
                        arr += df_list
        else:
            df = pd.concat([df] + dfs)

    if len(arr) >= 1:
        print(f'arr: {len(arr)}')
        return arr

    print(f'df:  {len(df)}')

    return df