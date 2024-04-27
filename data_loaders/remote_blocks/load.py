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

    if dry_run:
        outputs = outputs[:sample]
    
    print(f'sample: {sample}')
    print(f'dry_run: {dry_run}')
    print(f'outputs: {len(outputs)}')
    
    for dfs in outputs:
        print(f'dfs: {len(dfs)}')

        df = pd.concat([df] + dfs)

    print(f'df: {len(df)}')

    return df