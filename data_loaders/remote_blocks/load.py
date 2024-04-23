import numpy as np

from mage_ai.data_preparation.models.block.remote.models import RemoteBlock
from mage_ai.settings.repo import get_repo_path


@data_loader
def load_data(*args, **kwargs):
    sample = int(kwargs.get('sample', 2))
    dry_run = sample >= 1
    pipeline_uuid = kwargs.get('remote_pipeline_uuid')

    outputs = kwargs.get('remote_blocks')

    if not outputs:
        outputs = [
            RemoteBlock.load(
                block_uuid='export/mapping/files',
                pipeline_uuid=pipeline_uuid,
                repo_path=get_repo_path(),
            ).get_outputs(),
        ]

    documents_count = 0
    documents_list = []

    if dry_run:
        outputs = outputs[:sample]
    
    print(f'sample: {sample}')
    print(f'dry_run: {dry_run}')
    print(f'outputs: {len(outputs)}')
    for buckets in outputs:
        print(f'buckets: {len(buckets)}')

        if dry_run:
            buckets = buckets[:sample]

        for bucket in buckets:
            print(f'bucket: {len(bucket)}')
            
            if dry_run:
                bucket = bucket[:sample]

            documents_count += len(bucket)
            documents_list.append(bucket)

    print(f'documents_count: {documents_count}')
    print(f'documents_list: {len(documents_list)}')

    return [
        documents_list,
    ]