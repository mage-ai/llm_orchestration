from mage_ai.orchestration.triggers.api import trigger_pipeline
from mage_ai.settings.repo import get_repo_path


@custom
def trigger(*args, **kwargs):
    remote_block_uuid = kwargs.get('remote_block_uuid')
    pipeline_uuid = kwargs.get('remote_pipeline_uuid') or kwargs.get('pipeline_uuid')
    train = kwargs.get('train')
    trigger_pipeline_uuid = kwargs.get('trigger_pipeline_uuid')
    execution_partition = kwargs.get('execution_partition')
    sample = kwargs.get('sample')

    print('pipeline_uuid', pipeline_uuid)
    print('trigger_pipeline_uuid', trigger_pipeline_uuid)
    print('execution_partition', execution_partition)
    print('sample', sample)
    print('train', train)
    
    trigger_pipeline(
        trigger_pipeline_uuid, 
        check_status=False,
        error_on_failure=False,
        poll_interval=30,
        poll_timeout=None,
        remote_blocks=[
            dict(
                block_uuid=remote_block_uuid, 
                execution_partition=execution_partition, 
                pipeline_uuid=pipeline_uuid, 
                repo_path=get_repo_path(),
            ),
        ],
        return_remote_blocks=True,
        schedule_name=execution_partition,
        variables=dict(
            sample=sample,
            train=train,
        ),
        verbose=False,
    )