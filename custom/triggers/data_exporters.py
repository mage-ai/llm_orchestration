from mage_ai.orchestration.triggers.api import trigger_pipeline
from mage_ai.settings.repo import get_repo_path


@custom
def trigger(*args, **kwargs):
    execution_partition = kwargs.get('execution_partition')
    sample = kwargs.get('sample', 0)

    print('execution_partition', execution_partition)
    print('sample', sample)

    trigger_pipeline(
        'data_preparation_data_exporters', 
        check_status=False,
        error_on_failure=False,
        poll_interval=30,
        poll_timeout=None,
        schedule_name=execution_partition,
        verbose=False,
        remote_blocks=[
            dict(
                block_uuid='export/mapping/files', 
                execution_partition=execution_partition, 
                pipeline_uuid='data_preparation_transformer', 
                repo_path=get_repo_path(),
            ),
        ],
        return_remote_blocks=True,
        variables=dict(
            sample=sample,
        ),
    )