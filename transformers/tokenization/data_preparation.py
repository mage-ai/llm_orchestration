from typing import List

from default_repo.llm_orchestration.models.subword import get_train_transform


@transformer
def transform(documents: List[List[str]], model_file_path: str, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _ = factory_items_mapping['data_preparation/nlp']

    arr = []
    
    for document_id, document, metadata, chunk in documents:
        print(f'document_id: {document_id}')

        tokens = get_train_transform(
            nlp,
            execution_partition=kwargs.get('execution_partition'),
            transform_text=chunk,
        )['tokens']

        arr.append([
            document_id,
            document,
            metadata,
            chunk,
            tokens,
        ])

        print(f'{round(100 * len(arr) / len(documents))}% ({len(arr)}/{len(documents)})')

    return [
        arr,
    ]
