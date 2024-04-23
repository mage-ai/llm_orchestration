from typing import Dict, List, Union

from default_repo.llm_orchestration.models.topics import get_train_transform


@transformer
def transform(documents: List[List[Union[str, Dict]]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _ = factory_items_mapping['data_preparation/nlp']

    _lda_model, _dictionary, model_file_path, dictionary_file_path = get_train_transform(
        nlp,
        documents=[document[1] for document in documents],
        execution_partition=kwargs.get('execution_partition'),
        train=kwargs.get('train', 1) == 1,
    )

    return [
        [
            model_file_path,
            dictionary_file_path,
        ],
    ]
