from typing import Dict, List, Union

from default_repo.llm_orchestration.models.topics import get_topic_for_text, get_train_transform
from default_repo.llm_orchestration.utils.chunking import sliding_window
from default_repo.llm_orchestration.utils.tokenization import standardize


@transformer
def transform(documents: List[List[str]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _ = factory_items_mapping['data_preparation/nlp']

    arr = []
    for document_id, document, metadata in documents:
        data = get_train_transform(
            nlp,
            execution_partition=kwargs.get('execution_partition'),
            transform_text=document,
        )

        print(f'document_id: {document_id}')

        counter = 0
        for sentences in data['tokens']:
            chunks = sliding_window(nlp('\n'.join(sentences)))
            print(f'chunks: {len(chunks)}')

            for chunk in chunks:
                topic_data = get_topic_for_text(
                    data['model'], 
                    data['dictionary'],
                    standardize(nlp(chunk)),
                )
                metadata['topic'] = ' '.join(
                    [w.replace('\n', ' ').strip() for w in topic_data['words']],
                )

                arr.append([
                    document_id,
                    document,
                    metadata,
                    chunk,
                ])

                counter += 1
                print(f'{counter}/{len(chunks)}')

    return [
        arr,
    ]
