from typing import Dict, List, Union

from default_repo.llm_orchestration.models.topics import get_topic_for_text, get_train_transform
from default_repo.llm_orchestration.utils.chunking import sliding_window
from default_repo.llm_orchestration.utils.tokenization import standardize


@transformer
def transform(documents: List[List[str]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp, _ = factory_items_mapping['data_preparation/nlp']

    arr = []
    counter = 0
    for document_id, document, metadata in documents:
        print(f'document_id: {document_id}')

        data = get_train_transform(
            nlp,
            execution_partition=kwargs.get('execution_partition'),
            transform_text=document,
        )

        groups = data['tokens']
        print(f'groups: {len(groups)}')

        for idx, sentences in enumerate(groups):
            chunks = sliding_window(nlp('\n'.join(sentences)))
            print(f'chunks: {len(chunks)}')

            for idx1, chunk in enumerate(chunks):
                tokens = standardize(nlp(chunk))
                topic_data = None
                
                if tokens:
                    topic_data = get_topic_for_text(
                        data['model'], 
                        data['dictionary'],
                        tokens,
                    )
                
                if topic_data:
                    metadata['topic'] = ' '.join(
                        [w.replace('\n', ' ').strip() for w in topic_data['words']],
                    )
                else:
                    print(f'[WARNING] Tokens for chunk {idx1} is empty: {chunk}')
                    print(f'[WARNING] Skipping adding topic to metadata for document: {document_id}')

                arr.append([
                    document_id,
                    document,
                    metadata,
                    chunk,
                ])
        
        counter += 1
        print(f'{round(100 * counter / len(documents))}% ({counter}/{len(documents)})')

    return [
        arr,
    ]
