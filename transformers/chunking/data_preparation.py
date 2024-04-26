from typing import Dict, List, Union

from default_repo.llm_orchestration.models.topics import get_topic_for_text, get_train_transform
from default_repo.llm_orchestration.utils.chunking import chunk_markdown, sliding_window
from default_repo.llm_orchestration.utils.tokenization import standardize


@transformer
def transform(documents: List[List[str]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp = factory_items_mapping['data_preparation/nlp'][0]

    print(len(documents))

    arr = []
    counter = 0
    for document_id, document, metadata in documents:
        print(f'document_id: {document_id}')
        
        chunks = chunk_markdown(document)
        print(f'chunks: {len(chunks)}')
        for chunk in chunks:
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