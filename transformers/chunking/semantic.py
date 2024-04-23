from typing import Dict, List, Union

from default_repo.llm_orchestration.models.topics import get_train_transform
from default_repo.llm_orchestration.utils.chunking import sliding_window


def get_topic_for_text(model, dictionary, text: str, words_per_topic: int = 8) -> Dict:
    chunk_bow = dictionary.doc2bow([text])
    representation = model[chunk_bow]

    topic_words = []
    for topic in model.print_topics(num_words=words_per_topic):
        words = []
        for word in topic[1].split('+'):
            parts = word.split('*')
            words.append(parts[-1].strip().strip('"'))
        topic_words.append(words)

    topic_id, probability = sorted(representation, key=lambda t: t[1], reverse=True)[0]

    return dict(
        id=topic_id,
        probability=probability,
        words=topic_words[topic_id],
    )


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

        topics = data['tokens']
        model = data['model']
        dictionary = data['dictionary']

        print(f'document_id: {document_id}')

        counter = 0
        for sentences in topics:
            chunks = sliding_window(nlp('\n'.join(sentences)))
            print(f'chunks: {len(chunks)}')

            for chunk in chunks:
                topic_data = get_topic_for_text(
                    model, 
                    dictionary,
                    chunk,
                )

                print(' '.join([w.replace('\n', ' ').strip() for w in topic_data['words']]))

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
