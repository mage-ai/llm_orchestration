from typing import List

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


def split_sentences(nlp, text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


@transformer
def transform(documents: List[List[str]], *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    nlp = factory_items_mapping['data_preparation/nlp'][0]

    chunked_documents = []

    for document_id, document in documents:
        print('document_id', document_id)

        sentences = split_sentences(nlp, document)
        print('sentences', len(sentences))

        for sentence in sentences:
            chunked_documents.append([
                document_id,
                document,
                sentence,
            ])

    return [
        chunked_documents,
    ]
