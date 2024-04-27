from typing import Dict, List


def chunk_by_text_rank(nlp, text: str):
    doc = nlp(text)
    key_phrases = [span.text for span in doc._.phrases]
    return key_phrases


def chunk_sentences(nlp, text: str) -> List[str]:
    doc = nlp(text)
    sentence_chunks = [sent.text.strip() for sent in doc.sents]
    return sentence_chunks


@transformer
def transform(documents: List[Dict], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]

    print('Documents: ', len(documents))

    return [
        [[
            chunk_sentences(nlp, document['text']),
            document['document_id'],
            document['document'],
        ] for document in documents],
    ]