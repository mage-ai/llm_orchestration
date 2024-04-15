from typing import List


def sentences(nlp, chunks: List[str]):
    tokenized_chunks = []

    for chunk in chunks:
        doc = nlp(chunk)
        sentences = [sent.text for sent in doc.sents]
        tokenized_chunks += sentences

    return tokenized_chunks


def part_of_speech(nlp, chunks):
    tokenized_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [(token.text, token.pos_) for token in doc]
        tokenized_chunks.append(tokens)
    return tokenized_chunks


@transformer
def transform(chunks_arr: List[List[str]], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]

    return [
        part_of_speech(nlp, chunks_arr),
    ]