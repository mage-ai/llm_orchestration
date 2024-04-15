from typing import List


def sentences(nlp, chunks: List[str]):
    tokenized_chunks = []

    for chunk in chunks:
        doc = nlp(chunk)
        sentences = [sent.text for sent in doc.sents]
        tokenized_chunks.append(sentences)

    return tokenized_chunks