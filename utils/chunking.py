from typing import List

import spacy


def chunk_sentences(doc: spacy.tokens.doc.Doc) -> List[str]:
    return [sent.text.strip() for sent in doc.sents]


def sliding_window(
    doc: spacy.tokens.doc.Doc,
    chunk_size: int = 256,
    chunk_overlap: int = 64,
) -> List[str]:
    """
    Split text into chunks using a sliding window approach.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The desired size of each chunk in tokens.
        chunk_overlap (int): The number of overlapping tokens between consecutive chunks.

    Returns:
        list: A list of text chunks.
    """
    tokens = [token.text for token in doc]
    chunks = []
    start = 0
    end = chunk_size

    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        start = end - chunk_overlap
        end = start + chunk_size

    return chunks
