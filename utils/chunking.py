import re
from typing import List

import spacy


def chunk_sentences(doc: spacy.tokens.doc.Doc) -> List[str]:
    return [sent.text.strip() for sent in doc.sents]


def sliding_window(
    doc: spacy.tokens.doc.Doc,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
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
        chunk_text = ' '.join(chunk_tokens)
        chunks.append(chunk_text)
        start = end - chunk_overlap
        end = start + chunk_size

    return chunks


def chunk_markdown(
    markdown_text: str,
    chunk_size: int = 512,
    overlap: int = 128,
) -> List[str]:
    # Split the Markdown document into sections based on headers
    sections = re.split(r'(#+\s)', markdown_text)

    chunks = []
    current_chunk = ""

    for section in sections:
        # Check if the section is a header
        if re.match(r'#+\s', section):
            # If the current chunk is not empty, add it to the list of chunks
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            # Split the section into sentences
            sentences = re.findall(r'[^.!?]+[.!?]', section)

            for sentence in sentences:
                # If adding the sentence to the current chunk exceeds the chunk size,
                # add the current chunk to the list of chunks and start a new chunk
                if len(current_chunk) + len(sentence) > chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = current_chunk[-overlap:] + sentence
                else:
                    current_chunk += sentence

    # Add the last chunk to the list of chunks
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
