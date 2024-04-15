from typing import List


# def part_of_speech(nlp, chunks):
#     tokenized_chunks = []
#     for chunk in chunks:
#         doc = nlp(chunk)
#         tokens = [(token.text, token.pos_) for token in doc]
#         tokenized_chunks.append(tokens)
#     return tokenized_chunks


def tokenize_chunks(nlp, chunks):
    tokenized_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        tokens = [token.text for token in doc]
        tokenized_chunks.append(tokens)
    return tokenized_chunks


@transformer
def transform(chunks_for_documents: List[List[str]], *args, **kwargs):
    nlp = list(kwargs.get('factory_items_mapping').values())[0][0]
    """
    For every chunk in a document’s chunks, tokenize that 1 chunk. Then do it for all the other chunks. 
    Tokens can have multiple values for 1 chunk.

    Index level 0: document and it’s chunks
    Index level 1: single chunk

    Tokenization creates a List[str] for every chunk (str)
        Chunk = 'hey'
        Tokens = [...]

    Chunks are List[str], so tokenization will produce List[List[str]]
        Chunks = ['hey', 'yo]
        Tokens = [[...], [...]]
    """

    tokens_for_chunks_for_documents = [tokenize_chunks(nlp, chunks) for chunks in chunks_for_documents]

    print(len(chunks_for_documents[0]))  # The 1st document has 21 chunks
    print(len(tokens_for_chunks_for_documents[0][0]))  # The 1st chunk for the 1st document has 40 tokens

    return [
        tokens_for_chunks_for_documents,
    ]
