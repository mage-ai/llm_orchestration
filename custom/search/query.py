import os
import string
from contextlib import closing
from typing import List, Tuple

import faiss
import numpy as np

from mage_ai.settings.repo import get_repo_path

from default_repo.llm_orchestration.transformers.chunking.semantic import chunk_by_topic
from default_repo.llm_orchestration.transformers.topics import preprocess_sentences
from default_repo.llm_orchestration.transformers.tokenization.subword import (
    combo_tokens,
    named_entity_recognition,
    part_of_speech,
    preprocessing,
    subword_tokens,
)


class Index:
    def __init__(self, index_path: str = None):
        self.index = None
        self.index_path = index_path or os.path.join(
            get_repo_path(),
            'vector_database',
            'index',
            'index.faiss',
        )

    def search(self, query_vectors: List[List[float]], k: int = 12) -> Tuple:
        if not self.index:
            self.__load()

        D, I = self.index.search(np.array(query_vectors).astype('float32'), k)
        return D, I

    def __load(self):
        self.index = faiss.read_index(self.index_path)


import nltk.

def sentence_chunker(text, max_tokens_per_chunk=512):
    """
    Split text into chunks based on sentence boundaries.

    Args:
        text (str): The input text to be chunked.
        max_tokens_per_chunk (int): The maximum number of tokens allowed per chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(nltk.word_tokenize(sentence))
        if current_tokens + sentence_tokens > max_tokens_per_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


@custom
def retrieve(*args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    driver, connection = factory_items_mapping['database/drivers']
    nlp, stop_words = factory_items_mapping['data_preparation/nlp']
    _1, _2, openai = factory_items_mapping['embeddings/clients']

    documents_store_dir = kwargs.get(
        'documents_store',
        os.path.join(get_repo_path(), 'documents', 'store'),
    )

    k_nearest_neighbors = kwargs.get('k_nearest_neighbors', 2)
    query = kwargs.get('query', 'dynamic blocks')
    query = """
    how to clone blocks or replicate them
    """

    query = query.strip()

    if query:
        print(query)
    else:
        raise Exception('Query is required')

    error = None
    rows = []
    with closing(connection) as conn:
        try:
            conn.autocommit = True

            with conn, conn.cursor() as cur:
                # Fetch stored embeddings, document IDs, and chunk texts
                cur.execute("""
                SELECT chunk_text, metadata
                FROM embeddings
                ORDER BY
                    chunk_text_hash ASC
                    , document_id_hash ASC
                """)
                rows = cur.fetchall()
        except Exception as err:
            error = err
    if error:
        raise error

    model_file_path = '/home/src/default_repo/llm_orchestration/models/lda'
    dictionary_file_path = '/home/src/default_repo/llm_orchestration/dictionary/lda'
    subword_tokenizer = '/home/src/default_repo/llm_orchestration/models/subword_tokenizer'

    faiss_index = Index()
    nodes = []

    # Chunking
    sentences = sentence_chunker(query)
    for chunk in sentences:
        print('-------------------------------')
        print(f'chunk: {chunk}')

        tokens = combo_tokens(nlp, subword_tokenizer, stop_words, chunk)
        print(f'tokens: {tokens}')
        
        vector = openai.post(tokens)
        print(f'vector: {vector}')
        D, I = faiss_index.search([vector])

        with driver.session() as session:
            for i in range(k_nearest_neighbors):
                obj = I[0]
                if i >= len(obj):
                    i = -1
                index = obj[i]  # Index in the Faiss result to find the chunk index

                if index <= -1:
                    print(f'[WARNING] Nothing found')
                    continue

                row = rows[index]
                text, metadata = row

                print(f'index: {index}')
                print(metadata)
                print('==========')
                print(text)
                print('==========')

                # Execute a query using text to find its document via the relationship in the graph
                node = session.run("""
                MATCH (d:Document)-[:CONTAINS]->(c:Chunk {
                    text: $text
                })
                RETURN
                    c.text AS text
                    , d.document_id AS document_id
                """, text=text).single()

                if node:
                    document_id = node['document_id']
                    chunk_text = node['text']

                    print(f'document_id: {document_id}')

                    file_path = os.path.join(documents_store_dir, document_id)
                    with open(file_path, 'r') as f:
                        document = f.read()

                    nodes.append(dict(
                        chunk_text=chunk_text,
                        document=document,
                        document_id=document_id,
                    ))

        print('__________')
    # return [
    #     nodes,
    # ]
