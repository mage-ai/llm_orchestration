import os
import string
from contextlib import closing
from typing import List, Tuple

import faiss
import numpy as np

from mage_ai.settings.repo import get_repo_path

from default_repo.llm_orchestration.utils.chunking import chunk_sentences
from default_repo.llm_orchestration.utils.tokenization import embeddings_sum, named_entity_recognition_tokens


class Index:
    def __init__(self, index_path: str = None):
        self.index = None
        self.index_path = index_path or os.path.join(
            get_repo_path(),
            'assets', 
            'index', 
            'vectors',
            'index.faiss',
        )

    def search(self, query_vectors: List[List[float]], k: int = 12) -> Tuple:
        if not self.index:
            self.__load()

        D, I = self.index.search(np.array(query_vectors).astype('float32'), k)
        return D, I

    def __load(self):
        self.index = faiss.read_index(self.index_path)


@custom
def retrieve(*args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    driver, connection = factory_items_mapping['database/drivers']
    nlp, _ = factory_items_mapping['data_preparation/nlp']
    _, _, client = factory_items_mapping['embeddings/clients']

    documents_store_dir = kwargs.get(
        'documents_store',
        os.path.join(get_repo_path(), 'documents', 'store'),
    )

    k_nearest_neighbors = kwargs.get('k_nearest_neighbors', 2)
    query = kwargs.get('query', 'dynamic blocks')
    query = """
    charts for visualizing code
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

    faiss_index = Index()
    nodes = []

    # Chunking
    sentences = chunk_sentences(nlp(query))
    for chunk in sentences:
        print('-------------------------------')
        print(f'chunk: {chunk}')

        tokens_text, tokens_type = named_entity_recognition_tokens(nlp(chunk))
        print(f'tokens_text: {tokens_text}')
        print(f'tokens_type: {tokens_type}')
        
        vector_token_text = client.post(tokens_text)
        vector_token_type = client.post(tokens_type)
        vector = embeddings_sum([vector_token_text, vector_token_type])

        print(vector)
        
        D, I = faiss_index.search([vector])
        print(D, I)

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
