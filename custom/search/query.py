import os
import string
from contextlib import closing
from typing import List, Tuple

import faiss
import numpy as np

from mage_ai.settings.repo import get_repo_path

from default_repo.llm_orchestration.utils.matrix import flatten


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
        self.__load()

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
    _, model = factory_items_mapping['models/transformers']

    documents_store_dir = kwargs.get(
        'documents_store',
        os.path.join(get_repo_path(), 'documents', 'store'),
    )

    k_nearest_neighbors = kwargs.get('k_nearest_neighbors', 10)
    query = kwargs.get('query', 'dynamic blocks')
    query = """
distributed computing
    """
    query = query.strip()

    error = None
    rows = []
    with closing(connection) as conn:
        try:
            conn.autocommit = True

            with conn, conn.cursor() as cur:
                # Fetch stored embeddings, document IDs, and chunk texts
                cur.execute("""
                SELECT 
                    chunk_text
                    , metadata
                FROM documents
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
    vector_dim = faiss_index.index.d
    print('Dimensionality of vectors in the FAISS index:', vector_dim)
    nodes = []

    # Chunking
    chunk = query
    
    matrix = model([chunk])['embeddings'][0]
    vector = flatten(matrix, vector_dim)
    

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

            print(D[0][i], index)

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

                file_path = os.path.join(documents_store_dir, document_id)
                with open(file_path, 'r') as f:
                    document = f.read()

                nodes.append(dict(
                    chunk_text=chunk_text,
                    document=document,
                    document_id=document_id,
                ))

                print(metadata['source_document_id'])
                print(chunk_text)
                print()
                print('-' * 40)
                print()

    # return [
    #     nodes,
    # ]
