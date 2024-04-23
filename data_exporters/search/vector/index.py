import ast
import json
import os
from contextlib import closing
from typing import Dict, List, Union

import faiss
import numpy as np

from mage_ai.settings.repo import get_repo_path


def build_index(vectors: List[List[float]]):
    vectors = np.vstack(vectors)   # Stack all vectors into a NumPy array
    dim = vectors.shape[1]         # Number of dimensions of each vector
    nlist = min(100, len(vectors)) # This means that the index will partition the vector space into 100 distinct cells.

    # Create an IVF index
    quantizer = faiss.IndexFlatL2(dim)                 # Use the L2 distance for the quantizer
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # Initialize the IVF index

    print(f'vectors: {len(vectors)}')

    assert not index.is_trained
    index.train(vectors)  # Train the index
    assert index.is_trained

    index.add(vectors)  # Add vectors to the index

    path = os.path.join(get_repo_path(), 'vector_database', 'index')
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists

    # Write the index to disk
    print('Writing index to disk...')
    faiss.write_index(index, os.path.join(path, 'index.faiss'))


@data_exporter
def export_data(data, *args, **kwargs):
    _driver, connection = list(kwargs.get('factory_items_mapping').values())[0]

    error = None
    # This'll handle closing the connection
    with closing(connection) as conn:
        try:
            conn.autocommit = True

            # This'll handle the transaction and closing the cursor
            with conn, conn.cursor() as cur:
                cur.execute("""
                SELECT vector
                FROM embeddings
                ORDER BY 
                    chunk_text_hash ASC
                    , document_id_hash ASC
                """)
                rows = cur.fetchall()
                
                print(f'rows: {len(rows)}')

                build_index([json.loads(row[0]) for row in rows])
        except Exception as err:
            error = err

    if error:
        raise error