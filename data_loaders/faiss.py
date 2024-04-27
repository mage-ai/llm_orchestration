import ast
import json
from typing import Dict, List, Tuple

import numpy as np


def __pad_vectors(vectors: List[List[float]]) -> List[List[float]]:
    # Find the maximum length among all vectors
    max_length = max(len(vector) for vector in vectors)

    print('max_length', max_length)

    # Pad each vector with zeros to match the maximum length
    padded_vectors = []
    for vector in vectors:
        padding_length = max_length - len(vector)
        padded_vector = np.pad(vector, (0, padding_length), mode='constant')
        padded_vectors.append(padded_vector)

    return padded_vectors


@data_loader
def load_data(*args, **kwargs):
    _driver, conn = kwargs.get('factory_items_mapping')['database/drivers']

    cur = conn.cursor()
    # Fetch stored embeddings, document IDs, and chunk texts
    cur.execute('SELECT chunk_text, document_id, vector, metadata FROM embeddings')
    rows = cur.fetchall()

    # Initialize lists
    chunk_texts = []
    document_ids = []
    vector_list = []
    metadata_list = []

    for row in rows:
        chunk_texts.append(row[0])
        document_ids.append(row[1])

        # Safely evaluate the string representation of the list into an actual list of floats
        vector = np.array(ast.literal_eval(row[2]), dtype='float32')
        vector_list.append(vector)

        metadata = row[3]
        metadata_list.append(json.loads(metadata) if isinstance(metadata, str) else metadata)

    # Stack all vectors into a NumPy array
    vectors = np.vstack(__pad_vectors(vector_list))

    return [
        vectors, 
        chunk_texts, 
        document_ids, 
        metadata_list,
    ]