import ast  # For safely evaluating strings containing Python literals
import json
from typing import Dict, List, Tuple

import faiss
import numpy as np


def create_faiss_index(conn) -> Tuple[int, List[str], List[str], List[Dict]]:
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

    # Number of dimensions of each vector
    dim = vectors.shape[1]

    # Create a Faiss Index for L2 distance
    index = faiss.IndexFlatL2(dim)
    # Add vectors to the Faiss index
    index.add(vectors)

    return index, chunk_texts, document_ids, metadata_list


def query_knowledge_graph(
    driver,
    query_vector: List[float],
    faiss_index: int,
    chunk_texts: List[str],
    k: int = 2,
) -> List[Dict]:
    # Search for top-k most similar vectors in Faiss
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), 12)

    # Retrieve corresponding documents or chunks from Neo4j
    results = []

    with driver.session() as session:
        for i in range(k):
            obj = I[0]
            if i >= len(obj):
                i = -1
            index = obj[i]  # Index in the Faiss result to find the chunk index

            chunk_text = chunk_texts[index]  # Getting chunk_text using the index

            # Execute a query using chunk_text to find its document via the relationship in the graph
            result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk {chunk_text: $chunk_text})
            RETURN c.chunk_text , d.document_id
            """, chunk_text=chunk_text).single()

            if result:
                results.append(result)

    return results


def __pad_vectors(vectors: List[List[float]]) -> List[List[float]]:
    # Find the maximum length among all vectors
    max_length = max(len(vector) for vector in vectors)

    # Pad each vector with zeros to match the maximum length
    padded_vectors = []
    for vector in vectors:
        padding_length = max_length - len(vector)
        padded_vector = np.pad(vector, (0, padding_length), mode='constant')
        padded_vectors.append(padded_vector)

    return padded_vectors
