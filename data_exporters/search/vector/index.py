import ast
import json
import os
from contextlib import closing
from typing import Dict, List, Union

import faiss
import hnswlib
import numpy as np
# import scann
from annoy import AnnoyIndex

from mage_ai.settings.repo import get_repo_path


def build_index_hnsw(vectors: List[Union[str, List[float], List[List[float]]]]):
    # HNSW (Hierarchical Navigable Small World graphs)
    matrix = False

    # Convert vectors to a format suitable for hnswlib
    processed_vectors = []
    for vector in vectors:
        if isinstance(vector, str):
            vector = json.loads(vector)

        if vector and isinstance(vector[0], list):
            matrix = True
            vector = np.mean(vector, axis=0)  # Flattening vector matrices by averaging

        processed_vectors.append(vector)

    if matrix:
        print("Matrix format detected, vectors were averaged to 1D.")

    vectors = np.stack(processed_vectors).astype(np.float32)  # Ensure all vectors are numpy arrays of type float32

    # Define the dimension of the vectors
    dim = vectors.shape[1]

    # Create a new HNSW index
    index = hnswlib.Index(space='l2', dim=dim)  # Use L2 distance; for cosine use 'cosine'

    # Initialize the index - specify max elements and the other hyperparameters
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)

    # Add items to the index
    index.add_items(vectors)

    # Set ef parameter to a higher value for more accurate but slower search
    index.set_ef(50)  # ef should always be > k

    # Define the path to save the index
    path = os.path.join(get_repo_path(), 'assets', 'index', 'vectors')
    os.makedirs(path, exist_ok=True)

    # Save the index to disk
    index_path = os.path.join(path, 'hnsw_index.bin')
    print('Writing index to disk...')
    index.save_index(index_path)


# def build_index_scann(vectors: List[Union[str, List[float], List[List[float]]]]):
#     """
#     ScaNN (Scalable Nearest Neighbors) by Google, which combines techniques like quantization,
#     vector decomposition and graph-based search to achieve high speed and accuracy
#     """

#     # Convert all vectors to a consistent numpy format
#     processed_vectors = []
#     for vector in vectors:
#         if isinstance(vector, str):
#             vector = json.loads(vector)
#         if isinstance(vector, list):
#             if isinstance(vector[0], list):  # Nested list
#                 vector = np.mean(vector, axis=0)  # Example strategy to handle matrix: take mean
#             vector = np.array(vector).astype(np.float32)
#         processed_vectors.append(vector)

#     vectors = np.vstack(processed_vectors)

#     print(f'Vectors: {len(vectors)}')

#     # Normalize vectors for Angular Distance when using ScaNN
#     normalized_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

#     # Create a ScaNN searcher
#     searcher = scann.ScannBuilder(normalized_vectors, 10, "dot_product").tree(
#         num_leaves=2000,  # Number of leaves in the partitioning tree
#         num_leaves_to_search=100,  # At query time, search this many leaves
#         training_sample_size=len(normalized_vectors)
#     ).score_ah(
#         2, anisotropic_quantization_threshold=0.2
#     ).reorder(100).build()  # Top 100 candidates are re-ordered according to the true distance

#     # Save the ScaNN searcher to disk
#     path = os.path.join(get_repo_path(), 'assets', 'index', 'vectors')
#     os.makedirs(path, exist_ok=True)

#     index_path = os.path.join(path, 'searcher.scann')
#     print('Writing searcher to disk...')
#     with open(index_path, 'wb') as f:
#         searcher.serialize(f)


def build_index_annoy(vectors: List[Union[str, List[float], List[List[float]]]], metric='angular', n_trees=10):
    """
    Annoy (Approximate Nearest Neighbors Oh Yeah) by Spotify
    """

    dim = None  # Dimensionality of the vectors

    # Initialize an empty Annoy Index
    if vectors and isinstance(vectors[0], (list, str)):
        sample_vector = json.loads(vectors[0]) if isinstance(vectors[0], str) else vectors[0]
        dim = len(sample_vector)

    if dim is None:
        raise ValueError("Unable to determine vector dimensionality.")

    index = AnnoyIndex(dim, metric)

    for i, vector in enumerate(vectors):
        if isinstance(vector, str):
            vector = json.loads(vector)
        index.add_item(i, vector)

    index.build(n_trees)

    path = os.path.join(get_repo_path(), 'assets', 'index', 'vectors')
    os.makedirs(path, exist_ok=True)

    # Write the index to disk
    print('Writing index to disk...')
    index_file_path = os.path.join(path, 'index.annoy')
    index.save(index_file_path)

    return index_file_path  # Return the path of the saved index


def build_index(vectors: List[Union[str, List[float], List[List[float]]]]):
    matrix = False

    if vectors:
        arr = []
        for vector in vectors:
            if isinstance(vector, str):
                vector = json.loads(vector)

            if vector and isinstance(vector[0], list):
                matrix = True
                vector = np.array(vector)

            arr.append(vector)

        vectors = arr

    print(f'vectors: {len(vectors)}')

    # Stack all vectors into a NumPy array
    if matrix:
        # Find max dimensions for padding
        max_row_len = max(max(embedding.shape) for embedding in vectors if len(embedding.shape) > 0)
        max_col_len = max(max(embedding.shape) for embedding in vectors if len(embedding.shape) > 1)

        padded_vectors = []
        for embedding in vectors:
            if embedding.ndim == 2:  # Matrix needs potentially row and column padding
                padded = np.pad(
                    embedding,
                    ((0, max_row_len - embedding.shape[0]), (0, max_col_len - embedding.shape[1])),
                    'constant',
                    constant_values=0,
                )
            else:  # Flat vector, only pad columns
                padded = np.pad(
                    embedding,
                    (0, max_col_len - embedding.shape[0]),
                    'constant',
                    constant_values=0,
                )

            padded = padded.reshape(1,-1).astype(np.float64)
            padded_vectors.append(padded)

        # Ensure it's a 2D array
        vectors = np.vstack(padded_vectors)
    else:
        vectors = np.vstack(vectors)

    dim = vectors.shape[1]  # The number of dimensions of each vector
    nlist = round((len(vectors) / 2)**0.5)

    # Create an IVF index
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)

    vectors_count = len(vectors)
    print(f'dimensions: {dim}')
    print(f'vectors:    {vectors_count}')

    assert not index.is_trained
    index.train(vectors.astype('float32'))  # Convert to float32 and train the index
    assert index.is_trained

    index.add(vectors.astype('float32'))  # Add vectors to the index

    path = os.path.join(get_repo_path(), 'assets', 'index', 'vectors')
    os.makedirs(path, exist_ok=True)

    # Write the index to disk
    print('Writing index to disk...')
    faiss.write_index(index, os.path.join(path, 'index.faiss'))


@data_exporter
def export_data(*args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, connection = factory_items_mapping['database/drivers']

    error = None
    
    # This'll handle closing the connection
    with closing(connection) as conn:
        try:
            conn.autocommit = True

            # This'll handle the transaction and closing the cursor
            with conn, conn.cursor() as cur:
                cur.execute("""
                SELECT
                    vector
                    , embeddings
                FROM documents
                ORDER BY 
                    chunk_text_hash ASC
                    , document_id_hash ASC
                """)
                rows = cur.fetchall()
                
                print(f'rows: {len(rows)}')

                build_index([row[1] or row[0] for row in rows])
        except Exception as err:
            error = err

    if error:
        raise error
