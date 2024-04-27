import os
from typing import List, Union

import numpy as np
import voyageai
from sklearn.decomposition import PCA

MAX_TEXTS = 128
MAX_TOKENS = 120_000


def create_embedding(tokens: List[str], max_dimensions: int = 16_000) -> List[int]:
    embedding_vectors = __fetch_embeddings(tokens)
    embedding_vector = __reduce_dimensions(embedding_vectors, max_dimensions=max_dimensions)
    return embedding_vector


def batch_embeddings(
    tokenized_chunks: List[List[str]],
    verbosity: int = 1,
) -> List[List[List[int]]]:
    client = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
    model = 'voyage-large-2'

    embeddings_grouped = []  # This will store embeddings for each chunk.
    api_calls = 0  # To keep track of API calls made.

    for chunk in tokenized_chunks:
        # Count the tokens in this chunk to check against MAX_TOKENS;
        # adjust logic if handling of large chunks is required.
        chunk_token_count = sum(len(token) for token in chunk)

        # Check if this chunk exceeds MAX_TOKENS; if it does, you might split it,
        # but for this example, we'll process it as is, assuming it fits.
        if chunk_token_count > MAX_TOKENS:
            # Handle large chunks appropriately here.
            print(f"Warning: Chunk with {chunk_token_count} tokens exceeds the limit and needs special handling.")
            continue  # Skip this chunk or split it as needed.

        embeddings_result = client.embed(chunk, model=model)
        api_calls += 1
        if verbosity >= 1:
            print(f'API calls {api_calls}, processed chunk with {chunk_token_count} tokens')

        # Assuming `.embed` returns embeddings in the same order as texts,
        # we can directly append the result for this chunk.
        embeddings_grouped.append(embeddings_result.embeddings)

    return embeddings_grouped


def __fetch_embeddings(tokens: List[str]) -> List[List[int]]:
    client = voyageai.Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
    model = 'voyage-large-2'
    embeddings_result = client.embed(tokens[:MAX_TEXTS], model=model)

    return embeddings_result.embeddings


def __reduce_dimensions(embedding_vectors: List[List[int]], max_dimensions: int = 16_000):
    """
    Reduces the dimensionality of a list of embedding vectors using PCA.

    Args:
        embedding_vectors (list): A list of numpy arrays representing the embedding vectors.
        max_dimensions (int): The maximum number of dimensions allowed in the output.

    Returns:
        numpy.ndarray: The concatenated and dimensionality-reduced embedding vector.
    """
    # Convert the list of embedding vectors into a 2D numpy array
    embedding_matrix = np.array(embedding_vectors)
    shape = embedding_matrix.shape

    if shape[0] * shape[1] > max_dimensions:
        # Calculate the target number of dimensions
        target_dimensions = min(
            round(max_dimensions / shape[0]),
            shape[0],
        )

        # Apply PCA to reduce the dimensionality
        pca = PCA(n_components=target_dimensions)
        reduced_embeddings = pca.fit_transform(embedding_matrix)
    else:
        reduced_embeddings = embedding_matrix

    # Flatten the reduced embeddings into a 1D vector
    concatenated_embeddings = reduced_embeddings.flatten()

    return concatenated_embeddings
