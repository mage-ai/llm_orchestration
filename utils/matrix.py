from typing import List, Union

import numpy as np

from default_repo.llm_orchestration.utils.tokenization import embeddings_concatenate
from default_repo.llm_orchestration.utils.tokenization import embeddings_max_pooling
from default_repo.llm_orchestration.utils.tokenization import embeddings_mean
from default_repo.llm_orchestration.utils.tokenization import embeddings_sum


def flatten(matrix: List[List[float]], dimensions: int) -> List[float]:
    """
    matrix = [[1, 2], [3, 4], [5, 6]]

    flatten(matrix, 9)

    => array([1., 2., 0., 3., 4., 0., 5., 6., 0.])
    """
    embedding = np.array(matrix)

    rows, cols = embedding.shape

    if rows > cols:
        max_row_len = rows
        max_col_len = int(dimensions / rows)
    else:
        max_row_len = int(dimensions / cols)
        max_col_len = cols

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

    return padded.flatten().astype(np.float64)


def aggregate(matrix: Union[np.array, List[List[float]]]) -> List[float]:
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    vector = embeddings_concatenate([
        embeddings_mean(matrix),
        embeddings_max_pooling(matrix),
    ])

    return vector
