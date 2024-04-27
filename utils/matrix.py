from typing import List

import numpy as np


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
