from typing import Tuple

import numpy as np
import scipy.linalg as linalg


def svd(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize low-rank factors from a truncated SVD of the matrix.

    Args:
        matrix (np.ndarray): Matrix of shape (n_rows, n_cols).
        rank (int): Target rank for the approximation.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - left_factor: shape (n_rows, rank)
            - right_factor: shape (rank, n_cols)
    """
    (
        left_singular_vectors,
        singular_values,
        right_singular_vectors_t,
    ) = linalg.svd(matrix, full_matrices=False)

    left_vectors_truncated = left_singular_vectors[:, :rank]
    singular_values_truncated = singular_values[:rank]
    right_vectors_t_truncated = right_singular_vectors_t[:rank, :]

    left_factor = left_vectors_truncated * np.sqrt(singular_values_truncated[np.newaxis, :])
    right_factor = np.sqrt(singular_values_truncated[:, np.newaxis]) * right_vectors_t_truncated

    return left_factor, right_factor
