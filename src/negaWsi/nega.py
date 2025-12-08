"""
NEGA Module
==================

This module implements The Standard Non-Euclidean Gradient Algorithm for matrix completion..
"""

import numpy as np

from negaWsi.base import NegaBase
from negaWsi.utils import svd


class Nega(NegaBase):
    """
    Matrix completion based on the Standard Non-Euclidean Gradient Algorithm.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (h1 @ h2 - M) ||_F^2
            + 0.5 * λg * || h1 ||_F^2
            + 0.5 * λd * || h2 ||_F^2

    Attributes:
        h1 (np.ndarray): Latent factor matrix for genes (n x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x m).

    """

    def __init__(self, *args, svd_init: bool = False, **kwargs):
        """
        Initializes the session without side information.

        Args:
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
        """
        super().__init__(*args, **kwargs)

        if svd_init:
            # Apply the train mask: unobserved entries are set to zero
            observed_matrix = np.zeros_like(self.matrix)
            observed_matrix[self.train_mask] = self.matrix[self.train_mask]

            self.h1, self.h2 = svd(observed_matrix, self.rank)
            method = "using masked TruncatedSVD"
        else:
            nb_genes, nb_diseases = self.matrix.shape
            self.h1 = np.random.randn(nb_genes, self.rank)
            self.h2 = np.random.randn(self.rank, nb_diseases)
            method = "with random weights"

        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s %s",
            self.h1.shape,
            self.h2.shape,
            method,
        )

    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Computes the value of the kernel function h for a given matrix W and
        regularization parameter tau.

        The h function is defined as:
            h(W) = 0.25 * ||W||_F^4 + 0.5 * tau * ||W||_F^2

        Args:
            W (np.ndarray): The input matrix.
            tau (float): Regularization parameter.

        Returns:
            float: The computed value of the h function.
        """
        norm = np.linalg.norm(W, ord="fro")
        h_value = 0.25 * norm**4 + 0.5 * tau * norm**2
        return h_value

    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = h1 @ h2

        where:
        - h1 is the left factor matrix (shape: n x rank),
        - h2 is the right factor matrix (shape: rank x m).

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        return self.h1 @ self.h2

    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        where:
        - ∇_h1 = R @ h2.T + λg * h1,
        - ∇_h2 = h1.T @ R + λd * h2

        with R = (B ⊙ (h1 @ h2 - M))

        Returns:
            np.ndarray: The gradient of the latents ((n+m) x rank)
        """
        residuals = self.calculate_training_residual()
        self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] = np.linalg.norm(
            residuals, ord="fro"
        )
        self.loss_terms["|| h1 ||_F"] = np.linalg.norm(self.h1, ord="fro")
        self.loss_terms["|| h2 ||_F"] = np.linalg.norm(self.h2, ord="fro")

        grad_h1 = residuals @ self.h2.T + self.regularization_parameters["λg"] * self.h1
        grad_h2 = self.h1.T @ residuals + self.regularization_parameters["λd"] * self.h2
        return np.vstack([grad_h1, grad_h2.T])
