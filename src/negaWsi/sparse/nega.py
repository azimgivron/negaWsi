"""
NEGA Module
==================

This module implements The Standard Non-Euclidean Gradient Algorithm for matrix completion..
"""

import numpy as np
import scipy.sparse as sp

from negaWsi.sparse.base import NegaBase
from negaWsi.utils.utils import svd


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
            raise NotImplementedError(
                "SVD initialization not implemented for sparse matrices."
            )
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

    def calculate_loss(self) -> float:
        """
        Computes the loss function value for the training data.

        The loss is defined as the Frobenius norm of the residual matrix
        for observed entries only:
            Loss = 0.5 * || B ⊙ (h1 @ h2 - M) ||_F^2 + 0.5 * λg * || h1 ||_F^2 + 0.5 * λd * || h2 ||_F^2

        Returns:
            float: The computed loss value.
        """
        residuals = self.calculate_residual_entries()
        self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] = np.linalg.norm(residuals)
        self.loss_terms["|| h1 ||_F"] = np.linalg.norm(self.h1, ord="fro")
        self.loss_terms["|| h2 ||_F"] = np.linalg.norm(self.h2, ord="fro")

        loss = 0.5 * (
            self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] ** 2
            + self.regularization_parameters["λg"] * self.loss_terms["|| h1 ||_F"] ** 2
            + self.regularization_parameters["λd"] * self.loss_terms["|| h2 ||_F"] ** 2
        )
        return loss

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

    def predict_entries(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Predict entry R_ij

        Args:
            i (np.ndarray): The row index.
            j (np.ndarray): The column index.

        Returns:
            np.ndarray: The prediction ij.
        """
        return np.sum(self.h1[i, :] * self.h2[:, j].T, axis=1)

    def compute_grad_f_W(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        where:
        - ∇_h1 = R @ h2.T + λg * h1,
        - ∇_h2 = h1.T @ R + λd * h2

        with R = (B ⊙ (h1 @ h2 - M))

        Returns:
            np.ndarray: The gradient of the latents ((n+m) x rank)
        """
        residuals = self.calculate_residual_entries()

        grad_h1 = np.zeros_like(self.h1)
        grad_h2 = np.zeros_like(self.h2)

        # Accumulate per-observation contributions
        # grad_h1[i_t, :] += residuals_t * h2[:, j_t]^T
        np.add.at(
            grad_h1, self.train_i, residuals[:, None] * self.h2[:, self.train_j].T
        )
        # grad_h2[:, j_t] += residuals_t * h1[i_t, :]^T
        np.add.at(
            grad_h2.T, self.train_j, residuals[:, None] * self.h1[self.train_i, :]
        )

        grad_h1 += self.regularization_parameters["λg"] * self.h1
        grad_h2 += self.regularization_parameters["λd"] * self.h2
        return np.vstack([grad_h1, grad_h2.T])
