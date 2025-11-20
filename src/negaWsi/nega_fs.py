"""
NEGA with Side Information.
===========================

This module implements Non-Euclidean Gradient Algorithm by factorizing the latents
directly in the feature spaces.
"""

from typing import Tuple

import numpy as np

from negaWsi.base import NegaBase
from negaWsi.utils import svd


class NegaFS(NegaBase):
    """
    Matrix completion with side information following the Inductive
    Matrix Completion.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (X @ h1 @ h2 @ Y.T - M) ||_F^2
            + 0.5 * λg * || h1 ||_F^2
            + 0.5 * λd * || h2 ||_F^2

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (G ∈ R^{n x g}).
        disease_side_info (np.ndarray): Side information for diseases (D ∈ R^{m x d}).
        h1 (np.ndarray): Latent factor matrix for genes (g x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x d).

    """

    def __init__(
        self,
        *args,
        side_info: Tuple[np.ndarray, np.ndarray],
        svd_init: bool = False,
        **kwargs,
    ):
        """
        Initializes NegaFS model with side information.

        Args:
            side_info (Tuple[np.ndarray, np.ndarray]): Tuple containing
                (gene_feature_matrix, disease_feature_matrix).
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
        """
        super().__init__(*args, **kwargs)

        if side_info is None:
            raise ValueError("Side information must be provided for this session.")

        gene_feature_matrix, disease_feature_matrix = side_info

        if self.matrix.shape[0] != gene_feature_matrix.shape[0]:
            raise ValueError(
                "Number of rows in the matrix does not match gene features."
            )
        if self.matrix.shape[1] != disease_feature_matrix.shape[0]:
            raise ValueError(
                "Number of columns in the matrix does not match disease features."
            )

        self.gene_side_info = gene_feature_matrix
        self.disease_side_info = disease_feature_matrix

        if svd_init:
            # Masked matrix: only use observed training values
            observed_matrix = np.zeros_like(self.matrix)
            observed_matrix[self.train_mask] = self.matrix[self.train_mask]

            left_projection, right_projection = svd(observed_matrix, self.rank)

            # Backsolve for h1 and h2 using pseudoinverses
            # shape: (gene_feat_dim, rank)
            self.h1 = np.linalg.pinv(gene_feature_matrix) @ left_projection
            # shape: (rank, disease_feat_dim)
            self.h2 = right_projection @ np.linalg.pinv(disease_feature_matrix)
            method = "using masked TruncatedSVD"
        else:
            gene_feat_dim = self.gene_side_info.shape[1]
            disease_feat_dim = self.disease_side_info.shape[1]
            self.h1 = np.random.randn(gene_feat_dim, self.rank)
            self.h2 = np.random.randn(self.rank, disease_feat_dim)

            method = "with random weights"

        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s %s with side information",
            self.h1.shape,
            self.h2.shape,
            method,
        )

    @property
    def gene_latent(self) -> np.ndarray:
        """Compute gene latent matrix

        Returns:
            np.ndarray: The latent matrix. Shape is (k x n).
        """
        return self.gene_side_info @ self.h1
    
    @property
    def disease_latent(self) -> np.ndarray:
        """Compute disease latent matrix

        Returns:
            np.ndarray: The latent matrix. Shape is (k x m).
        """
        return self.h2 @ self.disease_side_info.T

    def init_tau(self) -> float:
        """
        Initialize tau value.

        Returns:
            float: tau value.
        """
        return np.linalg.norm(self.matrix, ord="fro") / 3

    def init_Wk(self) -> np.ndarray:
        """
        Initialize weight block matrix.

        Returns:
            np.ndarray: The weight block matrix.
        """
        return np.vstack([self.h1, self.h2.T])

    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix.
        """
        gene_feat_dim = self.h1.shape[0]
        self.h1 = weight_matrix[:gene_feat_dim, :]
        self.h2 = weight_matrix[gene_feat_dim:, :].T

    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Computes the value of the kernel function h for a given matrix W and
        regularization parameter tau.

        The h function is defined as:
            h(W) = 0.75 * (||X @ h1||_F^2 + ||h2 @ Y.T||_F^2)^2
            + 1.5 * tau * (||X @ h1||_F^2 + ||h2 @ Y.T||_F^2)

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
            M_pred = (X @ h1) @ (h2 @ Y.T)

        where:
        - X is the gene feature matrix (shape: n x g),
        - h1 is the left factor matrix (shape: g x rank),
        - h2 is the right factor matrix (shape: rank x d),
        - Y is the disease feature matrix (shape: m x d).

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        return self.gene_latent @ self.disease_latent

    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        with:
        - ∇_h1 = X.T @ (R @ (Y @ h2.T)) + λg * h1,
        - ∇_h2 = ((X @ h1).T @ R) @ Y + λd * h2

        with R = (B ⊙ ((X @ h1) @ (h2 @ Y.T) - M))

        Returns:
            np.ndarray: The gradient of the latents ((g+d) x rank)
        """
        residuals = self.calculate_training_residual()
        self.loss_terms["|| B ⊙ (X @ h1 @ h2 @ Y.T - M) ||_F"] = np.linalg.norm(
            residuals, ord="fro"
        )
        self.loss_terms["|| h1 ||_F"] = np.linalg.norm(self.h1, ord="fro")
        self.loss_terms["|| h2 ||_F"] = np.linalg.norm(self.h2, ord="fro")
        grad_h1 = (
            self.gene_side_info.T @ (residuals @ self.disease_latent.T)
            + self.regularization_parameters["λg"] * self.h1
        )
        grad_h2 = (
            (self.gene_latent.T @ residuals) @ self.disease_side_info
            + self.regularization_parameters["λd"] * self.h2
        )
        return np.vstack([grad_h1, grad_h2.T])
