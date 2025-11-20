"""
NEGA with Side Information.
===========================

This module implements Non-Euclidean Gradient Algorithm by factorizing the latents
directly in the feature spaces with additional graph laplacian contraint on
the PPI data.
"""

import numpy as np

from negaWsi.nega_fs import NegaFS


class ENegaFS(NegaFS):
    """
    Matrix completion with side information following the Inductive
    Matrix Completion with additional graph laplacian contraint on
    the PPI data.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (X @ h1 @ h2 @ Y.T - M) ||_F^2
            + 0.5 * λg * || h1 ||_F^2
            + 0.5 * λd * || h2 ||_F^2
            + 0.5 * λd * Tr(h1.T @ X.T @ L @ X @ h1)

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (G ∈ R^{n x g}).
        disease_side_info (np.ndarray): Side information for diseases (D ∈ R^{m x d}).
        h1 (np.ndarray): Latent factor matrix for genes (g x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x d).
        laplacian (np.ndarray): Graph Laplacian (L ∈ R^{n x n})

    """

    def __init__(
        self,
        *args,
        ppi_adjacency: np.ndarray,
        **kwargs,
    ):
        """
        Initializes ENegaFS model with side information.

        Args:
            ppi_adjacency (np.ndarray): PPI graph adjacency matrix. Shape is (n x n).
        """
        super().__init__(*args, **kwargs)
        self.laplacian = np.diag(ppi_adjacency.sum(axis=1)) - ppi_adjacency

    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        with:
        - ∇_h1 = X.T @ (R @ (Y @ h2.T)) + λg * h1 + X.T @ L @ X @ h1
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
        graph_term = self.gene_side_info.T @ (
            self.laplacian @ self.gene_latent
        )
        self.loss_terms["Tr(h1.T @ X.T @ L @ X @ h1)"] = np.sum(self.h1 * graph_term)
        grad_h1 = (
            self.gene_side_info.T @ (residuals @ self.disease_latent.T)
            + self.regularization_parameters["λg"] * self.h1
            + self.regularization_parameters["λG"] * graph_term
        )
        grad_h2 = (
            (self.gene_latent.T @ residuals)
        ) @ self.disease_side_info + self.regularization_parameters["λd"] * self.h2
        return np.vstack([grad_h1, grad_h2.T])
