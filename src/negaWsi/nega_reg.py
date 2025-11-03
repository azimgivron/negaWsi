# pylint: disable=C0103,R0914,R0801
"""
NEGA with Side Information as Regularization (NEGA-Reg).
=========================================================

This module implements Non-Euclidean Gradient Algorithm
with Side Information as Regularization.
"""
from typing import Tuple

import numpy as np

from negaWsi.base import NegaBase
from negaWsi.utils import svd


class NegaReg(NegaBase):
    """
    Matrix completion with side information following the formulation
    from GeneHound.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (h1 @ h2 - R) ||_F^2
            + 0.5 * λ * || P_n h1 - G @ β_G ||_F^2
            + 0.5 * λ * || h2 P_m - β_D.T @ D.T ||_F^2
            + 0.5 * λ' * || β_G ||_F^2
            + 0.5 * λ' * || β_D ||_F^2

        where:
            P_n = I_n - (1/n) * 1_n @ 1_n.T
            P_m = I_m - (1/m) * 1_m @ 1_m.T

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (G ∈ R^{n x g}).
        disease_side_info (np.ndarray): Side information for diseases (D ∈ R^{m x d}).
        h1 (np.ndarray): Latent factor matrix for genes (n x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x m).
        beta_g (np.ndarray): Link matrix for gene side information (g x k).
        beta_d (np.ndarray): Link matrix for disease side information (d x k).
    """

    def __init__(
        self,
        *args,
        side_info: Tuple[np.ndarray, np.ndarray],
        side_information_reg: float,
        svd_init: bool = False,
        **kwargs,
    ):
        """
        Initialize the NegaReg model with side information and regularization settings.

        Args:
            *args: Positional arguments forwarded to BaseNEGA.
            side_info (Tuple[np.ndarray, np.ndarray]):
                A tuple (G, D) of dense matrices containing gene side information
                G (n x g) and disease side information D (m x d).
            side_information_reg (float): Regularization weight for
                for the side information.
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            **kwargs: Additional keyword arguments forwarded to BaseNEGA.

        Raises:
            ValueError: If side_info is None or if matrix and side-info dimensions mismatch.
        """
        super().__init__(*args, **kwargs)

        self.side_information_reg = side_information_reg

        if side_info is None:
            raise ValueError("Side information must be provided.")
        gene_side_info, disease_side_info = side_info

        if self.matrix.shape[0] != gene_side_info.shape[0]:
            raise ValueError("Matrix rows and gene side info rows mismatch.")
        if self.matrix.shape[1] != disease_side_info.shape[0]:
            raise ValueError("Matrix columns and disease side info rows mismatch.")

        self.gene_side_info = gene_side_info
        self.disease_side_info = disease_side_info

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

        nb_gene_features = gene_side_info.shape[1]
        nb_disease_features = disease_side_info.shape[1]
        self.beta_g = np.random.randn(nb_gene_features, self.rank)
        self.beta_d = np.random.randn(nb_disease_features, self.rank)

        self.logger.debug(
            "Initialized h1 with shape %s and h2 with shape %s %s with side information",
            self.h1.shape,
            self.h2.shape,
            method,
        )

    def init_tau(self) -> float:
        """
        Initialize the tau parameter used in the kernel function.

        Returns:
            float: Initial tau value, set to ||X||_F / 3.
        """
        return np.linalg.norm(self.matrix, ord="fro") / 3

    def init_Wk(self) -> np.ndarray:
        """
        Initialize weight block matrix.

        Returns:
            np.ndarray: The weight block matrix.
        """
        return np.vstack([self.h1, self.h2.T, self.beta_g, self.beta_d])

    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix.
        """
        nb_genes, nb_diseases = self.matrix.shape
        gene_feat_dim = self.beta_g.shape[0]
        self.h1 = weight_matrix[0:nb_genes, :]
        self.h2 = weight_matrix[nb_genes : nb_genes + nb_diseases, :].T
        self.beta_g = weight_matrix[
            nb_genes + nb_diseases : nb_genes + nb_diseases + gene_feat_dim, :
        ]
        self.beta_d = weight_matrix[(nb_genes + nb_diseases + gene_feat_dim) :, :]

    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Compute the kernel function h(W) = 0.25 * ||W||_F^4 + 0.5 * tau * ||W||_F^2.

        Args:
            W (np.ndarray): Latent parameter matrix.
            tau (float): Regularization parameter.

        Returns:
            float: Kernel value.
        """
        norm = np.linalg.norm(W, ord="fro")
        return 0.25 * norm**4 + 0.5 * tau * norm**2

    def predict_all(self) -> np.ndarray:
        """
        Compute the full matrix reconstruction R_hat = h1 @ h2.

        Returns:
            np.ndarray: Reconstructed matrix of shape (n, m).
        """
        return self.h1 @ self.h2

    def compute_grad_f_W_k(self) -> np.ndarray:
        """
        Compute the stacked gradient of the objective function w.r.t. all variables:

        grad_f_W_k = (∇_h1, ∇_h2.T, ∇_beta_g, ∇_beta_d).T

        with:
            - ∇_h1 = R @ h2.T + λ * P_n @ (h1 - G @ β_G)
            - ∇_h2 = h1.T @ R + λ * (h2 - β_D.T @ D.T) @ P_m
            - ∇_beta_g = λ * G.T @ (P_n @ (h1 - G @ β_G)) + λ′ * β_G
            - ∇_beta_d = λ * D.T @ ((h2 - β_D.T @ D.T) @ P_m).T + λ′ * β_D

        where
            R = B ⊙ (h1 @ h2 - M)
            P_n = I_n - (1/n) * 1_n @ 1_n.T
            P_m = I_m - (1/m) * 1_m @ 1_m.T
        """
        residuals = self.calculate_training_residual()  # shape (nb_genes, nb_diseases)
        nb_genes, nb_diseases = self.matrix.shape

        gene_prediction = self.gene_side_info @ self.beta_g  # (nb_genes, k)
        h1_residual = self.h1 - gene_prediction  # (nb_genes, k)

        disease_prediction = (
            self.beta_d.T @ self.disease_side_info.T
        )  # (k, nb_diseases)
        h2_residual = self.h2 - disease_prediction  # (k, nb_diseases)

        # centering operators
        gene_centering = np.eye(nb_genes) - np.ones((nb_genes, nb_genes)) / nb_genes
        disease_centering = (
            np.eye(nb_diseases) - np.ones((nb_diseases, nb_diseases)) / nb_diseases
        )

        h1_residual_centered = gene_centering @ h1_residual  # P_n @ (h1 - G @ β_G)
        h2_residual_centered = (
            h2_residual @ disease_centering
        )  # (h2 - β_D.T @ D.T) @ P_m

        grad_h1 = (
            residuals @ self.h2.T + self.regularization_parameter * h1_residual_centered
        )
        grad_h2 = (
            self.h1.T @ residuals + self.regularization_parameter * h2_residual_centered
        )
        grad_beta_g = (
            self.regularization_parameter * self.gene_side_info.T @ h1_residual_centered
            + self.side_information_reg * self.beta_g
        )
        grad_beta_d = (
            self.regularization_parameter
            * self.disease_side_info.T
            @ h2_residual_centered.T
            + self.side_information_reg * self.beta_d
        )
        grad_Wk_next = np.vstack(
            [
                grad_h1,
                grad_h2.T,
                grad_beta_g,
                grad_beta_d,
            ]
        )
        return grad_Wk_next
