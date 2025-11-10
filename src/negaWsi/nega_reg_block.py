# pylint: disable=C0103,R0914,R0801
"""
NEGA with Side Information as Regularization (NEGA-Reg-Block).
=============================================================

This module implements a two-phase (block-coordinate) optimization for
matrix completion under the NEGA framework with side information.
"""
from typing import Tuple

import numpy as np

from negaWsi.nega_reg import NegaReg
from negaWsi.utils import svd


class NegaRegBlock(NegaReg):
    """
    Matrix completion with side information following the formulation
    from GeneHound.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (h1 @ h2 - M) ||_F^2
            + 0.5 * λg * || P_n @ h1 - G @ β_G ||_F^2
            + 0.5 * λd * || h2 @ P_m - β_D.T @ D.T ||_F^2
            + 0.5 * λ_βg * || β_G ||_F^2
            + 0.5 * λ_βd * || β_D ||_F^2

        where:
            P_n = I_n - (1/n) * 1_n @ 1_n.T
            P_m = I_m - (1/m) * 1_m @ 1_m.T

        Optimization is performed in two alternating phases:
            Phase A (Latents): update h1, h2 with NEGA/gradient steps given current β_G, β_D.
            Phase B (Links): update β_G, β_D in closed form (ridge) given current h1, h2.

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

        self.i = 0
        nb_gene_features = gene_side_info.shape[1]
        nb_disease_features = disease_side_info.shape[1]
        self.beta_g = np.empty((nb_gene_features, self.rank))
        self.beta_d = np.empty((nb_disease_features, self.rank))
        self.compute_betas()

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
        return np.vstack([self.h1, self.h2.T])

    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix.
        """
        nb_genes, _ = self.matrix.shape
        self.h1 = weight_matrix[:nb_genes, :]
        self.h2 = weight_matrix[nb_genes:, :].T

    def compute_grad_f_W_k(self) -> np.ndarray:
        """
        Phase A (Latents):
            Compute the stacked gradient of the objective
            function w.r.t. (h1, h2) holding (β_G, β_D) fixed.:

        grad_f_W_k = (∇_h1, ∇_h2.T).T

        with:
            * ∇_h1 = R @ h2.T + λg * P_n @ (P_n @ h1 - G @ β_G)
            * ∇_h2 = h1.T @ R + λd * (h2 @ P_m - β_D.T @ D.T) @ P_m

        where
            R = B ⊙ (h1 @ h2 - M)
            P_n = I_n - (1/n) * 1_n @ 1_n.T
            P_m = I_m - (1/m) * 1_m @ 1_m.T

        Returns:
            np.ndarray: The gradient of the latents ((m+n) x rank)
        """
        residuals = (
            self.calculate_training_residual()
        )  # B ⊙ (h1 @ h2 - M) shape (nb_genes, nb_diseases)
        self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] = np.linalg.norm(
            residuals, ord="fro"
        )
        nb_genes, nb_diseases = self.matrix.shape

        # centering operators
        gene_centering = np.eye(nb_genes) - np.ones((nb_genes, nb_genes)) / nb_genes
        disease_centering = (
            np.eye(nb_diseases) - np.ones((nb_diseases, nb_diseases)) / nb_diseases
        )

        self.compute_betas()

        gene_prediction = self.gene_side_info @ self.beta_g  # G @ β_G (nb_genes, k)
        h1_residual_centered = (
            gene_centering @ self.h1 - gene_prediction
        )  # P_n @ h1 - G @ β_G (nb_genes, k)
        self.loss_terms["|| P_n @ h1 - G @ β_G ||_F"] = np.linalg.norm(
            h1_residual_centered, ord="fro"
        )
        self.loss_terms["|| β_G ||_F"] = np.linalg.norm(self.beta_g, ord="fro")

        disease_prediction = (
            self.beta_d.T @ self.disease_side_info.T
        )  # β_D.T @ D.T (k, nb_diseases)
        h2_residual_centered = (
            self.h2 @ disease_centering - disease_prediction
        )  # h2 @ P_m - β_D.T @ D.T (k, nb_diseases)
        self.loss_terms["|| h2 @ P_m - β_D.T @ D.T ||_F"] = np.linalg.norm(
            h2_residual_centered, ord="fro"
        )
        self.loss_terms["|| β_D ||_F"] = np.linalg.norm(self.beta_d, ord="fro")

        grad_h1 = (
            residuals @ self.h2.T
            + self.regularization_parameters["λg"]
            * gene_centering
            @ h1_residual_centered
        )
        grad_h2 = (
            self.h1.T @ residuals
            + self.regularization_parameters["λd"]
            * h2_residual_centered
            @ disease_centering
        )
        grad_Wk_next = np.vstack(
            [
                grad_h1,
                grad_h2.T,
            ]
        )
        return grad_Wk_next

    def compute_betas(self):
        """
        Phase B (Links): closed-form ridge updates for (β_G, β_D) given (h1, h2).

        Closed-form ridge regression solutions:
            β_G = (G.T @ G + (λ_βg/λg) * I)^(-1) @ G.T @ P_n @ H1
            β_D = (D.T @ D + (λ_βd/λd) * I)^(-1) @ D.T @ P_m @ H2.T

        where:
            P_n = I_n - (1/n) * 1_n @ 1_n.T
            P_m = I_m - (1/m) * 1_m @ 1_m.T
        """
        if self.i % 10 == 0:
            num_genes, num_diseases = self.matrix.shape
            P_genes = np.eye(num_genes) - np.ones((num_genes, num_genes)) / num_genes
            P_diseases = (
                np.eye(num_diseases)
                - np.ones((num_diseases, num_diseases)) / num_diseases
            )
            ridge_G = (
                self.gene_side_info.T @ self.gene_side_info
                + self.regularization_parameters["λ_βg"]
                / self.regularization_parameters["λg"]
                * np.eye(self.gene_side_info.shape[1])
            )
            ridge_D = (
                self.disease_side_info.T @ self.disease_side_info
                + self.regularization_parameters["λ_βd"]
                / self.regularization_parameters["λd"]
                * np.eye(self.disease_side_info.shape[1])
            )
            self.beta_g = np.linalg.solve(
                ridge_G, self.gene_side_info.T @ P_genes @ self.h1
            )
            self.beta_d = np.linalg.solve(
                ridge_D, self.disease_side_info.T @ P_diseases @ self.h2.T
            )
        self.i += 1
