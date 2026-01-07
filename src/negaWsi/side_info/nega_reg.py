# pylint: disable=C0103,R0914,R0801
"""
NEGA with Side Information as Regularization (NEGA-Reg).
=========================================================

This module implements Non-Euclidean Gradient Algorithm
with Side Information as Regularization.
"""
from typing import Tuple

import numpy as np

from negaWsi.standard.base import NegaBase
from negaWsi.utils.utils import svd


class NegaReg(NegaBase):
    """
    Matrix completion with side information following the formulation
    from GeneHound.

    This model solves the following optimization problem:

        Minimize:
            0.5 * || B ⊙ (h1 @ h2 - M) ||_F^2
            + 0.5 * λg * || h1 - G @ β_G ||_F^2
            + 0.5 * λd * || h2 - β_D.T @ D.T ||_F^2
            + 0.5 * λ_βg * || β_G ||_F^2
            + 0.5 * λ_βd * || β_D ||_F^2

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
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            **kwargs: Additional keyword arguments forwarded to BaseNEGA.

        Raises:
            ValueError: If side_info is None or if matrix and side-info dimensions mismatch.
        """
        super().__init__(*args, **kwargs)

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

    def calculate_loss(self) -> float:
        """
        Computes the loss function value for the training data.

        The loss is defined as the Frobenius norm of the residual matrix
        for observed entries only:
            Loss = 0.5 * || B ⊙ (h1 @ h2 - M) ||_F^2
            + 0.5 * λg * || h1 - G @ β_G ||_F^2
            + 0.5 * λd * || h2 - β_D.T @ D.T ||_F^2
            + 0.5 * λ_βg * || β_G ||_F^2
            + 0.5 * λ_βd * || β_D ||_F^2

        Returns:
            float: The computed loss value.
        """
        residuals = self.calculate_training_residual()
        self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] = np.linalg.norm(
            residuals, ord="fro"
        )

        gene_prediction = self.gene_side_info @ self.beta_g  # G @ β_G (nb_genes, k)
        h1_residual = self.h1 - gene_prediction  # h1 - G @ β_G (nb_genes, k)
        self.loss_terms["|| h1 - G @ β_G ||_F"] = np.linalg.norm(h1_residual, ord="fro")
        self.loss_terms["|| β_G ||_F"] = np.linalg.norm(self.beta_g, ord="fro")

        disease_prediction = (
            self.beta_d.T @ self.disease_side_info.T
        )  # β_D.T @ D.T (k, nb_diseases)
        h2_residual = self.h2 - disease_prediction  # h2 - β_D.T @ D.T (k, nb_diseases)

        self.loss_terms["|| h2 - β_D.T @ D.T ||_F"] = np.linalg.norm(
            h2_residual, ord="fro"
        )
        self.loss_terms["|| β_D ||_F"] = np.linalg.norm(self.beta_d, ord="fro")

        loss = 0.5 * (
            self.loss_terms["|| B ⊙ (h1 @ h2 - M) ||_F"] ** 2
            + self.regularization_parameters["λg"]
            * self.loss_terms["|| h1 - G @ β_G ||_F"] ** 2
            + self.regularization_parameters["λd"]
            * self.loss_terms["|| h2 - β_D.T @ D.T ||_F"] ** 2
            + self.regularization_parameters["λ_βg"]
            * self.loss_terms["|| β_G ||_F"] ** 2
            + self.regularization_parameters["λ_βd"]
            * self.loss_terms["|| β_D ||_F"] ** 2
        )
        return loss

    def predict_all(self) -> np.ndarray:
        """
        Compute the full matrix reconstruction R_hat = h1 @ h2.

        Returns:
            np.ndarray: Reconstructed matrix of shape (n, m).
        """
        return self.h1 @ self.h2

    def compute_grad_f_W(self) -> np.ndarray:
        """
        Compute the stacked gradient of the objective function w.r.t. all variables:

        grad_f_W_k = (∇_h1, ∇_h2.T, ∇_beta_g, ∇_beta_d).T

        with:
            * ∇_h1 = R @ h2.T + λg * (h1 - G @ β_G)
            * ∇_h2 = h1.T @ R + λd * (h2 - β_D.T @ D.T)
            * ∇_beta_g = -λg * G.T @ (h1 - G @ β_G) + λ_βg * β_G
            * ∇_beta_d = -λd * D.T @ (h2 - β_D.T @ D.T).T + λ_βd * β_D

        where
            R = B ⊙ (h1 @ h2 - M)

        Returns:
            np.ndarray: The gradient of the latents ((m+n+g+d) x rank)
        """
        residuals = (
            self.calculate_training_residual()
        )  # B ⊙ (h1 @ h2 - M) shape (nb_genes, nb_diseases)

        gene_prediction = self.gene_side_info @ self.beta_g  # G @ β_G (nb_genes, k)
        h1_residual = self.h1 - gene_prediction  # h1 - G @ β_G (nb_genes, k)
        disease_prediction = (
            self.beta_d.T @ self.disease_side_info.T
        )  # β_D.T @ D.T (k, nb_diseases)
        h2_residual = self.h2 - disease_prediction  # h2 - β_D.T @ D.T (k, nb_diseases)

        grad_h1 = (
            residuals @ self.h2.T + self.regularization_parameters["λg"] * h1_residual
        )
        grad_h2 = (
            self.h1.T @ residuals + self.regularization_parameters["λd"] * h2_residual
        )
        grad_beta_g = (
            -self.regularization_parameters["λg"] * self.gene_side_info.T @ h1_residual
            + self.regularization_parameters["λ_βg"] * self.beta_g
        )
        grad_beta_d = (
            -self.regularization_parameters["λd"]
            * self.disease_side_info.T
            @ h2_residual.T
            + self.regularization_parameters["λ_βd"] * self.beta_d
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
