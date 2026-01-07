# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Inductive Matrix Completion Using Alternating Minimization
with conjugate gradient method.
===============================================================

This module implements IMC Algorithm.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from negaWsi.alternating_methods.standard.scipy_minimize import AMD


class IMC(AMD):
    """
    Inductive Matrix Completion model using alternating minimization and
    conjugate gradient updates for the latent factors.

    Attributes:
        gene_side_info (np.ndarray): Side information for genes (G ∈ R^{n x g}).
        disease_side_info (np.ndarray): Side information for diseases (D ∈ R^{m x d}).
        h1 (np.ndarray): Latent factor matrix for genes (g x k).
        h2 (np.ndarray): Latent factor matrix for diseases (k x d).

    """

    def __init__(self, side_info: Tuple[np.ndarray, np.ndarray], **kwargs) -> None:
        """Initialize an IMC model with data, masks, side information, and hyperparameters.

        Args:
            side_info (Tuple[np.ndarray, np.ndarray]): Tuple containing
                (gene_feature_matrix, disease_feature_matrix).
        """
        super().__init__(method="CG", **kwargs)

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

    def h1_step(
        self, h1_flat: np.ndarray, loss: Dict[str, List[float]] = defaultdict(list)
    ) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h1.

        Args:
            h1_flat (np.ndarray): Flattened h1 matrix passed by the optimizer.
            loss (Dict[str, List[float]]): Dictionary with train and test losses history.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h1.
        """
        self.h1 = h1_flat.reshape(self.h1.shape)
        residuals, test_loss_step, rmse_step, training_loss_step = self.evaluate()
        loss["test"].append(test_loss_step)
        loss["rmse"].append(rmse_step)
        loss["training"].append(training_loss_step)
        self.logger.debug(
            ("[Euclidean Gradient Step h1] Loss: %.6e"),
            loss["training"][-1],
        )
        grad_h1 = (
            self.gene_side_info.T @ (residuals @ self.disease_latent.T)
            + self.regularization_parameters["λg"] * self.h1
        )
        return loss["training"][-1], grad_h1.ravel()

    def h2_step(
        self, h2_flat: np.ndarray, loss: Dict[str, List[float]] = defaultdict(list)
    ) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h2.

        Args:
            h2_flat (np.ndarray): Flattened h2 matrix passed by the optimizer.
            loss (Dict[str, List[float]]): Dictionary with train and test losses history.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h2.
        """
        self.h2 = h2_flat.reshape(self.h2.shape)
        residuals, test_loss_step, rmse_step, training_loss_step = self.evaluate()
        loss["rmse"].append(rmse_step)
        loss["test"].append(test_loss_step)
        loss["training"].append(training_loss_step)
        self.logger.debug(
            ("[Euclidean Gradient Step h2] Loss: %.6e"),
            loss["training"][-1],
        )
        grad_h2 = (
            self.gene_latent.T @ residuals
        ) @ self.disease_side_info + self.regularization_parameters["λd"] * self.h2
        return loss["training"][-1], grad_h2.ravel()
