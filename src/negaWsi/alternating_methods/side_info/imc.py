# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Inductive Matrix Completion Using Alternating Minimization
with conjugate gradient method.
===============================================================

This module implements IMC Algorithm.
"""
import logging
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize

from negaWsi.utils.early_stopping import EarlyStopping
from negaWsi.utils.flip_labels import FlipLabels
from negaWsi.utils.result import Result
from negaWsi.utils.utils import svd


class IMC:
    """
    Inductive Matrix Completion model using alternating minimization and
    conjugate gradient updates for the latent factors.

    Attributes:
        matrix (np.ndarray): Input matrix to be approximated. Shape: (n, m),
            where n is the number of genes and m is the number of diseases.
        train_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for training. Shape: (n, m).
        test_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for testing. Shape: (n, m).
        rank (int): The target rank for the low-rank approximation.
        regularization_parameters (float): Regularization parameters used in the optimization
            objective.
        iterations (int): Maximum number of optimization iterations.
        max_inner_iter (int): Maximum number of iterations for the inner optimization loop.
        h1 (np.ndarray): Left factor matrix in the low-rank approximation.
        h2 (np.ndarray): Right factor matrix in the low-rank approximation.
        logger (logging.Logger): Logger instance for debugging and monitoring training progress.
        seed (int): Seed for reproducible random initialization.
        save_name (str or pathlib.Path or None): File path where the model will be saved.
            If None, the model will not be saved after training.
        flip_labels (FlipLabels or None): Object that simulates label noise by randomly flipping
            a fraction of positive (1) entries to negatives (0) in the training mask.
        early_stopping (EarlyStopping or None): Mechanism for monitoring training loss and
            triggering early termination of training if the performance does not improve.
        loss_terms (Dict[str, float]): Logger for loss terms tracking.

    """

    def __init__(
        self,
        matrix: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        rank: int,
        regularization_parameters: Dict[str, float],
        iterations: int,
        max_inner_iter: int,
        side_info: Tuple[np.ndarray, np.ndarray],
        svd_init: bool = False,
        seed: int = 123,
        flip_labels: FlipLabels = None,
        early_stopping: EarlyStopping = None,
    ) -> None:
        """Initialize an IMC model with data, masks, side information, and hyperparameters.

        Args:
            matrix (np.ndarray): Input matrix to be approximated. Shape: (n, m),
                where n is the number of genes and m is the number of diseases.
            train_mask (np.ndarray): Mask indicating observed entries in `matrix` for training.
                Shape: (n, m).
            test_mask (np.ndarray): Mask indicating observed entries in `matrix` for testing.
                Shape: (n, m).
            rank (int): Desired rank for the low-rank approximation.
            regularization_parameters (Dict[str, float]): Regularization parameters for the optimization
                objective, expects keys like ``λg`` and ``λd``.
            iterations (int): Maximum number of outer optimization iterations.
            max_inner_iter (int): Maximum number of iterations for the inner optimization loop.
            side_info (Tuple[np.ndarray, np.ndarray]): Tuple containing
                (gene_feature_matrix, disease_feature_matrix).
            svd_init (bool, optional): Whether to initialize the latent
                matrices with SVD decomposition. Default to False.
            seed (int, optional): Seed for reproducible random initialization. Defaults to 123.
            flip_labels (FlipLabels, optional): Object that simulates label noise by randomly
                flipping a fraction of positive (1) entries to negatives (0) in the training mask.
            early_stopping (EarlyStopping, optional): Early stopping object that implements
                a mechanism for monitoring the validation loss and triggering early termination
                if performance does not improve.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matrix = matrix
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.rank = rank
        self.regularization_parameters = regularization_parameters
        self.iterations = iterations
        self.h1 = None
        self.h2 = None
        self.flip_labels = flip_labels
        self.early_stopping = early_stopping
        self.loss_terms = {}
        self.max_inner_iter = max_inner_iter
        self.logs = defaultdict(list)
        self.ith_iteration = 0

        # Set random seed for reproducibility
        np.random.seed(seed)

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

    def calculate_training_residual(self) -> np.ndarray:
        """
        Compute the training residual from the input matrix M (m x n), the model's prediction
        M_pred and the binary training mask. Optionally, if positive_flip_fraction is set and
        positive_flip_fraction = d, a fraction 'd' of the positive entries (ones) in M
        (where P is 1) is flipped to 0, yielding a modified label matrix L. Otherwise, L = M.

        The training residual R is computed as:
            R = (M_pred - L) ⊙  mask
        where ⊙ represents hadamard product.

        Returns:
            np.ndarray: The residual matrix R (m x n).
        """
        if self.flip_labels is not None:
            labels = self.flip_labels(self.matrix)
        else:
            labels = self.matrix
        residual = self.predict_all() - labels
        residual[~self.train_mask] = 0
        return residual

    def calculate_loss(self) -> float:
        """
        Computes the loss function value for the training data.

        The loss is defined as the Frobenius norm of the residual matrix
        for observed entries only:
            Loss = 0.5 * ||M - M_pred||_F^2

        (only over observed entries indicated by `mask`)

        where:
        - M is the input matrix (shape: m x n),
        - M_pred is the predicted (reconstructed) matrix

        Returns:
            float: The computed loss value.
        """
        residuals = self.calculate_training_residual()
        return 0.5 * np.linalg.norm(residuals, ord="fro") ** 2

    def calculate_rmse(self, mask: np.ndarray) -> float:
        """
        Computes the Root Mean Square Error (RMSE).

        The RMSE measures the average deviation between the predicted matrix
        and the ground truth matrix, considering only the observed entries.

        Formula for RMSE:
            RMSE = sqrt( (1 / |Omega|) * sum( (M[i, j] - M_pred[i, j])^2 ) )

        Where:
            - Omega: The set of observed indices in `mask`.
            - M: The matrix containing actual values.
            - M_pred: The predicted matrix (low-rank approximation).

        Process:
        1. Extract the observed entries from both the matrix (`M`)
           and the predicted matrix (`M_pred`) using the `mask`.
        2. Compute the squared differences for the observed entries.
        3. Calculate the mean of these squared differences.
        4. Take the square root of the mean to obtain the RMSE.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed entries.
        """
        actual_values = self.matrix[mask]
        predictions = self.predict_all()[mask]
        rmse = np.sqrt(((actual_values - predictions) ** 2).mean())
        return rmse

    def callback(self) -> None:
        """
        Callback to add logs or whatever the user wants compute between 'log_freq'
        number of outer loop iterations.
        """

    def objective_and_grad_h1(self, h1_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h1.

        Minimizes:
            0.5 ||M - X h1 h2 Y^T||_F^2 + 0.5 λg ||h1||_F^2
        where X is gene side info and Y is disease side info .

        Args:
            h1_flat (np.ndarray): Flattened h1 matrix passed by the optimizer.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h1.
        """
        self.h1 = h1_flat.reshape(self.h1.shape)
        residuals = self.calculate_training_residual()
        data_loss = 0.5 * np.linalg.norm(residuals, ord="fro") ** 2
        loss_reg = (
            0.5
            * self.regularization_parameters["λg"]
            * np.linalg.norm(self.h1, ord="fro")
        )
        loss = data_loss + loss_reg
        grad_h1 = (
            self.gene_side_info.T @ (residuals @ self.disease_latent.T)
            + self.regularization_parameters["λg"] * self.h1
        )
        self.logger.debug(
            ("[Euclidean Gradient Step h1] Loss: %.6e"),
            loss,
        )

        self.logs["test"].append(self.calculate_rmse(self.test_mask))

        loss_reg2 = (
            0.5
            * self.regularization_parameters["λd"]
            * np.linalg.norm(self.h2, ord="fro")
        )
        self.logs["training"].append(loss + loss_reg2)

        return loss, grad_h1.ravel()

    def objective_and_grad_h2(self, h2_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h2.

        Minimizes:
            0.5 ||M - X h1 h2 Y^T||_F^2 + 0.5 λd ||h2||_F^2
        where X is gene side info and Y is disease side info.

        Args:
            h2_flat (np.ndarray): Flattened h2 matrix passed by the optimizer.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h2.
        """
        self.h2 = h2_flat.reshape(self.h2.shape)
        residuals = self.calculate_training_residual()
        data_loss = 0.5 * np.linalg.norm(residuals, ord="fro") ** 2
        loss_reg = (
            0.5
            * self.regularization_parameters["λd"]
            * np.linalg.norm(self.h2, ord="fro")
        )
        loss = data_loss + loss_reg
        grad_h2 = (
            self.gene_latent.T @ residuals
        ) @ self.disease_side_info + self.regularization_parameters["λd"] * self.h2
        self.logger.debug(
            ("[Euclidean Gradient Step h2] Loss: %.6e"),
            loss,
        )

        self.logs["test"].append(self.calculate_rmse(self.test_mask))

        loss_reg2 = (
            0.5
            * self.regularization_parameters["λg"]
            * np.linalg.norm(self.h1, ord="fro")
        )
        self.logs["training"].append(loss + loss_reg2)

        return loss, grad_h2.ravel()

    def run(self, log_freq: int = 1) -> Result:
        """
        Train the IMC model by alternating conjugate gradient updates on h1 and h2.

        Args:
            log_freq (int, optional): Period at which to log data in Tensorboard.
                Default to 1 (iterations).

        Returns:
            Result: A dataclass containing:
                - completed_matrix: The reconstructed matrix (low-rank approximation).
                - loss_history: List of loss values at each iteration.
                - rmse_history: List of RMSE values at each iteration.
                - runtime: Total runtime of the optimization process.
                - iterations: Total number of iterations performed.
        """
        # Start measuring runtime
        start_time = time.time()

        self.logger.debug("Starting optimization")
        rmse_history = []
        loss_history = []

        # Main optimization loop
        self.ith_iteration = 0
        for ith_iteration in range(self.iterations):
            for key, value in self.loss_terms.items():
                self.logger.debug(
                    ("[Main Loop] %s: %.6e"),
                    key,
                    value,
                )

            res_h1 = minimize(
                fun=lambda h1_flat: self.objective_and_grad_h1(h1_flat),
                x0=self.h1.ravel(),
                method="CG",
                jac=True,
                options={"maxiter": self.max_inner_iter},
            )
            self.h1 = res_h1.x.reshape(self.h1.shape)

            res_h2 = minimize(
                fun=lambda h2_flat: self.objective_and_grad_h2(h2_flat),
                x0=self.h2.ravel(),
                method="CG",
                jac=True,
                options={"maxiter": self.max_inner_iter},
            )
            self.h2 = res_h2.x.reshape(self.h2.shape)
            W_k = np.vstack([self.h1, self.h2.T])
            self.logger.debug(
                (
                    "[Main Loop] Iteration %d:"
                    " RMSE=%.6e (testing), Mean Loss=%.6e (training)"
                ),
                ith_iteration,
                self.logs["test"][-1],
                self.logs["training"][-1],
            )
            if self.early_stopping is not None and self.early_stopping(
                self.logs["test"][-1], W_k
            ):
                weight_matrix = self.early_stopping.best_weights
                gene_feat_dim = self.h1.shape[0]
                self.h1 = weight_matrix[:gene_feat_dim, :]
                self.h2 = weight_matrix[gene_feat_dim:, :].T
                self.logger.debug("[Early Stopping] Training interrupted.")
                break
            if (ith_iteration + 1) % log_freq == 0 or ith_iteration == 0:
                self.callback()
                rmse_history.extend(self.logs["test"])
                loss_history.extend(self.logs["training"])
                self.logs["test"].clear()
                self.logs["training"].clear()
        self.callback()
        # Compute runtime
        runtime = time.time() - start_time
        self.logger.debug(
            "[Completion] Optimization finished in %.2f seconds.", runtime
        )
        training_data = Result(
            loss_history=loss_history,
            iterations=self.ith_iteration,
            rmse_history=rmse_history,
            runtime=runtime,
        )
        return training_data
