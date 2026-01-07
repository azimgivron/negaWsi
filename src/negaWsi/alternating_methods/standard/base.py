# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Alternating Gradient Descent Algorithm for Matrix Completion
===============================================================

This module implements the template for Alternating Gradient Descent Algorithm.
"""
import abc
import logging
import time
from typing import Dict, Tuple, List
from collections import defaultdict

import numpy as np

from negaWsi.utils.early_stopping import EarlyStopping
from negaWsi.utils.result import Result

class McSolver(metaclass=abc.ABCMeta):
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    The class is designed for scenarios where the objective is to approximate a
    partially observed matrix by a low-rank factorization. The problem is of
    the form
    
    Minimizes:
            0.5 ||M - h1 @ h2||_F^2 + 0.5 λg ||h1||_F^2 + 0.5 λd ||h2||_F^2

    Attributes:
        matrix (np.ndarray): Input matrix to be approximated. Shape: (n, m),
            where n is the number of genes and m is the number of diseases.
        train_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for training. Shape: (n, m).
        test_mask (np.ndarray): Boolean mask indicating observed entries in `matrix`
            for testing. Shape: (n, m).
        rank (int): The target rank for the low-rank approximation.
        regularization_parameters (Dict[str, float]): Regularization parameters used in the optimization
            objective.
        iterations (int): Maximum number of optimization iterations.
        h1 (np.ndarray): Left factor matrix in the low-rank approximation. Shape: (n, rank).
        h2 (np.ndarray): Right factor matrix in the low-rank approximation. Shape: (rank, m).
        logger (logging.Logger): Logger instance for debugging and monitoring training progress.
        seed (int): Seed for reproducible random initialization.
        early_stopping (EarlyStopping or None): Mechanism for monitoring training loss and
            triggering early termination of training if the performance does not improve.

    """

    def __init__(
        self,
        matrix: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        rank: int,
        regularization_parameters: Dict[str, float],
        iterations: int,
        seed: int = 123,
        early_stopping: EarlyStopping = None,
        max_inner_iter: int = 100
    ):
        """
        Initializes the AGD instance with the provided configuration
        parameters for matrix approximation.

        Args:
            matrix (np.ndarray): Input matrix to be approximated. Shape: (n, m),
                where n is the number of genes and m is the number of diseases.
            train_mask (np.ndarray): Mask indicating observed entries in `matrix` for training.
                Shape: (n, m).
            test_mask (np.ndarray): Mask indicating observed entries in `matrix` for testing.
                Shape: (n, m).
            rank (int): Desired rank for the low-rank approximation.
            regularization_parameters (Dict[str, float]): Regularization parameters for the optimization
                objective.
            iterations (int): Maximum number of optimization iterations.
            seed (int, optional): Seed for reproducible random initialization. Defaults to 123.
            early_stopping (EarlyStopping, optional): Early stopping object that implements
                a mechanism for monitoring the validation loss and triggering early termination
                if performance does not improve.
        """
        self.matrix = matrix
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.rank = rank
        self.regularization_parameters = regularization_parameters
        self.iterations = iterations
        self.early_stopping = early_stopping
        self.max_inner_iter = max_inner_iter

        # Set random seed for reproducibility
        np.random.seed(seed)

        self.logger = logging.getLogger(self.__class__.__name__)

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

    @abc.abstractmethod
    def step(self, loss: Dict[str, List[float]]) -> np.ndarray:
        """_summary_

        Args:
            loss (Dict[str, List[float]]): _description_

        Returns:
            np.ndarray: _description_
        """

    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Mathematically, the completed matrix is computed as:
            M_pred = h1 @ h2

        where:
        - h1 is the left factor matrix (shape: n x rank),
        - h2 is the right factor matrix (shape: m x rank),

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        return self.gene_latent @ self.disease_latent

    def evaluate(self) -> Tuple[float, float, float]:
        residuals = self.calculate_training_residual()
        data_loss = 0.5 * np.linalg.norm(residuals, ord="fro") ** 2
        loss_reg_h1 = (
            0.5
            * self.regularization_parameters["λg"]
            * np.linalg.norm(self.h1, ord="fro")
        )
        loss_reg_h2 = (
            0.5
            * self.regularization_parameters["λd"]
            * np.linalg.norm(self.h2, ord="fro")
        )
        test_loss = self.calculate_rmse(self.test_mask)
        training_loss = data_loss + loss_reg_h1 + loss_reg_h2
        return residuals, test_loss, training_loss

    def h1_step(self, h1_flat: np.ndarray, loss: Dict[str, List[float]] = defaultdict(list)) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h1.

        Args:
            h1_flat (np.ndarray): Flattened h1 matrix passed by the optimizer.
            loss (Dict[str, List[float]]): Dictionary with train and test losses history.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h1.
        """
        self.h1 = h1_flat.reshape(self.h1.shape)
        residuals, test_loss, training_loss = self.evaluate()
        loss["test"].append(test_loss)
        loss["training"].append(training_loss)
        self.logger.debug(
            ("[Euclidean Gradient Step h1] Loss: %.6e"),
            loss["training"][-1],
        )
        grad_h1 = (
            residuals @ self.disease_latent.T
            + self.regularization_parameters["λg"] * self.h1
        )
        return loss["training"][-1], grad_h1.ravel()

    def h2_step(self, h2_flat: np.ndarray, loss: Dict[str, List[float]] = defaultdict(list)) -> Tuple[float, np.ndarray]:
        """
        Compute objective and gradient for h2.

        Args:
            h2_flat (np.ndarray): Flattened h2 matrix passed by the optimizer.
            loss (Dict[str, List[float]]): Dictionary with train and test losses history.

        Returns:
            Tuple[float, np.ndarray]: The scalar loss and the flattened gradient for h2.
        """
        self.h2 = h2_flat.reshape(self.h2.shape)
        residuals, test_loss, training_loss = self.evaluate()
        loss["test"].append(test_loss)
        loss["training"].append(training_loss)
        self.logger.debug(
            ("[Euclidean Gradient Step h2] Loss: %.6e"),
            loss["training"][-1],
        )
        grad_h2 = (
            self.gene_latent.T @ residuals 
            + self.regularization_parameters["λd"] * self.h2
        )
        return loss["training"][-1], grad_h2.ravel()

    @property
    def gene_latent(self) -> np.ndarray:
        """Compute gene latent matrix

        Returns:
            np.ndarray: The latent matrix. Shape is (n x k).
        """
        return self.h1

    @property
    def disease_latent(self) -> np.ndarray:
        """Compute disease latent matrix

        Returns:
            np.ndarray: The latent matrix. Shape is (k x m).
        """
        return self.h2

    def calculate_training_residual(self) -> np.ndarray:
        """
        The training residual R is computed as:
            R = B ⊙ (M_pred - M)
        where ⊙ represents hadamard product.

        Returns:
            np.ndarray: The residual matrix R (n x m).
        """
        residual = self.predict_all() - self.matrix
        residual[~self.train_mask] = 0
        return residual

    def calculate_rmse(self, mask: np.ndarray) -> float:
        """
        Computes the Root Mean Square Error (RMSE).

        Args:
            mask (np.ndarray): The mask to apply.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed entries.
        """
        actual_values = self.matrix[mask]
        predictions = self.predict_all()[mask]
        rmse = np.sqrt(((actual_values - predictions) ** 2).mean())
        return rmse

    def callback(
        self,
        ith_iteration: int,
        training_loss: np.ndarray,
        testing_loss: np.ndarray,
    ):
        """
        Callback to add logs or whatever the user wants compute between 'log_freq'
        number of outer loop iterations.

        Args:
            ith_iteration (int): The current iteration index at which the logging is performed.
            training_loss (np.ndarray): The computed loss value for the training dataset at the
                current iteration.
            testing_loss (np.ndarray): The computed loss value for the testing dataset at the
                current iteration.
        """

    def run(self, log_freq: int = 1) -> Result:
        """
        Train the AGD model by alternating gradient updates on h1 and h2.

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
        _, test_loss, training_loss = self.evaluate()
        rmse_history = [test_loss]
        loss_history = [training_loss]
        loss = defaultdict(list)

        # Main optimization loop
        self.ith_iteration = 0
        for ith_iteration in range(self.iterations):
            W_k = self.step(loss)
            self.logger.debug(
                (
                    "[Main Loop] Iteration %d:"
                    " RMSE=%.6e (testing), Mean Loss=%.6e (training)"
                ),
                ith_iteration,
                loss["test"][-1],
                loss["training"][-1],
            )
            if self.early_stopping is not None and self.early_stopping(
                loss["test"][-1], W_k
            ):
                weight_matrix = self.early_stopping.best_weights
                gene_feat_dim = self.h1.shape[0]
                self.h1 = weight_matrix[:gene_feat_dim, :]
                self.h2 = weight_matrix[gene_feat_dim:, :].T
                self.logger.debug("[Early Stopping] Training interrupted.")
                break
            if (ith_iteration + 1) % log_freq == 0 or ith_iteration == 0:
                self.callback(
                    ith_iteration,
                    loss["training"],
                    loss["test"]
                )
                rmse_history.append(loss["test"][-1])
                loss_history.append(loss["training"][-1])
                loss["test"].clear()
                loss["training"].clear()
        self.callback(
                    ith_iteration,
                    loss["training"],
                    loss["test"]
                )

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
