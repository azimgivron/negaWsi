# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Non-Euclidean Gradient Algorithm Template for Matrix Completion
===============================================================

This module implements the template for Non-Euclidean Gradient Algorithm.
"""
import abc
import logging
import time
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp

import negaWsi.standard.base as nb
from negaWsi.utils.early_stopping import EarlyStopping
from negaWsi.utils.flip_labels import FlipLabels
from negaWsi.utils.result import Result


class NegaBase(nb.NegaBase):
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    The class is designed for scenarios where the objective is to approximate a
    partially observed matrix by a low-rank factorization.

    Attributes:
        matrix (sp.csr_matrix): Input matrix to be approximated. Shape: (n, m),
            where n is the number of genes and m is the number of diseases.
        train_mask (sp.csr_matrix): Boolean mask indicating observed entries in `matrix`
            for training. Shape: (n, m).
        test_mask (sp.csr_matrix): Boolean mask indicating observed entries in `matrix`
            for testing. Shape: (n, m).
        rank (int): The target rank for the low-rank approximation.
        regularization_parameters (float): Regularization parameters used in the optimization
            objective.
        iterations (int): Maximum number of optimization iterations.
        symmetry_parameter (float): Parameter used to adjust gradient symmetry during
            the optimization process.
        lipschitz_smoothness (float): Initial smoothness parameter for the optimization steps.
        rho_increase (float): Factor used to dynamically increase the optimization step size.
        rho_decrease (float): Factor used to dynamically decrease the optimization step size.
        tau (float):
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
        matrix: sp.csr_matrix,
        train_mask: sp.csr_matrix,
        test_mask: sp.csr_matrix,
        rank: int,
        regularization_parameters: Dict[str, float],
        iterations: int,
        symmetry_parameter: float,
        lipschitz_smoothness: float,
        rho_increase: float,
        rho_decrease: float,
        tau: float = None,
        seed: int = 123,
        flip_labels: FlipLabels = None,
        early_stopping: EarlyStopping = None,
    ):
        """
        Initializes the BaseNEGA instance with the provided configuration
        parameters for matrix approximation.

        Args:
            matrix (sp.csr_matrix: Input matrix to be approximated. Shape: (n, m),
                where n is the number of genes and m is the number of diseases.
            train_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for training.
                Shape: (n, m).
            test_mask (sp.csr_matrix): Mask indicating observed entries in `matrix` for testing.
                Shape: (n, m).
            rank (int): Desired rank for the low-rank approximation.
            regularization_parameters (Dict[str, float],): Regularization parameters for the optimization
                objective.
            iterations (int): Maximum number of optimization iterations.
            symmetry_parameter (float): Parameter for adjusting gradient symmetry during
                optimization.
            lipschitz_smoothness (float): Initial smoothness parameter for optimization steps.
            rho_increase (float): Multiplicative factor to dynamically increase the optimization
                step size.
            rho_decrease (float): Multiplicative factor to dynamically decrease the optimization
                step size.
            tau (float):
            seed (int, optional): Seed for reproducible random initialization. Defaults to 123.
            flip_labels (FlipLabels, optional): Object that simulates label noise by randomly
                flipping a fraction of positive (1) entries to negatives (0) in the training mask.
            early_stopping (EarlyStopping, optional): Early stopping object that implements
                a mechanism for monitoring the validation loss and triggering early termination
                if performance does not improve.
        """
        if flip_labels is not None:
            raise NotImplementedError(
                "Flip Label is not implemented for sparse matrices."
            )

        self.matrix = matrix
        self.train_mask = train_mask
        self.test_mask = test_mask

        # indices of observed entries
        self.train_i, self.train_j = self.train_mask.nonzero()
        self.test_i, self.test_j = self.test_mask.nonzero()

        self.train_y = np.asarray(self.matrix[self.train_i, self.train_j]).ravel()
        self.test_y = np.asarray(self.matrix[self.test_i, self.test_j]).ravel()

        self.rank = rank
        self.regularization_parameters = regularization_parameters
        self.iterations = iterations
        self.symmetry_parameter = symmetry_parameter
        self.lipschitz_smoothness = lipschitz_smoothness
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.h1 = None
        self.h2 = None
        self.early_stopping = early_stopping
        self.loss_terms = {}
        if tau is not None:
            self.tau = tau
        else:
            vals = self.matrix[self.train_i, self.train_j]
            self.tau = np.linalg.norm(vals) / 3
        # Set random seed for reproducibility
        np.random.seed(seed)

        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def predict_entries(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Predict entry R_ij

        Args:
            i (np.ndarray): The row index.
            j (np.ndarray): The column index.

        Returns:
            np.ndarray: The prediction ij.
        """

    def calculate_training_residual_entries(self) -> np.ndarray:
        """
        Compute the training residual from the input matrix M (m x n), the model's prediction
        M_pred and the binary training mask.

        The training residual R is computed as:
            R = B ⊙ (M_pred - M)
        where ⊙ represents hadamard product.

        Returns:
            np.ndarray: The residual matrix R, shape: (nnz_train,).
        """
        yhat = self.predict_entries(self.train_i, self.train_j)
        return yhat - self.train_y

    def calculate_rmse(self, mask: sp.csr_matrix) -> float:
        """
        Computes the Root Mean Square Error (RMSE).

        Args:
            mask (sp.csr_matrix): The mask to apply.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed entries.
        """
        if mask is self.train_mask:
            i, j, y = self.train_i, self.train_j, self.train_y
        elif mask is self.test_mask:
            i, j, y = self.test_i, self.test_j, self.test_y
        else:
            i, j = mask.nonzero()
            y = np.asarray(self.matrix[i, j]).ravel()

        yhat = self.predict_entries(i, j)
        return float(np.sqrt(np.mean((y - yhat) ** 2)))
