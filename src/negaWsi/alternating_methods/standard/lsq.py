# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Alternating Least Squares Descent Algorithm for Matrix Completion
===============================================================

This module implements the template for Alternating Least Squares Algorithm.
"""
from typing import Dict, List

import numpy as np

from negaWsi.alternating_methods.standard.base import McSolver


class ALSQ(McSolver):
    """
    Alternating Least Squares descent solver for matrix completion.

    This solver alternates between optimizing the left and right factor matrices.
    """

    def __init__(self, **kwargs):
        """
        Initializes the solver with parameters forwarded to `McSolver`.

        Args:
            **kwargs: See `McSolver` for the full list of supported parameters.
        """
        super().__init__(**kwargs)
        m, n = self.matrix.shape
        self.obs_cols_by_row = [np.flatnonzero(self.train_mask[i, :]) for i in range(m)]
        self.obs_rows_by_col = [np.flatnonzero(self.train_mask[:, j]) for j in range(n)]

    def step(self, loss: Dict[str, List[float]]) -> float:
        """
        Performs one alternating optimization step over h1 and h2.

        Args:
            loss (Dict[str, List[float]]): Accumulator for training and test loss
                values.

        Returns:
            np.ndarray: Stacked weight matrix with shape (n + m, rank).
        """
        m, n = self.matrix.shape
        I = np.eye(self.rank)
        # --- Update H1 row-wise ---
        for i in range(m):
            idx = self.obs_cols_by_row[i]
            A = (
                self.h2[:, idx] @ self.h2[:, idx].T
                + self.regularization_parameters["λg"] * I
            )
            b = self.h2[:, idx] @ self.matrix[i, idx]
            self.h1[i, :] = np.linalg.solve(A, b)
        _, test_loss_step, rmse_step, training_loss_step = self.evaluate()
        loss["test"].append(test_loss_step)
        loss["rmse"].append(rmse_step)
        loss["training"].append(training_loss_step)
        self.logger.debug(
            ("[ALS Step h1] Loss: %.6e"),
            loss["training"][-1],
        )

        # ---  Update H2 column-wise ---
        for j in range(n):
            idx = self.obs_rows_by_col[j]
            A = (
                self.h1[idx, :].T @ self.h1[idx, :]
                + self.regularization_parameters["λd"] * I
            )
            b = self.h1[idx, :].T @ self.matrix[idx, j]
            self.h2[:, j] = np.linalg.solve(A, b)
        _, test_loss_step, rmse_step, training_loss_step = self.evaluate()
        loss["test"].append(test_loss_step)
        loss["rmse"].append(rmse_step)
        loss["training"].append(training_loss_step)
        self.logger.debug(
            ("[ALS Step h2] Loss: %.6e"),
            loss["training"][-1],
        )

        W_k = np.vstack([self.h1, self.h2.T])
        return W_k
