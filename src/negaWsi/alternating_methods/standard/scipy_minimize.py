# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Alternating Minimize Descent Algorithm for Matrix Completion
===============================================================

This module implements the template for Alternating Minimize Descent Algorithm.
"""
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from negaWsi.alternating_methods.standard.base import McSolver


class AMD(McSolver):
    """
    Alternating minimize descent solver for matrix completion.

    This solver alternates between optimizing the left and right factor matrices.
    """

    def __init__(self, method: str, **kwargs):
        """
        Initializes the solver with parameters forwarded to `McSolver`.

        Args:
            method (str): Method available in scipy minimize function.
            **kwargs: See `McSolver` for the full list of supported parameters.
        """
        super().__init__(**kwargs)
        self.method = method

    def step(self, loss: Dict[str, List[float]]) -> float:
        """
        Performs one alternating optimization step over h1 and h2.

        Args:
            loss (Dict[str, List[float]]): Accumulator for training and test loss
                values.

        Returns:
            np.ndarray: Stacked weight matrix with shape (n + m, rank).
        """
        res_h1 = minimize(
            fun=lambda h1_flat: self.h1_step(h1_flat),
            x0=self.h1.ravel(),
            args=[loss],
            method=self.method,
            jac=True,
            options={"maxiter": self.max_inner_iter},
        )
        self.h1 = res_h1.x.reshape(self.h1.shape)
        res_h2 = minimize(
            fun=lambda h2_flat: self.h2_step(h2_flat),
            x0=self.h2.ravel(),
            args=[loss],
            method=self.method,
            jac=True,
            options={"maxiter": self.max_inner_iter},
        )
        self.h2 = res_h2.x.reshape(self.h2.shape)
        W_k = np.vstack([self.h1, self.h2.T])
        return W_k
