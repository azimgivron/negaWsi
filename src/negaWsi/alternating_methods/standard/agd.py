# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Alternating Gradient Descent Algorithm for Matrix Completion
===============================================================

This module implements the template for Alternating Gradient Descent Algorithm.
"""

import numpy as np
from typing import Dict, List
from scipy.optimize import line_search
from negaWsi.alternating_methods.standard.base import McSolver

class AGD(McSolver):
    """
    Alternating gradient descent solver for matrix completion.

    This solver alternates between optimizing the left and right factor matrices
    using gradient descent steps.
    """

    def __init__(
        self,
        **kwargs
    ):
        """
        Initializes the solver with parameters forwarded to `McSolver`.

        Args:
            **kwargs: See `McSolver` for the full list of supported parameters.
        """
        super().__init__(**kwargs)

    def step(self, loss: Dict[str, List[float]]) -> float:
        """
        Performs one alternating optimization step over h1 and h2.

        Args:
            loss (Dict[str, List[float]]): Accumulator for training and test loss
                values.

        Returns:
            np.ndarray: Stacked weight matrix with shape (n + m, rank).
        """
        armijo_c = 1e-4
        curvature_c = 0.9

        def line_search_update(step_fn, x_flat):
            loss_val, grad = step_fn(x_flat, loss)
            direction = -grad
            f = lambda x: step_fn(x)[0]
            grad_f = lambda x: step_fn(x)[1]
            alpha, *_ = line_search(
                f,
                grad_f,
                x_flat,
                direction,
                gfk=grad,
                old_fval=loss_val,
                c1=armijo_c,
                c2=curvature_c,
                maxiter=self.max_inner_iter
            )
            return x_flat + alpha * direction
            
        h1_flat = self.h1.ravel()
        h1_flat = line_search_update(self.h1_step, h1_flat)
        self.h1 = h1_flat.reshape(self.h1.shape)
        h2_flat = self.h2.ravel()
        h2_flat = line_search_update(self.h2_step, h2_flat)
        self.h2 = h2_flat.reshape(self.h2.shape)
        W_k = np.vstack([self.h1, self.h2.T])
        return W_k
