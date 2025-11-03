"""
Result Module
================================

This module provides an data stucture to
storing training results.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Result:
    """
    Stores the results of the matrix completion process.

    Attributes:
        loss_history (List[float]):
            A list of loss values recorded at each iteration of the training process.
        rmse_history (List[float]):
            A list of RMSE (Root Mean Squared Error) values recorded during each
            iteration on the test set to monitor performance.
        runtime (float):
            The total runtime of the optimization process, measured in seconds.
        iterations (int):
            The total number of iterations performed during the optimization process.
    """

    loss_history: List[float]
    rmse_history: List[float]
    runtime: float
    iterations: int
