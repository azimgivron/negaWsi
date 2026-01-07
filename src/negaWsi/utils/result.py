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
        training_loss_history (List[float]):
            A list of training loss values recorded at each iteration of the training process.
        test_loss_history (List[float]):
            A list of test loss values recorded at each iteration of the training process.
        test_rmse_history (List[float]):
            A list of RMSE (Root Mean Squared Error) values recorded during each
            iteration on the test set to monitor performance.
        runtime (float):
            The total runtime of the optimization process, measured in seconds.
        iterations (int):
            The total number of iterations performed during the optimization process.
    """

    training_loss_history: List[float]
    test_loss_history: List[float]
    test_rmse_history: List[float]
    runtime: float
    iterations: int
