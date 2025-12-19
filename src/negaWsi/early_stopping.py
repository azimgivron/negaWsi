# pylint: disable=R0903
"""
EarlyStopping Module
======================

Provides tools for monitoring validation loss and early stopping for machine learning models.
The EarlyStopping class tracks recent losses and corresponding model weights. If no improvement
is observed within a specified patience interval, it signals for early termination of training.
"""
import numpy as np


class EarlyStopping:
    """
    A callable class to determine early stopping during model training based on loss improvement.

    This class tracks the losses and corresponding model parameters over a specified
    number of iterations (patience). If the losses do not show a favorable improvement
    over the tracked iterations, the early stopping condition is met.

    Attributes:
        patience (int): Number of epochs/iterations to wait for an improvement before stopping.
        best_loss (float): Best (minimum) loss observed so far.
        since_improved (int): Number of consecutive epochs/iterations since the last improvement.
        best_weights (Tuple[np.ndarray, np.ndarray] | None): Copies of model parameters (h1, h2)
            at the best observed loss. None if no weights have been saved yet.
    """

    def __init__(self, patience: int):
        """
        Initializes the EarlyStopping instance with a fixed patience parameter.

        Args:
            patience (int): The number of recent epochs/iterations to consider when evaluating the
                stopping condition.
        """
        self.patience = patience
        self.best_loss = np.inf
        self.since_improved = 0
        self.best_weights = None

    def __call__(self, loss: float, weight_matrix: np.ndarray) -> bool:
        """
        Update the early stopping criteria with a new loss and model weights.

        On each call, the new loss and a copy of the provided weights
        are compared against the best observed metrics. If the loss improves, the
        best_loss, since_improved counter, and best_weights are updated. If no
        improvement is observed for more than 'patience' iterations, early stopping
        is triggered.

        Args:
            loss (float): The latest computed loss value.
            weight_matrix (np.ndarray): The weight matrix.

        Returns:
            bool: True if early stopping condition is met (i.e., no improvement observed
                  within 'patience' iterations), False otherwise.
        """
        if loss < self.best_loss and not np.isclose(loss, self.best_loss, atol=1e-3):
            self.best_loss = loss
            self.since_improved = 0
            # Store copies of the weights at the best observed loss
            self.best_weights = weight_matrix.copy()
        else:
            self.since_improved += 1

        return self.since_improved > self.patience
