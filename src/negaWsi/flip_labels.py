# pylint: disable=R0903,R0902
"""
Flip Labels Module
====================

Provides functionality to simulate label noise by flipping positive labels to negative labels
in matrix-based datasets. The API is designed to be straightforward and intuitive, allowing
users to apply noise on training masks at configurable intervals.
"""
import numpy as np


class FlipLabels:
    """
    A callable class for simulating label noise by randomly flipping a fraction of positive (1)
    entries to negatives (0) in a training mask.

    This class is useful for experiments where adding noise to the training labels is desired.
    The flipping operation is performed every `flip_frequency` iterations. On the designated
    iterations, a copy of the provided label matrix is made and a computed number of its positive
    entries are randomly selected and set to 0.

    Attributes:
        fraction (float): Fraction of observed positive entries (ones) in the training mask that
            will be flipped to negatives (zeros) to simulate label noise. Must be between 0 and 1.
        ones_indices (np.ndarray): Array of indices (row, col) identifying the locations of
            positive entries observed in the training mask.
        n_ones (int): Total number of positive entries observed in the training mask.
        n_to_zero (int): The number of positive entries to flip to negatives, computed as
            int(fraction * n_ones).
        flip_frequency (int): Frequency (in iterations) at which to resample the observed positive
            entries for flipping.
        flip_iteration (int): Counter tracking the number of iterations where the flipping
            operation has been performed or checked. It is updated each time the object is called.
        tmp_labels (np.ndarray): Temporary label matrix holding simulated label noise (not actively
            used in the __call__ method).
    """

    def __init__(self, fraction: float, flip_frequency: int, ones_indices: np.ndarray):
        """
        Initialize a FlipLabels instance.

        Args:
            fraction (float): Fraction of observed positive entries (ones) in the training mask
                that will be flipped to negatives (zeros) to simulate label noise. Must be between
                0 and 1.
            flip_frequency (int): The frequency (in iterations) at which to resample the observed
                positive entries for flipping.
            ones_indices (np.ndarray): Array of indices (row, col) identifying the locations of
                positive entries observed in the training mask.

        Raises:
            ValueError: If `fraction` is not strictly between 0 and 1.
        """
        if not 0 < fraction < 1:
            raise ValueError("Fraction must be between 0 and 1 (exclusive).")
        self.fraction = fraction
        self.ones_indices = ones_indices
        self.n_ones = self.ones_indices.shape[0]
        self.n_to_zero = int(self.fraction * self.n_ones)
        self.flip_iteration = 0
        self.flip_frequency = flip_frequency
        self.labels = None

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply simulated label noise to the provided label matrix
        by flipping a subset of the positive entries.

        Args:
            matrix (np.ndarray): The input label matrix to which simulated label
            noise will be applied.

        Returns:
            np.ndarray: The label matrix with simulated noise (flipped positives) applied.
        """
        if self.flip_iteration % self.flip_frequency == 0:
            self.labels = matrix.copy()
            selected_indices = np.random.choice(
                self.n_ones,
                size=self.n_to_zero,
                replace=False,
            )
            coords_to_zero = self.ones_indices[selected_indices]
            self.labels[coords_to_zero[:, 0], coords_to_zero[:, 1]] = 0
        self.flip_iteration += 1
        return self.labels
