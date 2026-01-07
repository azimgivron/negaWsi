# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Non-Euclidean Gradient Algorithm Template for Matrix Completion
===============================================================

This module implements the template for Non-Euclidean Gradient Algorithm.
"""
import abc
import logging
import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from negaWsi.utils.early_stopping import EarlyStopping
from negaWsi.utils.flip_labels import FlipLabels
from negaWsi.utils.result import Result


@dataclass
class State:
    """Container for optimization state across iterations.

    Attributes:
        W_k (np.ndarray): Current stacked factor matrix. Shape: (n + m, rank).
        W_k_next (np.ndarray): Next stacked factor matrix after a step. Shape: (n + m, rank).
        grad_f_W_k (np.ndarray): Gradient of the objective at W_k. Shape: (n + m, rank).
        loss_W_k (float): Loss value at W_k.
        loss_W_k_next (float): Loss value at W_k_next.
    """

    W_k: np.ndarray = None
    W_k_next: np.ndarray = None
    grad_f_W_k: np.ndarray = None
    loss_W_k: float = None
    loss_W_k_next: float = None
    step_size: float = None

    def update(self):
        """Advance the state to the next iterate."""
        self.W_k = self.W_k_next
        self.loss_W_k = self.loss_W_k_next


class NegaBase(metaclass=abc.ABCMeta):
    """
    Manages the configuration, training, and evaluation of a matrix completion model.

    The class is designed for scenarios where the objective is to approximate a
    partially observed matrix by a low-rank factorization.

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
        symmetry_parameter (float): Parameter used to adjust gradient symmetry during
            the optimization process.
        lipschitz_smoothness (float): Initial smoothness parameter for the optimization steps.
        rho_increase (float): Factor used to dynamically increase the optimization step size.
        rho_decrease (float): Factor used to dynamically decrease the optimization step size.
        tau (float): Kernel regularization parameter used in h(W).
        h1 (np.ndarray): Left factor matrix in the low-rank approximation. Shape: (n, rank).
        h2 (np.ndarray): Right factor matrix in the low-rank approximation. Shape: (rank, m).
        logger (logging.Logger): Logger instance for debugging and monitoring training progress.
        seed (int): Seed for reproducible random initialization.
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
            symmetry_parameter (float): Parameter for adjusting gradient symmetry during
                optimization.
            lipschitz_smoothness (float): Initial smoothness parameter for optimization steps.
            rho_increase (float): Multiplicative factor to dynamically increase the optimization
                step size.
            rho_decrease (float): Multiplicative factor to dynamically decrease the optimization
                step size.
            tau (float, optional): Kernel regularization parameter used in h(W).
            seed (int, optional): Seed for reproducible random initialization. Defaults to 123.
            flip_labels (FlipLabels, optional): Object that simulates label noise by randomly
                flipping a fraction of positive (1) entries to negatives (0) in the training mask.
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
        self.symmetry_parameter = symmetry_parameter
        self.lipschitz_smoothness = lipschitz_smoothness
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.h1 = None
        self.h2 = None
        self.flip_labels = flip_labels
        self.early_stopping = early_stopping
        self.loss_terms = {}
        if tau is not None:
            self.tau = tau
        else:
            mat = self.matrix.copy()
            mat[~self.train_mask] = 0
            self.tau = np.linalg.norm(mat, ord="fro") / 3

        # Set random seed for reproducibility
        np.random.seed(seed)

        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def predict_all(self) -> np.ndarray:
        """
        Computes the reconstructed matrix from the factor matrices h1 and h2.

        Returns:
            np.ndarray: The reconstructed (completed) matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_grad_f_W(self) -> np.ndarray:
        """Compute the gradients for each latent as:

        grad_f(W_k) = (∇_h1, ∇_h2).T

        Returns:
            np.ndarray: The gradient of the latents (n+m x rank)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_loss(self, mask: np.ndarray) -> float:
        """
        Computes the loss function value for the training data.

        Args:
            mask (np.ndarray): The binary mask.

        Returns:
            float: The computed loss value.
        """
        raise NotImplementedError

    def kernel(self, W: np.ndarray) -> float:
        """
        Computes the value of the kernel function h for a given matrix W and
        regularization parameter tau.

        The h function is defined as:
            h(W) = 0.25 * ||W||_F^4 + tau/2 * ||W||_F^2

        Args:
            W (np.ndarray): The input matrix.

        Returns:
            float: The computed value of the h function.
        """
        norm = np.linalg.norm(W, ord="fro")
        h_value = 0.25 * norm**4 + self.tau / 2 * norm**2
        return h_value

    def init_Wk(self) -> np.ndarray:
        """
        Initialize weight block matrix.

        Returns:
            np.ndarray: The weight block matrix. Shape: (n + m, rank).
        """
        return np.vstack([self.h1, self.h2.T])

    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix. Shape: (n + m, rank).
        """
        nb_genes = self.h1.shape[0]
        self.h1 = weight_matrix[:nb_genes, :]
        self.h2 = weight_matrix[nb_genes:, :].T

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

    def cardano(self, delta: float) -> float:
        """
        Solve the cubic equation:
            s^3 - tau * s^2 - delta = 0
        using Cardano's method.

        Args:
            delta (float): Constant term δ in the equation.

        Returns:
            float: The unique real root s of the cubic.
        """
        tau1 = -(self.tau**2) / 3
        tau2 = (-2 * self.tau**3 - 27 * delta) / 27
        discr = (tau2 / 2) ** 2 + (tau1 / 3) ** 3
        sqrt_disc = np.sqrt(discr, dtype=np.complex128)
        t_k = (self.tau / 3) + (
            np.power(-tau2 / 2 + sqrt_disc, 1 / 3, dtype=np.complex128)
            + np.power(-tau2 / 2 - sqrt_disc, 1 / 3, dtype=np.complex128)
        ).real
        return t_k

    def bregman_distance(self, W_next: np.ndarray, W_current: np.ndarray) -> float:
        """
        Computes the Bregman distance:
            D_h = h(W_next) - h(W_current) - <grad_h(W_current), W_next - W_current>

        Args:
            W_next (np.ndarray): The updated input sparse matrix.
            W_current (np.ndarray): The current input sparse matrix.

        Returns:
            float: The computed Bregman distance.
        """
        h_W_next = self.kernel(W_next)
        h_W_current = self.kernel(W_current)
        linear_approx = np.vdot(self.compute_grad_h_W(W_current), (W_next - W_current))
        dist = h_W_next - (h_W_current + linear_approx)
        assert (
            dist >= -1e-3
        ), f"Bregman distance must always be positive: D_h = {dist:.4e}"
        return dist

    def calculate_residual(self, mask: np.ndarray = None) -> np.ndarray:
        """
        Compute the residual from the input matrix M (m x n), the model's prediction
        M_pred and the binary mask. Optionally, if positive_flip_fraction is set and
        positive_flip_fraction = d, a fraction 'd' of the positive entries (ones) in M
        (where P is 1) is flipped to 0, yielding a modified label matrix L. Otherwise, L = M.

        The residual R is computed as:
            R = B ⊙ (M_pred - M)
        where ⊙ represents hadamard product.

        Args:
            mask (np.ndarray): The binary mask.

        Returns:
            np.ndarray: The residual matrix R (n x m).
        """
        if self.flip_labels is not None:
            labels = self.flip_labels(self.matrix)
        else:
            labels = self.matrix
        residual = self.predict_all() - labels
        mask = mask if mask is not None else self.train_mask
        residual[~mask] = 0
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
        grad_f_W_k: np.ndarray,
        step_size: float,
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
            grad_f_W_k (np.ndarray): The gradient of the loss with respect to the model weights
                at the current iteration.
            step_size (float): The step size.
        """

    def compute_grad_h_W(self, W: np.ndarray) -> np.ndarray:
        """Compute the gradients for in the kernel space.

        ∇_h(W) = (||W||_F^2 + tau) * W

        Args:
            W (np.ndarray): Stacked factor matrices.

        Returns:
            np.ndarray: The gradient of the latents in the kernel space.
        """
        return (np.linalg.norm(W, ord="fro") ** 2 + self.tau) * W

    def step(self, state: State) -> float:
        """
        Performs a single step in the optimization process to update the factor matrices.

        This step calculates the next iterate W_{k+1} using the gradient of the objective
        function and an adaptive step size.

        Steps in the Process:

        1. Compute the Gradient Step: step = grad_h_W_k - step_size * grad_f_W_k
        2. Solve the Cubic Equation for the Step Size t.
        3. Update the Next Iterate W_{k+1}: W_{k+1} = (1 / t) * step
        4. Split W_{k+1} into Factor Matrices.

        Args:
            state (State): Current optimization state.

        Returns:
            float: Loss value f(W_{k+1}).
        """
        step = self.compute_grad_h_W(state.W_k) - (state.step_size * state.grad_f_W_k)
        delta = np.linalg.norm(step, ord="fro") ** 2
        # Solve the cubic s³ – τ s² – Δ = 0 by Cardano
        t_k = self.cardano(delta)
        state.W_k_next = (1 / t_k) * step
        self.set_weights(state.W_k_next)
        state.loss_W_k_next = self.calculate_loss(self.train_mask)

    def non_euclidean_descent_lemma_cond(self, state: State):
        """Check the non-Euclidean descent lemma condition.

        Args:
            state (State): Current optimization state.

        Returns:
            bool: True if the descent lemma condition is satisfied.
        """
        linear_approx = np.vdot(state.grad_f_W_k, (state.W_k_next - state.W_k))
        bregman = self.bregman_distance(state.W_k_next, state.W_k)
        left = state.loss_W_k_next
        right = (state.loss_W_k + linear_approx) + self.lipschitz_smoothness * bregman
        cond = left <= right
        self.logger.debug(
            "[Descent Lemma] Satisfied %s: %.4e <= %.4e", cond, left, right
        )
        return cond

    def backtrack(self, state: State):
        """Adjust the step size until the descent condition is satisfied.

        Args:
            state (State): Current optimization state.
        """
        # Inner loop to adjust step size
        flag = 0
        while not self.non_euclidean_descent_lemma_cond(state):
            flag = 1
            # Adjust step size
            old_step_size = state.step_size
            self.lipschitz_smoothness *= self.rho_increase
            state.step_size = (1 + self.symmetry_parameter) / self.lipschitz_smoothness

            self.logger.debug(
                "[Step Size Decrease] %.4e -> %.4e", old_step_size, state.step_size
            )
            if state.step_size <= 1e-10:
                self.logger.warning(
                    "[Step Size Too Small] Step size of %.4e can lead to overflow. Training interrupted.",
                    state.step_size,
                )
                state.loss_W_k_next = np.nan
                break

            self.step(state)
        if flag == 1:
            # Adjust step size
            self.lipschitz_smoothness *= self.rho_decrease

    def update_grad_f_W(self, state: State):
        """Update gradient and log tracked loss terms.

        Args:
            state (State): Current optimization state.
        """
        state.grad_f_W_k = self.compute_grad_f_W()
        for key, value in self.loss_terms.items():
            self.logger.debug(
                ("[Main Loop] %s: %.6e"),
                key,
                value,
            )

    def run(self, log_freq: int = 10) -> Result:
        """
        Performs matrix completion using adaptive step size optimization.

        The optimization process minimizes the objective function:
            f(W) = 0.5 * ||M - M_pred(W)||_F^2 + 0.5 * mu * ||W||_F^2

        where:
            - M: The input matrix (shape: m x n).
            - M_pred: The low-rank approximation.
            - ||W||_F: The Frobenius norm of the stacked factor matrix W.
            - mu: The regularization parameter.

        Key Steps in the Optimization Process:

        1. Initialization: Initialize the factor matrices
        2. Iterative Updates:
           Update the stacked factor matrix W_k using gradients and adaptive
           step sizes
        3. Dynamic Step Size Adjustment:
           Adjust the step size if the new loss does not satisfy the improvement condition:
               f(W_{k+1}) <= f(W_k) + ∇f(W_k) (W_{k+1} - W_k) + L_f * D_h(W_{k+1}, W_k, τ),
           where:
               - L_f is the smoothness parameter,
               - D_h is the Bregmann distance.
        4. Termination:
           Stop when the maximum number of iterations is reached or the loss converges.

        Args:
            log_freq (int, optional): Period at which to log data in Tensorboard.
                Default to 10 (iterations).

        Returns:
            Result: A dataclass containing:
                - completed_matrix: The reconstructed matrix (low-rank approximation).
                - training_loss_history: List of training loss values at each iteration.
                - test_loss_history: List of test loss values at each iteration.
                - rmse_history: List of RMSE values at each iteration.
                - runtime: Total runtime of the optimization process.
                - iterations: Total number of iterations performed.
        """
        # Start measuring runtime
        start_time = time.time()
        state = State()

        # Stack h1 and h2 for optimization
        state.W_k = self.init_Wk()
        state.step_size = 1 / self.lipschitz_smoothness
        self.logger.debug(
            "Starting optimization with tau=%f, step_size=%f", self.tau, state.step_size
        )
        # Initialize loss and RMSE history
        state.loss_W_k = self.calculate_loss(self.train_mask)
        training_loss = [state.loss_W_k]
        test_loss = [self.calculate_loss(self.test_mask)]
        rmse = [self.calculate_rmse(self.test_mask)]

        # Main optimization loop
        for ith_iteration in range(self.iterations):
            self.logger.debug(
                (
                    "[Main Loop] Iteration %d:"
                    " RMSE=%.6e (testing), Mean Loss=%.6e (training)"
                ),
                ith_iteration,
                rmse[-1],
                training_loss[-1],
            )
            # compute gradients and update W
            self.update_grad_f_W(state)
            self.step(state)

            # log
            if (ith_iteration + 1) % log_freq == 0 or ith_iteration == 0:
                self.callback(
                    ith_iteration,
                    state.loss_W_k_next,
                    rmse[-1],
                    state.grad_f_W_k,
                    state.step_size,
                )

            self.backtrack(state)

            # Break if loss becomes NaN
            if np.isnan(state.loss_W_k_next):
                self.logger.warning(
                    "[NaN Loss] Iteration %d: Loss is NaN, exiting loop.", ith_iteration
                )
                break

            # Update variables for the next iteration
            state.update()
            training_loss.append(state.loss_W_k)
            test_loss.append(self.calculate_loss(self.test_mask))
            rmse.append(self.calculate_rmse(self.test_mask))

            if self.early_stopping is not None and self.early_stopping(
                rmse[-1], state.W_k
            ):
                self.set_weights(self.early_stopping.best_weights)
                self.logger.debug("[Early Stopping] Training interrupted.")
                if ith_iteration % log_freq != 0:
                    self.callback(
                        ith_iteration,
                        state.loss_W_k_next,
                        rmse[-1],
                        state.grad_f_W_k,
                        state.step_size,
                    )
                break

        # Compute runtime
        runtime = time.time() - start_time
        self.logger.debug(
            "[Completion] Optimization finished in %.2f seconds.", runtime
        )
        training_data = Result(
            training_loss_history=training_loss,
            test_loss_history=test_loss,
            iterations=ith_iteration,
            test_rmse_history=rmse,
            runtime=runtime,
        )
        return training_data
