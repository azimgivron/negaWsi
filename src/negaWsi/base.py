# pylint: disable=C0103,R0913,R0914,R0915,R0902,R0903,R0912
"""
Non-Euclidean Gradient Algorithm Template for Matrix Completion
===============================================================

This module implements the template for Non-Euclidean Gradient Algorithm.
"""
import abc
import logging
import time
from typing import Tuple

import numpy as np

from negaWsi.early_stopping import EarlyStopping
from negaWsi.flip_labels import FlipLabels
from negaWsi.result import Result


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
        regularization_parameter (float): Regularization parameter used in the optimization
            objective.
        iterations (int): Maximum number of optimization iterations.
        symmetry_parameter (float): Parameter used to adjust gradient symmetry during
            the optimization process.
        smoothness_parameter (float): Initial smoothness parameter for the optimization steps.
        rho_increase (float): Factor used to dynamically increase the optimization step size.
        rho_decrease (float): Factor used to dynamically decrease the optimization step size.
        threshold (int): Maximum iterations allowed for the inner optimization loop.
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
    """

    def __init__(
        self,
        matrix: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        rank: int,
        regularization_parameter: float,
        iterations: int,
        symmetry_parameter: float,
        smoothness_parameter: float,
        rho_increase: float,
        rho_decrease: float,
        threshold: int,
        seed: int = 123,
        flip_labels: FlipLabels = None,
        early_stopping: EarlyStopping = None,
    ):
        """
        Initializes the BaseNEGA instance with the provided configuration
        parameters for matrix approximation.

        Args:
            matrix (np.ndarray: Input matrix to be approximated. Shape: (n, m),
                where n is the number of genes and m is the number of diseases.
            train_mask (np.ndarray): Mask indicating observed entries in `matrix` for training.
                Shape: (n, m).
            test_mask (np.ndarray): Mask indicating observed entries in `matrix` for testing.
                Shape: (n, m).
            rank (int): Desired rank for the low-rank approximation.
            regularization_parameter (float): Regularization parameter for the optimization
                objective.
            iterations (int): Maximum number of optimization iterations.
            symmetry_parameter (float): Parameter for adjusting gradient symmetry during
                optimization.
            smoothness_parameter (float): Initial smoothness parameter for optimization steps.
            rho_increase (float): Multiplicative factor to dynamically increase the optimization
                step size.
            rho_decrease (float): Multiplicative factor to dynamically decrease the optimization
                step size.
            threshold (int): Maximum number of iterations for the inner optimization loop.
            writer (tf.summary.SummaryWriter, optional): TensorFlow summary writer for logging
                training summaries. Defaults to None.
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
        self.regularization_parameter = regularization_parameter
        self.iterations = iterations
        self.symmetry_parameter = symmetry_parameter
        self.smoothness_parameter = smoothness_parameter
        self.rho_increase = rho_increase
        self.rho_decrease = rho_decrease
        self.threshold = threshold
        self.h1 = None
        self.h2 = None
        self.flip_labels = flip_labels
        self.early_stopping = early_stopping

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
    def compute_grad_f_W_k(self) -> np.ndarray:
        """Compute the gradients for for each latent as:

        grad_f_W_k = (∇_h1, ∇_h2).T

        Returns:
            np.ndarray: The gradient of the latents (n+m x rank)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def kernel(self, W: np.ndarray, tau: float) -> float:
        """
        Computes the value of the kernel function h.

        Args:
            W (np.ndarray): The input matrix.
            tau (float): Regularization parameter.

        Returns:
            float: The computed value of the h function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_tau(self) -> float:
        """
        Initialize tau value.

        Returns:
            float: tau value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_Wk(self) -> np.ndarray:
        """
        Initialize weight block matrix.

        Returns:
            np.ndarray: The weight block matrix.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_weights(self, weight_matrix: np.ndarray):
        """
        Set the weights individually from the stacked block matrix.

        Args:
            weight_matrix (np.ndarray): The stacked block matrix.
        """
        raise NotImplementedError

    def cardano(self, tau: float, delta: float) -> float:
        """
        Solve the cubic equation:
            s^3 - tau * s^2 - delta = 0
        using Cardano's method.

        Args:
            tau (float): Coefficient τ in the equation.
            delta (float): Constant term δ in the equation.

        Returns:
            float: The unique real root s of the cubic.
        """
        # Compute intermediate parameters
        tau1 = -(tau**2) / 3
        tau2 = (-2 * tau**3 - 27 * delta) / 27

        # Discriminant for Cardano's formula
        discr = (tau2 / 2) ** 2 + (tau1 / 3) ** 3
        sqrt_disc = np.sqrt(discr, dtype=np.complex128)

        # Compute the real root
        t_k = (tau / 3) + (
            np.power(-tau2 / 2 + sqrt_disc, 1 / 3, dtype=np.complex128)
            + np.power(-tau2 / 2 - sqrt_disc, 1 / 3, dtype=np.complex128)
        ).real

        return t_k

    def bregman_distance(
        self, W_next: np.ndarray, W_current: np.ndarray, tau: float
    ) -> float:
        """
        Computes the Bregman distance:
            D_h = h(W_next) - h(W_current) - <grad_h(W_next), W_next - W_current>

        Args:
            W_next (np.ndarray): The updated input sparse matrix.
            W_current (np.ndarray): The current input sparse matrix.
            tau (float): Regularization parameter.

        Returns:
            float: The computed Bregman distance.
        """
        h_W_next = self.kernel(W_next, tau)
        h_W_current = self.kernel(W_current, tau)
        grad_h_W_current = (np.linalg.norm(W_current, ord="fro") ** 2 + tau) * W_current
        linear_approx = (grad_h_W_current.T @ (W_next - W_current)).sum()
        dist = h_W_next - h_W_current - linear_approx
        return dist

    def calculate_training_residual(self) -> np.ndarray:
        """
        Compute the training residual from the input matrix M (m x n), the model's prediction
        M_pred and the binary training mask. Optionally, if positive_flip_fraction is set and
        positive_flip_fraction = d, a fraction 'd' of the positive entries (ones) in M
        (where P is 1) is flipped to 0, yielding a modified label matrix L. Otherwise, L = M.

        The training residual R is computed as:
            R = (M_pred - L) ⊙  mask
        where ⊙ represents hadamard product.

        Returns:
            np.ndarray: The residual matrix R (m x n).
        """
        if self.flip_labels is not None:
            labels = self.flip_labels(self.matrix)
        else:
            labels = self.matrix
        residual = self.predict_all() - labels
        residual[~self.train_mask] = 0
        return residual

    def calculate_loss(self) -> float:
        """
        Computes the loss function value for the training data.

        The loss is defined as the Frobenius norm of the residual matrix
        for observed entries only:
            Loss = 0.5 * ||M - M_pred||_F^2

        (only over observed entries indicated by `mask`)

        where:
        - M is the input matrix (shape: m x n),
        - M_pred is the predicted (reconstructed) matrix

        Returns:
            float: The computed loss value.
        """
        residuals = self.calculate_training_residual()
        return 0.5 * np.linalg.norm(residuals, ord="fro") ** 2

    def calculate_rmse(self, mask: np.ndarray) -> float:
        """
        Computes the Root Mean Square Error (RMSE).

        The RMSE measures the average deviation between the predicted matrix
        and the ground truth matrix, considering only the observed entries.

        Formula for RMSE:
            RMSE = sqrt( (1 / |Omega|) * sum( (M[i, j] - M_pred[i, j])^2 ) )

        Where:
            - Omega: The set of observed indices in `mask`.
            - M: The matrix containing actual values.
            - M_pred: The predicted matrix (low-rank approximation).

        Process:
        1. Extract the observed entries from both the matrix (`M`)
           and the predicted matrix (`M_pred`) using the `mask`.
        2. Compute the squared differences for the observed entries.
        3. Calculate the mean of these squared differences.
        4. Take the square root of the mean to obtain the RMSE.

        Returns:
            float: The computed RMSE value, representing the prediction error
                   on the observed entries.
        """
        actual_values = self.matrix[mask]
        predictions = self.predict_all()[mask]
        rmse = np.sqrt((actual_values - predictions) ** 2).mean()
        return rmse

    def callback(
        self, ith_iteration: int, testing_loss: np.ndarray, grad_f_W_k: np.ndarray
    ):
        """
        Callback to add logs or whatever the user wants compute between 'log_freq'
        number of outer loop iterations.

        Args:
            ith_iteration (int): The current iteration index at which the logging is performed.
            testing_loss (np.ndarray): The computed loss value for the testing dataset at the
                current iteration.
            grad_f_W_k (np.ndarray): The gradient of the loss with respect to the model weights
                at the current iteration.
        """

    def substep(
        self,
        W_k: np.ndarray,
        tau: float,
        step_size: float,
        grad_f_W_k: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Performs a single substep in the optimization process to update the factor matrices.

        This substep calculates the next iterate W_{k+1} using the gradient of the objective
        function and an adaptive step size.

        Steps in the Substep Process:

        1. Compute the Gradient Step:
           grad = (||W_k||_F^2 + tau) * W_k - step_size * grad_f_W_k

           - ||W_k||_F: Frobenius norm of the current matrix W_k.
           - tau: Regularization parameter.
           - step_size: Learning rate for the gradient step.
           - grad_f_W_k: Gradient of the objective function at W_k.

        2. Solve the Cubic Equation for the Step Size t:
           Δ = tau_2^2 + (tau1/3)^3
           if Δ >= 0:

           t = (tau / 3) + cube_root(T_1) + cube_root(T_2)

           - T_1 = -tau_2 + sqrt(Δ)
           - T_2 = -tau_2 - sqrt(Δ)
           - tau_2 = (-2 * tau^3 - 27 * ||grad||_F^2) / 27

           The cubic root function ensures stability, even for negative T_1 and T_2.

        3. Update the Next Iterate W_{k+1}:
           W_{k+1} = (1 / t) * grad

        4. Split W_{k+1} into Factor Matrices h1 and h2:
           - h1 = W_{k+1}[:m, :]
           - h2 = W_{k+1}[m:, :].T

        Args:
            W_k (np.ndarray): Current stacked factor matrices.
            tau (float): Regularization parameter.
            step_size (float): Learning rate for the gradient step.
            grad_f_W_k (np.ndarray): Gradient of the objective function at W_k.

        Returns:
            Tuple[np.ndarray, float]:
                - Updated stacked matrix W_{k+1}.
                - New loss value f(W_{k+1}).
        """
        step = (np.linalg.norm(W_k, ord="fro") ** 2 + tau) * W_k - (
            step_size * grad_f_W_k
        )
        delta = np.linalg.norm(step, ord="fro") ** 2
        # Solve the cubic s³ – τ s² – Δ = 0 by Cardano
        t_k = self.cardano(tau, delta)

        W_k_next = (1 / t_k) * step
        self.set_weights(W_k_next)
        loss = self.calculate_loss()
        return W_k_next, loss

    def run(self, log_freq: int = 10) -> Result:
        """
        Performs matrix completion using adaptive step size optimization.

        The optimization process minimizes the objective function:
            f(W) = 0.5 * ||M - M_pred||_F^2 + mu * ||W||_F^2

        where:
            - M: The input matrix (shape: m x n).
            - M_pred: The low-rank approximation.
            - ||W||_F: The Frobenius norm of the stacked factor matrix W.
            - mu: The regularization parameter.

        Key Steps in the Optimization Process:

        1. Initialization:
           Initialize the factor matrices:
               - h1,
               - h2

        2. Iterative Updates:
           Update the stacked factor matrix W_k = [h1; h2.T] using gradients and adaptive
           step sizes:
               - ∇_h1,
               - ∇_h2

        3. Dynamic Step Size Adjustment:
           Adjust the step size if the new loss does not satisfy the improvement condition:
               f(W_{k+1}) <= f(W_k) + ∇f(W_k).T (W_{k+1} - W_k) + η * D_h(W_{k+1}, W_k, τ),
           where:
               - η is the step size/smoothness adjustment parameter,
               - D_h is the Bregmann distance.

        4. Termination:
           Stop when the maximum number of iterations is reached or the loss converges.

        Args:
            log_freq (int, optional): Period at which to log data in Tensorboard.
                Default to 10 (iterations).

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
        rows, columns = self.matrix.shape
        nb_elements = rows * columns

        # Initialize loss and RMSE history
        res_norm = self.calculate_loss()
        training_loss = res_norm / nb_elements
        testing_loss = self.calculate_rmse(self.test_mask)
        loss = [training_loss]
        rmse = [testing_loss]

        # Stack h1 and h2 for optimization
        W_k = self.init_Wk()
        step_size = 1 / self.smoothness_parameter
        tau = self.init_tau()
        self.logger.debug(
            "Starting optimization with tau=%f, step_size=%f", tau, step_size
        )

        # Main optimization loop
        iterations_count = 0
        for ith_iteration in range(self.iterations):
            iterations_count = ith_iteration
            inner_loop_it = 0
            flag = 0

            self.logger.debug(
                (
                    "[Main Loop] Iteration %d, Inner Loop %d:"
                    " RMSE=%.6e (testing), Mean Loss=%.6e (training)"
                ),
                ith_iteration,
                inner_loop_it,
                rmse[-1],
                loss[-1],
            )
            grad_f_W_k = self.compute_grad_f_W_k()

            substep_res = self.substep(W_k, tau, step_size, grad_f_W_k)
            if substep_res is None:
                break
            W_k_next, res_norm_next_it = substep_res
            # Inner loop to adjust step size
            linear_approx = (grad_f_W_k.T @ (W_k_next - W_k)).sum()
            bregman = self.bregman_distance(W_k_next, W_k, tau)
            non_euclidean_descent_lemma_cond = (
                res_norm_next_it
                <= res_norm + linear_approx + self.smoothness_parameter * bregman
            )
            if (ith_iteration + 1) % log_freq == 0 or ith_iteration == 0:
                self.callback(ith_iteration, testing_loss, grad_f_W_k)
            while not non_euclidean_descent_lemma_cond:
                flag = 1
                inner_loop_it += 1
                # Detect overflow and exit if necessary
                if self.rho_increase**inner_loop_it > np.finfo(float).max:
                    self.logger.warning(
                        "[Overflow Detected] Iteration %d, Inner Loop %d: "
                        "Exiting to prevent overflow.",
                        ith_iteration,
                        inner_loop_it,
                    )
                    break
                if inner_loop_it == self.threshold:
                    break
                # Adjust step size
                self.smoothness_parameter *= self.rho_increase**inner_loop_it
                step_size = (1 + self.symmetry_parameter) / self.smoothness_parameter

                substep_res = self.substep(W_k, tau, step_size, grad_f_W_k)
                if substep_res is None:
                    break
                W_k_next, res_norm_next_it = substep_res
                linear_approx = (grad_f_W_k.T @ (W_k_next - W_k)).sum()
                bregman = self.bregman_distance(W_k_next, W_k, tau)
                # Detect overflow and exit if necessary
                if self.rho_increase**inner_loop_it > np.finfo(float).max:
                    self.logger.warning(
                        "[Overflow Detected] Iteration %d, Inner Loop %d: "
                        "Exiting to prevent overflow.",
                        ith_iteration,
                        inner_loop_it,
                    )
                    break
                non_euclidean_descent_lemma_cond = (
                    res_norm_next_it
                    <= res_norm + linear_approx + self.smoothness_parameter * bregman
                )
            # Break if loss becomes NaN
            if np.isnan(res_norm_next_it):
                self.logger.warning(
                    "[NaN Loss] Iteration %d: Loss is NaN, exiting loop.", ith_iteration
                )
                break
            if flag == 1:
                # Adjust step size
                self.smoothness_parameter *= self.rho_decrease

            # Update variables for the next iteration
            W_k = W_k_next
            step_size = (1 + self.symmetry_parameter) / self.smoothness_parameter

            training_loss = res_norm_next_it / nb_elements
            testing_loss = self.calculate_rmse(self.test_mask)
            loss.append(training_loss)
            rmse.append(testing_loss)
            res_norm = res_norm_next_it
            if self.early_stopping is not None and self.early_stopping(
                testing_loss, W_k
            ):
                self.set_weights(self.early_stopping.best_weights)
                self.logger.debug("[Early Stopping] Training interrupted.")
                if ith_iteration % log_freq != 0:
                    self.callback(ith_iteration, testing_loss, grad_f_W_k)
                break
        # Compute runtime
        runtime = time.time() - start_time
        self.logger.debug(
            "[Completion] Optimization finished in %.2f seconds.", runtime
        )
        training_data = Result(
            loss_history=loss,
            iterations=iterations_count,
            rmse_history=rmse,
            runtime=runtime,
        )
        return training_data
