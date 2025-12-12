from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from negaWsi.nega import Nega

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


def test_nega_recovers_rank2_matrix(
    rank2_case: Tuple[Matrix, Mask, Mask],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega: Dict[str, Dict[str, float | bool]],
):
    """Ensure Nega reconstructs the small rank-2 matrix.

    Args:
        rank2_case: Fixture providing (matrix, train_mask, test_mask).
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega: Search space for λg and λd.
    """
    R, train_mask, test_mask = rank2_case
    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=test_mask,
        rank=2,
        iterations=5000,
        symmetry_parameter=0.99,
        smoothness_parameter=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        threshold=100,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(Nega, reg_space_nega, kwargs, n_trials=8)
    model = Nega(regularization_parameters=best_params, **kwargs)
    _ = model.run()

    R_hat = model.predict_all()
    assert np.allclose(R_hat, R, atol=1e-3), (
        f"Reconstruction mismatch with tuned params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}"
    )

def test_nega_recovers_rank2_matrix_without_masks(
    rank2_case: Tuple[Matrix, Mask, Mask],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega: Dict[str, Dict[str, float | bool]],
):
    """Ensure Nega converges when train/test masks cover the full matrix.

    Args:
        rank2_case: Fixture providing the synthetic matrix.
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega: Search space for λg and λd.
    """
    R, _, _ = rank2_case
    train_mask = np.ones_like(R, dtype=bool)

    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=train_mask.copy(),
        rank=2,
        iterations=1000,
        symmetry_parameter=0.99,
        smoothness_parameter=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        threshold=100,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(Nega, reg_space_nega, kwargs, n_trials=8)
    model = Nega(regularization_parameters=best_params, **kwargs)
    _ = model.run()

    R_hat = model.predict_all()
    assert np.allclose(R_hat, R, atol=1e-3), (
        f"Reconstruction mismatch with tuned params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}"
    )
