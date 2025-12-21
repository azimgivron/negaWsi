from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from negaWsi.nega_fs import NegaFS

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


def test_nega_fs(
    side_info_case: Tuple[Matrix, Mask, Mask, Matrix, Matrix, int],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega: Dict[str, Dict[str, float | bool]],
):
    """Ensure NegaFS reconstructs the matrix when provided side information.

    Args:
        side_info_case: Fixture providing (matrix, train_mask, test_mask, X, Y, rank).
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega: Search space for λg and λd.
    """
    R, train_mask, val_maks, _, X, Y, rank = side_info_case
    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=val_maks,
        rank=rank,
        side_info=(X, Y),
        iterations=20_000,
        symmetry_parameter=0.99,
        smoothness_parameter=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(NegaFS, reg_space_nega, kwargs, n_trials=20)
    model = NegaFS(regularization_parameters=best_params, **kwargs)
    _ = model.run()

    R_hat = model.predict_all()
    assert np.allclose(R_hat, R, atol=1e-2), (
        f"Reconstruction mismatch with tuned params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}\training mask=\n{train_mask}"
    )