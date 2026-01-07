from typing import Callable, Dict, Tuple

import numpy as np
from negaWsi.standard.nega import Nega
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


def test_nega_without_masks(
    nega_case: Tuple[Matrix, Mask, Mask, int],
):
    """Ensure Nega converges when train/test masks cover the full matrix.

    Args:
        nega_case: Fixture providing the synthetic matrix.
    """
    R, _, _, _, rank = nega_case
    train_mask = np.ones_like(R, dtype=bool)

    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=train_mask.copy(),
        rank=rank,
        iterations=700,
        symmetry_parameter=0.99,
        lipschitz_smoothness=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        seed=0,
        svd_init=False,
        regularization_parameters={"λg": 0, "λd": 0},
    )

    model = Nega(**kwargs)
    _ = model.run()

    R_hat = model.predict_all()
    nrmse = np.sqrt(((R_hat.ravel() - R.ravel()) ** 2).mean()) / (R.max() - R.min())
    nrmse_random = np.sqrt(
        (
            (np.random.uniform(R.min(), R.max(), R_hat.shape).ravel() - R.ravel()) ** 2
        ).mean()
    ) / (R.max() - R.min())
    assert nrmse < 0.1 * nrmse_random, (
        f"Reconstruction mismatch:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}\training mask=\n{train_mask}\nNRMSE:{nrmse} while random has {nrmse_random}"
    )


def test_nega_with_masks(
    nega_case: Tuple[Matrix, Mask, Mask, int],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega: Dict[str, Dict[str, float | bool]],
):
    """Ensure Nega reconstructs the small rank-2 matrix.

    Args:
        nega_case: Fixture providing (matrix, train_mask, test_mask, rank).
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega: Search space for λg and λd.
    """
    R, train_mask, val_mask, _, rank = nega_case
    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=val_mask,
        rank=rank,
        iterations=1000,
        symmetry_parameter=0.99,
        lipschitz_smoothness=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(Nega, reg_space_nega, kwargs, n_trials=20)
    model = Nega(regularization_parameters=best_params, **kwargs)
    _ = model.run()

    R_hat = model.predict_all()
    nrmse = np.sqrt(((R_hat.ravel() - R.ravel()) ** 2).mean()) / (R.max() - R.min())
    nrmse_random = np.sqrt(
        (
            (np.random.uniform(R.min(), R.max(), R_hat.shape).ravel() - R.ravel()) ** 2
        ).mean()
    ) / (R.max() - R.min())
    assert nrmse < 0.1 * nrmse_random, (
        f"Reconstruction mismatch with tuned params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}\training mask=\n{train_mask}\nNRMSE:{nrmse} while random has {nrmse_random}"
    )
