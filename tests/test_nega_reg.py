from typing import Callable, Dict, Tuple

import numpy as np
from negaWsi.side_info.nega_reg import NegaReg
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


def test_nega_reg(
    reg_side_info_case: Tuple[Matrix, Mask, Mask, Matrix, Matrix, int],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega_reg: Dict[str, Dict[str, float | bool]],
):
    """Ensure NegaReg leverages side information via regularization to reconstruct.

    Args:
        reg_side_info_case: Fixture providing (matrix, train_mask, test_mask, X, Y, rank).
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega_reg: Search space for λg, λd, λ_βg, and λ_βd.
    """
    R, train_mask, val_mask, _, X, Y, rank = reg_side_info_case
    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=val_mask,
        rank=rank,
        side_info=(X, Y),
        iterations=1000,
        symmetry_parameter=0.99,
        lipschitz_smoothness=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(NegaReg, reg_space_nega_reg, kwargs, n_trials=50)
    model = NegaReg(regularization_parameters=best_params, **kwargs)
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
