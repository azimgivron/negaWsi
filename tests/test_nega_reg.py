from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from negaWsi.nega_reg import NegaReg

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


def test_nega_reg_uses_side_info_regularization(
    reg_side_info_case: Tuple[Matrix, Mask, Mask, Matrix, Matrix],
    tune_regularization: Callable[[type, Dict, Dict, int], Dict[str, float]],
    reg_space_nega_reg: Dict[str, Dict[str, float | bool]],
):
    """Ensure NegaReg leverages side information via regularization to reconstruct.

    Args:
        reg_side_info_case: Fixture providing (matrix, train_mask, test_mask, X, Y).
        tune_regularization: Fixture that tunes regularization parameters with Optuna.
        reg_space_nega_reg: Search space for λg, λd, λ_βg, and λ_βd.
    """
    R, train_mask, val_mask, test_mask, X, Y, rank = reg_side_info_case
    kwargs = dict(
        matrix=R,
        train_mask=train_mask,
        test_mask=val_mask,
        rank=rank,
        side_info=(X, Y),
        iterations=10_000,
        symmetry_parameter=0.99,
        smoothness_parameter=0.001,
        rho_increase=10.0,
        rho_decrease=0.1,
        seed=0,
        svd_init=False,
    )

    best_params = tune_regularization(NegaReg, reg_space_nega_reg, kwargs, n_trials=100)
    kwargs["test_mask"] = test_mask
    model = NegaReg(regularization_parameters=best_params, **kwargs)
    training_history = model.run()

    R_hat = model.predict_all()
    assert np.allclose(R_hat, R, atol=1e-3), (
        f"Reconstruction mismatch with tuned params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}"
    )
    rel_rmse = np.linalg.norm(R_hat - R) / np.linalg.norm(R)
    assert rel_rmse < 1e-3, (
        f"Nega reconstruction mismatch with params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}\ntrain mask=\n{train_mask}\n"
        f"RelRMSE=\n{rel_rmse}\n"
    )
    assert training_history.rmse_history[-1] < 1e-3, (
        f"Nega reconstruction mismatch with params {best_params}:\n"
        f"pred=\n{R_hat}\ntruth=\n{R}\ntrain mask=\n{train_mask}\n"
        f"Final RMSE on Test=\n{training_history.rmse_history[-1]}\n"
    )
