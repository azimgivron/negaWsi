from typing import Callable, Dict, Tuple

import numpy as np
import optuna
import pytest
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


@pytest.fixture
def tune_regularization() -> Callable[
    [type, Dict[str, Dict[str, float | bool]], Dict, int], Dict[str, float]
]:
    """Return a tuner that runs a small Optuna search for stable regularization.

    Args:
        model_cls: Model class supporting `regularization_parameters` and `run`.
        reg_space: Search space describing bounds for each regularization key.
        kwargs: Keyword arguments passed to the model constructor.
        n_trials: Number of Optuna trials to execute.

    Returns:
        Callable: Function that accepts (model_cls, reg_space, kwargs, n_trials)
        and returns the best hyperparameter dictionary.
    """

    def _tune(
        model_cls: type,
        reg_space: Dict[str, Dict[str, float | bool]],
        kwargs: Dict,
        n_trials: int,
    ) -> Dict[str, float]:
        def objective(trial: optuna.Trial) -> float:
            reg = {
                name: trial.suggest_float(name, **params)
                for name, params in reg_space.items()
            }
            model = model_cls(regularization_parameters=reg, **kwargs)
            training_status = model.run()
            return training_status.rmse_history[-1]

        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        return study.best_params

    return _tune


@pytest.fixture
def rank2_case() -> Tuple[Matrix, Mask, Mask]:
    """Provide the base rank-2 reconstruction toy problem.

    Returns:
        Tuple[Matrix, Mask, Mask]: (matrix, train_mask, test_mask).
    """
    h1_true = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )  # (4, 2)

    h2_true = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )  # (2, 4)

    R = h1_true @ h2_true  # (4,4)

    train_mask = np.array(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=bool,
    )

    test_mask = ~train_mask.copy()
    return R, train_mask, test_mask


@pytest.fixture
def side_info_case() -> Tuple[Matrix, Mask, Mask, Matrix, Matrix]:
    """Provide the inductive side-information toy problem for NegaFS.

    Returns:
        Tuple[Matrix, Mask, Mask, Matrix, Matrix]: (matrix, train_mask,
        test_mask, gene_features, disease_features).
    """
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )  # (4,3)

    Y = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )  # (3,4)

    H1_true = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
        ]
    )  # (3, 2)

    H2_true = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
        ]
    )  # (2, 3)

    R = X @ H1_true @ H2_true @ Y  # (4,4)

    train_mask = np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=bool,
    )

    test_mask = ~train_mask.copy()
    return R, train_mask, test_mask, X, Y.T


@pytest.fixture
def reg_side_info_case() -> Tuple[Matrix, Mask, Mask, Matrix, Matrix]:
    """Provide the side-information-regularized toy problem for NegaReg.

    Returns:
        Tuple[Matrix, Mask, Mask, Matrix, Matrix]: (matrix, train_mask,
        test_mask, gene_features, disease_features).
    """
    H1_true = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )  # (4,2)

    H2_true = np.array(
        [
            [1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    )  # (2,4)

    A = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )  # (2,3)

    B = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )  # (3,2)

    X = H1_true @ A  # (4,3)
    Y = B @ H2_true  # (3,4)

    R = H1_true @ H2_true  # (4,4)

    train_mask = np.array(
        [
            [True, True, False, True],
            [True, True, True, True],
            [True, False, True, True],
            [True, True, True, False],
        ],
        dtype=bool,
    )

    test_mask = ~train_mask.copy()

    return R, train_mask, test_mask, X, Y.T


@pytest.fixture
def reg_space_nega() -> Dict[str, Dict[str, float | bool]]:
    """Regularization search space for the base Nega model.

    Returns:
        Dict[str, Dict[str, float | bool]]: Search bounds keyed by parameter.
    """
    return {
        "λg": {"low": 1e-4, "high": 1e2, "log": True},
        "λd": {"low": 1e-4, "high": 1e2, "log": True},
    }


@pytest.fixture
def reg_space_nega_reg() -> Dict[str, Dict[str, float | bool]]:
    """Regularization search space for the NegaReg model.

    Returns:
        Dict[str, Dict[str, float | bool]]: Search bounds keyed by parameter.
    """
    return {
        "λg": {"low": 1e-4, "high": 1e2, "log": True},
        "λd": {"low": 1e-4, "high": 1e2, "log": True},
        "λ_βg": {"low": 1e-2, "high": 1e2, "log": True},
        "λ_βd": {"low": 1e-2, "high": 1e2, "log": True},
    }
