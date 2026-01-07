from typing import Callable, Dict, Tuple

import numpy as np
import optuna
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
Mask = NDArray[np.bool_]


@pytest.fixture
def tune_regularization() -> (
    Callable[[type, Dict[str, Dict[str, float | bool]], Dict, int], Dict[str, float]]
):
    """Return a tuner that runs a small Optuna search for stable regularization.

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
        """Fine tuning function.

        Args:
            model_cls: Model class supporting `regularization_parameters` and `run`.
            reg_space: Search space describing bounds for each regularization key.
            kwargs: Keyword arguments passed to the model constructor.
            n_trials: Number of Optuna trials to execute.

        Returns:
            Dict[str, float]: The best parameters.
        """

        def objective(trial: optuna.Trial) -> float:
            reg = {
                name: trial.suggest_float(name, **params)
                for name, params in reg_space.items()
            }
            model = model_cls(regularization_parameters=reg, **kwargs)
            training_status = model.run()
            return training_status.test_rmse_history[-1]

        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)
        return study.best_params

    return _tune


def _masks(R: Matrix, p_testval: float, p_val_within: float) -> Tuple[Mask, Mask, Mask]:
    """Create train/val/test masks for a matrix given split proportions.

    Args:
        R: Input matrix to match mask shapes.
        p_testval: Fraction of entries reserved for validation+test.
        p_val_within: Fraction of testval entries used for validation.

    Returns:
        Tuple[Mask, Mask, Mask]: (train_mask, val_mask, test_mask).
    """
    N = R.size

    n_testval = int(np.ceil(N * p_testval))
    n_val = int(np.ceil(n_testval * p_val_within))

    rng = np.random.default_rng(seed=0)
    idx = rng.permutation(N)  # all indices, shuffled
    testval = idx[:n_testval]
    val_idx = testval[:n_val]
    test_idx = testval[n_val:]

    train_mask = np.ones(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)

    train_mask[testval] = False
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    train_mask = train_mask.reshape(R.shape)
    val_mask = val_mask.reshape(R.shape)
    test_mask = test_mask.reshape(R.shape)
    return train_mask, val_mask, test_mask


@pytest.fixture(
    params=[
        {"n": 10, "m": 10, "k": 4, "seed": 0},
        {"n": 50, "m": 50, "k": 18, "seed": 0},
        {"n": 10, "m": 10, "k": 4, "seed": 1},
        {"n": 50, "m": 50, "k": 18, "seed": 1},
    ]
)
def nega_case(request: pytest.FixtureRequest) -> Tuple[Matrix, Mask, Mask, Mask, int]:
    """Provide the reconstruction toy problem.

    Returns:
        Tuple[Matrix, Mask, Mask, Mask, int]: (matrix, train_mask,
        val_mask, test_mask, rank).
    """
    params = request.param
    n = params["n"]
    m = params["m"]
    k = params["k"]
    seed = params["seed"]

    rng = np.random.default_rng(seed)
    H1_true = rng.random((n, k))
    H2_true = rng.random((k, m))
    R = H1_true @ H2_true

    p_testval = 0.1
    p_val_within = 0.40

    train_mask, val_mask, test_mask = _masks(R, p_testval, p_val_within)

    return R, train_mask, val_mask, test_mask, k


@pytest.fixture(
    params=[
        {"n": 10, "m": 10, "k": 4, "seed": 0},
        {"n": 50, "m": 50, "k": 18, "seed": 0},
        {"n": 10, "m": 10, "k": 4, "seed": 1},
        {"n": 50, "m": 50, "k": 18, "seed": 1},
    ]
)
def sparse_nega_case(
    request: pytest.FixtureRequest,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, int]:
    """Provide the reconstruction toy problem.

    Returns:
        Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, int]: (matrix, train_mask,
        val_mask, test_mask, rank).
    """
    params = request.param
    n = params["n"]
    m = params["m"]
    k = params["k"]
    seed = params["seed"]

    rng = np.random.default_rng(seed)
    success_prob = 0.35
    H1_true = rng.binomial(1, p=success_prob, size=(n, k))
    H2_true = rng.binomial(1, p=success_prob, size=(k, m))
    R = H1_true @ H2_true

    p_testval = 0.1
    p_val_within = 0.40

    train_mask, val_mask, test_mask = _masks(R, p_testval, p_val_within)
    R = sp.csr_matrix(R.astype(float))
    train_mask = sp.csr_matrix(train_mask)
    val_mask = sp.csr_matrix(val_mask)
    test_mask = sp.csr_matrix(test_mask)

    return R, train_mask, val_mask, test_mask, k


@pytest.fixture(
    params=[
        {"n": 5, "m": 5, "p": 6, "q": 6, "k": 4},
    ]
)
def side_info_case(
    request: pytest.FixtureRequest,
) -> Tuple[Matrix, Mask, Mask, Matrix, Matrix, int]:
    """Provide the inductive side-information toy problem for NegaFS.

    Returns:
        Tuple[Matrix, Mask, Mask, Mask, Matrix, Matrix, int]: (matrix, train_mask,
        val_mask, test_mask, gene_features, disease_features, rank).
    """
    params = request.param
    n = params["n"]
    m = params["m"]
    p = params["p"]
    q = params["q"]
    k = params["k"]

    rng = np.random.default_rng(0)
    X = rng.random((n, p))
    Y = rng.random((q, m))
    H1_true = rng.random((p, k))
    H2_true = rng.random((k, q))
    R = X @ H1_true @ H2_true @ Y

    p_testval = 0.1
    p_val_within = 0.40

    train_mask, val_mask, test_mask = _masks(R, p_testval, p_val_within)

    return R, train_mask, val_mask, test_mask, X, Y.T, k


@pytest.fixture(
    params=[
        {"n": 5, "m": 5, "p": 6, "q": 6, "k": 4},
    ]
)
def reg_side_info_case(
    request: pytest.FixtureRequest,
) -> Tuple[Matrix, Mask, Mask, Matrix, Matrix, int]:
    """Provide the side-information-regularized toy problem for NegaReg.

    Returns:
        Tuple[Matrix, Mask, Mask, Mask, Matrix, Matrix, int]: (matrix, train_mask,
        val_mask, test_mask, gene_features, disease_features, rank).
    """
    params = request.param
    n = params["n"]
    m = params["m"]
    p = params["p"]
    q = params["q"]
    k = params["k"]

    rng = np.random.default_rng(0)
    A = rng.random((k, p))
    B = rng.random((q, k))
    H1_true = rng.random((n, k))
    H2_true = rng.random((k, m))

    X = H1_true @ A
    Y = B @ H2_true
    R = H1_true @ H2_true

    p_testval = 0.1
    p_val_within = 0.40

    train_mask, val_mask, test_mask = _masks(R, p_testval, p_val_within)

    return R, train_mask, val_mask, test_mask, X, Y.T, k


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
