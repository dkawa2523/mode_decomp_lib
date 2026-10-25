from __future__ import annotations

import numpy as np
import pytest

from ml_platform_mode_decomp.regressors import build_regressor


@pytest.mark.parametrize("name", ["ridge", "elasticnet", "multitask_elasticnet", "multitask_lasso", "gpr"])
def test_failed_refit_invalidates_predictions_with_the_mutated_scaler(name):
    model = build_regressor({"name": name, **({"optimizer": None} if name == "gpr" else {})})
    conditions = np.arange(8.0)[:, None]
    targets = np.column_stack((conditions[:, 0], conditions[:, 0] ** 2))
    model.fit(conditions, targets)
    assert np.isfinite(model.predict(conditions[:2])).all()
    invalid = targets.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(conditions + 100, invalid)
    with pytest.raises(ValueError, match="must be fit"):
        model.predict(conditions[:2])
    with pytest.raises(ValueError, match="before fit"):
        model.state()
    model.fit(conditions, targets)
    assert np.isfinite(model.predict(conditions[:2])).all()


@pytest.mark.parametrize("outputs", [1, 3])
def test_ridge_single_and_multi_output_predictions(outputs) -> None:
    rng = np.random.default_rng(10)
    X = rng.normal(size=(9, 3)).astype(np.float32)
    weights = np.arange(1, 3 * outputs + 1).reshape(3, outputs)
    y = X @ weights

    model = build_regressor(
        {
            "name": "ridge",
            "alpha": 0.0,
            "cond_scaler": "standardize",
            "seed": 0,
        }
    )
    model.fit(X, y[:, 0] if outputs == 1 else y)
    query = rng.normal(size=(2, 3)).astype(np.float32)
    pred = model.predict(query)

    assert pred.shape == (2, outputs)
    np.testing.assert_allclose(pred, query @ weights, atol=1e-5)


def test_elasticnet_multi_output_wrapper_matches_orthogonal_design_solution() -> None:
    rng = np.random.default_rng(12)
    X = np.vstack((np.eye(4), -np.eye(4)))
    coefficients = rng.normal(size=(4, 3))
    offsets = np.array([1.0, -2.0, 3.0])
    Y = X @ coefficients + offsets

    model = build_regressor(
        {
            "name": "elasticnet",
            "alpha": 0.1,
            "l1_ratio": 0.3,
            "cond_scaler": "none",
            "seed": 0,
        }
    )
    model.fit(X, Y)
    query = rng.normal(size=(3, 4))
    # X.T @ X / n = I/4, so each target has a closed-form shrinkage solution.
    shrunk = np.sign(coefficients) * np.maximum(np.abs(coefficients) - 4 * 0.1 * 0.3, 0) / (1 + 4 * 0.1 * 0.7)
    np.testing.assert_allclose(model.predict(query), query @ shrunk + offsets, atol=1e-12)
