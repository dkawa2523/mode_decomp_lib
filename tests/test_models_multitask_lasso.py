from __future__ import annotations

import numpy as np

from mode_decomp_ml.plugins.models import build_regressor


def test_multitask_lasso_multi_output_shapes() -> None:
    rng = np.random.default_rng(20)
    X = rng.normal(size=(16, 4)).astype(np.float32)
    Y = rng.normal(size=(16, 3)).astype(np.float32)

    model = build_regressor(
        {
            "name": "multitask_lasso",
            "target_space": "a",
            "alpha": 0.05,
            "cond_scaler": "standardize",
            "seed": 0,
        }
    )
    model.fit(X, Y)
    pred = model.predict(X)

    assert pred.shape == Y.shape
    assert np.isfinite(pred).all()


def test_multitask_lasso_promotes_shared_sparsity() -> None:
    rng = np.random.default_rng(21)
    n_samples = 40
    n_features = 5
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    weights = np.array([[2.0, -1.5]], dtype=np.float32)
    Y = X[:, :1] @ weights

    model = build_regressor(
        {
            "name": "multitask_lasso",
            "target_space": "a",
            "alpha": 0.5,
            "cond_scaler": "none",
            "seed": 0,
        }
    )
    model.fit(X, Y)

    fitted = getattr(model, "_model", None)
    assert fitted is not None
    coef = np.asarray(fitted.coef_)

    assert coef.shape == (2, n_features)
    assert np.all(np.abs(coef[:, 1:]) < 1.0e-3)
