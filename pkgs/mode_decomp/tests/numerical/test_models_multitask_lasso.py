from __future__ import annotations

import numpy as np

from ml_platform_mode_decomp.regressors import build_regressor


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
            "alpha": 0.5,
            "cond_scaler": "none",
            "seed": 0,
        }
    )
    model.fit(X, Y)

    effects = model.predict(np.eye(n_features)) - model.predict(np.zeros((1, n_features)))
    assert np.all(np.abs(effects[0]) > 0.5)
    assert np.all(np.abs(effects[1:]) < 1.0e-3)
