from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.models import build_regressor


def test_ridge_multi_output_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4)).astype(np.float32)
    Y = rng.normal(size=(12, 3)).astype(np.float32)

    model = build_regressor(
        {
            "name": "ridge",
            "target_space": "a",
            "alpha": 1.0,
            "cond_scaler": "standardize",
            "seed": 0,
        }
    )
    model.fit(X, Y)
    Y_hat = model.predict(X)

    assert Y_hat.shape == Y.shape
    assert np.isfinite(Y_hat).all()


def test_ridge_reproducible_seed() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3)).astype(np.float32)
    Y = rng.normal(size=(10, 2)).astype(np.float32)

    cfg = {
        "name": "ridge",
        "target_space": "z",
        "alpha": 0.5,
        "cond_scaler": "none",
        "seed": 42,
    }

    model_a = build_regressor(cfg)
    model_b = build_regressor(cfg)
    model_a.fit(X, Y)
    model_b.fit(X, Y)

    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)

    assert np.allclose(pred_a, pred_b)


def test_ridge_target_space_validation() -> None:
    with pytest.raises(ValueError):
        _ = build_regressor({"name": "ridge", "target_space": "bad"})
