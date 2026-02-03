from __future__ import annotations

import numpy as np

from mode_decomp_ml.plugins.models import build_regressor


def test_ridge_single_output_shape() -> None:
    rng = np.random.default_rng(10)
    X = rng.normal(size=(9, 3)).astype(np.float32)
    y = rng.normal(size=9).astype(np.float32)

    model = build_regressor(
        {
            "name": "ridge",
            "target_space": "a",
            "alpha": 1.0,
            "cond_scaler": "standardize",
            "seed": 0,
        }
    )
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (9, 1)


def test_gpr_single_output_shape() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(7, 2)).astype(np.float32)
    y = rng.normal(size=7).astype(np.float32)

    model = build_regressor(
        {
            "name": "gpr",
            "target_space": "z",
            "cond_scaler": "standardize",
            "kernel": "rbf",
            "kernel_constant": 1.0,
            "kernel_constant_bounds": (1.0e-3, 1.0e3),
            "length_scale": 1.0,
            "length_scale_bounds": (1.0e-2, 1.0e2),
            "white_noise": True,
            "noise_level": 1.0e-6,
            "noise_level_bounds": (1.0e-8, 1.0e1),
            "alpha": 1.0e-6,
            "normalize_y": True,
            "optimizer": None,
            "n_restarts_optimizer": 0,
            "seed": 0,
        }
    )
    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (7, 1)


def test_elasticnet_multi_output_wrapper_shape() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(10, 4)).astype(np.float32)
    Y = rng.normal(size=(10, 3)).astype(np.float32)

    model = build_regressor(
        {
            "name": "elasticnet",
            "target_space": "a",
            "alpha": 0.1,
            "l1_ratio": 0.3,
            "cond_scaler": "none",
            "seed": 0,
        }
    )
    model.fit(X, Y)
    pred = model.predict(X)

    assert pred.shape == Y.shape
