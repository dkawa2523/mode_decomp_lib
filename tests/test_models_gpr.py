from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.plugins.models import build_regressor


def _base_gpr_cfg() -> dict[str, object]:
    return {
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


def test_gpr_multi_output_shapes() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3)).astype(np.float32)
    Y = rng.normal(size=(8, 2)).astype(np.float32)

    model = build_regressor(_base_gpr_cfg())
    model.fit(X, Y)
    Y_hat = model.predict(X)

    assert Y_hat.shape == Y.shape
    assert np.isfinite(Y_hat).all()


def test_gpr_reproducible_seed() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2)).astype(np.float32)
    Y = rng.normal(size=(6, 1)).astype(np.float32)

    cfg = _base_gpr_cfg()
    cfg["seed"] = 123

    model_a = build_regressor(cfg)
    model_b = build_regressor(cfg)
    model_a.fit(X, Y)
    model_b.fit(X, Y)

    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)

    assert np.allclose(pred_a, pred_b)


def test_gpr_predict_with_std_shapes() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(7, 3)).astype(np.float32)
    Y = rng.normal(size=(7, 2)).astype(np.float32)

    model = build_regressor(_base_gpr_cfg())
    model.fit(X, Y)
    mean, std = model.predict_with_std(X)

    assert mean.shape == Y.shape
    assert std is not None
    assert std.shape == Y.shape
    assert np.isfinite(std).all()
    assert (std >= 0).all()


def test_gpr_kernel_validation() -> None:
    cfg = _base_gpr_cfg()
    cfg["kernel"] = "bad_kernel"
    with pytest.raises(ValueError):
        _ = build_regressor(cfg)
