from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.plugins.models import build_regressor


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(8, 4)).astype(np.float32)
    Y = rng.normal(size=(8, 3)).astype(np.float32)
    return X, Y


def test_xgb_multioutput_shape() -> None:
    pytest.importorskip("xgboost")
    X, Y = _make_data(10)
    model = build_regressor(
        {
            "name": "xgb",
            "target_space": "z",
            "cond_scaler": "none",
            "seed": 0,
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.2,
        }
    )
    model.fit(X, Y)
    pred = model.predict(X)

    assert pred.shape == Y.shape


def test_lgbm_multioutput_shape() -> None:
    pytest.importorskip("lightgbm")
    X, Y = _make_data(11)
    model = build_regressor(
        {
            "name": "lgbm",
            "target_space": "z",
            "cond_scaler": "none",
            "seed": 0,
            "n_estimators": 10,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 1,
            "learning_rate": 0.2,
        }
    )
    model.fit(X, Y)
    pred = model.predict(X)

    assert pred.shape == Y.shape


def test_catboost_multioutput_shape() -> None:
    pytest.importorskip("catboost")
    X, Y = _make_data(12)
    model = build_regressor(
        {
            "name": "catboost",
            "target_space": "z",
            "cond_scaler": "none",
            "seed": 0,
            "n_estimators": 10,
            "depth": 4,
            "learning_rate": 0.2,
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    model.fit(X, Y)
    pred = model.predict(X)

    assert pred.shape == Y.shape
