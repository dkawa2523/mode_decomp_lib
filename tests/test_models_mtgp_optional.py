from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.plugins.models import build_regressor


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 3)).astype(np.float32)
    Y = rng.normal(size=(6, 2)).astype(np.float32)
    return X, Y


def test_mtgp_predict_std_shapes() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    X, Y = _make_data(5)
    model = build_regressor(
        {
            "name": "mtgp",
            "target_space": "z",
            "cond_scaler": "none",
            "seed": 0,
            "training_iter": 3,
            "learning_rate": 0.1,
            "rank": 1,
        }
    )
    model.fit(X, Y)
    pred, std = model.predict_with_std(X)

    assert pred.shape == Y.shape
    assert std is not None
    assert std.shape == Y.shape
