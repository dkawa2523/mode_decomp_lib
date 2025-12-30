from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.evaluate import compute_metrics


def test_div_curl_rmse() -> None:
    height, width = 5, 7
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    field_true = np.stack([xx, yy], axis=-1)
    field_pred = np.stack([xx, 2.0 * yy], axis=-1)

    metrics = compute_metrics(
        ["div_rmse", "curl_rmse"],
        field_true=field_true,
        field_pred=field_pred,
        grid_spacing=(1.0, 1.0),
    )

    assert metrics["div_rmse"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["curl_rmse"] == pytest.approx(0.0, abs=1e-6)
