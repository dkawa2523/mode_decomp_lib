from __future__ import annotations

import numpy as np
import pytest

from ml_platform_mode_decomp.regressors import build_regressor


@pytest.mark.parametrize("outputs", [1, 2])
def test_gpr_interpolates_single_and_multioutput_targets_with_finite_deviation(outputs) -> None:
    x = np.linspace(-1, 1, 7)[:, None]
    targets = np.column_stack([np.sin(x[:, 0]), 10 * np.cos(x[:, 0])])[:, :outputs]
    model = build_regressor({"name": "gpr", "optimizer": None, "white_noise": False, "alpha": 1e-10})
    model.fit(x, targets[:, 0] if outputs == 1 else targets)
    mean, std = model.predict_with_std(x)
    np.testing.assert_allclose(mean, targets, atol=1e-5)
    assert std is not None and std.shape == targets.shape
    assert np.isfinite(std).all()
    assert (std >= 0).all()


def test_gpr_kernel_validation() -> None:
    with pytest.raises(ValueError):
        build_regressor({"name": "gpr", "kernel": "bad_kernel"})
