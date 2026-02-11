from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def _rbf_cfg(*, kernel: str, stride: int, length_scale: float, ridge_alpha: float):
    return {
        "name": "rbf_expansion",
        "kernel": str(kernel),
        "centers": "stride",
        "stride": int(stride),
        "length_scale": float(length_scale),
        "ridge_alpha": float(ridge_alpha),
        "mask_policy": "ignore_masked_points",
    }


def test_rbf_expansion_roundtrip_known_coeffs_no_mask() -> None:
    domain = _rectangle_domain(9, 9)
    cfg = _rbf_cfg(kernel="gaussian", stride=3, length_scale=0.25, ridge_alpha=0.0)
    decomposer = build_decomposer(cfg)

    # Initialize coeff shape for inverse_transform.
    _ = decomposer.transform(np.zeros((9, 9, 1), dtype=np.float64), mask=None, domain_spec=domain)

    # Create a field from known coefficients.
    k = len(range(0, 9, 3)) ** 2
    coeff_true = np.linspace(-0.5, 0.5, k, dtype=np.float64).reshape(1, k)
    field = decomposer.inverse_transform(coeff_true, domain_spec=domain)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)

    max_err = float(np.max(np.abs(np.asarray(coeff) - coeff_true)))
    assert max_err < 1e-8


def test_rbf_expansion_roundtrip_with_varying_mask() -> None:
    domain = _rectangle_domain(17, 17)
    cfg = _rbf_cfg(kernel="thin_plate", stride=8, length_scale=0.5, ridge_alpha=1e-6)
    decomposer = build_decomposer(cfg)

    # Initialize coeff shape for inverse_transform.
    _ = decomposer.transform(np.zeros((17, 17, 1), dtype=np.float64), mask=None, domain_spec=domain)

    k = len(range(0, 17, 8)) ** 2
    rng = np.random.default_rng(0)
    coeff_true = rng.normal(size=(1, k)).astype(np.float64)
    field = decomposer.inverse_transform(coeff_true, domain_spec=domain)

    # Simulate per-sample missingness: only a subset is observed.
    mask = rng.random((17, 17)) > 0.35
    field_obs = np.asarray(field).copy()
    field_obs[~mask] = 0.0

    coeff = decomposer.transform(field_obs, mask=mask, domain_spec=domain)
    # Under sufficient observations, coefficients should be close.
    max_err = float(np.max(np.abs(np.asarray(coeff) - coeff_true)))
    assert max_err < 1e-2
