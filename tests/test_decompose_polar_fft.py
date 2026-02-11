from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _disk_domain(height: int, width: int):
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _polar_cfg(n_r: int, n_theta: int):
    return {
        "name": "polar_fft",
        "n_r": int(n_r),
        "n_theta": int(n_theta),
        "radial_transform": "dct",
        "angular_transform": "fft",
        "interpolation": "bilinear",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }


def test_polar_fft_roundtrip_constant_approx() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    field = np.ones((height, width, 1), dtype=np.float32)

    decomposer = build_decomposer(_polar_cfg(n_r=33, n_theta=64))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    meta = decomposer.coeff_meta()

    assert meta["complex_format"] == "complex"
    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[..., 0][domain.mask] - 1.0)))
    assert max_err < 2e-2

