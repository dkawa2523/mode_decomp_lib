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


def test_fourier_bessel_neumann_includes_dc_mode() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    assert domain.mask is not None

    # Constant field inside disk, zero outside.
    field = np.zeros((height, width, 1), dtype=np.float64)
    field[..., 0][domain.mask] = 1.0

    cfg = {
        "name": "fourier_bessel",
        "m_max": 0,
        "n_max": 1,
        "ordering": "m_then_n",
        "normalization": "orthonormal",
        "boundary_condition": "neumann",
        "mask_policy": "ignore_masked_points",
    }
    decomposer = build_decomposer(cfg)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    max_err = float(np.max(np.abs(field_hat[..., 0][domain.mask] - 1.0)))
    assert max_err < 1e-3

    meta = decomposer.coeff_meta()
    assert meta["boundary_condition"] == "neumann"
    mode_list = meta["mode_list"]
    assert len(mode_list) == 1
    assert mode_list[0][0] == 0  # m
    assert mode_list[0][1] == 0  # n (DC)
    assert abs(float(mode_list[0][3]) - 0.0) < 1e-12  # root

