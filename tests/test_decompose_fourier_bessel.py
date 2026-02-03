from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jv

from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


def _disk_domain(height: int, width: int):
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _fb_cfg(m_max: int, n_max: int, normalization: str):
    return {
        "name": "fourier_bessel",
        "m_max": m_max,
        "n_max": n_max,
        "ordering": "m_then_n",
        "normalization": normalization,
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }


def test_fourier_bessel_roundtrip_single_mode() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    r = domain.coords["r"]
    theta = domain.coords["theta"]
    root = float(jn_zeros(1, 1)[0])
    field = jv(1, root * r) * np.cos(theta)
    if domain.mask is not None:
        field = field.copy()
        field[~domain.mask] = 0.0
    field = field.astype(np.float64)

    decomposer = build_decomposer(_fb_cfg(2, 2, normalization="none"))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[domain.mask] - field[domain.mask])))
    assert max_err < 1e-3


def test_fourier_bessel_coeff_meta() -> None:
    domain = _disk_domain(21, 21)
    field = np.zeros((21, 21), dtype=np.float32)

    decomposer = build_decomposer(_fb_cfg(3, 2, normalization="orthonormal"))
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()

    expected_modes = 2 * (1 + 2 * 3)
    assert meta["coeff_shape"] == [1, expected_modes]
    assert meta["m_max"] == 3
    assert meta["n_max"] == 2
    assert meta["ordering"] == "m_then_n"
    assert meta["boundary_condition"] == "dirichlet"
    assert meta["normalization"] == "orthonormal"
    assert meta["mask_policy"] == "ignore_masked_points"
    assert len(meta["mode_list"]) == expected_modes
