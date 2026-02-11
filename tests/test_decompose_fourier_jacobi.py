from __future__ import annotations

import numpy as np
from scipy.special import eval_jacobi

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


def _fj_cfg(*, m_max: int, k_max: int, normalization: str):
    return {
        "name": "fourier_jacobi",
        "m_max": int(m_max),
        "k_max": int(k_max),
        "ordering": "m_then_k",
        "normalization": str(normalization),
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }


def test_fourier_jacobi_roundtrip_single_mode() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    r = np.asarray(domain.coords["r"], dtype=np.float64)
    theta = np.asarray(domain.coords["theta"], dtype=np.float64)
    m, k = 2, 1
    t = 2.0 * (r * r) - 1.0
    radial = np.power(r, m) * eval_jacobi(k, 0.0, float(m), t)
    field = radial * np.cos(m * theta)
    if domain.mask is not None:
        field = field.copy()
        field[~domain.mask] = 0.0

    decomposer = build_decomposer(_fj_cfg(m_max=3, k_max=3, normalization="none"))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[domain.mask] - field[domain.mask])))
    assert max_err < 1e-3


def test_fourier_jacobi_constant_roundtrip_and_meta() -> None:
    domain = _disk_domain(21, 21)
    field = np.ones((21, 21), dtype=np.float32)

    decomposer = build_decomposer(_fj_cfg(m_max=3, k_max=2, normalization="orthonormal"))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    meta = decomposer.coeff_meta()

    expected_modes = (2 + 1) * (1 + 2 * 3)
    assert meta["coeff_shape"] == [1, expected_modes]
    assert len(meta["mode_list"]) == expected_modes
    assert meta["ordering"] == "m_then_k"
    assert meta["boundary_condition"] == "dirichlet"
    assert meta["normalization"] == "orthonormal"

    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[domain.mask] - 1.0)))
    assert max_err < 1e-3

