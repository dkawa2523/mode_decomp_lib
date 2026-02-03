from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _annulus_domain(height: int, width: int, *, r_inner: float, r_outer: float):
    cfg = {
        "name": "annulus",
        "center": [0.0, 0.0],
        "r_inner": r_inner,
        "r_outer": r_outer,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _disk_domain(height: int, width: int):
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _annular_cfg(n_max: int, m_max: int):
    return {
        "name": "annular_zernike",
        "n_max": n_max,
        "m_max": m_max,
        "ordering": "n_then_m",
        "normalization": "orthonormal",
        "boundary_condition": "unit_annulus",
    }


def _zernike_cfg(n_max: int):
    return {
        "name": "zernike",
        "n_max": n_max,
        "ordering": "n_then_m",
        "normalization": "orthonormal",
        "boundary_condition": "unit_disk",
    }


def test_annular_zernike_roundtrip_constant() -> None:
    height, width = 33, 33
    domain = _annulus_domain(height, width, r_inner=0.4, r_outer=1.0)
    field = np.ones((height, width, 1), dtype=np.float32)

    decomposer = build_decomposer(_annular_cfg(4, 4))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[..., 0][domain.mask] - 1.0)))
    assert max_err < 1e-3


def test_annular_zernike_matches_disk_when_inner_zero() -> None:
    height, width = 25, 25
    disk_domain = _disk_domain(height, width)
    annulus_domain = _annulus_domain(height, width, r_inner=0.0, r_outer=1.0)
    rng = np.random.default_rng(0)
    field = rng.normal(size=(height, width)).astype(np.float32)

    zernike = build_decomposer(_zernike_cfg(3))
    annular = build_decomposer(_annular_cfg(3, 3))

    coeff_disk = zernike.transform(field, mask=None, domain_spec=disk_domain)
    coeff_annulus = annular.transform(field, mask=None, domain_spec=annulus_domain)

    assert np.allclose(coeff_disk, coeff_annulus, atol=1e-5)
