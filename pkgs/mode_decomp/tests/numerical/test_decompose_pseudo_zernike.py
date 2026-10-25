from __future__ import annotations

import numpy as np

from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.decomposition import build_decomposer


def _disk_domain(height: int, width: int):
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _pz_cfg(n_max: int, *, m_max: int | None = None):
    cfg = {
        "name": "pseudo_zernike",
        "n_max": int(n_max),
        "ordering": "n_then_m",
        "normalization": "orthonormal",
        "boundary_condition": "unit_disk",
    }
    if m_max is not None:
        cfg["m_max"] = int(m_max)
    return cfg


def test_pseudo_zernike_roundtrip_constant() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    field = np.ones((height, width, 1), dtype=np.float32)

    decomposer = build_decomposer(_pz_cfg(6))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[..., 0][domain.mask] - 1.0)))
    assert max_err < 1e-3
