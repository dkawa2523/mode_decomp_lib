from __future__ import annotations

import numpy as np

from mode_decomp_ml.decompose import build_decomposer
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


def _zernike_cfg(n_max: int):
    return {
        "name": "zernike",
        "n_max": n_max,
        "ordering": "n_then_m",
        "normalization": "orthonormal",
        "boundary_condition": "unit_disk",
    }


def test_zernike_roundtrip_constant() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    field = np.ones((height, width, 1), dtype=np.float32)

    decomposer = build_decomposer(_zernike_cfg(4))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert domain.mask is not None
    max_err = float(np.max(np.abs(field_hat[..., 0][domain.mask] - 1.0)))
    assert max_err < 1e-3


def test_zernike_coeff_meta_and_truncation() -> None:
    domain = _disk_domain(21, 21)
    field = np.zeros((21, 21), dtype=np.float32)

    decomposer = build_decomposer(_zernike_cfg(3))
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()

    expected_modes = (3 + 1) * (3 + 2) // 2
    assert meta["coeff_shape"] == [1, expected_modes]
    assert len(meta["nm_list"]) == expected_modes
    assert meta["ordering"] == "n_then_m"
    assert meta["boundary_condition"] == "unit_disk"

    coeff_low = coeff[: expected_modes // 2]
    field_low = decomposer.inverse_transform(coeff_low, domain_spec=domain)
    assert field_low.shape == field.shape
