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


def _polar_cfg(*, mask_aware_sampling: bool) -> dict[str, object]:
    return {
        "name": "polar_fft",
        "n_r": 33,
        "n_theta": 64,
        "radial_transform": "dct",
        "angular_transform": "fft",
        "interpolation": "bilinear",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
        "mask_aware_sampling": bool(mask_aware_sampling),
    }


def test_polar_fft_mask_aware_sampling_improves_disk_boundary() -> None:
    height, width = 33, 33
    domain = _disk_domain(height, width)
    assert domain.mask is not None

    # Field is 1 inside disk, 0 outside. Naive bilinear sampling mixes outside zeros near the boundary.
    field = np.zeros((height, width, 1), dtype=np.float32)
    field[..., 0][domain.mask] = 1.0

    dec_naive = build_decomposer(_polar_cfg(mask_aware_sampling=False))
    coeff_naive = dec_naive.transform(field, mask=None, domain_spec=domain)
    hat_naive = dec_naive.inverse_transform(coeff_naive, domain_spec=domain)
    rmse_naive = float(np.sqrt(np.mean((hat_naive[..., 0][domain.mask] - 1.0) ** 2)))

    dec_mask = build_decomposer(_polar_cfg(mask_aware_sampling=True))
    coeff_mask = dec_mask.transform(field, mask=None, domain_spec=domain)
    hat_mask = dec_mask.inverse_transform(coeff_mask, domain_spec=domain)
    rmse_mask = float(np.sqrt(np.mean((hat_mask[..., 0][domain.mask] - 1.0) ** 2)))

    assert rmse_mask < rmse_naive
    assert rmse_mask < 2.0e-1

