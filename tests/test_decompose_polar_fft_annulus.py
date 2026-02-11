from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _annulus_domain(height: int, width: int):
    cfg = {
        "name": "annulus",
        "center": [0.0, 0.0],
        "r_inner": 0.4,
        "r_outer": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def test_polar_fft_annulus_constant_roundtrip() -> None:
    height, width = 33, 33
    domain = _annulus_domain(height, width)
    mask = np.asarray(domain.mask).astype(bool)
    r = np.asarray(domain.coords["r"], dtype=np.float64)
    r0 = float(domain.meta["r_inner_norm"])
    # Smooth field that vanishes at inner/outer boundaries to reduce interpolation/Gibbs artifacts.
    base = (r - r0) * (1.0 - r)
    field = np.where(mask, base, 0.0).astype(np.float32)[..., None]

    cfg = {
        "name": "polar_fft",
        "n_r": 16,
        "n_theta": 32,
        "radial_transform": "dct",
        "angular_transform": "fft",
        "interpolation": "bilinear",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }
    decomposer = build_decomposer(cfg)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    err = np.abs(field_hat[..., 0] - field[..., 0])
    rmse = float(np.sqrt(np.mean((err[mask]) ** 2)))
    assert rmse < 2.0e-2

    meta = decomposer.coeff_meta()
    assert meta["method"] == "polar_fft"
    assert meta["r_inner_norm"] is not None
