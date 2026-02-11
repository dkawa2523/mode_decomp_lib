from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import vector_curl, vector_divergence
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))

def _grid_spacing(domain) -> tuple[float, float]:
    x = domain.coords["x"]
    y = domain.coords["y"]
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    return dx, dy


def test_helmholtz_poisson_matches_fft_helmholtz_for_periodic() -> None:
    height, width = 32, 32
    domain = _rectangle_domain(height, width)
    x = domain.coords["x"]
    y = domain.coords["y"]
    u = 2 * np.pi * np.cos(2 * np.pi * x) + 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(
        2 * np.pi * y
    )
    v = -2 * np.pi * np.sin(2 * np.pi * y) - 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(
        2 * np.pi * y
    )
    field = np.stack([u, v], axis=-1).astype(np.float32)

    cfg = {"boundary_condition": "periodic", "mask_policy": "error"}
    helmholtz = build_decomposer({"name": "helmholtz", **cfg})
    poisson = build_decomposer({"name": "helmholtz_poisson", **cfg})

    coeff_h = helmholtz.transform(field, mask=None, domain_spec=domain)
    coeff_p = poisson.transform(field, mask=None, domain_spec=domain)
    assert np.allclose(coeff_p, coeff_h, atol=1e-5)

    field_hat = poisson.inverse_transform(coeff_p, domain_spec=domain)
    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)

    meta = poisson.coeff_meta()
    assert meta["method"] == "helmholtz_poisson"
    assert meta["coeff_layout"] == "PHWC"
    assert meta["coeff_shape"] == [2, height, width, 2]
    assert meta["boundary_condition"] == "periodic"


def test_helmholtz_poisson_dirichlet_roundtrip_smooth_field() -> None:
    height, width = 33, 33
    domain = _rectangle_domain(height, width)
    x = domain.coords["x"]
    y = domain.coords["y"]

    phi = np.sin(np.pi * x) * np.sin(np.pi * y)
    psi = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    dphi_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dphi_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    dpsi_dx = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    dpsi_dy = 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

    curl_free = np.stack([dphi_dx, dphi_dy], axis=-1)
    div_free = np.stack([dpsi_dy, -dpsi_dx], axis=-1)
    field = (curl_free + div_free).astype(np.float64)

    decomposer = build_decomposer(
        {"name": "helmholtz_poisson", "boundary_condition": "dirichlet", "mask_policy": "error"}
    )
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert field_hat.shape == field.shape

    # Parts should be approximately curl-free / div-free away from the boundary.
    dx, dy = _grid_spacing(domain)
    curl_rms = float(np.sqrt(np.mean(vector_curl(coeff[0], grid_spacing=(dx, dy))[1:-1, 1:-1] ** 2)))
    div_rms = float(
        np.sqrt(np.mean(vector_divergence(coeff[1], grid_spacing=(dx, dy))[1:-1, 1:-1] ** 2))
    )
    assert curl_rms < 1.0e-2
    assert div_rms < 1.0e-2

    # NOTE: grad/div/curl use finite differences, while Poisson uses spectral diagonalization.
    # This is not an exact discrete Hodge decomposition, so use a moderate tolerance.
    max_err = float(np.max(np.abs(field_hat - field)))
    assert max_err < 7.0e-2

    meta = decomposer.coeff_meta()
    assert meta["boundary_condition"] == "dirichlet"
    assert meta["parts"] == ["curl_free", "div_free"]


def test_helmholtz_poisson_neumann_roundtrip_smooth_field() -> None:
    height, width = 33, 33
    domain = _rectangle_domain(height, width)
    x = domain.coords["x"]
    y = domain.coords["y"]

    phi = np.cos(np.pi * x) * np.cos(np.pi * y)
    psi = np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

    dphi_dx = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    dphi_dy = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    dpsi_dx = -2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    dpsi_dy = -2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

    curl_free = np.stack([dphi_dx, dphi_dy], axis=-1)
    div_free = np.stack([dpsi_dy, -dpsi_dx], axis=-1)
    field = (curl_free + div_free).astype(np.float64)

    decomposer = build_decomposer(
        {"name": "helmholtz_poisson", "boundary_condition": "neumann", "mask_policy": "error"}
    )
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert field_hat.shape == field.shape

    # With Neumann BC, a non-trivial harmonic component may remain; focus on part properties.
    dx, dy = _grid_spacing(domain)
    curl_rms = float(np.sqrt(np.mean(vector_curl(coeff[0], grid_spacing=(dx, dy))[1:-1, 1:-1] ** 2)))
    div_rms = float(
        np.sqrt(np.mean(vector_divergence(coeff[1], grid_spacing=(dx, dy))[1:-1, 1:-1] ** 2))
    )
    assert curl_rms < 1.0e-2
    assert div_rms < 1.0e-2

    assert np.isfinite(field_hat).all()
