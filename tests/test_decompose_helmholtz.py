from __future__ import annotations

import numpy as np

from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import vector_curl, vector_divergence


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def _grid_spacing(domain) -> tuple[float, float]:
    x = domain.coords["x"]
    y = domain.coords["y"]
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    return dx, dy


def test_helmholtz_roundtrip_and_parts() -> None:
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

    decomposer = build_decomposer(
        {"name": "helmholtz", "boundary_condition": "periodic", "mask_policy": "error"}
    )
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)

    meta = decomposer.coeff_meta()
    assert meta["method"] == "helmholtz"
    assert meta["coeff_shape"] == [2, height, width, 2]

    coeff_tensor = np.asarray(coeff)
    if coeff_tensor.shape != tuple(meta["coeff_shape"]):
        coeff_tensor = coeff_tensor.reshape(meta["coeff_shape"])
    curl_free = coeff_tensor[0]
    div_free = coeff_tensor[1]
    dx, dy = _grid_spacing(domain)
    curl = vector_curl(curl_free, grid_spacing=(dx, dy))
    div = vector_divergence(div_free, grid_spacing=(dx, dy))
    # NOTE: finite-difference curl/div introduce boundary artifacts; use RMS tolerance.
    curl_rms = float(np.sqrt(np.mean(curl**2)))
    div_rms = float(np.sqrt(np.mean(div**2)))
    assert curl_rms < 1e-1
    assert div_rms < 1e-1
