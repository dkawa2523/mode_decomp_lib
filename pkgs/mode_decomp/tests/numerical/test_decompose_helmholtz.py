from __future__ import annotations

import numpy as np

from ml_platform_mode_decomp.decomposition import build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec


def _rectangle_domain(height: int, width: int):
    cfg = {
        "name": "rectangle",
        "axes": {
            "x": {"start": 0, "step": 1 / width, "count": width, "periodic": True},
            "y": {"start": 0, "step": 1 / height, "count": height, "periodic": True},
        },
    }
    return build_domain_spec(cfg, (height, width))


def test_helmholtz_roundtrip_and_parts() -> None:
    height, width = 32, 32
    domain = _rectangle_domain(height, width)
    x = domain.coords["x"]
    y = domain.coords["y"]
    u = 2 * np.pi * np.cos(2 * np.pi * x) + 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    v = -2 * np.pi * np.sin(2 * np.pi * y) - 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    field = np.stack([u, v], axis=-1).astype(np.float32)

    decomposer = build_decomposer({"name": "helmholtz", "boundary_condition": "periodic", "mask_policy": "error"})
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
    # Known periodic potentials: phi=sin(2*pi*x)+cos(2*pi*y),
    # psi=sin(2*pi*x)*sin(2*pi*y). Check the analytic parts, rather
    # than mixing the FFT operator with one-sided finite differences.
    expected_curl_free = np.stack((2 * np.pi * np.cos(2 * np.pi * x), -2 * np.pi * np.sin(2 * np.pi * y)), axis=-1)
    np.testing.assert_allclose(curl_free, expected_curl_free, atol=2e-6)
    np.testing.assert_allclose(div_free, field - expected_curl_free, atol=2e-6)
