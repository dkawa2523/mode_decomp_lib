from __future__ import annotations

import numpy as np
import pytest

pywt = pytest.importorskip("pywt")

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def test_wavelet2d_keep_approx_roundtrip_matches_zeroed_details() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(32, 32)).astype(np.float32)
    domain = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, field.shape)

    cfg = {
        "name": "wavelet2d",
        "wavelet": "db2",
        "level": 2,
        "mode": "symmetric",
        "keep": "approx",
        "mask_policy": "error",
    }
    decomposer = build_decomposer(cfg)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()

    assert meta["keep"] == "approx"
    approx_shape = tuple(int(x) for x in meta["approx_shape"])
    assert int(np.prod(approx_shape)) == int(meta["coeff_shape"][1])

    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert field_hat.shape == field.shape

    # Reference: wavedec2 then zero-out details and reconstruct.
    coeffs = pywt.wavedec2(field, wavelet="db2", mode="symmetric", level=2)
    cA = coeffs[0]
    zeros_details = []
    for detail in coeffs[1:]:
        cH, cV, cD = detail
        zeros_details.append((np.zeros_like(cH), np.zeros_like(cV), np.zeros_like(cD)))
    coeffs_zero = [cA] + zeros_details
    ref = pywt.waverec2(coeffs_zero, wavelet="db2", mode="symmetric")
    ref = np.asarray(ref)[: field.shape[0], : field.shape[1]]

    assert np.allclose(field_hat, ref, atol=1e-6)

