from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pywt")

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _domain_cfg() -> dict[str, object]:
    return {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}


def test_wavelet2d_roundtrip_scalar() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(16, 16)).astype(np.float32)
    domain = build_domain_spec(_domain_cfg(), field.shape)

    decomposer = build_decomposer(
        {
            "name": "wavelet2d",
            "wavelet": "db2",
            "level": 2,
            "mode": "symmetric",
            "mask_policy": "error",
        }
    )
    codec = build_coeff_codec({"name": "wavelet_pack_v1", "dtype_policy": "float32"})

    raw_coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    raw_meta = decomposer.coeff_meta()
    vector = codec.encode(raw_coeff, raw_meta)
    raw_back = codec.decode(vector, raw_meta)
    field_hat = decomposer.inverse_transform(raw_back, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-6)
