from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.plugins.decomposers.base import ChannelwiseAdapter


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_channelwise_adapter_roundtrip() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(8, 6, 2)).astype(np.float32)
    domain = _rectangle_domain(8, 6)

    cfg = {"name": "dct2", "disk_policy": "error"}
    adapter = ChannelwiseAdapter(cfg=cfg, decomposer_factory=build_decomposer, label="dct2_channelwise")

    coeff = adapter.transform(field, mask=None, domain_spec=domain)
    field_hat = adapter.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)
    meta = adapter.coeff_meta()
    assert meta["channels"] == 2
    assert meta["base_method"] == "dct2"
