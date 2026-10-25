from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from ml_platform_mode_decomp.data.schema import FieldSample
from ml_platform_mode_decomp.decomposition import build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec


def _domain_cfg() -> dict[str, object]:
    return {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}


def _decompose_cfg(latent_dim: int) -> dict[str, object]:
    return {
        "name": "autoencoder",
        "latent_dim": int(latent_dim),
        "hidden_channels": [4],
        "activation": "relu",
        "epochs": 2,
        "batch_size": 2,
        "lr": 1.0e-2,
        "weight_decay": 0.0,
        "mask_policy": "error",
        "device": "cpu",
        "seed": 0,
    }


def test_autoencoder_roundtrip() -> None:
    height, width, channels = 8, 8, 1
    rng = np.random.default_rng(0)
    dataset = [FieldSample(cond=None, field=rng.normal(size=(height, width, channels)), mask=None) for _ in range(4)]
    domain = build_domain_spec(_domain_cfg(), (height, width, channels))
    decomposer = build_decomposer(_decompose_cfg(latent_dim=4))

    decomposer.fit(dataset=dataset, domain_spec=domain)
    sample = dataset[0]
    coeff = decomposer.transform(sample.field, mask=sample.mask, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == sample.field.shape
    assert np.isfinite(field_hat).all()
    meta = decomposer.coeff_meta()
    assert meta["method"] == "autoencoder"
    # Default behavior includes per-channel mean + latent.
    assert meta["include_mean"] is True
    assert meta["coeff_shape"] == [5]
