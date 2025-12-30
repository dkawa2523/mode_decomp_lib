from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.decompose import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


def _domain_cfg() -> dict[str, object]:
    return {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}


def _dataset_cfg(height: int, width: int, channels: int, num_samples: int) -> dict[str, object]:
    return {
        "name": "synthetic",
        "num_samples": int(num_samples),
        "cond_dim": 2,
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "mask_policy": "allow_none",
        "mask_mode": "none",
    }


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
    dataset = build_dataset(_dataset_cfg(height, width, channels, num_samples=4), domain_cfg=_domain_cfg(), seed=0)
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
    assert meta["coeff_shape"] == [4]
