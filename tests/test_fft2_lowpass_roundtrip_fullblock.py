from __future__ import annotations

import numpy as np


def test_fft2_lowpass_roundtrip_full_block_rectangle() -> None:
    from mode_decomp_ml.domain import build_domain_spec
    from mode_decomp_ml.plugins.decomposers.fft2_lowpass import FFT2LowpassDecomposer

    h = w = 8
    domain_cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    domain_spec = build_domain_spec(domain_cfg, (h, w, 1))

    rng = np.random.default_rng(0)
    field = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)

    decomp = FFT2LowpassDecomposer(cfg={"name": "fft2_lowpass", "disk_policy": "error", "block_size": h})
    coeff = decomp.transform(field, mask=None, domain_spec=domain_spec)
    field_hat = decomp.inverse_transform(coeff, domain_spec=domain_spec)

    assert field_hat.shape == field.shape
    assert np.max(np.abs(field_hat - field)) < 1e-5


def test_fft2_lowpass_disk_mask_zero_fill_smoke() -> None:
    from mode_decomp_ml.domain import build_domain_spec
    from mode_decomp_ml.plugins.decomposers.fft2_lowpass import FFT2LowpassDecomposer

    h = w = 16
    domain_cfg = {
        "name": "disk",
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
        "center": [0.0, 0.0],
        "radius": 1.0,
    }
    domain_spec = build_domain_spec(domain_cfg, (h, w, 1))

    field = np.ones((h, w), dtype=np.float32)
    decomp = FFT2LowpassDecomposer(cfg={"name": "fft2_lowpass", "disk_policy": "mask_zero_fill", "block_size": 8})

    coeff = decomp.transform(field, mask=None, domain_spec=domain_spec)
    field_hat = decomp.inverse_transform(coeff, domain_spec=domain_spec)

    assert field_hat.shape == field.shape
