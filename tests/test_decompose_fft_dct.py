from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mode_decomp_ml.decompose import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_fft2_roundtrip_sine() -> None:
    height, width = 32, 32
    x = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    field = np.sin(2 * np.pi * 3 * xx) + 0.5 * np.cos(2 * np.pi * 2 * yy)
    field = field.astype(np.float32)[..., None]

    domain = _rectangle_domain(height, width)
    decomposer = build_decomposer({"name": "fft2", "disk_policy": "error"})
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    max_err = float(np.max(np.abs(field_hat - field)))
    assert max_err < 1e-5


def test_dct2_roundtrip_vector() -> None:
    rng = np.random.default_rng(0)
    field = rng.normal(size=(8, 6, 2)).astype(np.float32)

    domain = _rectangle_domain(8, 6)
    decomposer = build_decomposer({"name": "dct2", "disk_policy": "error"})
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)


def test_coeff_meta_saved(tmp_path: Path) -> None:
    field = np.zeros((4, 5, 1), dtype=np.float32)
    domain = _rectangle_domain(4, 5)
    decomposer = build_decomposer({"name": "fft2", "disk_policy": "error"})

    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta_path = decomposer.save_coeff_meta(tmp_path)

    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["method"] == "fft2"
    assert payload["coeff_shape"] == [1, 4, 5, 2]
