from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Sample:
    cond: np.ndarray
    field: np.ndarray
    mask: np.ndarray | None
    meta: dict[str, Any]


class _Dataset:
    def __init__(self, cond: np.ndarray, field: np.ndarray, mask: np.ndarray | None) -> None:
        self.name = "test"
        self._cond = np.asarray(cond)
        self._field = np.asarray(field)
        self._mask = None if mask is None else np.asarray(mask)

    def __len__(self) -> int:
        return int(self._cond.shape[0])

    def __getitem__(self, idx: int) -> _Sample:
        i = int(idx)
        return _Sample(
            cond=self._cond[i],
            field=self._field[i],
            mask=None if self._mask is None else self._mask[i],
            meta={"sample_id": f"s{i:02d}"},
        )


def test_offset_residual_roundtrip_dct2() -> None:
    from mode_decomp_ml.domain import build_domain_spec
    from mode_decomp_ml.plugins.codecs import build_coeff_codec
    from mode_decomp_ml.plugins.decomposers import build_decomposer

    rng = np.random.default_rng(0)
    n = 4
    h = 8
    w = 8
    c = 2

    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    base = np.sin(np.pi * xx) + 0.5 * np.cos(np.pi * yy)
    base = base - float(np.mean(base))
    base = base.astype(np.float32)

    offsets = np.stack(
        [
            np.linspace(10.0, 11.0, n, dtype=np.float32),
            np.linspace(-5.0, -4.5, n, dtype=np.float32),
        ],
        axis=1,
    )
    fields = np.zeros((n, h, w, c), dtype=np.float32)
    for i in range(n):
        for ch in range(c):
            noise = (0.001 * offsets[i, ch]) * rng.normal(size=(h, w)).astype(np.float32)
            fields[i, :, :, ch] = offsets[i, ch] + 0.05 * offsets[i, ch] * base + noise

    cond = rng.normal(size=(n, 3)).astype(np.float32)
    dataset = _Dataset(cond=cond, field=fields, mask=None)

    domain_spec = build_domain_spec(
        {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        fields.shape[1:],
    )

    decomposer = build_decomposer(
        {
            "name": "offset_residual",
            "enabled": True,
            "f_offset": 5.0,
            "max_samples": 4,
            "seed": 0,
            "min_residual_rms": 1e-8,
            "offset_def": "mean_per_channel",
            "agg": "median",
            "inner": {"name": "dct2", "disk_policy": "error"},
        }
    )
    decomposer.fit(dataset=dataset, domain_spec=domain_spec)

    codec = build_coeff_codec({"name": "auto_codec_v1"})
    raw_coeff = decomposer.transform(fields[0], None, domain_spec)
    raw_meta = decomposer.coeff_meta()
    vec = codec.encode(raw_coeff, raw_meta)
    decoded = codec.decode(vec, raw_meta)
    assert isinstance(decoded, dict)
    assert "offset" in decoded and "residual" in decoded

    # Offset should match weighted mean (uniform weights on rectangle).
    expected_offset = fields[0].mean(axis=(0, 1))
    assert np.allclose(np.asarray(decoded["offset"]), expected_offset, atol=1e-5)

    field_hat = decomposer.inverse_transform(decoded, domain_spec=domain_spec)
    assert field_hat.shape == fields[0].shape
    assert np.allclose(field_hat, fields[0], atol=1e-4)

