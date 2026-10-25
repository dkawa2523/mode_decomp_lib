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
    def __init__(self, cond: np.ndarray, field: np.ndarray) -> None:
        self.name = "test"
        self._cond = np.asarray(cond)
        self._field = np.asarray(field)

    def __len__(self) -> int:
        return int(self._cond.shape[0])

    def __getitem__(self, idx: int) -> _Sample:
        i = int(idx)
        return _Sample(
            cond=self._cond[i],
            field=self._field[i],
            mask=None,
            meta={"sample_id": f"s{i:02d}"},
        )


def _make_field(*, n: int, h: int, w: int, offset: float, residual_amp: float) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pat = np.sin(np.pi * xx) + 0.5 * np.cos(np.pi * yy)
    pat = (pat - float(np.mean(pat))).astype(np.float32)
    field = np.zeros((n, h, w, 1), dtype=np.float32)
    for i in range(n):
        field[i, :, :, 0] = float(offset) + float(residual_amp) * pat
    return field


def test_offset_split_auto_enables_when_dominant() -> None:
    from ml_platform_mode_decomp.geometry import build_domain_spec
    from ml_platform_mode_decomp.decomposition import build_decomposer

    n, h, w = 4, 8, 8
    cond = np.zeros((n, 1), dtype=np.float32)
    fields = _make_field(n=n, h=h, w=w, offset=10.0, residual_amp=0.2)
    dataset = _Dataset(cond=cond, field=fields)
    domain_spec = build_domain_spec(
        {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        fields.shape[1:],
    )

    dec = build_decomposer(
        {
            "name": "offset_residual",
            "enabled": "auto",
            "f_offset": 5.0,
            "max_samples": 4,
            "seed": 0,
            "min_residual_rms": 1e-8,
            "offset_def": "mean_per_channel",
            "agg": "median",
            "inner": {"name": "dct2", "disk_policy": "error"},
        }
    )
    dec.fit(dataset=dataset, domain_spec=domain_spec)
    raw = dec.transform(fields[0], None, domain_spec)
    meta = dec.coeff_meta()
    assert isinstance(raw, dict)
    assert str(meta.get("coeff_format", "")).strip().lower() == "offset_residual_v1"


def test_offset_split_auto_disables_when_not_dominant() -> None:
    from ml_platform_mode_decomp.geometry import build_domain_spec
    from ml_platform_mode_decomp.decomposition import build_decomposer

    n, h, w = 4, 8, 8
    cond = np.zeros((n, 1), dtype=np.float32)
    fields = _make_field(n=n, h=h, w=w, offset=1.0, residual_amp=2.0)
    dataset = _Dataset(cond=cond, field=fields)
    domain_spec = build_domain_spec(
        {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        fields.shape[1:],
    )

    dec = build_decomposer(
        {
            "name": "offset_residual",
            "enabled": "auto",
            "f_offset": 5.0,
            "max_samples": 4,
            "seed": 0,
            "min_residual_rms": 1e-8,
            "offset_def": "mean_per_channel",
            "agg": "median",
            "inner": {"name": "dct2", "disk_policy": "error"},
        }
    )
    dec.fit(dataset=dataset, domain_spec=domain_spec)
    raw = dec.transform(fields[0], None, domain_spec)
    meta = dec.coeff_meta()
    assert not isinstance(raw, dict)
    assert str(meta.get("coeff_format", "")).strip().lower() != "offset_residual_v1"


def test_offset_inner_explicit_seed_has_priority():
    from copy import deepcopy
    from ml_platform_mode_decomp.catalog import build_decomposer
    from ml_platform_mode_decomp.geometry import build_domain_spec

    fields = np.random.default_rng(5).normal(size=(24, 8, 8, 1))
    dataset = _Dataset(cond=np.zeros((24, 1)), field=fields)
    domain = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, (8, 8))
    inner = {"name": "pod", "params": {"solver": "randomized", "n_modes": 3, "seed": 7}}
    original = deepcopy(inner)
    wrapped = build_decomposer({"name": "offset_residual", "enabled": False, "seed": 31, "inner": inner})
    direct = build_decomposer(inner)
    wrapped.fit(dataset, domain)
    direct.fit(dataset, domain)
    np.testing.assert_array_equal(wrapped.transform(fields[0], None, domain), direct.transform(fields[0], None, domain))
    assert inner == original


def test_offset_inner_randomized_pod_is_reproducible():
    from ml_platform_mode_decomp.catalog import build_decomposer
    from ml_platform_mode_decomp.geometry import build_domain_spec

    rng = np.random.default_rng(5)
    fields = rng.normal(size=(24, 8, 8, 1))
    dataset = _Dataset(cond=np.zeros((24, 1)), field=fields)
    domain = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, (8, 8))
    config = {
        "name": "offset_residual",
        "enabled": False,
        "seed": 31,
        "inner": {"name": "pod", "backend": "sklearn", "solver": "randomized", "n_modes": 3, "mask_policy": "error"},
    }
    first, same = build_decomposer(config), build_decomposer(config)
    first.fit(dataset, domain)
    same.fit(dataset, domain)
    np.testing.assert_array_equal(first.transform(fields[0], None, domain), same.transform(fields[0], None, domain))
