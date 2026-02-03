from __future__ import annotations

import numpy as np

from mode_decomp_ml.data.datasets import FieldSample
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


class _ArrayDataset:
    def __init__(self, fields: list[np.ndarray]) -> None:
        self._fields = fields

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index: int) -> FieldSample:
        field = self._fields[index]
        return FieldSample(cond=np.zeros(1, dtype=np.float32), field=field, mask=None, meta={})


def _make_low_rank_fields(height: int, width: int, n_modes: int, n_samples: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(height * width, n_modes)).astype(np.float64)
    coeffs = rng.normal(size=(n_samples, n_modes)).astype(np.float64)
    fields = coeffs @ modes.T
    fields = fields.reshape(n_samples, height, width)
    return [field.astype(np.float32) for field in fields]


def test_gappy_pod_reconstructs_observed_points() -> None:
    height, width = 8, 6
    fields = _make_low_rank_fields(height, width, n_modes=2, n_samples=6)
    domain = _rectangle_domain(height, width)
    cfg = {
        "name": "gappy_pod",
        "mask_policy": "require",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 2,
        "reg_lambda": 1.0e-6,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)

    rng = np.random.default_rng(1)
    mask = rng.random(size=(height, width)) > 0.3
    coeff = decomposer.transform(fields[0], mask=mask, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    diff = (field_hat - fields[0])[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    assert rmse < 1e-4


def test_gappy_pod_requires_mask() -> None:
    fields = _make_low_rank_fields(6, 5, n_modes=2, n_samples=4)
    domain = _rectangle_domain(6, 5)
    cfg = {
        "name": "gappy_pod",
        "mask_policy": "require",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 2,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    try:
        decomposer.transform(fields[0], mask=None, domain_spec=domain)
    except ValueError as exc:
        assert "requires a mask" in str(exc)
    else:
        raise AssertionError("gappy_pod should require a mask")
