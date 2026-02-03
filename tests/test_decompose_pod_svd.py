from __future__ import annotations

import numpy as np

from mode_decomp_ml.data.datasets import FieldSample
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


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


def test_pod_svd_roundtrip_training_sample() -> None:
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(6, 5, 2)).astype(np.float32) for _ in range(4)]
    domain = _rectangle_domain(6, 5)

    decomposer = build_decomposer({"name": "pod_svd", "mask_policy": "error"})
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == fields[0].shape
    assert np.allclose(field_hat, fields[0], atol=1e-5)


def test_pod_svd_rank_select_energy() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(5, 4, 1)).astype(np.float32)
    fields = [(0.25 * base).astype(np.float32) for _ in range(6)]
    domain = _rectangle_domain(5, 4)
    cfg = {
        "name": "pod_svd",
        "mask_policy": "error",
        "n_modes": None,
        "options": {
            "rank_select": {"enable": True, "method": "energy", "energy": 0.9, "max_modes": 5},
        },
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    assert meta.get("n_modes") == 1
    assert coeff.shape[1] == 1
