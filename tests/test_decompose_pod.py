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


def test_pod_roundtrip_scalar() -> None:
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(4)]
    domain = _rectangle_domain(6, 5)

    decomposer = build_decomposer({"name": "pod", "mask_policy": "error"})
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == fields[0].shape
    assert np.allclose(field_hat, fields[0], atol=1e-5)


def test_pod_roundtrip_vector_channelwise() -> None:
    rng = np.random.default_rng(1)
    fields = [rng.normal(size=(4, 3, 2)).astype(np.float32) for _ in range(3)]
    domain = _rectangle_domain(4, 3)

    decomposer = build_decomposer({"name": "pod", "mask_policy": "error"})
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == fields[0].shape
    assert np.allclose(field_hat, fields[0], atol=1e-5)


def test_pod_sklearn_randomized_seed_reproducible() -> None:
    rng = np.random.default_rng(2)
    fields = [rng.normal(size=(5, 4)).astype(np.float32) for _ in range(6)]
    domain = _rectangle_domain(5, 4)
    cfg = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "randomized",
        "n_modes": 3,
        "seed": 123,
    }

    decomposer_a = build_decomposer(cfg)
    decomposer_a.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff_a = decomposer_a.transform(fields[0], mask=None, domain_spec=domain)

    decomposer_b = build_decomposer(cfg)
    decomposer_b.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff_b = decomposer_b.transform(fields[0], mask=None, domain_spec=domain)

    assert decomposer_a.mean is not None
    assert decomposer_a.modes is not None
    assert decomposer_a.eigvals is not None
    assert np.allclose(coeff_a, coeff_b, atol=1e-6)


def test_pod_sklearn_rmse_decreases_with_k() -> None:
    rng = np.random.default_rng(3)
    fields = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(8)]
    domain = _rectangle_domain(6, 5)

    rmses: list[float] = []
    for k in (1, 2, 3):
        cfg = {
            "name": "pod",
            "mask_policy": "error",
            "backend": "sklearn",
            "solver": "direct",
            "n_modes": k,
            "seed": 0,
        }
        decomposer = build_decomposer(cfg)
        decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
        coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
        field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
        rmse = float(np.sqrt(np.mean((field_hat - fields[0]) ** 2)))
        rmses.append(rmse)

    assert rmses[1] <= rmses[0] + 1e-8
    assert rmses[2] <= rmses[1] + 1e-8


def test_pod_incremental_metadata_and_rmse() -> None:
    rng = np.random.default_rng(4)
    fields = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(10)]
    domain = _rectangle_domain(6, 5)

    cfg_inc = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "incremental",
        "n_modes": 3,
        "batch_size": 3,
    }
    decomposer_inc = build_decomposer(cfg_inc)
    decomposer_inc.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff_inc = decomposer_inc.transform(fields[0], mask=None, domain_spec=domain)
    field_inc = decomposer_inc.inverse_transform(coeff_inc, domain_spec=domain)
    rmse_inc = float(np.sqrt(np.mean((field_inc - fields[0]) ** 2)))

    meta = decomposer_inc.coeff_meta()
    metrics = meta.get("metrics", {})
    assert metrics.get("batch_size") == 3
    assert metrics.get("n_batches") is not None
    assert metrics.get("fit_time") is not None

    cfg_full = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 3,
    }
    decomposer_full = build_decomposer(cfg_full)
    decomposer_full.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff_full = decomposer_full.transform(fields[0], mask=None, domain_spec=domain)
    field_full = decomposer_full.inverse_transform(coeff_full, domain_spec=domain)
    rmse_full = float(np.sqrt(np.mean((field_full - fields[0]) ** 2)))

    assert rmse_inc <= rmse_full * 5.0 + 1e-8


def test_pod_rank_select_energy() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(size=(6, 5)).astype(np.float32)
    fields = [(0.5 * base + 0.01 * rng.normal(size=(6, 5))).astype(np.float32) for _ in range(6)]
    domain = _rectangle_domain(6, 5)
    cfg = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": None,
        "options": {
            "rank_select": {"enable": True, "method": "energy", "energy": 0.9, "max_modes": 5},
        },
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    assert 0 < int(meta.get("n_modes")) < 5
    assert coeff.shape[1] == int(meta.get("n_modes"))


def test_pod_mode_weight_eigval_scale_roundtrip() -> None:
    rng = np.random.default_rng(6)
    fields = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(5)]
    domain = _rectangle_domain(6, 5)
    cfg = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 5,
        "options": {"mode_weight": {"enable": True, "method": "eigval_scale"}},
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert np.allclose(field_hat, fields[0], atol=1e-5)
    meta = decomposer.coeff_meta()
    assert meta.get("mode_weight", {}).get("enable") is True
