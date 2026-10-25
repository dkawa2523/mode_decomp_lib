from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace
from sklearn.decomposition import PCA

from ml_platform_mode_decomp.data.schema import FieldSample
from ml_platform_mode_decomp.decomposition import build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec


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


@pytest.mark.parametrize("channels", [1, 2])
def test_incremental_pod_recovers_weighted_low_rank_with_uneven_batches(channels) -> None:
    rng = np.random.default_rng(4)
    mean = rng.normal(size=(6, 5, channels))
    directions = rng.normal(size=(2, 6, 5, channels))
    fields = mean + np.einsum("nk,khwc->nhwc", rng.normal(size=(17, 2)), directions)
    mask = np.ones((6, 5), dtype=bool)
    mask[0] = False
    fields[:, ~mask] = np.nan
    weights = np.arange(30.0).reshape(6, 5)
    weights[1, 0] = 0
    domain = replace(_rectangle_domain(6, 5), mask=mask, weights=weights)
    original = fields.copy()
    model = build_decomposer(
        {"name": "pod", "solver": "incremental", "n_modes": 2, "batch_size": 5, "inner_product": "domain_weights"}
    ).fit(_ArrayDataset(list(fields)), domain)
    restored = np.stack([model.inverse_transform(model.transform(field, None, domain), domain) for field in fields])
    positive_measure = mask & (weights > 0)
    np.testing.assert_allclose(restored[:, positive_measure], fields[:, positive_measure], atol=2e-12)
    np.testing.assert_array_equal(restored[:, ~mask], 0)
    np.testing.assert_allclose(restored[:, 1, 0] - np.mean(fields[:, 1, 0], axis=0), 0, atol=1e-14)
    assert model.coeff_meta()["diagnostics"]["retained_energy_fraction"] == pytest.approx(1)
    np.testing.assert_array_equal(fields, original)


@pytest.mark.parametrize("solver", ["direct", "randomized"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pod_owned_work_matches_previous_pca_projection(solver, dtype) -> None:
    """Work-buffer changes retain the old solver, sign convention and physical energy."""
    rng = np.random.default_rng(11)
    fields = (3 + rng.normal(size=(40, 6, 7))).astype(dtype)
    mask = np.ones((6, 7), dtype=bool)
    mask[0] = False
    fields[:, ~mask] = np.nan
    weights = np.arange(42.0).reshape(6, 7)
    weights[1, 0] = 0
    domain = replace(_rectangle_domain(6, 7), mask=mask, weights=weights)
    original = fields.copy()
    matrix = np.asarray(fields[:, mask], dtype=np.float64)
    mean = matrix.mean(axis=0)
    sqrt_weights = np.sqrt(weights[mask])
    centered = (matrix - mean) * sqrt_weights
    pca = PCA(n_components=4, svd_solver="full" if solver == "direct" else "randomized", random_state=5)
    pca.fit(centered)
    expected_coeff = centered @ pca.components_.T
    model = build_decomposer(
        {"name": "pod", "solver": solver, "n_modes": 4, "inner_product": "domain_weights", "seed": 5}
    ).fit_batch(fields[..., None], None, domain)
    actual_coeff = np.stack([model.transform(field, None, domain) for field in fields])
    np.testing.assert_allclose(actual_coeff, expected_coeff, atol=2e-11, rtol=2e-11)
    diagnostics = model.coeff_meta()["diagnostics"]
    assert diagnostics["total_training_energy"] == pytest.approx(np.sum(centered**2), rel=2e-14)
    assert diagnostics["retained_training_energy"] == pytest.approx(np.sum(expected_coeff**2), rel=2e-14)
    field_hat = model.inverse_transform(actual_coeff[0], domain)
    residual = np.zeros_like(mean)
    np.divide(expected_coeff[0] @ pca.components_, sqrt_weights, out=residual, where=sqrt_weights > 0)
    np.testing.assert_allclose(field_hat[mask], mean + residual, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(fields, original)


def test_pod_mode_weight_eigval_scale_roundtrip() -> None:
    rng = np.random.default_rng(6)
    fields = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(5)]
    domain = _rectangle_domain(6, 5)
    cfg = {
        "name": "pod",
        "mask_policy": "error",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 4,
        "options": {"mode_weight": {"enable": True, "method": "eigval_scale"}},
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert np.allclose(field_hat, fields[0], atol=1e-5)
    meta = decomposer.coeff_meta()
    assert meta.get("mode_weight", {}).get("enable") is True
