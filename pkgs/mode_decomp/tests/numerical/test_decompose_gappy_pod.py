from __future__ import annotations

import numpy as np
import pytest

from ml_platform_mode_decomp.data.schema import FieldSample
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.decomposition import build_decomposer


def test_unregularized_gappy_query_requires_observations_that_identify_the_modes():
    from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

    fields = np.array([[1.0, 2.0], [1.0, -2.0], [-1.0, 2.0], [-1.0, -2.0]])[:, None, :, None]
    training = FieldDataset.from_arrays(fields)
    observed = np.array([[True, False]])
    config = {"decomposition": {"name": "gappy_pod", "params": {"n_modes": 2, "reg_lambda": 0}}}
    model = FieldRepresentation(config).fit(training)
    with pytest.raises(ValueError, match="does not identify all fitted modes"):
        model.transform(fields[:1], observed)
    np.testing.assert_allclose(model.reconstruct(training), fields, atol=1e-12)
    config["decomposition"]["params"]["reg_lambda"] = 0.1
    regularized = FieldRepresentation(config).fit(training)
    assert np.isfinite(regularized.transform(fields[:1], observed)).all()


@pytest.mark.parametrize("varying", [False, True])
@pytest.mark.parametrize("regularization", [0.0, 0.2])
def test_gappy_batch_keeps_weighted_partial_observation_solution(varying, regularization):
    from dataclasses import replace

    from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

    rng = np.random.default_rng(517)
    spatial = rng.normal(size=(2, 3, 4))
    values = 7 + np.einsum("nk,khw->nhw", rng.normal(size=(20, 2)), spatial)
    fields = np.stack((np.full_like(values, 3), values), axis=-1)
    geometry = replace(_rectangle_domain(3, 4), weights=np.arange(1.0, 13.0).reshape(3, 4))
    data = FieldDataset.from_arrays(fields, geometry=geometry)
    representation = FieldRepresentation(
        {
            "decomposition": {
                "name": "gappy_pod",
                "params": {
                    "inner_product": "domain_weights",
                    "reg_lambda": regularization,
                },
            }
        }
    ).fit(data)
    query = fields[:7, :, ::-1, :]
    query.setflags(write=False)
    mask = np.ones((7, 3, 4), dtype=bool)
    mask[:, 0, :2] = False
    if varying:
        mask[::2, 2, -1] = False
    mask.setflags(write=False)
    expected = []
    for field, observed in zip(query, mask, strict=True):
        component = representation.decomposer._impl._decomposers[1]
        basis = component.modes[observed.reshape(-1)]
        weights = geometry.weights[observed]
        truth = field[..., 1][observed] - component.mean[observed.reshape(-1)]
        # Independent small, well-conditioned normal-equation oracle; production
        # continues to use augmented least squares without squaring conditioning.
        normal = basis.T @ (weights[:, None] * basis) + regularization * np.eye(basis.shape[1])
        expected.append(np.linalg.solve(normal, basis.T @ (weights * truth)))
    actual = representation.transform(query, mask, space="a")
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    reconstructed = representation.inverse_transform(actual, space="a")
    np.testing.assert_allclose(reconstructed[..., 0], 3)


@pytest.mark.parametrize("inner_product", ["euclidean", "domain_weights"])
def test_gappy_learns_the_fixed_domain_without_requiring_truth_outside_it(inner_product):
    from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

    geometry = build_domain_spec(
        {"name": "disk", "radius": 1, "x_range": [-1, 1], "y_range": [-1, 1], "center": [0, 0]}, (9, 9)
    )
    x, y = geometry.coords["x"], geometry.coords["y"]
    q = np.random.default_rng(16).normal(size=(18, 2))
    fields = 2 + q[:, 0, None, None] * x + q[:, 1, None, None] * y
    fields[:, ~geometry.mask] = np.nan
    data = FieldDataset.from_arrays(fields[..., None], geometry=geometry)
    representation = FieldRepresentation(
        {
            "decomposition": {
                "name": "gappy_pod",
                "params": {
                    "n_modes": 2,
                    "reg_lambda": 0,
                    "inner_product": inner_product,
                },
            }
        }
    ).fit(data)
    observed = geometry.mask.copy()
    observed[::2, ::2] = False
    query = np.where(observed[None, ..., None], data.fields[:5], np.nan)
    prediction = representation.inverse_transform(representation.transform(query, observed))
    np.testing.assert_allclose(prediction[:, geometry.mask, :], data.fields[:5, geometry.mask, :], atol=1e-12)
    np.testing.assert_array_equal(prediction[:, ~geometry.mask, :], 0)
    incomplete = np.broadcast_to(geometry.mask, fields.shape).copy()
    incomplete[0, 4, 4] = False
    with pytest.raises(ValueError, match="complete observations inside"):
        representation.fit(FieldDataset.from_arrays(fields[..., None], geometry=geometry, observation_mask=incomplete))


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


def test_gappy_pod_none_mask_means_fully_observed() -> None:
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
    complete = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    explicit = decomposer.transform(fields[0], mask=np.ones(domain.grid_shape, dtype=bool), domain_spec=domain)
    np.testing.assert_allclose(complete, explicit)
