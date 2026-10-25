"""Rank, energy, physical projection, and partial-observation regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ml_platform_mode_decomp.catalog import build_decomposer
from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.geometry import DomainSpec, build_domain_spec
from ml_platform_mode_decomp.geometry.sphere_grid import sphere_grid_domain_cfg
from ml_platform_mode_decomp.representation import FieldRepresentation


def _rectangle(height=2, width=2):
    x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    return DomainSpec("rectangle", {"x": x, "y": y}, None, np.ones_like(x), {})


def _samples(matrix):
    return [SimpleNamespace(field=row.reshape(2, 2, -1), mask=None) for row in matrix]


@pytest.mark.parametrize("method,channels", [("pod_em", 1), ("pod_joint_em", 2)])
@pytest.mark.parametrize("center,initialization", [(True, "mean_fill"), (False, "zero_fill")])
def test_weighted_em_matches_explicit_small_als_with_readonly_missing_snapshots(
    method, channels, center, initialization
):
    rng = np.random.default_rng(58)
    fields = rng.normal(size=(7, 4, 6, channels)).astype(np.float32)[:, ::2, ::2]
    masks = rng.random((7, 2, 3)) > 0.3
    masks[:, 0, :] = True
    domain_mask = np.array([[True, True, True], [True, True, False]])
    domain = replace(_rectangle(2, 3), weights=np.array([[0.0, 1.0, 3.0], [2.0, 4.0, 1.0]]), mask=domain_mask)
    masks &= domain_mask
    fields[~masks] = np.nan
    fields.flags.writeable = masks.flags.writeable = False
    samples = [SimpleNamespace(field=field, mask=mask) for field, mask in zip(fields, masks, strict=True)]
    model = build_decomposer(
        {
            "name": method,
            "n_modes": 2,
            "center": center,
            "n_iter": 2,
            "ridge_alpha": 0.01,
            "init": initialization,
            "inner_product": "domain_weights",
            "mask_policy": "allow",
        }
    ).fit(samples, domain)

    # Independent small ALS reference: normal equations, explicit full matrices,
    # and direct SVD, rather than production ridge/work-buffer helpers.
    truth = fields.reshape(len(fields), -1).astype(float)
    observed = np.repeat(masks.reshape(len(fields), -1), channels, axis=1)
    counts = observed.sum(axis=0)
    mean = np.divide(np.where(observed, truth, 0).sum(axis=0), counts, out=np.zeros(truth.shape[1]), where=counts > 0)
    filled = np.where(observed, truth, mean if initialization == "mean_fill" else 0.0)
    if not center:
        mean = np.zeros_like(mean)
    scale = np.repeat(np.sqrt(domain.weights.reshape(-1)), channels)
    basis = np.linalg.svd((filled - mean) * scale, full_matrices=False)[2][:2].T
    for _ in range(2):
        for index in range(len(fields)):
            selected = observed[index]
            matrix = basis[selected]
            target = (truth[index, selected] - mean[selected]) * scale[selected]
            coefficients = np.linalg.solve(matrix.T @ matrix + 0.01 * np.eye(2), matrix.T @ target)
            weighted = basis @ coefficients
            restored = mean + np.divide(weighted, scale, out=np.zeros_like(weighted), where=scale > 0)
            filled[index, ~selected] = restored[~selected]
        if center:
            mean = filled.mean(axis=0)
        basis = np.linalg.svd((filled - mean) * scale, full_matrices=False)[2][:2].T
    selected = observed[0]
    matrix = basis[selected]
    target = (truth[0, selected] - mean[selected]) * scale[selected]
    coefficients = np.linalg.solve(matrix.T @ matrix + 0.01 * np.eye(2), matrix.T @ target)
    weighted = basis @ coefficients
    expected = (mean + np.divide(weighted, scale, out=np.zeros_like(weighted), where=scale > 0)).reshape(2, 3, channels)
    expected[~domain_mask] = 0.0
    actual = model.inverse_transform(model.transform(fields[0], masks[0], domain), domain)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


_POD_BACKENDS = [
    ("sklearn", "direct"),
    ("sklearn", "snapshots"),
    ("sklearn", "randomized"),
    ("sklearn", "incremental"),
    ("modred", "snapshots"),
]


@pytest.mark.parametrize("backend,solver", _POD_BACKENDS)
@pytest.mark.parametrize("threshold,expected_met", [(0.5, True), (0.9, False)])
def test_pod_energy_uses_complete_training_denominator(backend, solver, threshold, expected_met):
    if backend == "modred":
        pytest.importorskip("modred")
    diagonal = np.diag(np.sqrt([4.0, 3.0, 2.0, 1.0]))
    samples = _samples(np.concatenate((diagonal, -diagonal)))
    domain = _rectangle()
    model = build_decomposer(
        {
            "name": "pod",
            "backend": backend,
            "solver": solver,
            "mask_policy": "error",
            "options": {
                "rank_select": {
                    "enable": True,
                    "energy": threshold,
                    "max_modes": 2,
                }
            },
        }
    )
    model.fit(samples, domain)
    restored = [model.inverse_transform(model.transform(s.field, None, domain), domain) for s in samples]
    diagnostics = model.coeff_meta()["diagnostics"]
    actual = 1 - sum(np.sum((field - sample.field) ** 2) for field, sample in zip(restored, samples, strict=True)) / 20
    assert diagnostics["selected_modes"] == 2
    assert diagnostics["total_training_energy"] == pytest.approx(20)
    assert diagnostics["retained_training_energy"] == pytest.approx(14)
    assert diagnostics["retained_energy_fraction"] == pytest.approx(actual)
    assert actual == pytest.approx(0.7)
    assert diagnostics["energy_threshold_met"] is expected_met


@pytest.mark.parametrize(
    "method,backend,solver",
    [("pod", backend, solver) for backend, solver in _POD_BACKENDS] + [("pod_svd", "sklearn", "direct")],
)
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_pod_rank_is_relative_and_rejects_nonexistent_modes(method, backend, solver, scale):
    if backend == "modred":
        pytest.importorskip("modred")
    matrix = scale * np.array([-2.0, -1.0, 1.0, 2.0])[:, None] * np.arange(1.0, 5.0)[None, :]
    config = {"name": method, "n_modes": 3, "mask_policy": "error"}
    if method == "pod":
        config.update(backend=backend, solver=solver)
    model = build_decomposer(config)
    with pytest.raises(ValueError, match="rank"):
        model.fit(_samples(matrix), _rectangle())


@pytest.mark.parametrize("solver", ["direct", "randomized"])
def test_pod_large_offset_does_not_create_a_centering_rank(solver):
    amplitudes = np.array([-7.0, -3.0, -1.0, 2.0, 4.0, 6.0])
    direction = np.array([1.0, 2.0, 4.0, 8.0])
    matrix = 1e12 + amplitudes[:, None] * direction
    model = build_decomposer({"name": "pod", "solver": solver, "n_modes": 2, "seed": 7})
    with pytest.raises(ValueError, match="rank"):
        model.fit(_samples(matrix), _rectangle())


@pytest.mark.parametrize("method", ["pod", "pod_svd"])
def test_zero_rank_preserves_a_zero_field_without_padding(method):
    model = build_decomposer({"name": method, "mask_policy": "error"})
    domain = _rectangle()
    model.fit(_samples(np.zeros((4, 4))), domain)
    coeff = model.transform(np.zeros((2, 2, 1)), None, domain)
    assert coeff.shape == (0,)
    np.testing.assert_array_equal(model.inverse_transform(coeff, domain), np.zeros((2, 2, 1)))


@pytest.mark.parametrize("method", ["pod", "pod_svd"])
def test_requested_domain_inner_product_requires_an_actual_measure(method):
    from dataclasses import replace

    model = build_decomposer({"name": method, "inner_product": "domain_weights"})
    domain = replace(_rectangle(), weights=None)
    with pytest.raises(ValueError, match="requires domain integration weights"):
        model.fit(_samples(np.eye(4)), domain)


@pytest.mark.parametrize("method", ["pod", "pod_svd"])
def test_channelwise_rank_counts_have_no_padding_modes(method):
    first = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])[:, None] * np.arange(1.0, 5.0)[None, :]
    second = np.zeros((5, 4))
    second[:, 0] = [-2, -1, 0, 1, 2]
    second[:, 1] = [1, -1, 0, -1, 1]
    fields = np.stack((first, second), axis=-1).reshape(5, 2, 2, 2)
    samples = [SimpleNamespace(field=field, mask=None) for field in fields]
    model = build_decomposer({"name": method, "mask_policy": "error"})
    model.fit(samples, _rectangle())
    coeff = model.transform(fields[0], None, _rectangle())
    assert coeff.shape == (3,)
    diagnostics = model.coeff_meta()["diagnostics"]
    assert diagnostics["channel_effective_ranks"] == [1, 2]
    assert diagnostics["channel_selected_modes"] == [1, 2]
    assert diagnostics["selected_modes"] == 3
    assert diagnostics["retained_energy_fraction"] == pytest.approx(1)
    for field in fields:
        np.testing.assert_allclose(
            model.inverse_transform(model.transform(field, None, _rectangle()), _rectangle()), field, atol=1e-12
        )


def test_channelwise_energy_selection_preserves_independent_counts():
    first = np.array([[10.0, 0, 0, 0], [-10.0, 0, 0, 0], [0, 1.0, 0, 0], [0, -1.0, 0, 0]])
    second = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, -1.0, 0, 0]])
    fields = np.stack((first, second), axis=-1).reshape(4, 2, 2, 2)
    model = build_decomposer(
        {
            "name": "pod",
            "mask_policy": "error",
            "options": {
                "rank_select": {"enable": True, "energy": 0.9},
            },
        }
    )
    model.fit([SimpleNamespace(field=field, mask=None) for field in fields], _rectangle())
    assert model.transform(fields[0], None, _rectangle()).shape == (3,)
    assert model.coeff_meta()["component_modes"] == [1, 2]
    assert model.coeff_meta()["diagnostics"]["retained_energy_fraction"] == pytest.approx(204 / 206)


@pytest.mark.parametrize("method", ["helmholtz"])
@pytest.mark.parametrize("shape", [(7, 9), (8, 8), (7, 8), (8, 9)])
def test_periodic_helmholtz_is_a_real_orthogonal_projection(method, shape):
    domain = _rectangle(*shape)
    field = np.random.default_rng(9).normal(size=(*shape, 2))
    model = build_decomposer({"name": method, "boundary_condition": "periodic", "mask_policy": "error"})
    parts = model.transform(field, None, domain)
    curl_free, div_free = parts
    np.testing.assert_allclose(curl_free + div_free, field, atol=1e-12)
    np.testing.assert_allclose(model.transform(curl_free, None, domain)[0], curl_free, atol=1e-12)
    np.testing.assert_allclose(model.transform(div_free, None, domain)[0], 0, atol=1e-12)
    assert np.sum(curl_free * div_free) == pytest.approx(0, abs=1e-12)
    assert np.sum(parts**2) == pytest.approx(np.sum(field**2), abs=1e-10)
    assert model.coeff_meta()["derivative_convention"] == "spectral_first_derivative_nyquist_zero"


def test_helmholtz_nyquist_derivative_nullspace_is_assigned_once():
    y, x = np.mgrid[:8, :8]
    checkerboard = (-1.0) ** (x + y)
    field = np.stack((checkerboard, -checkerboard), axis=-1)
    model = build_decomposer({"name": "helmholtz", "boundary_condition": "periodic", "mask_policy": "error"})
    parts = model.transform(field, None, _rectangle(8, 8))
    np.testing.assert_allclose(parts[0], 0, atol=1e-12)
    np.testing.assert_allclose(parts[1], field, atol=1e-12)


def test_slepian_uses_only_the_positive_discrete_subspace():
    domain = build_domain_spec(sphere_grid_domain_cfg(5, 8), (5, 8))
    config = {
        "name": "spherical_slepian",
        "l_max": 6,
        "k": 49,
        "backend": "scipy",
        "region_cap": {"lat": 0.0, "lon": 0.0, "radius": 90.0, "angle_unit": "deg"},
    }
    with pytest.raises(ValueError, match="effective.*rank"):
        build_decomposer(config).fit(None, domain)
    model = build_decomposer({**config, "k": 25})
    model.fit(None, domain)
    gram = model._basis.T @ (domain.integration_weights().reshape(-1, 1) * model._basis)
    np.testing.assert_allclose(gram, np.eye(25), atol=1e-12)
    field = np.random.default_rng(4).normal(size=(5, 8))
    coeff = model.transform(field, None, domain)
    restored = model.inverse_transform(coeff, domain)
    np.testing.assert_allclose(model.transform(restored, None, domain), coeff, atol=1e-8)
    assert model.coeff_meta()["diagnostics"]["effective_rank"] == 25
    with pytest.raises(ValueError, match="domain spec"):
        model.transform(field, None, replace(domain, weights=2 * domain.weights))


@pytest.mark.parametrize("components", [1, 2])
def test_gappy_representation_accepts_full_observation_and_uses_physical_basis(components):
    domain = build_domain_spec({"kind": "rectangle", "x_range": [0, 3], "y_range": [0, 2]}, (3, 4))
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(3, 4, components))
    fields = np.stack([2 + amplitude * direction for amplitude in [-2.0, -1.0, 1.0, 2.0]])
    dataset = FieldDataset(fields=fields, geometry=domain, sample_ids=np.array(["a", "b", "c", "d"]))
    decomposer = build_decomposer(
        {"name": "gappy_pod", "n_modes": 1, "reg_lambda": 0.0, "inner_product": "domain_weights"}
    )
    decomposer.fit(dataset, domain)
    np.testing.assert_allclose(
        decomposer.inverse_transform(decomposer.transform(fields[0], None, domain), domain), fields[0], atol=1e-12
    )
    representation = FieldRepresentation(
        {
            "decomposition": {
                "name": "gappy_pod",
                "params": {
                    "n_modes": 1,
                    "reg_lambda": 0.0,
                    "inner_product": "domain_weights",
                },
            }
        }
    ).fit(dataset)
    np.testing.assert_allclose(representation.reconstruct(dataset), fields, atol=1e-12)
    masks = np.ones(fields.shape[:-1], dtype=bool)
    masks[:, 0, :] = False
    partial = np.where(masks[..., None], fields, np.nan)
    np.testing.assert_allclose(representation.reconstruct(partial, masks), fields, atol=1e-12)
