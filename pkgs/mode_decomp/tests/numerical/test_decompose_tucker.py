"""Tucker factors preserve samples, physical measures and identifiable observations."""

from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.decomposition.base import FieldBatch
from ml_platform_mode_decomp.decomposition.tucker import TuckerDecomposer
from ml_platform_mode_decomp.model_artifact import load_model, save_model
from ml_platform_mode_decomp.representation import FieldRepresentation


def _low_rank(*, weighted=False):
    rng = np.random.default_rng(417)
    shape, ranks = (9, 8, 3), (2, 3, 2)
    x, y = np.linspace(0, 1, shape[1]) ** 2, np.linspace(0, 2, shape[0]) ** 1.7
    empty = FieldDataset.from_arrays(np.zeros((1, *shape)), x=x, y=y)
    weights = empty.geometry.weights if weighted else np.ones(shape[:2])
    factors = [np.linalg.qr(rng.normal(size=(size, rank)))[0] for size, rank in zip(shape, ranks, strict=True)]
    baseline = rng.normal(size=shape)

    def synthesize(cores):
        return (
            np.einsum("nabc,ya,xb,dc->nyxd", cores, *factors, optimize=True) / np.sqrt(weights)[None, ..., None]
            + baseline
        )

    train = synthesize(rng.normal(size=(13, *ranks)))
    query = synthesize(rng.normal(size=(4, *ranks)))
    return FieldDataset.from_arrays(train, x=x, y=y), query, ranks


@pytest.mark.parametrize("weighted", [False, True])
def test_tucker_unseen_cores_roundtrip_and_orthogonal_projection(weighted):
    data, query, ranks = _low_rank(weighted=weighted)
    original = data.fields.copy()
    product = "domain_weights" if weighted else "euclidean"
    model = TuckerDecomposer(ranks=ranks, inner_product=product).fit(data, data.geometry)
    coefficients = np.asarray(model.transform_batch(query, None, data.geometry))
    assert coefficients.shape == (len(query), *ranks)
    np.testing.assert_allclose(model.inverse_transform_batch(coefficients, data.geometry), query, atol=2e-12)
    np.testing.assert_array_equal(data.fields, original)
    assert model.coeff_meta()["sample_axis_compressed"] is False
    assert model.coeff_meta()["core_axes"] == ["y", "x", "component"]
    for factor in model._factors:
        np.testing.assert_allclose(factor.T @ factor, np.eye(factor.shape[1]), atol=1e-12)

    # Independently form physical decoder columns; complete projection must be
    # the same weighted least-squares solution, rather than an unweighted fit.
    dimension = int(np.prod(ranks))
    columns = np.stack(
        [model.inverse_transform(row.reshape(ranks), data.geometry) - model._mean for row in np.eye(dimension)], axis=-1
    ).reshape(-1, dimension)
    spatial = data.geometry.weights if weighted else np.ones(data.fields.shape[1:3])
    root_weight = np.repeat(np.sqrt(spatial).reshape(-1), data.fields.shape[-1])
    arbitrary = np.random.default_rng(7).normal(size=query[0].shape)
    expected = np.linalg.lstsq(
        columns * root_weight[:, None], (arbitrary - model._mean).reshape(-1) * root_weight, rcond=None
    )[0]
    actual = model.transform(arbitrary, None, data.geometry).reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=2e-12)
    np.testing.assert_allclose(columns.T @ (root_weight[:, None] ** 2 * columns), np.eye(dimension), atol=2e-12)


def test_hooi_improves_hosvd_without_reducing_the_sample_axis():
    fields = np.random.default_rng(209).normal(size=(5, 7, 6, 3))
    data = FieldDataset.from_arrays(fields)
    initial = TuckerDecomposer(ranks=[2, 2, 2], center=False, n_iter=0).fit(data, data.geometry)
    refined = TuckerDecomposer(ranks=[2, 2, 2], center=False, n_iter=20).fit(data, data.geometry)
    initial_fields = initial.inverse_transform_batch(
        initial.transform_batch(fields, None, data.geometry), data.geometry
    )
    refined_fields = refined.inverse_transform_batch(
        refined.transform_batch(fields, None, data.geometry), data.geometry
    )
    initial_error = np.linalg.norm(fields - initial_fields)
    refined_error = np.linalg.norm(fields - refined_fields)
    assert refined_error < initial_error - 1e-5
    assert refined.coeff_meta()["training_samples"] == len(fields)
    assert 0 < refined.coeff_meta()["diagnostics"]["iterations"] <= 20


@pytest.mark.parametrize("weighted", [False, True])
def test_tucker_partial_observation_recovers_identifiable_field_without_hidden_truth(weighted):
    data, query, ranks = _low_rank(weighted=weighted)
    model = TuckerDecomposer(ranks=ranks, inner_product="domain_weights" if weighted else "euclidean").fit(
        data, data.geometry
    )
    mask = np.random.default_rng(99).random(query.shape[1:3]) > 0.35
    observed = np.where(mask[..., None], query[0], np.nan)
    metadata = dict(model.coeff_meta())
    core = model.transform(observed, mask, data.geometry)
    np.testing.assert_allclose(model.inverse_transform(core, data.geometry), query[0], atol=4e-12)
    assert model.coeff_meta() == metadata
    altered_hidden = np.where(mask[..., None], observed, 1e10)
    np.testing.assert_array_equal(model.transform(altered_hidden, mask, data.geometry), core)
    queries = np.where(mask[None, ..., None], query, np.nan)
    shared = model.transform_batch(queries, mask, data.geometry)
    per_sample = model.transform_batch(queries, np.broadcast_to(mask, query.shape[:3]), data.geometry)
    np.testing.assert_array_equal(shared, per_sample)
    np.testing.assert_allclose(model.inverse_transform_batch(shared, data.geometry), query, atol=4e-12)


def test_tucker_unidentifiable_query_requires_explicit_ridge_and_complete_projection_is_unshrunk():
    data, query, ranks = _low_rank()
    plain = TuckerDecomposer(ranks=ranks).fit(data, data.geometry)
    ridge = TuckerDecomposer(ranks=ranks, ridge_alpha=0.4).fit(data, data.geometry)
    mask = np.zeros(query.shape[1:3], dtype=bool)
    mask[2, 3] = True
    with pytest.raises(ValueError, match="rank-deficient"):
        plain.transform(query[0], mask, data.geometry)
    core = ridge.transform(query[0], mask, data.geometry)
    assert core.shape == ranks and np.isfinite(core).all()
    dimension = int(np.prod(ranks))
    design = np.column_stack(
        [
            (ridge.inverse_transform(row.reshape(ranks), data.geometry) - ridge._mean)[mask].reshape(-1)
            for row in np.eye(dimension)
        ]
    )
    target = (query[0] - ridge._mean)[mask].reshape(-1)
    expected = np.linalg.lstsq(
        np.vstack((design, np.sqrt(0.4) * np.eye(dimension))), np.r_[target, np.zeros(dimension)], rcond=None
    )[0]
    np.testing.assert_allclose(core.reshape(-1), expected, atol=1e-12)
    np.testing.assert_array_equal(
        ridge.transform(query[0], None, data.geometry), plain.transform(query[0], None, data.geometry)
    )


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 5, 2), (6, 1, 1)])
def test_tucker_singleton_axes_and_constant_centered_training(shape):
    rng = np.random.default_rng(2)
    baseline = rng.normal(size=shape)
    fields = np.repeat(baseline[None], 3, axis=0)
    data = FieldDataset.from_arrays(fields)
    model = TuckerDecomposer(inner_product="domain_weights").fit(data, data.geometry)
    core = model.transform(fields[0], None, data.geometry)
    assert core.shape == tuple(min(8, size) for size in shape)
    np.testing.assert_allclose(core, 0, atol=1e-14)
    np.testing.assert_allclose(model.inverse_transform(core, data.geometry), baseline, atol=1e-14)


def test_tucker_thin_orthonormal_completion_and_scalar_legacy_layout():
    # The x-rank is larger than N*ry*rc during its HOOI update, and the
    # original y unfolding is tall. Neither needs a full axis-by-axis Q.
    fields = np.random.default_rng(86).normal(size=(1, 100, 4))
    geometry = FieldDataset.from_arrays(fields[..., None]).geometry
    model = TuckerDecomposer(ranks=(7, 4, 1), center=False, n_iter=2).fit_batch(fields, None, geometry)
    for factor in model._factors:
        np.testing.assert_allclose(factor.T @ factor, np.eye(factor.shape[1]), atol=1e-12)
    core = model.transform(fields[0], None, geometry)
    restored = model.inverse_transform(core, geometry)
    assert restored.shape == fields[0].shape
    np.testing.assert_allclose(restored, fields[0], atol=2e-12)
    assert model.coeff_meta()["field_layout"] == "HW"
    with pytest.raises(ValueError, match="layout"):
        model.transform(fields[0, ..., None], None, geometry)


@pytest.mark.parametrize(
    "options",
    [
        {"ranks": [1, 2]},
        {"ranks": [0, 1, 1]},
        {"ranks": [True, 1, 1]},
        {"ranks": [1.5, 1, 1]},
        {"n_iter": -1},
        {"n_iter": True},
        {"tol": float("nan")},
        {"ridge_alpha": -1},
        {"center": "false"},
        {"inner_product": "typo"},
    ],
)
def test_tucker_rejects_invalid_numerical_options(options):
    with pytest.raises(ValueError, match="tucker"):
        TuckerDecomposer(**options)


def test_tucker_rejects_incomplete_training_irregular_domains_and_nonseparable_weights():
    data = FieldDataset.from_arrays(np.random.default_rng(5).normal(size=(3, 4, 5, 2)))
    mask = np.ones(data.fields.shape[:3], dtype=bool)
    mask[0, 1, 1] = False
    with pytest.raises(ValueError, match="complete training"):
        TuckerDecomposer().fit_batch(data.fields, mask, data.geometry)
    with pytest.raises(ValueError, match="ranks cannot exceed"):
        TuckerDecomposer(ranks=[5, 1, 1]).fit(data, data.geometry)
    with pytest.raises(ValueError, match="rectangular geometry"):
        TuckerDecomposer().fit(data, replace(data.geometry, name="arbitrary_mask"))
    with pytest.raises(ValueError, match="irregular domain mask"):
        TuckerDecomposer().fit(data, replace(data.geometry, mask=mask[0]))
    invalid_weights = data.geometry.weights.copy()
    invalid_weights[1, 2] *= 2
    with pytest.raises(ValueError, match="separable"):
        TuckerDecomposer(inner_product="domain_weights").fit(data, replace(data.geometry, weights=invalid_weights))
    invalid_weights[1, 2] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        TuckerDecomposer(inner_product="domain_weights").fit(data, replace(data.geometry, weights=invalid_weights))
    nonfinite = data.fields.copy()
    nonfinite[1, 1, 1, 0] = np.inf
    with pytest.raises(ValueError, match="finite training"):
        TuckerDecomposer().fit(FieldBatch(nonfinite), data.geometry)


def test_tucker_fixed_geometry_and_failed_refit_invalidate_old_state():
    data, query, ranks = _low_rank()
    model = TuckerDecomposer(ranks=ranks).fit(data, data.geometry)
    shifted = replace(data.geometry, coords={**data.geometry.coords, "x": data.geometry.coords["x"] + 1})
    with pytest.raises(ValueError, match="Geometry differs"):
        model.transform(query[0], None, shifted)
    with pytest.raises(ValueError, match="observed"):
        model.transform(query[0], np.zeros(query.shape[1:3], dtype=bool), data.geometry)
    observed_nan = query[0].copy()
    observed_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.transform(observed_nan, None, data.geometry)
    with pytest.raises(ValueError, match="nonempty"):
        model.fit(None, data.geometry)
    with pytest.raises(ValueError, match="fit must be called"):
        model.transform(query[0], None, data.geometry)


def test_tucker_public_representation_codec_modes_and_saved_bundle(tmp_path):
    data, query, ranks = _low_rank(weighted=True)
    representation = FieldRepresentation(
        {"decomposition": {"name": "tucker", "params": {"ranks": list(ranks), "inner_product": "domain_weights"}}}
    )
    coefficients = representation.fit_transform(data)
    assert coefficients.shape == (len(data), int(np.prod(ranks)))
    assert representation.layout.synthesis_kind == "affine"
    np.testing.assert_allclose(representation.reconstruct(query), query, atol=4e-12)
    modes = representation.mode_fields([0, 3])
    assert modes.shape == (2, *query.shape[1:])
    path, _ = save_model(representation, tmp_path / "tucker.joblib")
    restored = load_model(path)
    np.testing.assert_array_equal(restored.transform(query), representation.transform(query))
    np.testing.assert_array_equal(
        restored.inverse_transform(coefficients), representation.inverse_transform(coefficients)
    )
