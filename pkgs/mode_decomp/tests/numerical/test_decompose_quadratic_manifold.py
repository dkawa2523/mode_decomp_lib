"""Scientific invariants for the POD encoder and its quadratic decoder."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp.decomposition.quadratic_manifold import QuadraticManifoldDecomposer
from ml_platform_mode_decomp.geometry import DomainSpec, build_domain_spec


def domain():
    return build_domain_spec({"name": "rectangle", "x_range": [-1, 1], "y_range": [-1, 1]}, (2, 2, 1))


def parabola(coordinates):
    values = np.asarray(coordinates)
    linear = np.asarray([1, 1, -1, -1], dtype=float).reshape(2, 2, 1) / 2
    quadratic = np.asarray([1, -1, 1, -1], dtype=float).reshape(2, 2, 1) * 0.15
    return values[:, None, None, None] * linear + values[:, None, None, None] ** 2 * quadratic


@pytest.mark.parametrize("center", [True, False])
def test_exact_quadratic_recovers_unseen_parabola_with_one_coordinate(center):
    geometry = domain()
    model = QuadraticManifoldDecomposer(n_modes=1, center=center, quadratic_alpha=0)
    training = parabola(np.linspace(-1, 1, 11))
    model.fit_batch(training, None, geometry)
    fields = parabola([-0.71, -0.19, 0.43, 0.88])
    coefficients = model.transform_batch(fields, None, geometry)
    restored = model.inverse_transform_batch(coefficients, geometry)
    np.testing.assert_allclose(restored, fields, atol=5e-15)
    assert model.synthesis_kind == "nonlinear"
    assert model.coeff_meta()["coeff_shape"] == [1]
    assert model.coeff_meta()["diagnostics"]["linear_training_squared_error"] > 0.01
    assert model.coeff_meta()["diagnostics"]["quadratic_training_squared_error"] < 1e-25
    assert model.coeff_meta()["quadratic_feature_centering"] == ("training_mean" if center else "none")


def test_linear_encoder_inverts_arbitrary_decoded_coordinates_and_preserves_metadata():
    geometry = domain()
    model = QuadraticManifoldDecomposer(n_modes=1, quadratic_alpha=1e-6)
    model.fit_batch(parabola(np.linspace(-1, 1, 9)), None, geometry)
    expected = np.asarray([[-0.4], [0.0], [0.72], [1.6]])
    decoded = model.inverse_transform_batch(expected, geometry)
    np.testing.assert_allclose(model.transform_batch(decoded, None, geometry), expected, atol=2e-15)
    metadata = deepcopy(model.coeff_meta())
    observed = np.asarray([[True, True], [True, False]])
    incomplete = decoded[0].copy()
    incomplete[~observed] = np.nan
    model.transform(incomplete, observed, geometry)
    assert model.coeff_meta() == metadata


def test_partial_observations_use_nonlinear_solve_and_ignore_hidden_truth():
    geometry = domain()
    model = QuadraticManifoldDecomposer(n_modes=1, quadratic_alpha=0, tol=1e-12)
    model.fit_batch(parabola(np.linspace(-1, 1, 11)), None, geometry)
    truth = parabola([0.67])[0]
    expected = model.transform(truth, None, geometry)
    observed = np.asarray([[True, True], [True, False]])
    incomplete = truth.copy()
    incomplete[~observed] = np.nan
    observed.setflags(write=False)
    before = incomplete.copy()
    actual = model.transform(incomplete, observed, geometry)
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    np.testing.assert_allclose(model.inverse_transform(actual, geometry), truth, atol=2e-11)
    np.testing.assert_array_equal(incomplete, before)


def test_shared_and_sample_masks_match_individual_query_encodings():
    geometry = domain()
    model = QuadraticManifoldDecomposer(n_modes=1, quadratic_alpha=0, tol=1e-12)
    model.fit_batch(parabola(np.linspace(-1, 1, 11)), None, geometry)
    fields = parabola([-0.31, 0.67])
    first = np.asarray([[True, True], [True, False]])
    second = np.asarray([[True, False], [True, True]])
    masks = np.stack((first, second))
    queries = np.where(masks[..., None], fields, np.nan)
    expected = model.transform_batch(fields, None, geometry)
    np.testing.assert_allclose(model.transform_batch(queries, masks, geometry), expected, atol=2e-10)
    np.testing.assert_allclose(model.transform_batch(fields, first, geometry), expected, atol=2e-10)


def test_weighted_joint_fields_on_fixed_domain_preserve_projection_and_zero_outside():
    active = np.asarray([[True, True], [True, False]])
    geometry = replace(domain(), mask=active, weights=np.asarray([[0.3, 0.7], [1.4, 0.0]]))
    rng = np.random.default_rng(17)
    training = rng.normal(size=(12, 2, 2, 2))
    training[:, ~active] = np.nan
    original = training.copy()
    model = QuadraticManifoldDecomposer(n_modes=2, inner_product="domain_weights")
    model.fit_batch(training, active, geometry)
    coefficients = np.asarray([[0.3, -0.9], [-0.7, 0.6]])
    reconstructed = model.inverse_transform_batch(coefficients, geometry)
    np.testing.assert_allclose(model.transform_batch(reconstructed, None, geometry), coefficients, atol=5e-15)
    assert np.count_nonzero(reconstructed[:, ~active]) == 0
    np.testing.assert_array_equal(training, original)
    np.testing.assert_allclose(model._encoder.T @ model._correction, 0, atol=2e-15)


def test_constant_training_has_rank_zero_without_dummy_coordinates():
    geometry = domain()
    profile = np.asarray([0.1, 1.3, -0.7, 2.1]).reshape(2, 2, 1)
    training = np.repeat(profile[None], 9, axis=0)
    model = QuadraticManifoldDecomposer().fit_batch(training, None, geometry)
    coefficients = model.transform_batch(training[:2], None, geometry)
    assert np.asarray(coefficients).shape == (2, 0)
    np.testing.assert_array_equal(model.inverse_transform(np.empty(0), geometry), profile)
    observed = np.asarray([[True, False], [False, False]])
    incomplete = np.where(observed[..., None], profile, np.nan)
    assert model.transform(incomplete, observed, geometry).shape == (0,)


def test_mesh_internal_layout_and_measure_use_the_same_decoder_contract():
    geometry = DomainSpec(
        "mesh",
        {"vertices": np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]), "faces": np.asarray([[0, 1, 2], [1, 3, 2]])},
        None,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        {"vertex_count": 4},
    )
    training = parabola(np.linspace(-1, 1, 9)).reshape(9, 4, 1, 1)
    model = QuadraticManifoldDecomposer(n_modes=1, inner_product="domain_weights").fit_batch(training, None, geometry)
    coefficients = np.asarray([[-0.2], [0.9]])
    decoded = model.inverse_transform_batch(coefficients, geometry)
    np.testing.assert_allclose(model.transform_batch(decoded, None, geometry), coefficients, atol=2e-15)


def test_partial_query_rejects_nonidentifiable_coordinates_without_regularization():
    geometry = domain()
    rng = np.random.default_rng(5)
    training = rng.normal(size=(8, 2, 2, 1))
    model = QuadraticManifoldDecomposer(n_modes=2).fit_batch(training, None, geometry)
    observed = np.asarray([[True, False], [False, False]])
    with pytest.raises(ValueError, match="not identifiable"):
        model.transform(training[0], observed, geometry)
    regularized = QuadraticManifoldDecomposer(n_modes=2, ridge_alpha=0.1).fit_batch(training, None, geometry)
    assert np.isfinite(regularized.transform(training[0], observed, geometry)).all()


def test_partial_query_reports_nonconvergence_instead_of_returning_initial_guess():
    geometry = domain()
    model = QuadraticManifoldDecomposer(n_modes=1, quadratic_alpha=0, max_nfev=1)
    model.fit_batch(parabola(np.linspace(-1, 1, 11)), None, geometry)
    observed = np.asarray([[True, True], [True, False]])
    with pytest.raises(RuntimeError, match="did not converge"):
        model.transform(parabola([0.67])[0], observed, geometry)


def test_cross_products_identify_a_missing_linear_coordinate_from_quadratic_observations():
    geometry = build_domain_spec({"name": "rectangle", "x_range": [-1, 1], "y_range": [-1, 1]}, (2, 3, 1))
    aa, bb = np.meshgrid(np.linspace(-0.5, 0.5, 7), np.linspace(-0.5, 0.5, 7))

    def fields(a, b):
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        return np.stack((a, b, 0.2 * a**2, 0.2 * a * b, 0.2 * b**2, np.zeros_like(a)), axis=1).reshape(-1, 2, 3, 1)

    model = QuadraticManifoldDecomposer(n_modes=2, quadratic_alpha=0, tol=1e-12)
    model.fit_batch(fields(aa, bb), None, geometry)
    truth = fields([0.31], [-0.27])[0]
    expected = model.transform(truth, None, geometry)
    np.testing.assert_allclose(model.inverse_transform(expected, geometry), truth, atol=2e-15)
    observed = np.asarray([[True, False, True], [True, True, False]])
    query = np.where(observed[..., None], truth, np.nan)
    actual = model.transform(query, observed, geometry)
    np.testing.assert_allclose(actual, expected, atol=2e-10)
    np.testing.assert_allclose(model.inverse_transform(actual, geometry), truth, atol=2e-10)


def test_rank_deficient_observed_jacobian_and_failed_refit_do_not_return_arbitrary_coordinates():
    geometry = domain()
    training = np.zeros((9, 2, 2, 1))
    training[:, 0, 0, 0] = np.linspace(-1, 1, 9)
    model = QuadraticManifoldDecomposer(n_modes=1).fit_batch(training, None, geometry)
    observed = np.asarray([[False, True], [False, True]])
    with pytest.raises(ValueError, match="Jacobian is rank-deficient"):
        model.transform(training[0], observed, geometry)
    invalid = np.zeros((9, 2, 2, 1))
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite training"):
        model.fit_batch(invalid, None, geometry)
    with pytest.raises(ValueError, match="fit must be called"):
        model.transform(training[0], None, geometry)


def test_geometry_layout_finiteness_and_training_observations_are_validated():
    geometry = domain()
    training = parabola(np.linspace(-1, 1, 9))
    model = QuadraticManifoldDecomposer(n_modes=1)
    with pytest.raises(ValueError, match="complete training"):
        model.fit_batch(training, np.asarray([[True, False], [True, True]]), geometry)
    model.fit_batch(training, None, geometry)
    model.transform(training[0], None, geometry)
    with pytest.raises(ValueError, match="Geometry differs"):
        model.inverse_transform(np.ones(1), replace(geometry, weights=geometry.weights * 2))
    with pytest.raises(ValueError, match="layout"):
        model.transform(np.zeros((2, 2, 2)), None, geometry)
    with pytest.raises(ValueError, match="finite values"):
        model.transform(np.full((2, 2, 1), np.nan), None, geometry)
    with pytest.raises(ValueError, match="real"):
        model.transform(training[0].astype(complex), None, geometry)
    with pytest.raises(ValueError, match="finite real"):
        model.inverse_transform(np.asarray([np.inf]), geometry)
    with pytest.raises(ValueError, match="overflow"):
        model.inverse_transform(np.asarray([1e308]), geometry)
    with pytest.raises(ValueError, match="size"):
        model.inverse_transform(np.ones(2), geometry)
    with pytest.raises(ValueError, match="training rank"):
        QuadraticManifoldDecomposer(n_modes=3).fit_batch(training, None, geometry)


@pytest.mark.parametrize(
    "options",
    [
        {"n_modes": True},
        {"n_modes": 1.5},
        {"n_modes": 0},
        {"center": "false"},
        {"inner_product": "area"},
        {"quadratic_alpha": -1},
        {"ridge_alpha": np.nan},
        {"max_nfev": True},
        {"max_nfev": 0},
        {"tol": 0},
        {"tol": np.inf},
    ],
)
def test_invalid_constructor_options_are_rejected(options):
    with pytest.raises(ValueError):
        QuadraticManifoldDecomposer(**options)
