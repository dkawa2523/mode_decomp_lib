"""Regressions for inputs whose former coercions silently changed their meaning."""

from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp import FieldDataset
from ml_platform_mode_decomp.coefficients.post.basic import PCACoeffPost, QuantileCoeffPost
from ml_platform_mode_decomp.data.splitting import cross_validation_folds
from ml_platform_mode_decomp.decomposition.fourier_jacobi import FourierJacobiDecomposer
from ml_platform_mode_decomp.decomposition.gappy_graph_fourier import GappyGraphFourierDecomposer
from ml_platform_mode_decomp.decomposition.base import FieldBatch
from ml_platform_mode_decomp.decomposition.pod import PODDecomposer
from ml_platform_mode_decomp.decomposition.pod_em import PODEMDecomposer
from ml_platform_mode_decomp.decomposition.pod_joint import JointPODDecomposer
from ml_platform_mode_decomp.decomposition.pod_joint_em import JointPODEMDecomposer
from ml_platform_mode_decomp.decomposition.rbf_expansion import RBFExpansionDecomposer
from ml_platform_mode_decomp.evaluation import score_fields
from ml_platform_mode_decomp.evaluation.objectives import prepare_objective
from ml_platform_mode_decomp.field_preprocessing import build_preprocess
from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.regressors.sklearn import GPRRegressor, RidgeRegressor


@pytest.fixture(scope="module")
def graph():
    x, y = np.meshgrid(np.arange(24.0), np.arange(24.0))
    domain = DomainSpec("rectangle", {"x": x, "y": y}, None, np.ones((24, 24)), {})
    model = GappyGraphFourierDecomposer(n_modes=16, ridge_alpha=0.0, solver="dense").fit(domain_spec=domain)
    return model, domain


def test_gappy_graph_ill_conditioned_observations_preserve_known_coefficients(graph):
    model, domain = graph
    coefficients = np.arange(1.0, 17.0) / 16
    truth = (model._basis @ coefficients).reshape(24, 24)
    observed = np.zeros((24, 24), dtype=bool)
    observed[:5, :5] = True
    # A full-rank observed design with condition number about 1e8 used to
    # lose most of its reconstruction accuracy through normal equations.
    query = np.where(observed, truth, np.nan)
    actual = model.transform(query, observed, domain)
    np.testing.assert_allclose(actual[0], coefficients, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model.inverse_transform(actual, domain), truth, rtol=1e-6, atol=1e-7)


def test_gappy_graph_rejects_unidentifiable_observed_basis(graph):
    model, domain = graph
    observed = np.zeros((24, 24), dtype=bool)
    observed[:16, 0] = True
    with pytest.raises(ValueError, match="rank-deficient"):
        model.transform(np.ones((24, 24)), observed, domain)


@pytest.mark.parametrize("value", [-1.0, np.nan])
def test_gappy_graph_does_not_silently_drop_invalid_measure(graph, value):
    model, domain = graph
    weights = np.ones((24, 24))
    weights[0, 0] = value
    with pytest.raises(ValueError, match="finite and non-?negative"):
        model.transform(np.ones((24, 24)), None, replace(domain, weights=weights))


@pytest.mark.parametrize("options", [{"seed": 1.9}, {"random_state": True}, {"max_iter": 2.5}, {"alpha": np.nan}])
def test_ridge_does_not_truncate_counts_or_accept_nonfinite_parameters(options):
    with pytest.raises(ValueError):
        RidgeRegressor(**options)


@pytest.mark.parametrize(
    "constructor,options", [(RidgeRegressor, {"fit_intercept": "false"}), (GPRRegressor, {"normalize_y": "false"})]
)
def test_quoted_boolean_does_not_enable_a_false_option(constructor, options):
    with pytest.raises(ValueError, match="bool"):
        constructor(**options)


@pytest.mark.parametrize("settings", [{"n_splits": 2.5}, {"n_splits": True}, {"seed": 1.5}])
def test_public_cv_rejects_fractional_counts_and_seeds(settings):
    dataset = FieldDataset.from_arrays(np.ones((6, 2, 2, 1)))
    with pytest.raises(ValueError, match="integer"):
        cross_validation_folds(dataset, {"method": "random", "n_splits": 2, "seed": 3, **settings})


def test_hidden_truth_never_uses_field_metric_aliases():
    reference = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="hidden_\\*"):
        prepare_objective(
            reference,
            {"reference": "hidden", "names": ["field_rmse"]},
            evaluation_mask=np.ones_like(reference.effective_mask),
        )
    report = prepare_objective(
        reference,
        {"reference": "hidden", "names": ["hidden_rmse"]},
        evaluation_mask=np.ones_like(reference.effective_mask),
    ).score(reference.fields)
    assert report.metrics["hidden_rmse"] == 0
    assert "field_rmse" not in report.metrics


def test_primary_r2_is_evaluated_even_without_a_redundant_names_setting():
    reference = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="constant channel"):
        score_fields(reference, reference.fields, {"primary_metric": "field_r2"})


def test_complex_metric_weights_cannot_succeed_after_discarding_imaginary_parts():
    reference = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="real"):
        score_fields(reference, reference.fields, {"channel_weights": np.array([1.0 + 2.0j])})


@pytest.mark.parametrize("value", [2.5, "2", True])
@pytest.mark.parametrize(
    "constructor,key",
    [(FourierJacobiDecomposer, "k_max"), (PCACoeffPost, "n_components"), (RBFExpansionDecomposer, "n_centers")],
)
def test_count_parameters_keep_their_integer_meaning_at_each_constructor_entry(constructor, key, value):
    with pytest.raises(ValueError, match="integer"):
        constructor(**{key: value})
    with pytest.raises(ValueError, match="integer"):
        constructor(cfg={key: value})


@pytest.mark.parametrize(
    "constructor,options",
    [
        (QuantileCoeffPost, {"n_quantiles": 6, "subsample": 12}),
        (PCACoeffPost, {"n_components": 2, "svd_solver": "randomized"}),
    ],
)
def test_coeff_post_random_state_keyword_has_the_same_effect_as_seed(constructor, options):
    values = np.random.default_rng(2).normal(size=(50, 8))
    seeded = constructor(seed=9, **options).fit(values)
    aliased = constructor(random_state=9, **options).fit(values)
    assert aliased.state()["random_state"] == 9
    np.testing.assert_array_equal(aliased.transform(values), seeded.transform(values))
    with pytest.raises(ValueError, match="must match"):
        constructor(seed=8, random_state=9, **options)
    with pytest.raises(ValueError, match="must match"):
        constructor(seed=8, cfg={"random_state": 9}, **options)


@pytest.mark.parametrize("value", [0.2, np.nan, 1.0j])
def test_nonbinary_masks_cannot_become_observations_by_truthiness(graph, value):
    model, domain = graph
    mask = np.ones(domain.grid_shape, dtype=np.result_type(value))
    mask[0, 0] = value
    with pytest.raises(ValueError, match="0/1"):
        model.transform(np.ones(domain.grid_shape), mask, domain)
    with pytest.raises(ValueError, match="0/1"):
        build_preprocess({"name": "none"}).fit(np.ones((*domain.grid_shape, 1)), mask)


def test_exact_binary_float_masks_preserve_observation_meaning(graph):
    model, domain = graph
    mask = np.ones(domain.grid_shape, dtype=float)
    field = np.ones((*domain.grid_shape, 1))
    model.transform(field[..., 0], mask, domain)
    pre = build_preprocess({"name": "none"}).fit(field, mask)
    _, actual_mask = pre.transform(field, mask)
    assert actual_mask.dtype == bool
    assert actual_mask.all()


@pytest.mark.parametrize("constructor", [PODEMDecomposer, JointPODDecomposer, JointPODEMDecomposer])
def test_centering_aliases_cannot_silently_disagree(constructor):
    x, y = np.meshgrid(np.arange(3.0), np.arange(2.0))
    domain = DomainSpec("rectangle", {"x": x, "y": y}, None, np.ones((2, 3)), {})
    data = FieldBatch(np.random.default_rng(8).normal(size=(4, 2, 3, 1)))
    constructor(n_modes=1, center=False).fit(data, domain)
    with pytest.raises(ValueError, match="center and mean_centered must match"):
        constructor(n_modes=1, center=False, mean_centered=True).fit(data, domain)


@pytest.mark.parametrize("options", [{"names": ["field_typo"]}, {"primary_metric": "field_typo"}])
def test_python_scoring_rejects_unknown_requested_metrics(options):
    reference = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="known field_"):
        score_fields(reference, reference.fields, options)
    with pytest.raises(ValueError, match="known field_"):
        prepare_objective(reference, options)


def test_grid_only_scorer_cannot_ignore_an_observation_reference_request():
    reference = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="requires metrics.reference: grid"):
        score_fields(reference, reference.fields, {"reference": "observations"})


@pytest.mark.parametrize("solver", ["direct", "incremental"])
def test_field_batch_training_cannot_drop_complex_values_during_real_snapshot_conversion(solver):
    x, y = np.meshgrid(np.arange(2.0), np.arange(2.0))
    domain = DomainSpec("rectangle", {"x": x, "y": y}, None, np.ones((2, 2)), {})
    values = np.arange(16.0).reshape(4, 2, 2, 1)
    fields = FieldBatch(values + 1j * (values + 16))
    with pytest.raises(ValueError, match="real numeric"):
        PODDecomposer(n_modes=1, solver=solver, batch_size=2).fit(fields, domain)
