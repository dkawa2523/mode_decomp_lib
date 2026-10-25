"""Scientific meaning of field scores, support and paired comparisons."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.evaluation import analyze_surrogate, mean_field_prediction, observation_mask, score_fields
from ml_platform_mode_decomp.evaluation.comparison import paired_comparisons
from ml_platform_mode_decomp.evaluation.fields import prepare_reference
from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.surrogate import FieldSurrogate


def field_data(values, *, weights=None, mask=None):
    values = np.asarray(values, dtype=float)
    h, w = values.shape[1:3]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    geometry = DomainSpec("rectangle", {"x": x, "y": y}, None, weights, {})
    return FieldDataset(values, geometry, np.asarray([f"s{i}" for i in range(len(values))]), observation_mask=mask)


def test_optional_undefined_r2_does_not_discard_valid_scores():
    data = field_data(np.ones((3, 2, 2, 1)))
    report = score_fields(data, data.fields + 2, {"names": ["field_rmse", "field_r2"]})
    assert report.metrics["field_rmse"] == 2
    assert "field_r2" not in report.metrics
    assert report.summary["metric_status"]["field_r2"] == "undefined_for_constant_channel"
    with pytest.raises(ValueError, match="Primary R2"):
        score_fields(data, data.fields, {"names": ["field_r2"], "primary_metric": "field_r2"})


def test_score_only_surrogate_matches_detailed_total():
    from ml_platform_mode_decomp.evaluation import score_surrogate

    rng = np.random.default_rng(101)
    data = FieldDataset.from_arrays(rng.normal(size=(12, 3, 4, 1)), rng.normal(size=(12, 2)))
    training, reference = data.subset(range(8)), data.subset(range(8, 12))
    model = FieldSurrogate(
        {"representation": {"decomposition": {"name": "dct2"}}, "regression": {"name": "ridge"}}
    ).fit(training)
    report = score_surrogate(model, training, reference)
    detailed = analyze_surrogate(model, training, reference)
    assert report.metrics["field_rmse"] == pytest.approx(detailed.metrics["field_rmse"])
    np.testing.assert_allclose(report.predictions, detailed.predictions)


@pytest.mark.parametrize("post", ["none", "pca"])
def test_omitting_detailed_fields_keeps_the_same_surrogate_errors_and_predictions(post):
    rng = np.random.default_rng(619)
    data = FieldDataset.from_arrays(rng.normal(size=(18, 3, 4, 2)), rng.normal(size=(18, 2)))
    train, reference = data.subset(range(12)), data.subset(range(12, 18))
    model = FieldSurrogate(
        {
            "representation": {
                "decomposition": {"name": "dct2"},
                "coeff_post": {"name": post, "params": {"n_components": 3} if post == "pca" else {}},
            }
        }
    ).fit(train)
    detailed = analyze_surrogate(model, train, reference, retain_coefficients=True)
    compact = analyze_surrogate(model, train, reference, retain_coefficients=True, retain_fields=False)
    assert compact.fields == {}
    assert set(detailed.fields) == {"decomposition", "representation"}
    assert compact.metrics == detailed.metrics
    pd.testing.assert_frame_equal(compact.per_sample, detailed.per_sample)
    np.testing.assert_array_equal(compact.predictions, detailed.predictions)
    np.testing.assert_array_equal(compact.packed_coefficients, detailed.packed_coefficients)


@pytest.mark.parametrize("post", ["none", "pca"])
def test_representation_prediction_time_includes_reconstruction_and_excludes_diagnostic_inverse(monkeypatch, post):
    import ml_platform_mode_decomp.evaluation.analysis as analysis
    from ml_platform_mode_decomp.evaluation.objectives import prepare_objective
    from ml_platform_mode_decomp.representation import FieldRepresentation

    rng = np.random.default_rng(317)
    data = FieldDataset.from_arrays(rng.normal(size=(12, 2, 3, 1)))
    model = FieldRepresentation(
        {
            "decomposition": {"name": "dct2"},
            "coeff_post": {"name": post, "params": {"n_components": 2} if post == "pca" else {}},
        }
    ).fit(data)
    objective = prepare_objective(data)
    elapsed = [0.0]

    def timed(operation, cost):
        def wrapped(*args, **kwargs):
            result = operation(*args, **kwargs)
            elapsed[0] += cost
            return result

        return wrapped

    monkeypatch.setattr(analysis, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(model, "transform", timed(model.transform, 2))
    monkeypatch.setattr(model.post, "transform", timed(model.post.transform, 3))
    monkeypatch.setattr(model, "inverse_transform", timed(model.inverse_transform, 5))
    monkeypatch.setattr(objective, "score", timed(objective.score, 29))
    report = analysis.analyze_representation(model, data, data, prepared=objective)
    assert report.summary["prediction_seconds"] == 10


@pytest.mark.parametrize("kind", ["representation", "surrogate"])
def test_score_prediction_time_excludes_field_scoring(monkeypatch, kind):
    import ml_platform_mode_decomp.evaluation.analysis as analysis
    from ml_platform_mode_decomp.evaluation.objectives import prepare_objective

    rng = np.random.default_rng(319)
    data = FieldDataset.from_arrays(rng.normal(size=(12, 2, 3, 1)), rng.normal(size=(12, 2)))
    surrogate = FieldSurrogate({"representation": {"decomposition": {"name": "dct2"}}}).fit(data)
    objective = prepare_objective(data)
    elapsed = [0.0]

    def timed(operation, cost):
        def wrapped(*args, **kwargs):
            result = operation(*args, **kwargs)
            elapsed[0] += cost
            return result

        return wrapped

    monkeypatch.setattr(analysis, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(objective, "score", timed(objective.score, 29))
    if kind == "surrogate":
        monkeypatch.setattr(surrogate, "predict", timed(surrogate.predict, 7))
        report = analysis.score_surrogate(surrogate, data, data, prepared=objective)
    else:
        model = surrogate.representation
        monkeypatch.setattr(model, "reconstruct", timed(model.reconstruct, 7))
        report = analysis.score_representation(model, data, data, prepared=objective)
    assert report.summary["prediction_seconds"] == 7


def test_explicit_sensor_mask_aligns_raw_array_rows_then_subsets(tmp_path):
    import yaml
    from ml_platform_mode_decomp.data import load_field_dataset
    from ml_platform_mode_decomp.evaluation import load_observation_mask

    np.save(tmp_path / "field.npy", np.arange(9.0).reshape(3, 1, 3, 1))
    raw_mask = np.eye(3, dtype=bool).reshape(3, 1, 3)
    np.save(tmp_path / "sensors.npy", raw_mask)
    (tmp_path / "samples.csv").write_text("sample_id,array_index\nthird,2\nfirst,0\nsecond,1\n", encoding="utf-8")
    (tmp_path / "field_dataset.yaml").write_text(
        yaml.safe_dump(
            {"schema": "field_dataset_v1", "fields": {"path": "field.npy"}, "samples": {"path": "samples.csv"}}
        ),
        encoding="utf-8",
    )
    data = {"local_path": str(tmp_path)}
    dataset = load_field_dataset(data)
    aligned = load_observation_mask(data, dataset, "sensors.npy")
    np.testing.assert_array_equal(aligned, raw_mask[[2, 0, 1]])
    selected = dataset.subset([2, 0])
    observed = observation_mask(selected, 0.3, 41, explicit=aligned[[2, 0]])
    np.testing.assert_array_equal(observed, raw_mask[[1, 2]])
    from ml_platform_mode_decomp.evaluation.analysis import prepare_imputation_reference

    prepared = prepare_imputation_reference(dataset, selected, observed, {"reference": "hidden"})
    assert np.isnan(prepared.input_data.fields[~observed]).all()
    report = prepared.hidden.score(np.zeros_like(selected.fields))
    assert report.summary["evaluation_reference"] == "hidden"
    assert "hidden_rmse" in report.metrics and "field_rmse" not in report.metrics


@pytest.mark.parametrize("section", ["tuning", "uncertainty"])
def test_unknown_feature_keys_fail_even_when_enabled_is_absent(section, tmp_path):
    from ml_platform_mode_decomp.policy import resolve_config

    with pytest.raises(ValueError, match=f"Unknown {section}"):
        resolve_config(
            {"task": "mode_decomp_pipeline", "data": {"local_path": str(tmp_path)}, section: {"enbaled": True}}
        )


def test_objective_namespace_defaults_and_explicit_mask_settings_are_unambiguous(tmp_path):
    from ml_platform_mode_decomp.policy import resolve_config

    raw = {
        "task": "mode_decomp_pipeline",
        "data": {"local_path": str(tmp_path)},
        "metrics": {"reference": "observations"},
    }
    cfg = resolve_config(raw)
    assert cfg["model"]["selection_metric"] == "observation_rmse"
    assert cfg["metrics"]["names"] == ["observation_rmse", "observation_mae"]
    with pytest.raises(ValueError, match="namespace"):
        resolve_config({**raw, "model": {"selection_metric": "field_rmse"}})
    raw = {
        "task": "mode_decomp_analysis",
        "data": {"local_path": str(tmp_path)},
        "analysis": {"kind": "imputation", "observation_mask": "sensors.npy"},
    }
    cfg = resolve_config(raw)
    assert cfg["metrics"]["reference"] == "hidden"
    assert cfg["metrics"]["names"] == ["hidden_rmse", "hidden_mae"]
    explicit = {**raw, "metrics": {"reference": "hidden", "names": ["hidden_rmse"]}}
    assert resolve_config(explicit)["metrics"]["names"] == ["hidden_rmse"]
    with pytest.raises(ValueError, match="reference: hidden"):
        resolve_config({**raw, "metrics": {"reference": "hidden", "names": ["field_rmse"]}})
    with pytest.raises(ValueError, match="reference: hidden"):
        resolve_config({**raw, "metrics": {"names": ["field_rmse"]}})
    with pytest.raises(ValueError, match="random mask"):
        resolve_config({**raw, "analysis": {**raw["analysis"], "observed_fraction": 0.5}})


def test_static_spatial_profile_has_high_global_r2_but_no_conditional_skill():
    profile = np.array([[-1000.0, 1000.0]])[None, ..., None]
    truth = profile + np.array([-1, -0.5, 0.5, 1])[:, None, None, None]
    reference = field_data(truth)
    report = score_fields(
        reference,
        np.broadcast_to(profile, truth.shape),
        {"names": ["field_r2"]},
        baseline=mean_field_prediction(reference, reference),
    )
    assert report.metrics["field_r2"] > 0.999999
    assert report.metrics["field_skill"] == 0
    assert report.metrics["field_rmse"] == pytest.approx(np.sqrt(0.625))
    np.testing.assert_allclose(report.per_sample.mse, [1, 0.25, 0.25, 1])


def test_hidden_truth_is_scored_separately_from_model_observations():
    truth = np.tile(np.array([0.0, 100.0])[None, None, :, None], (2, 1, 1, 1))
    reference = field_data(truth)
    observed = np.tile([True, False], (2, 1, 1))
    prediction = np.zeros_like(truth)
    seen = score_fields(reference, prediction, evaluation_mask=observed)
    hidden = score_fields(reference, prediction, evaluation_mask=~observed)
    full = score_fields(reference, prediction)
    assert seen.metrics["field_rmse"] == 0
    assert hidden.metrics["field_rmse"] == 100
    assert full.metrics["field_rmse"] == pytest.approx(np.sqrt(5000))
    with pytest.raises(ValueError, match="without known reference"):
        score_fields(replace(reference, observation_mask=observed), prediction, evaluation_mask=~observed)


def test_sample_and_channel_weights_have_one_explicit_measure():
    reference = field_data(np.zeros((2, 1, 2, 2)), weights=np.array([[1.0, 3.0]]))
    reference = replace(reference, sample_metadata={"importance": np.array([1.0, 3.0])})
    prediction = np.array([[[[2.0, 4.0], [0.0, 0.0]]], [[[0.0, 0.0], [2.0, 4.0]]]])
    report = score_fields(reference, prediction, {"sample_weight_column": "importance", "channel_scales": [1, 2]})
    # Per-sample scaled MSE: 1 and 3; sample weights: 1/4 and 3/4.
    assert report.metrics["field_rmse"] == pytest.approx(np.sqrt(2.5))
    assert report.metrics["channel_1_rmse"] == pytest.approx(np.sqrt(10))
    assert report.summary["sample_weight_column"] == "importance"


def test_zero_area_locations_do_not_make_training_baseline_undefined():
    training = field_data(
        np.array([[[[0.0], [999.0]]]]), weights=np.array([[1.0, 0.0]]), mask=np.array([[True, False]])
    )
    reference = replace(training, fields=np.array([[[[2.0], [20.0]]]]), observation_mask=None)
    prediction = np.array([[[[3.0], [100.0]]]])
    report = score_fields(reference, prediction, baseline=mean_field_prediction(training, reference))
    assert report.metrics["field_rmse"] == 1
    assert report.metrics["field_skill"] == pytest.approx(0.75)


@pytest.mark.parametrize("selected", [[8, 2, 5], [1, 2, 3], [5, 2, 5]])
def test_indexed_training_baseline_uses_only_selected_observed_rows(selected):
    rng = np.random.default_rng(842)
    values = rng.normal(size=(9, 4, 7, 2))[:, :, ::2]
    observed = rng.random(values.shape[:-1]) > 0.2
    observed[:, 0, 0] = False
    values[~observed] = np.nan
    values.setflags(write=False)
    training = replace(field_data(values, mask=observed), sample_metadata={"importance": np.zeros(len(values))})
    selected = np.asarray(selected)
    reference = training.subset([1, 6])
    baseline = mean_field_prediction(training, reference, indices=selected)
    known = observed[selected]
    expected = np.full(values.shape[1:], np.nan)
    np.divide(
        np.where(known[..., None], values[selected], 0).sum(axis=0),
        known.sum(axis=0)[..., None],
        out=expected,
        where=known.sum(axis=0)[..., None] > 0,
    )
    np.testing.assert_allclose(baseline, np.broadcast_to(expected, reference.fields.shape), atol=1e-15)


def test_artificial_observations_leave_positive_measure_for_both_supports():
    data = field_data(np.ones((2, 2, 2, 1)), weights=np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="at least two"):
        observation_mask(data, 0.5, 42)
    data = replace(data, geometry=replace(data.geometry, weights=np.array([[1.0, 0.0], [0.0, 1.0]])))
    observed = observation_mask(data, 0.5, 42)
    assert observed.sum() == 2
    score_fields(data, data.fields, evaluation_mask=observed)
    score_fields(data, data.fields, evaluation_mask=data.effective_mask & ~observed)


def test_nonorthogonal_regularized_representation_keeps_the_error_cross_term():
    rng = np.random.default_rng(8)
    conditions = rng.normal(size=(30, 2))
    y, x = np.mgrid[:6, :6]
    fields = np.einsum("np,phw->nhw", conditions, np.stack((x / 5, np.sin(y))))[..., None]
    data = field_data(fields)
    data = replace(data, conditions=conditions, condition_columns=("a", "b"))
    train, reference = data.subset(np.arange(20)), data.subset(np.arange(20, 30))
    model = FieldSurrogate(
        {
            "representation": {
                "decomposition": {"name": "rbf_expansion", "params": {"stride": 3, "length_scale": 3, "ridge_alpha": 1}}
            },
            "regression": {"name": "ridge", "params": {"alpha": 1}},
        }
    ).fit(train)
    report = analyze_surrogate(model, train, reference)
    values = report.metrics
    assert abs(values["error_cross_term_mse"]) > 1e-8
    assert values["field_rmse"] ** 2 == pytest.approx(
        values["representation_rmse"] ** 2 + values["regression_rmse"] ** 2 + values["error_cross_term_mse"], abs=1e-12
    )
    np.testing.assert_allclose(
        report.per_sample.mse,
        report.per_sample.representation_mse
        + report.per_sample.regression_mse
        + report.per_sample.error_cross_term_mse,
        atol=1e-12,
    )


def test_paired_interval_resamples_independent_groups_and_does_not_pool_splits(monkeypatch):
    monkeypatch.setattr("ml_platform_mode_decomp.evaluation.comparison._BOOTSTRAP_VALUES", 14)
    tables = []
    for split_seed in (10, 20):
        for candidate_id, losses in (("a", [1, 4, 9, 16]), ("b", [4, 9, 16, 25])):
            tables.append(
                pd.DataFrame(
                    {
                        "sample_id": list("abcd"),
                        "group": ["one", "one", "two", "two"],
                        "mse": losses,
                        "sample_weight": [0.25] * 4,
                        "coverage": [1] * 4,
                        "valid_points": [10] * 4,
                        "candidate_id": candidate_id,
                        "split_seed": split_seed,
                        "model_seed": 42,
                    }
                )
            )
    result = paired_comparisons(pd.concat(tables), draws=100, seed=11)
    assert len(result) == 2
    assert set(result.independent_units) == {2}
    assert set(result.resampling_unit) == {"group"}
    assert np.all(result.rmse_a_minus_b < 0)
    assert np.all(result.ci_low <= result.ci_high)
    rng = np.random.default_rng(11)
    for row in result.itertuples():
        indices = rng.integers(0, 2, (100, 2))
        # Each whole group carries half of the test measure.
        expected = np.sqrt(np.array([2.5, 12.5])[indices].mean(axis=1))
        expected -= np.sqrt(np.array([6.5, 20.5])[indices].mean(axis=1))
        np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(expected, [0.025, 0.975]), atol=1e-15)


@pytest.mark.parametrize("draws", [0, 100])
def test_time_dependence_does_not_get_an_iid_confidence_interval(draws):
    data = pd.DataFrame(
        {
            "sample_id": ["a", "b", "a", "b"],
            "mse": [1, 2, 2, 3],
            "sample_weight": [0.5] * 4,
            "coverage": [1] * 4,
            "valid_points": [4] * 4,
            "candidate_id": ["first", "first", "second", "second"],
            "split_seed": [1] * 4,
            "model_seed": [42] * 4,
        }
    )
    result = paired_comparisons(data, draws=draws, independent_samples=False)
    assert result.ci_low.isna().all()
    assert set(result.resampling_unit) == {"not_assumed_independent"}


@pytest.mark.parametrize("block_values", [23, 10_000])
def test_field_reductions_match_dense_weighted_formulas_on_strided_masked_data(monkeypatch, block_values):
    monkeypatch.setattr("ml_platform_mode_decomp.evaluation.fields._BLOCK_VALUES", block_values)
    rng = np.random.default_rng(51)
    truth = rng.normal(size=(5, 8, 12, 3)).astype(np.float32)[:, ::2, ::2]
    predicted = rng.normal(size=(5, 8, 12, 3)).astype(np.float32)[:, ::2, ::2]
    mask = rng.uniform(size=truth.shape[:-1]) > 0.3
    mask[0] = True  # A known training value remains at every spatial location.
    truth[~mask] = np.nan
    predicted[~mask] = np.nan
    area = rng.uniform(size=truth.shape[1:-1])
    area[0, 0] = 0
    sw = np.array([0, 1, 2, 3, 4], dtype=float)
    reference = replace(field_data(truth, mask=mask, weights=area), fields=truth, sample_metadata={"importance": sw})
    baseline = mean_field_prediction(reference, reference)
    options = {
        "names": ["field_r2"],
        "sample_weight_column": "importance",
        "channel_scales": [1, 2, 3],
        "channel_weights": [1, 0, 3],
    }
    prepared = prepare_reference(reference, options, baseline=baseline)
    report = prepared.score(predicted)
    valid = mask & (area > 0)
    weights = np.where(valid, area, 0)
    weights /= weights.sum(axis=(1, 2), keepdims=True)
    sw /= sw.sum()
    cw = np.array([0.25, 0, 0.75])
    scales = np.array([1, 2, 3])
    error = np.where(valid[..., None], predicted.astype(float) - truth, 0)
    squared = np.sum(weights[..., None] * error**2, axis=(1, 2))
    mse = (squared / scales**2) @ cw
    mae = (np.sum(weights[..., None] * abs(error), axis=(1, 2)) / scales) @ cw
    active = sw > 0
    np.testing.assert_allclose(report.per_sample.mse[active], mse[active], rtol=1e-14)
    np.testing.assert_allclose(report.per_sample.mae[active], mae[active], rtol=1e-14)
    assert report.per_sample.loc[~active, ["mse", "mae"]].isna().all().all()
    assert report.metrics["field_rmse"] == pytest.approx(np.sqrt(sw @ mse), rel=1e-14)
    known = np.where(valid[..., None], truth, 0)
    mean = np.einsum("n,nhw,nhwc->c", sw, weights, known)
    centered = np.where(valid[..., None], truth - mean, 0)
    variance = np.einsum("n,nhw,nhwc->c", sw, weights, centered**2)
    expected_r2 = 1 - np.average((sw @ squared)[cw > 0] / variance[cw > 0], weights=cw[cw > 0])
    assert report.metrics["field_r2"] == pytest.approx(expected_r2, abs=1e-14)
    expected_mean = np.sum(np.where(mask[..., None], truth, 0), axis=0, dtype=float) / mask.sum(axis=0)[..., None]
    np.testing.assert_allclose(baseline[0], expected_mean, rtol=1e-14, atol=1e-15)
    represented = baseline + 0.25
    cross, samples = prepared.inner_product(represented, predicted, first_reference=truth, second_reference=represented)
    regression, regression_samples = prepared.loss(predicted, truth=represented)
    representation, representation_samples = prepared.loss(represented)
    np.testing.assert_allclose(
        mse[active], (representation_samples + regression_samples + 2 * samples)[active], atol=1e-14
    )
    assert sw @ mse == pytest.approx(representation + regression + 2 * cross, abs=1e-14)
    broken = predicted.copy()
    valid_broken = np.argwhere(valid[1])[0]
    broken[(1, *valid_broken, 0)] = np.nan
    with pytest.raises(ValueError, match="finite reference and predictions"):
        prepared.score(broken)


def test_finite_large_weights_have_the_same_measure_as_their_ratios():
    reference = field_data(np.zeros((2, 1, 2, 2)), weights=np.array([[1e308, 3e307]]))
    reference = replace(reference, sample_metadata={"importance": np.array([1e308, 1e308])})
    prediction = np.broadcast_to([1.0, 3.0], reference.fields.shape)
    with np.errstate(over="raise", invalid="raise"):
        report = score_fields(
            reference,
            prediction,
            {
                "sample_weight_column": "importance",
                "channel_weights": [1e308, 1e308],
            },
        )
    assert report.metrics["field_rmse"] == pytest.approx(np.sqrt(5))
    assert report.metrics["field_mae"] == 2
    np.testing.assert_array_equal(report.per_sample.sample_weight, [0.5, 0.5])
    sparse = field_data(
        np.array([[[[np.nan], [0.0]]]]), mask=np.array([[False, True]]), weights=np.array([[1e308, 1e-308]])
    )
    sparse_report = score_fields(sparse, np.array([[[[np.nan], [2.0]]]]))
    assert sparse_report.metrics["field_rmse"] == 2


def test_zero_weight_samples_are_unscored_and_do_not_require_finite_predictions():
    reference = replace(field_data(np.zeros((2, 1, 2, 1))), sample_metadata={"importance": [0.0, 1.0]})
    prediction = np.array([[[[np.nan], [np.nan]]], [[[2.0], [2.0]]]])
    baseline = np.array([[[[np.nan], [np.nan]]], [[[1.0], [1.0]]]])
    report = score_fields(reference, prediction, {"sample_weight_column": "importance"}, baseline=baseline)
    assert report.metrics["field_rmse"] == report.metrics["field_mae"] == 2
    assert report.metrics["field_skill"] == -3
    assert np.isnan(report.per_sample.mse.iloc[0])
    assert report.per_sample.mse.iloc[1] == 4


def test_zero_weight_validation_fold_does_not_invalidate_the_pooled_cv_objective():
    from ml_platform_mode_decomp.training.search import prepare_folds, _trial_report
    from ml_platform_mode_decomp.policy import candidate_config

    controls = np.arange(8.0)[:, None]
    data = FieldDataset.from_arrays(
        np.broadcast_to(controls[:, None, None], (8, 2, 2, 1)).copy(),
        controls,
        sample_metadata={"importance": [1.0, 0, 0, 0, 0, 0, 0, 0]},
    )
    options = {"names": ["field_rmse"], "primary_metric": "field_rmse", "sample_weight_column": "importance"}
    folds = prepare_folds(data, {"method": "random", "n_splits": 2, "seed": 1}, options)
    assert any(not fold.evaluation.measure.sample_weights.any() for fold in folds.folds)
    candidate = candidate_config(
        {
            "id": "linear",
            "representation": {"decomposition": {"name": "dct2"}},
            "regression": {"name": "ridge", "params": {"alpha": 0.0}},
        },
        42,
    )
    report = _trial_report(candidate, data, folds, options)
    assert report.metrics["field_rmse"] < 1e-12
    assert report.per_sample.loc[report.per_sample.sample_weight == 0, "mse"].isna().all()


def test_paired_comparison_excludes_unscored_rows_and_zero_weight_units():
    data = pd.DataFrame(
        {
            "sample_id": ["unused", "first", "second"] * 2,
            "mse": [np.nan, 1.0, 4.0, np.nan, 4.0, 9.0],
            "sample_weight": [0.0, 0.5, 0.5] * 2,
            "coverage": [1.0] * 6,
            "valid_points": [2] * 6,
            "candidate_id": ["a"] * 3 + ["b"] * 3,
            "split_seed": [1] * 6,
            "model_seed": [42] * 6,
        }
    )
    result = paired_comparisons(data, draws=100).iloc[0]
    assert result.independent_units == 2
    assert result.rmse_a_minus_b == pytest.approx(np.sqrt(2.5) - np.sqrt(6.5))
    assert np.isfinite(result.ci_low) and np.isfinite(result.ci_high)
