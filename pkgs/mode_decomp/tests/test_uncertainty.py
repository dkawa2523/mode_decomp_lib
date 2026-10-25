"""Exact finite-rank, support, and physical-unit checks for residual field bands."""

from dataclasses import replace
import json

import numpy as np
import pytest

from ml_platform_mode_decomp.data import FieldDataset
from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.uncertainty import FieldCalibrator


def dataset(values, *, prefix="cal", weights=None, mask=None, units=None):
    values = np.asarray(values)
    count, points, channels = values.shape
    x, y = np.meshgrid(np.arange(points, dtype=float), [0.0])
    geometry = DomainSpec(
        "rectangle",
        {"x": x, "y": y},
        None,
        np.ones((1, points)) if weights is None else np.asarray(weights, dtype=float).reshape(1, points),
        {},
    )
    return FieldDataset(
        fields=values.reshape(count, 1, points, channels),
        geometry=geometry,
        sample_ids=np.asarray([f"{prefix}_{i}" for i in range(count)]),
        observation_mask=mask,
        components=tuple(f"c{i}" for i in range(channels)),
        manifest={} if units is None else {"fields": {"units": list(units)}},
    )


def test_corrected_rank_is_an_order_statistic_and_not_an_interpolated_quantile():
    calibration = dataset(np.arange(1, 10).reshape(9, 1, 1))
    calibrator = FieldCalibrator(coverage=0.9).fit(calibration, np.zeros_like(calibration.fields))
    assert calibrator.summary()["order_statistic_rank"] == 9
    assert calibrator.summary()["threshold"] == 9
    points = np.array([[[[1.0]]], [[[2.0]]]])
    untouched = points.copy()
    lower, upper = calibrator.intervals(points)
    np.testing.assert_array_equal(lower, points - 9)
    np.testing.assert_array_equal(upper, points + 9)
    np.testing.assert_array_equal(points, untouched)
    json.dumps(calibrator.summary(), allow_nan=False)


def test_exchangeable_finite_population_has_exact_simultaneous_coverage():
    values = np.arange(1, 11, dtype=float)
    complete = dataset(np.stack((values, values / 2), axis=1)[..., None])
    covered = []
    for heldout in range(10):
        calibration = complete.subset([i for i in range(10) if i != heldout])
        future = complete.subset([heldout])
        calibrator = FieldCalibrator(coverage=0.9).fit(calibration, np.zeros_like(calibration.fields))
        covered.append(calibrator.evaluate(future, np.zeros_like(future.fields))["sample_simultaneous_coverage"])
    assert sum(covered) == 9  # One unseen rank out of ten falls beyond the calibration maximum.


def test_coverage_normalizes_finite_large_sample_channel_and_area_weights():
    calibration = dataset(np.ones((9, 2, 2)), weights=[1e308, 1e308])
    calibrator = FieldCalibrator(coverage=0.9).fit(calibration, np.zeros_like(calibration.fields))
    reference = dataset(
        np.array([[[0.5, 0.5], [0.5, 0.5]], [[2.0, 2.0], [2.0, 2.0]]]), prefix="future", weights=[1e308, 1e308]
    )
    reference = replace(reference, sample_metadata={"importance": np.array([1e308, 1e308])})
    with np.errstate(over="raise", invalid="raise"):
        result = calibrator.evaluate(
            reference,
            np.zeros_like(reference.fields),
            {"sample_weight_column": "importance", "channel_weights": [1e308, 1e308]},
        )
    assert result["weighted_sample_simultaneous_coverage"] == 0.5
    assert result["pointwise_coverage"] == 0.5
    assert result["channel_mean_width"] == [2.0, 2.0]


def test_pointwise_and_simultaneous_scopes_are_distinct():
    calibration = dataset(np.array([[1, 10], [2, 20], [30, 3]])[..., None])
    future = dataset(np.array([[3, 11]])[..., None], prefix="test")
    zero = np.zeros_like(calibration.fields)
    pointwise = FieldCalibrator(coverage=0.5, scope="pointwise").fit(calibration, zero)
    simultaneous = FieldCalibrator(coverage=0.5).fit(calibration, zero)
    np.testing.assert_array_equal(pointwise.half_width.reshape(-1), [2, 10])
    np.testing.assert_array_equal(simultaneous.half_width.reshape(-1), [20, 20])
    assert pointwise.evaluate(future, np.zeros_like(future.fields))["sample_simultaneous_coverage"] == 0
    assert simultaneous.evaluate(future, np.zeros_like(future.fields))["sample_simultaneous_coverage"] == 1


def test_insufficient_calibration_is_rejected_and_failed_refit_invalidates_old_bands():
    calibration = dataset(np.ones((9, 1, 1)))
    calibrator = FieldCalibrator().fit(calibration, np.zeros_like(calibration.fields))
    short = calibration.subset(np.arange(8))
    with pytest.raises(ValueError, match="Insufficient calibration"):
        calibrator.fit(short, np.zeros_like(short.fields))
    with pytest.raises(ValueError, match="must be fitted"):
        calibrator.intervals(np.zeros((1, 1, 1, 1)))


def test_decimal_coverage_does_not_add_a_spurious_quantile_rank():
    calibration = dataset(np.arange(99).reshape(99, 1, 1))
    calibrator = FieldCalibrator(coverage=0.07).fit(calibration, np.zeros_like(calibration.fields))
    assert calibrator.summary()["order_statistic_rank"] == 7
    assert calibrator.summary()["threshold"] == 6


def test_masked_nan_and_zero_measure_do_not_enter_scores_or_bands():
    calibration = dataset(
        np.tile(np.array([1.0, np.nan])[None, :, None], (9, 1, 1)),
        weights=[1, 0],
        mask=np.asarray([[True, False]]),
    )
    predictions = np.zeros_like(calibration.fields)
    predictions[:, :, 1] = np.nan
    with np.errstate(invalid="raise"):
        calibrator = FieldCalibrator().fit(calibration, predictions)
        lower, upper = calibrator.intervals(predictions)
    assert np.isnan(lower[:, :, 1]).all() and np.isnan(upper[:, :, 1]).all()
    np.testing.assert_array_equal(upper[:, :, 0], np.ones((9, 1, 1)))
    assert calibrator.evaluate(calibration, predictions)["sample_simultaneous_coverage"] == 1


@pytest.mark.parametrize("scope", ["pointwise", "simultaneous"])
def test_unknown_positive_measure_truth_is_not_a_full_field_calibration(scope):
    calibration = dataset(np.ones((9, 2, 1)), mask=np.asarray([[True, False]]))
    with pytest.raises(ValueError, match="complete truth"):
        FieldCalibrator(scope=scope).fit(calibration, np.zeros_like(calibration.fields))


def test_channel_scales_use_only_disjoint_proper_training_and_widths_keep_units():
    training = dataset(np.array([[[0.0, 0.0]], [[2.0, 20.0]]]), prefix="train", units=["m", "Pa"])
    calibration = dataset(np.tile([[[1.0, 10.0]]], (9, 1, 1)), units=["m", "Pa"])
    calibrator = FieldCalibrator(normalization="channel").fit(
        calibration, np.zeros_like(calibration.fields), training=training
    )
    np.testing.assert_array_equal(calibrator.channel_scales, [1, 10])
    evaluation = calibrator.evaluate(calibration, np.zeros_like(calibration.fields))
    np.testing.assert_array_equal(evaluation["channel_mean_width"], [2, 20])
    assert evaluation["mean_width"] is None and evaluation["mean_scaled_width"] == 2
    changed = replace(calibration, fields=calibration.fields * 1000)
    other = FieldCalibrator(normalization="channel").fit(changed, np.zeros_like(changed.fields), training=training)
    np.testing.assert_array_equal(other.channel_scales, calibrator.channel_scales)
    with pytest.raises(ValueError, match="disjoint"):
        FieldCalibrator(normalization="channel").fit(
            calibration, np.zeros_like(calibration.fields), training=calibration
        )


def test_empirical_coverage_distinguishes_sample_events_from_weighted_points():
    calibration = dataset(np.ones((1, 2, 1)), weights=[1, 3])
    calibrator = FieldCalibrator(coverage=0.5).fit(calibration, np.zeros_like(calibration.fields))
    future = dataset(np.array([[1.0, 2.0], [0.0, 0.0]])[..., None], prefix="test", weights=[1, 3])
    future = replace(future, sample_metadata={"weight": np.asarray([9.0, 1.0])})
    report = calibrator.evaluate(future, np.zeros_like(future.fields), {"sample_weight_column": "weight"})
    assert report["sample_simultaneous_coverage"] == 0.5
    assert report["weighted_sample_simultaneous_coverage"] == pytest.approx(0.1)
    assert report["pointwise_coverage"] == pytest.approx(0.325)
    assert report["channel_mean_width"] == [2]


def test_unsigned_residuals_do_not_wrap_and_prediction_schema_is_checked():
    calibration = dataset(np.zeros((9, 1, 1), dtype=np.uint8))
    calibrator = FieldCalibrator().fit(calibration, np.full_like(calibration.fields, 255))
    assert calibrator.summary()["threshold"] == 255
    with pytest.raises(ValueError, match="real nonempty"):
        calibrator.intervals(np.zeros((1, 1, 1, 1), dtype=complex))
    with pytest.raises(ValueError, match="matching"):
        calibrator.intervals(np.zeros((1, 2, 2, 1)))
    shifted = replace(
        calibration.geometry, coords={**calibration.geometry.coords, "x": calibration.geometry.coords["x"] + 1}
    )
    with pytest.raises(ValueError, match="geometry/components/units"):
        calibrator.evaluate(replace(calibration, geometry=shifted), np.zeros_like(calibration.fields))


def test_normalization_needs_training_and_a_constant_channel_uses_unit_scale():
    calibration = dataset(np.ones((9, 1, 2)))
    with pytest.raises(ValueError, match="proper training"):
        FieldCalibrator(normalization="channel").fit(calibration, np.zeros_like(calibration.fields))
    training = dataset(np.array([[[0.0, 2.0]], [[2.0, 2.0]]]), prefix="train")
    calibrator = FieldCalibrator(normalization="channel").fit(
        calibration, np.zeros_like(calibration.fields), training=training
    )
    assert calibrator.summary()["constant_training_channels_using_unit_scale"] == [1]


def test_mesh_bands_preserve_public_nvc_shape_and_zero_mass_vertices():
    geometry = DomainSpec(
        "mesh",
        {"x": np.arange(3.0), "y": np.zeros(3)},
        None,
        np.asarray([1.0, 1.0, 0.0]),
        {"vertex_count": 3},
    )
    calibration = FieldDataset(
        fields=np.tile(np.asarray([[[1.0], [2.0], [0.0]]]), (9, 1, 1)),
        geometry=geometry,
        sample_ids=np.asarray([f"cal_{i}" for i in range(9)]),
    )
    calibrator = FieldCalibrator(scope="pointwise").fit(calibration, np.zeros_like(calibration.fields))
    lower, upper = calibrator.intervals(np.zeros((2, 3, 1)))
    assert lower.shape == upper.shape == (2, 3, 1)
    np.testing.assert_array_equal(upper[:, :2, 0], [[1, 2], [1, 2]])
    assert np.isnan(upper[:, 2]).all()


@pytest.mark.parametrize("scope", ["simultaneous", "pointwise"])
def test_chunked_calibration_and_reporting_match_full_order_statistics_and_saved_endpoints(monkeypatch, scope):
    from types import SimpleNamespace

    import ml_platform_mode_decomp.uncertainty as uncertainty
    from ml_platform_mode_decomp.uncertainty_reporting import add_uncertainty_report

    monkeypatch.setattr(uncertainty, "_BATCH_VALUES", 20)
    rng = np.random.default_rng(61)
    weights = np.array([0, 1, 2, 3, 4])
    train_values = rng.normal(size=(17, 5, 2))
    observed = np.ones((17, 1, 5), dtype=bool)
    observed[::2, :, 1] = False
    training = dataset(train_values, prefix="train", weights=weights, mask=observed)
    calibration = dataset(rng.normal(size=(43, 5, 2)), weights=weights)
    predictions = rng.normal(size=calibration.fields.shape).astype(np.float32)
    calibrator = FieldCalibrator(coverage=0.8, scope=scope, normalization="channel").fit(
        calibration, predictions, training=training
    )
    support = weights > 0
    scales = np.std(training.fields[training.effective_mask & support[None, :]], axis=0)
    np.testing.assert_allclose(calibrator.channel_scales, scales, atol=1e-14, rtol=1e-14)
    scores = abs(predictions[:, :, support, :].astype(float) - calibration.fields[:, :, support, :]) / scales
    scores = scores.reshape(len(calibration), -1, 2)
    if scope == "simultaneous":
        scores = scores.max(axis=(1, 2))
    threshold = np.partition(scores, calibrator.rank - 1, axis=0)[calibrator.rank - 1]
    np.testing.assert_allclose(calibrator.threshold, threshold, atol=1e-14, rtol=1e-14)

    future = dataset(rng.normal(size=(11, 5, 2)), prefix="future", weights=weights)
    point = rng.normal(size=future.fields.shape).astype(np.float32)
    lower, upper = calibrator.intervals(point)
    # Include exact endpoints and their next representable neighbours in the truth.
    truth = future.fields.copy()
    truth[0, :, support, :] = upper[0, :, support, :]
    truth[1, :, support, :] = np.nextafter(upper[1, :, support, :], np.inf)
    truth[2, :, support, :] = lower[2, :, support, :]
    truth[3, :, support, :] = np.nextafter(lower[3, :, support, :], -np.inf)
    future = replace(future, fields=truth, sample_metadata={"weight": np.arange(1.0, 12.0)})
    options = {"sample_weight_column": "weight", "channel_weights": [1, 3]}
    from ml_platform_mode_decomp.evaluation import score_fields

    report = score_fields(future, point, options)
    add_uncertainty_report(report, SimpleNamespace(calibrator=calibrator), future, options)
    inside = ((lower <= truth) & (truth <= upper))[:, :, support, :].reshape(len(future), -1, 2)
    area = weights[support] / weights.sum()
    sample_weights = np.arange(1.0, 12.0) / np.arange(1.0, 12.0).sum()
    evaluation = report.summary["field_uncertainty"]["evaluation"]
    expected_pointwise = (sample_weights @ np.sum(inside * area[None, :, None], axis=1)) @ np.array([0.25, 0.75])
    assert evaluation["pointwise_coverage"] == pytest.approx(expected_pointwise)
    assert evaluation["sample_simultaneous_coverage"] == inside.all(axis=(1, 2)).mean()
    np.testing.assert_array_equal(report.per_sample.interval_simultaneous_inside, inside.all(axis=(1, 2)))
    np.testing.assert_allclose(
        report.per_sample.interval_pointwise_coverage,
        np.sum(inside * area[None, :, None], axis=1) @ np.array([0.25, 0.75]),
    )
    for channel in range(2):
        np.testing.assert_allclose(
            report.per_sample[f"channel_{channel}_interval_coverage"],
            np.sum(inside * area[None, :, None], axis=1)[:, channel],
        )
    assert evaluation["target"] == "grid_fixed_support"
    assert evaluation == calibrator.evaluate(future, point, options)


@pytest.mark.parametrize("mesh", [False, True])
def test_pointwise_coordinate_blocks_preserve_strided_grid_and_mesh_order_statistics(monkeypatch, mesh):
    import ml_platform_mode_decomp.uncertainty as uncertainty

    rng = np.random.default_rng(72)
    count = 31
    if mesh:
        fields = rng.normal(size=(count, 24, 4)).astype(np.float32)[:, ::2, ::2]
        predictions = rng.normal(size=(count, 24, 4)).astype(np.float32)[:, ::2, ::2]
        x, y = np.arange(12.0), np.zeros(12)
    else:
        fields = rng.normal(size=(count, 6, 8, 4)).astype(np.float32)[:, ::2, ::2, ::2]
        predictions = rng.normal(size=(count, 6, 8, 4)).astype(np.float32)[:, ::2, ::2, ::2]
        x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    weights = np.ones(fields.shape[1:-1])
    weights.flat[0] = 0
    domain = np.ones(weights.shape, dtype=bool)
    domain.flat[-1] = False
    support = (weights > 0) & domain
    fields[:, ~support, :] = np.nan
    predictions[:, ~support, :] = np.nan
    geometry = DomainSpec(
        "mesh" if mesh else "rectangle", {"x": x, "y": y}, domain, weights, {"vertex_count": 12} if mesh else {}
    )
    calibration = FieldDataset(
        fields, geometry, np.asarray([f"cal_{i}" for i in range(count)]), observation_mask=support
    )
    before_fields, before_predictions = fields.copy(), predictions.copy()
    fields.flags.writeable = predictions.flags.writeable = False
    monkeypatch.setattr(uncertainty, "_BATCH_VALUES", count * fields.shape[-1] * 2)
    calibrator = FieldCalibrator(scope="pointwise").fit(calibration, predictions)
    scores = abs(predictions[:, support, :].astype(float) - fields[:, support, :].astype(float))
    expected = np.partition(scores, calibrator.rank - 1, axis=0)[calibrator.rank - 1]
    np.testing.assert_array_equal(calibrator.threshold, expected)
    np.testing.assert_array_equal(calibrator.half_width[support], expected)
    assert np.isnan(calibrator.half_width[~support]).all()
    np.testing.assert_array_equal(fields, before_fields)
    np.testing.assert_array_equal(predictions, before_predictions)


@pytest.mark.parametrize("coverage", [True, 0, 1, np.nan, np.inf])
def test_invalid_coverage_is_rejected(coverage):
    with pytest.raises(ValueError, match="coverage"):
        FieldCalibrator(coverage=coverage)


@pytest.mark.parametrize("coverage", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("scope", ["pointwise", "simultaneous"])
def test_one_pass_calibration_batches_match_independent_sorted_residuals(monkeypatch, coverage, scope):
    import ml_platform_mode_decomp.uncertainty as uncertainty

    rng = np.random.default_rng(62)
    fields = rng.normal(size=(31, 8, 4))[:, ::2, ::2]
    predictions = rng.normal(size=(31, 1, 8, 4))[:, :, ::2, ::2]
    support = np.array([False, True, True, True])
    fields[:, ~support, :] = predictions[:, 0, ~support, :] = np.nan
    calibration = dataset(fields, weights=[0, 1, 2, 3], mask=support[None, :])
    training = dataset(rng.normal(size=(17, 4, 2)), prefix="train", weights=[0, 1, 2, 3])
    predictions.flags.writeable = calibration.fields.flags.writeable = False
    monkeypatch.setattr(uncertainty, "_BATCH_VALUES", 24)
    batches = (predictions[start:stop] for start, stop in [(0, 1), (1, 8), (8, 11), (11, 28), (28, 31)])
    calibrator = FieldCalibrator(coverage=coverage, scope=scope, normalization="channel").fit_batches(
        calibration, batches, training=training
    )
    scales = np.std(training.fields[:, :, support, :].reshape(-1, 2), axis=0)
    scores = abs(predictions[:, 0, support, :] - fields[:, support, :]) / scales
    if scope == "simultaneous":
        scores = scores.max(axis=(1, 2))
    expected = np.sort(scores, axis=0)[calibrator.rank - 1]
    np.testing.assert_allclose(calibrator.threshold, expected, rtol=1e-14, atol=1e-14)
    assert np.isnan(calibrator.half_width[:, ~support, :]).all()


@pytest.mark.parametrize("count", [8, 10])
def test_calibration_batch_count_must_match_reference_and_failed_refit_invalidates_bands(count):
    calibration = dataset(np.ones((9, 2, 1)))
    calibrator = FieldCalibrator().fit(calibration, np.zeros_like(calibration.fields))
    with pytest.raises(ValueError, match="one row per calibration sample"):
        calibrator.fit_batches(calibration, iter([np.zeros((count, 1, 2, 1))]))
    with pytest.raises(ValueError, match="must be fitted"):
        calibrator.intervals(np.zeros((1, 1, 2, 1)))
