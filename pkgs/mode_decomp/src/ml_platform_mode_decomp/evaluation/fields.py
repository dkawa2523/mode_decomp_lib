"""One field-space loss definition for training, representation and imputation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data.schema import FieldDataset
from ..geometry.identity import geometry_fingerprint


# Bound working arrays even when one spatial field is larger than a batch.
_BLOCK_VALUES = 262_144


def _field_blocks(shape):
    remaining = max(1, _BLOCK_VALUES // shape[-1])
    steps = []
    for length in reversed(shape[:-1]):
        step = min(length, remaining)
        steps.append(step)
        remaining = max(1, remaining // step)
    steps.reverse()
    for starts in product(*(range(0, length, step) for length, step in zip(shape[:-1], steps, strict=True))):
        yield tuple(
            slice(start, min(start + step, length))
            for start, step, length in zip(starts, steps, shape[:-1], strict=True)
        )


@dataclass
class ScoreWeights:
    """Sample/channel measure, independent of any grid quadrature."""

    sample_weights: np.ndarray
    scales: np.ndarray
    channel_weights: np.ndarray
    summary: dict[str, Any]


@dataclass
class _FieldMeasure(ScoreWeights):
    truth: np.ndarray
    valid: np.ndarray
    area: np.ndarray
    area_scale: np.ndarray
    total: np.ndarray
    coverage: np.ndarray
    counts: np.ndarray

    def blocks(self):
        for index in _field_blocks(self.truth.shape):
            active = self.sample_weights[index[0]] > 0
            valid = self.valid[index] & active.reshape((-1,) + (1,) * (self.valid.ndim - 1))
            weights = np.where(valid, self.area[index[1:]], 0.0)
            weights /= self.area_scale[index[0]].reshape((-1,) + (1,) * (valid.ndim - 1))
            weights /= self.total[index[0]].reshape((-1,) + (1,) * (valid.ndim - 1))
            yield index, valid, weights


@dataclass
class ScoreResult:
    metrics: dict[str, float]
    per_sample: pd.DataFrame
    summary: dict[str, Any]
    predictions: np.ndarray | None
    fields: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    packed_coefficients: np.ndarray | None = field(default=None, repr=False)


# The detailed analysis API extends the same result with requested fields and
# coefficient diagnostics; existing callers keep their public result name.
EvaluationReport = ScoreResult


@dataclass
class PreparedReference:
    """One operation's fixed truth, support and measure, shared across predictions."""

    reference: FieldDataset
    options: Mapping[str, Any]
    measure: _FieldMeasure
    baseline: np.ndarray | None = None

    def score(self, predictions, *, baseline=None):
        return _score(self, predictions, self.baseline if baseline is None else baseline)

    def loss(self, predictions, *, truth=None):
        measure = self.measure
        squared, _ = _reduce_errors(measure, predictions, truth=truth)
        per_sample = (squared / measure.scales**2) @ measure.channel_weights
        return _weighted_mean(measure.sample_weights, per_sample), _sample_values(measure, per_sample)

    def inner_product(self, first, second, *, first_reference=None, second_reference=None):
        """Integrate products, optionally forming either field difference inside each block."""
        measure = self.measure
        arrays = [_prediction(values, measure.truth.shape) for values in (first, second)]
        references = [
            None if values is None else _prediction(values, measure.truth.shape)
            for values in (first_reference, second_reference)
        ]
        products = np.zeros((len(measure.truth), measure.truth.shape[-1]))
        for index, valid, weights in measure.blocks():
            left, right = (
                _difference(None if actual is None else actual[index], values[index], valid)
                for actual, values in zip(references, arrays, strict=True)
            )
            left *= right
            products[index[0]] += _spatial_sum(left, weights)
        per_sample = (products / measure.scales**2) @ measure.channel_weights
        return _weighted_mean(measure.sample_weights, per_sample), _sample_values(measure, per_sample)


def prepare_reference(reference, options=None, *, evaluation_mask=None, baseline=None, allow_zero_weight=False):
    options = dict(options or {})
    return PreparedReference(
        reference, options, _measure(reference, options, evaluation_mask, allow_zero_weight=allow_zero_weight), baseline
    )


def normalized_weights(values, count, label, *, allow_zero=False):
    if np.iscomplexobj(values):
        raise ValueError(f"{label} must be real")
    weights = np.asarray(values, dtype=float)
    if weights.shape != (count,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{label} must be finite, nonnegative and have positive sum")
    maximum = float(weights.max()) if len(weights) else 0.0
    if maximum == 0:
        if allow_zero:
            return np.zeros(count)
        raise ValueError(f"{label} must be finite, nonnegative and have positive sum")
    scaled = weights / maximum
    return scaled / scaled.sum()


def _weighted_mean(weights, values):
    active = weights > 0
    return float(weights[active] @ np.asarray(values)[active])


def _sample_values(measure, values):
    return np.where(measure.sample_weights > 0, values, np.nan)


def _validate_metric_options(options: Mapping[str, Any], *, reference: str) -> None:
    prefix = {"grid": "field", "observations": "observation", "hidden": "hidden"}[reference]
    suffixes = ("rmse", "mae", "skill") if reference == "observations" else ("rmse", "mae", "r2", "skill")
    allowed = {f"{prefix}_{suffix}" for suffix in suffixes}
    names = options.get("names", ())
    if not isinstance(names, (list, tuple)):
        raise ValueError("metrics.names must be a list of metric names")
    requested = [*names]
    if options.get("primary_metric") is not None:
        requested.append(options["primary_metric"])
    if any(not isinstance(name, str) or name not in allowed for name in requested):
        raise ValueError(f"metrics.reference: {reference} requires known {prefix}_* metric names: {sorted(allowed)}")


def score_weights(reference: FieldDataset, options: Mapping[str, Any], *, allow_zero=False):
    channels = reference.fields.shape[-1]
    for key in ("channel_scales", "channel_weights"):
        value = options.get(key)
        if value is not None and np.iscomplexobj(value):
            raise ValueError("Channel scales and weights must be real")
    scales = np.asarray(options.get("channel_scales", np.ones(channels)), dtype=float)
    channel_weights = np.asarray(options.get("channel_weights", np.ones(channels)), dtype=float)
    if scales.shape != (channels,) or channel_weights.shape != (channels,):
        raise ValueError("Channel scales and weights must match the channel count")
    if not np.isfinite(scales).all() or not np.isfinite(channel_weights).all():
        raise ValueError("Channel scales and weights must be finite")
    if np.any(scales <= 0) or np.any(channel_weights < 0) or not np.any(channel_weights > 0):
        raise ValueError("Channel scales must be positive and channel weights nonnegative with positive sum")
    channel_weights = normalized_weights(channel_weights, channels, "Channel weights")
    units = reference.field_units
    if units:
        if len(set(units)) > 1 and "channel_scales" not in options:
            raise ValueError("Mixed channel units require explicit channel_scales for aggregate metrics")
    sample_column = options.get("sample_weight_column")
    if sample_column is None:
        sample_weights = np.ones(len(reference))
    else:
        if sample_column not in reference.sample_metadata:
            raise ValueError("sample_weight_column must name reference sample metadata")
        sample_weights = np.asarray(reference.sample_metadata[sample_column])
        if sample_weights.shape != (len(reference),):
            raise ValueError("Sample weights must have one value per sample")
    sample_weights = normalized_weights(sample_weights, len(reference), "Sample weights", allow_zero=allow_zero)
    summary = {
        "reference_scope": options.get("reference_scope", "reference"),
        "sample_count": len(reference),
        "sample_weight_column": sample_column,
        "channel_scales": scales.tolist(),
        "channel_weights": channel_weights.tolist(),
        "field_components": list(reference.components),
        "field_units": [] if units is None else list(units),
        "geometry_sha256": geometry_fingerprint(reference.geometry),
        "dataset_revision": reference.revision,
        "aggregate_units": "scaled" if "channel_scales" in options else "field_units",
    }
    return ScoreWeights(sample_weights, scales, channel_weights, summary)


def _measure(reference: FieldDataset, options: Mapping[str, Any], evaluation_mask=None, *, allow_zero_weight=False):
    common = score_weights(reference, options, allow_zero=allow_zero_weight)
    truth = np.asarray(reference.fields)
    spatial = truth.shape[1:-1]
    area = reference.geometry.integration_weights()
    if area is not None and np.iscomplexobj(area):
        raise ValueError("Evaluation weights must be real")
    area = np.ones(spatial) if area is None else np.asarray(area, dtype=float).reshape(spatial)
    if not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError("Evaluation weights must be finite and nonnegative")
    valid = np.array(reference.effective_mask, copy=True)
    if evaluation_mask is not None:
        requested = np.asarray(evaluation_mask)
        if requested.shape not in (spatial, truth.shape[:-1]) or requested.dtype.kind not in "bui":
            raise ValueError("Evaluation mask must be spatial or sample-by-spatial boolean values")
        if not np.isin(requested, (0, 1)).all():
            raise ValueError("Evaluation mask must contain boolean or 0/1 values")
        requested = np.broadcast_to(requested.astype(bool), valid.shape)
        if np.any(requested & ~valid):
            raise ValueError("Evaluation cannot use locations without known reference values")
        valid &= requested
    valid &= area > 0
    total = np.zeros(len(reference))
    area_scale = np.zeros(len(reference))
    counts = np.zeros(len(reference), dtype=np.int64)
    for index in _field_blocks(truth.shape):
        axes = tuple(range(1, valid.ndim))
        values = np.where(valid[index], area[index[1:]], 0.0)
        updated = np.maximum(area_scale[index[0]], values.reshape(len(values), -1).max(axis=1))
        divisor = np.where(updated > 0, updated, 1)
        total[index[0]] *= area_scale[index[0]] / divisor
        total[index[0]] += np.sum(values / divisor.reshape((-1,) + (1,) * len(axes)), axis=axes)
        area_scale[index[0]] = updated
        counts[index[0]] += np.sum(valid[index], axis=axes)
    if np.any((total <= 0) & (common.sample_weights > 0)):
        raise ValueError("Each positive-weight evaluation sample requires positive valid weight")
    domain = np.ones(spatial, dtype=bool)
    if reference.geometry.mask is not None:
        domain &= np.asarray(reference.geometry.mask).reshape(spatial)
    domain_scale = float(np.max(np.where(domain, area, 0.0)))
    domain_area = float(np.sum(area[domain] / domain_scale)) if domain_scale > 0 else 0.0
    summary = {
        **common.summary,
        "reference_support": "explicit_known_reference" if evaluation_mask is not None else "known_reference",
        "spatial_measure": "per_sample_normalized_geometry_weights",
        "r2_definition": "channel-wise variance across samples and space; not conditional prediction skill",
        "skill_definition": "1 - model_mse / training_mean_profile_mse on identical reference support",
    }
    coverage = (area_scale / domain_scale) * (total / domain_area) if domain_area > 0 else np.zeros(len(reference))
    return _FieldMeasure(
        common.sample_weights,
        common.scales,
        common.channel_weights,
        summary,
        truth,
        valid,
        area,
        np.where(area_scale > 0, area_scale, 1),
        np.where(total > 0, total, 1),
        coverage,
        counts,
    )


def _prediction(prediction, shape):
    values = np.asarray(prediction)
    if values.dtype.kind not in "fiu" or values.shape != shape:
        raise ValueError("Predictions must be a real numeric array matching reference field shape")
    return values


def _difference(truth, values, valid):
    if not np.isfinite(values).all(where=valid[..., None]) or (
        truth is not None and not np.isfinite(truth).all(where=valid[..., None])
    ):
        raise ValueError("Evaluation requires finite reference and predictions at valid points")
    errors = np.zeros(values.shape, dtype=np.float64)
    if truth is None:
        np.copyto(errors, values, where=valid[..., None])
    else:
        np.subtract(values, truth, out=errors, where=valid[..., None], dtype=np.float64)
    return errors


def _spatial_sum(values, weights):
    return np.einsum("np,npc->nc", weights.reshape(len(weights), -1), values.reshape(len(values), -1, values.shape[-1]))


def _reduce_errors(measure, predictions, *, truth=None, absolute=False):
    predictions = _prediction(predictions, measure.truth.shape)
    actual = measure.truth if truth is None else _prediction(truth, measure.truth.shape)
    squared = np.zeros((len(actual), actual.shape[-1]))
    absolute_sum = np.zeros_like(squared)
    for index, valid, weights in measure.blocks():
        errors = _difference(actual[index], predictions[index], valid)
        if absolute:
            absolute_sum[index[0]] += _spatial_sum(np.abs(errors), weights)
        np.square(errors, out=errors)
        squared[index[0]] += _spatial_sum(errors, weights)
    return squared, absolute_sum


def _channel_variance(measure):
    mean = np.zeros(measure.truth.shape[-1])
    for index, valid, weights in measure.blocks():
        known = _difference(None, measure.truth[index], valid)
        mean += measure.sample_weights[index[0]] @ _spatial_sum(known, weights)
    variance = np.zeros_like(mean)
    for index, valid, weights in measure.blocks():
        centered = _difference(np.broadcast_to(mean, measure.truth[index].shape), measure.truth[index], valid)
        np.square(centered, out=centered)
        variance += measure.sample_weights[index[0]] @ _spatial_sum(centered, weights)
    return variance


def score_fields(
    reference: FieldDataset,
    predictions: np.ndarray,
    options: Mapping[str, Any] | None = None,
    *,
    evaluation_mask: np.ndarray | None = None,
    baseline: np.ndarray | None = None,
) -> EvaluationReport:
    """Score known grid reference values on explicitly selected support.

    An input observation mask is never guessed here. Use ``prepare_objective``
    for the hidden or scattered-observation namespaces.
    """
    options = dict(options or {})
    if options.get("reference", "grid") != "grid":
        raise ValueError("score_fields requires metrics.reference: grid; use prepare_objective for other references")
    _validate_metric_options(options, reference="grid")
    prepared = prepare_reference(reference, options, evaluation_mask=evaluation_mask, baseline=baseline)
    return prepared.score(predictions)


def _score(prepared, predictions, baseline):
    reference, options = prepared.reference, prepared.options
    measure = prepared.measure
    truth = measure.truth
    sw, scales, cw = measure.sample_weights, measure.scales, measure.channel_weights
    summary = dict(measure.summary)
    sample_channel_mse, sample_channel_mae = _reduce_errors(measure, predictions, absolute=True)
    sample_mse = (sample_channel_mse / scales**2) @ cw
    sample_mae = (sample_channel_mae / scales) @ cw
    mse = float(sw @ sample_mse)
    metrics = {"field_rmse": float(np.sqrt(mse)), "field_mae": float(sw @ sample_mae)}
    rows = {
        "sample_id": reference.sample_ids,
        "mse": _sample_values(measure, sample_mse),
        "rmse": _sample_values(measure, np.sqrt(sample_mse)),
        "mae": _sample_values(measure, sample_mae),
        "sample_weight": sw,
        "valid_points": measure.counts,
        "coverage": measure.coverage,
    }
    for channel in range(truth.shape[-1]):
        metrics[f"channel_{channel}_rmse"] = float(np.sqrt(sw @ sample_channel_mse[:, channel]))
        metrics[f"channel_{channel}_mae"] = float(sw @ sample_channel_mae[:, channel])
        rows[f"channel_{channel}_rmse"] = _sample_values(measure, np.sqrt(sample_channel_mse[:, channel]))
        rows[f"channel_{channel}_mae"] = _sample_values(measure, sample_channel_mae[:, channel])
    if (
        "field_r2" in options.get("names", [])
        or "hidden_r2" in options.get("names", [])
        or options.get("primary_metric") in {"field_r2", "hidden_r2"}
    ):
        variance = _channel_variance(measure)
        active = cw > 0
        if np.any(variance[active] == 0):
            reason = "undefined_for_constant_channel"
            if options.get("primary_metric") in {"field_r2", "hidden_r2"}:
                raise ValueError("Primary R2 metric is undefined for a constant channel; use RMSE or MAE")
            summary.setdefault("metric_status", {})["field_r2"] = reason
        else:
            channel_mse = sw @ sample_channel_mse
            metrics["field_r2"] = float(1 - np.average(channel_mse[active] / variance[active], weights=cw[active]))
    summary["baseline_status"] = "not_provided"
    if baseline is not None:
        candidate = np.asarray(baseline)
        if candidate.shape != truth.shape:
            raise ValueError("Baseline field shape differs from reference fields")
        if all(np.isfinite(candidate[index]).all(where=valid[..., None]) for index, valid, _ in measure.blocks()):
            baseline_squared, _ = _reduce_errors(measure, candidate)
            baseline_sample_mse = (baseline_squared / scales**2) @ cw
            baseline_mse = float(sw @ baseline_sample_mse)
            rows["baseline_mse"] = _sample_values(measure, baseline_sample_mse)
            metrics["baseline_field_rmse"] = float(np.sqrt(baseline_mse))
            summary["baseline_status"] = "defined" if baseline_mse > 0 else "zero_error_skill_undefined"
            if baseline_mse > 0:
                metrics["field_skill"] = float(1 - mse / baseline_mse)
        else:
            summary["baseline_status"] = "training_mean_undefined_at_evaluated_locations"
    summary["coverage_mean"] = float(sw @ measure.coverage)
    requested_primary = options.get("primary_metric")
    if requested_primary in {"field_skill", "hidden_skill"} and "field_skill" not in metrics:
        raise ValueError("Primary skill metric is undefined for this training baseline")
    if "field_skill" not in metrics:
        summary.setdefault("metric_status", {})["field_skill"] = summary["baseline_status"]
    if not np.any(sw > 0):
        metrics.clear()
        summary["evaluation_status"] = "no_positive_weight_samples"
    if any(not np.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation loss is not finite; rescale field values or channel scales")
    return EvaluationReport(metrics, pd.DataFrame(rows), summary, np.asarray(predictions))


def evaluate_fields(dataset: FieldDataset, predictions: np.ndarray, options=None) -> dict[str, float]:
    return score_fields(dataset, predictions, options).metrics


def mean_field_prediction(training: FieldDataset, reference: FieldDataset, *, indices=None) -> np.ndarray:
    """Fit an observed training mean, gathering only a bounded batch of rows.

    ``indices`` selects fit rows from the parent dataset without materializing a
    complete CV training subset just to compute its condition-free baseline.
    """
    total = np.zeros(training.fields.shape[1:], dtype=np.float64)
    count = np.zeros(training.fields.shape[1:-1], dtype=np.int64)
    batch_size = max(1, _BLOCK_VALUES // int(np.prod(training.fields.shape[1:])))
    for batch in training.iter_batches(indices, batch_size=batch_size):
        mask = batch.effective_mask
        for index in _field_blocks(batch.fields.shape):
            total[index[1:]] += np.sum(batch.fields[index], axis=0, dtype=np.float64, where=mask[index][..., None])
            count[index[1:]] += np.sum(mask[index], axis=0)
    np.divide(total, count[..., None], out=total, where=count[..., None] > 0)
    np.copyto(total, np.nan, where=count[..., None] == 0)
    return np.broadcast_to(total, reference.fields.shape)


def error_inner_product(reference, first, second, options=None, evaluation_mask=None) -> tuple[float, np.ndarray]:
    return prepare_reference(reference, options, evaluation_mask=evaluation_mask).inner_product(first, second)
