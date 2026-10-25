"""Three explicit scientific objectives, sharing one sample/channel measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from ..data.schema import FieldDataset
from .fields import (
    PreparedReference,
    ScoreResult,
    ScoreWeights,
    _channel_variance,
    _sample_values,
    _weighted_mean,
    mean_field_prediction,
    normalized_weights,
    prepare_reference,
    score_weights,
)
from .observations import _PreparedScatteredEvaluation


@dataclass
class PreparedObjective:
    """Fixed reference/support for one fit, validation or final-test role."""

    reference: FieldDataset
    options: dict
    measure: ScoreWeights
    baseline: np.ndarray | None = None
    reference_kind: str = "grid"
    grid: PreparedReference | None = None
    observations: _PreparedScatteredEvaluation | None = None

    @property
    def metric_prefix(self):
        return {"grid": "field", "observations": "observation", "hidden": "hidden"}[self.reference_kind]

    def _points(self, field):
        assert self.observations is not None
        values = np.asarray(field)
        if values.shape != self.reference.fields.shape or values.dtype.kind not in "fiu":
            raise ValueError("Predictions must match the fixed observation reference field shape")
        for index, (grid, sample) in enumerate(zip(values, self.observations.samples, strict=True)):
            support = sample.support & sample.observed
            weights = np.ones(len(support)) if sample.point_weights is None else sample.point_weights
            support &= weights > 0
            if self.measure.sample_weights[index] == 0:
                yield sample, np.zeros_like(support), weights, np.empty((0, values.shape[-1]))
                continue
            stencils = grid.reshape(-1, grid.shape[-1])[sample.vertices]
            required = sample.weights > 1e-13
            if not np.isfinite(stencils).all(where=(support[:, None] & required)[..., None]):
                raise ValueError("Predictions must be finite on the fixed scattered scoring support")
            predicted = np.einsum("qv,qvc->qc", sample.weights, np.where(required[..., None], stencils, 0.0))
            yield sample, support, weights, predicted

    def _reduce(self, predictions, *, truth=None, absolute=False):
        if self.reference_kind != "observations":
            raise ValueError("Point reduction requires the observations objective")
        actual = None if truth is None or truth is self.reference.fields else self._points(truth)
        result = np.zeros((len(self.reference), self.reference.fields.shape[-1]))
        mae = np.zeros_like(result)
        for index, (sample, support, weights, predicted) in enumerate(self._points(predictions)):
            target = sample.truth if actual is None else next(actual)[3]
            if not support.any():
                continue  # Only a zero-weight sample can have empty primary support.
            error = predicted[support] - target[support]
            point_weights = normalized_weights(weights[support], int(support.sum()), "Point weights")
            result[index] = np.average(error**2, weights=point_weights, axis=0)
            if absolute:
                mae[index] = np.average(np.abs(error), weights=point_weights, axis=0)
        return result, mae

    def loss(self, predictions, *, truth=None):
        if self.reference_kind != "observations":
            assert self.grid is not None
            return self.grid.loss(predictions, truth=truth)
        squared, _ = self._reduce(predictions, truth=truth)
        samples = (squared / self.measure.scales**2) @ self.measure.channel_weights
        return _weighted_mean(self.measure.sample_weights, samples), _sample_values(self.measure, samples)

    def inner_product(self, first, second, *, first_reference=None, second_reference=None):
        if self.reference_kind != "observations":
            assert self.grid is not None
            return self.grid.inner_product(
                first, second, first_reference=first_reference, second_reference=second_reference
            )
        references = []
        for field in (first_reference, second_reference):
            references.append(None if field is None or field is self.reference.fields else self._points(field))
        products = np.zeros((len(self.reference), self.reference.fields.shape[-1]))
        for index, (left, right) in enumerate(zip(self._points(first), self._points(second), strict=True)):
            sample, support, weights, lvalues = left
            rvalues = right[3]
            actual = [None if iterator is None else next(iterator)[3] for iterator in references]
            if not support.any():
                continue
            if first_reference is not None:
                lvalues = lvalues - (sample.truth if references[0] is None else actual[0])
            if second_reference is not None:
                rvalues = rvalues - (sample.truth if references[1] is None else actual[1])
            point_weights = normalized_weights(weights[support], int(support.sum()), "Point weights")
            products[index] = np.average(lvalues[support] * rvalues[support], axis=0, weights=point_weights)
        samples = (products / self.measure.scales**2) @ self.measure.channel_weights
        return _weighted_mean(self.measure.sample_weights, samples), _sample_values(self.measure, samples)

    def score(self, predictions, *, baseline=None):
        if self.reference_kind != "observations":
            assert self.grid is not None
            report = self.grid.score(predictions, baseline=baseline)
            if self.reference_kind == "hidden":
                renamed = {key.replace("field_", "hidden_", 1): value for key, value in report.metrics.items()}
                legacy = self.options.get("_legacy_hidden_aliases", False)
                report.metrics = {**report.metrics, **renamed} if legacy else renamed
                report.summary["legacy_hidden_field_aliases"] = bool(legacy)
                if "metric_status" in report.summary:
                    report.summary["metric_status"] = {
                        key.replace("field_", "hidden_", 1): value
                        for key, value in report.summary["metric_status"].items()
                    }
            report.summary["evaluation_reference"] = self.reference_kind
            return report
        squared, absolute = self._reduce(predictions, absolute=True)
        assert self.observations is not None
        measure = self.measure
        sw, scales, cw = measure.sample_weights, measure.scales, measure.channel_weights
        mse, mae = (squared / scales**2) @ cw, (absolute / scales) @ cw
        metrics = {"observation_rmse": float(np.sqrt(sw @ mse)), "observation_mae": float(sw @ mae)}
        counts, coverage = [], []
        for sample in self.observations.samples:
            weights = np.ones(len(sample.observed)) if sample.point_weights is None else sample.point_weights
            known = sample.observed & (weights > 0)
            count = np.count_nonzero(known & sample.support)
            counts.append(count)
            coverage.append(count / np.count_nonzero(known) if known.any() else 0.0)
        rows = {
            "sample_id": self.reference.sample_ids,
            "mse": _sample_values(measure, mse),
            "rmse": _sample_values(measure, np.sqrt(mse)),
            "mae": _sample_values(measure, mae),
            "sample_weight": sw,
            "valid_points": counts,
            "coverage": coverage,
        }
        for channel in range(squared.shape[1]):
            metrics[f"observation_rmse_ch{channel}"] = float(np.sqrt(sw @ squared[:, channel]))
            metrics[f"observation_mae_ch{channel}"] = float(sw @ absolute[:, channel])
            rows[f"observation_rmse_ch{channel}"] = _sample_values(measure, np.sqrt(squared[:, channel]))
            rows[f"observation_mae_ch{channel}"] = _sample_values(measure, absolute[:, channel])
        summary: dict[str, Any] = {
            **measure.summary,
            "evaluation_reference": "observations",
            "reference_support": "fixed_original_observation_support",
            "spatial_measure": "per_sample_normalized_point_weights",
            "coverage_mean": float(sw @ coverage),
            "baseline_status": "not_provided",
        }
        candidate = self.baseline if baseline is None else baseline
        if candidate is not None:
            try:
                baseline_squared, _ = self._reduce(candidate)
                baseline_samples = (baseline_squared / scales**2) @ cw
                baseline_mse = float(sw @ baseline_samples)
                rows["baseline_mse"] = _sample_values(measure, baseline_samples)
                summary["baseline_status"] = "defined" if baseline_mse > 0 else "zero_error_skill_undefined"
                if baseline_mse > 0:
                    metrics["observation_skill"] = float(1 - float(sw @ mse) / baseline_mse)
            except ValueError:
                summary["baseline_status"] = "training_mean_undefined_at_evaluated_locations"
        if "observation_skill" not in metrics:
            summary["metric_status"] = {"observation_skill": summary["baseline_status"]}
            if self.options.get("primary_metric") == "observation_skill":
                raise ValueError("Primary observation_skill is undefined for this training baseline")
        if not np.any(sw > 0):
            metrics.clear()
            summary["evaluation_status"] = "no_positive_weight_samples"
        if any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError("Observation loss is not finite; rescale field values or channel scales")
        return ScoreResult(metrics, pd.DataFrame(rows), summary, np.asarray(predictions))

    def pool(self, reports):
        """Reduce out-of-fold sample statistics without retaining field predictions."""
        table = pd.concat([report.per_sample for report in reports], ignore_index=True)
        if table.sample_id.astype(str).tolist() != self.reference.sample_ids.astype(str).tolist():
            raise ValueError("Pooled scores must follow the prepared validation sample order")
        sw, cw = self.measure.sample_weights, self.measure.channel_weights
        table["sample_weight"] = sw
        prefix = self.metric_prefix
        metrics = {
            f"{prefix}_rmse": float(np.sqrt(_weighted_mean(sw, table.mse.to_numpy()))),
            f"{prefix}_mae": _weighted_mean(sw, table.mae.to_numpy()),
        }
        channel_mse = []
        for channel in range(self.reference.fields.shape[-1]):
            rmse_name = f"observation_rmse_ch{channel}" if prefix == "observation" else f"channel_{channel}_rmse"
            mae_name = f"observation_mae_ch{channel}" if prefix == "observation" else f"channel_{channel}_mae"
            squared = table[rmse_name].to_numpy() ** 2
            channel_mse.append(_weighted_mean(sw, squared))
            metrics[rmse_name] = float(np.sqrt(_weighted_mean(sw, squared)))
            metrics[mae_name] = _weighted_mean(sw, table[mae_name].to_numpy())
        summary: dict[str, Any] = {
            **self.measure.summary,
            "evaluation_reference": self.reference_kind,
            "aggregation": "pooled_out_of_fold_sample_losses_on_validation_samples",
            "coverage_mean": _weighted_mean(sw, table.coverage.to_numpy()),
            "spatial_measure": "per_sample_normalized_point_weights"
            if prefix == "observation"
            else "per_sample_normalized_geometry_weights",
            "metric_status": {},
        }
        if f"{prefix}_r2" in self.options.get("names", []) or (
            prefix == "hidden" and "field_r2" in self.options.get("names", [])
        ):
            assert self.grid is not None
            variance = _channel_variance(self.grid.measure)
            active = cw > 0
            if np.any(variance[active] == 0):
                if self.options.get("primary_metric") in {f"{prefix}_r2", "field_r2"}:
                    raise ValueError("Primary R2 metric is undefined for a constant channel")
                summary["metric_status"][f"{prefix}_r2"] = "undefined_for_constant_channel"
            else:
                metrics[f"{prefix}_r2"] = float(
                    1 - np.average(np.asarray(channel_mse)[active] / variance[active], weights=cw[active])
                )
        summary["baseline_status"] = "not_provided"
        if "baseline_mse" in table:
            if np.isfinite(table.baseline_mse.to_numpy()[sw > 0]).all():
                baseline_mse = _weighted_mean(sw, table.baseline_mse.to_numpy())
                summary["baseline_status"] = "defined" if baseline_mse > 0 else "zero_error_skill_undefined"
                if baseline_mse > 0:
                    metrics[f"{prefix}_skill"] = float(1 - _weighted_mean(sw, table.mse.to_numpy()) / baseline_mse)
            else:
                summary["baseline_status"] = "training_mean_undefined_at_evaluated_locations"
        if f"{prefix}_skill" not in metrics:
            summary["metric_status"][f"{prefix}_skill"] = summary["baseline_status"]
        if self.options.get("primary_metric") == f"{prefix}_skill" and f"{prefix}_skill" not in metrics:
            raise ValueError("Primary skill metric is undefined for the pooled training baselines")
        if prefix == "hidden" and self.options.get("_legacy_hidden_aliases", False):
            metrics.update(
                {
                    key.replace("hidden_", "field_", 1): value
                    for key, value in list(metrics.items())
                    if key.startswith("hidden_")
                }
            )
        if any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError("Pooled evaluation loss is not finite; rescale field values or channel scales")
        return ScoreResult(metrics, table, summary, None)


def prepare_objective(
    reference, options=None, *, training=None, data=None, evaluation_mask=None, baseline=None, allow_zero_weight=False
):
    options = dict(options or {})
    kind = options.get("reference", "grid")
    if kind not in {"grid", "observations", "hidden"}:
        raise ValueError("metrics.reference must be grid, observations, or hidden")
    if baseline is None and training is not None:
        baseline = mean_field_prediction(training, reference)
    grid = None
    observations = None
    if kind == "observations":
        measure = score_weights(reference, options, allow_zero=allow_zero_weight)
        if not data or not reference.manifest.get("scattered"):
            raise ValueError("metrics.reference: observations requires preserved original scattered observations")
        from ..data.input_spec import field_input_spec
        from .observations import prepare_scattered_evaluation

        _, specification = field_input_spec(data)
        observations = prepare_scattered_evaluation(data["local_path"], reference, source_manifest=specification)
        for weight, sample in zip(measure.sample_weights, observations.samples, strict=True):
            support = sample.support & sample.observed
            if sample.point_weights is not None:
                support &= sample.point_weights > 0
            if weight > 0 and not support.any():
                raise ValueError("Every positive-weight reference sample requires scored original observations")
    else:
        grid_options = dict(options)
        grid_options["names"] = [name.replace("hidden_", "field_", 1) for name in options.get("names", [])]
        grid = prepare_reference(
            reference,
            grid_options,
            evaluation_mask=evaluation_mask,
            baseline=baseline,
            allow_zero_weight=allow_zero_weight,
        )
        measure = grid.measure
    return PreparedObjective(reference, options, measure, baseline, kind, grid, observations)
