"""Split-conformal residual bands in original field units, without posterior assumptions."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, ROUND_CEILING
from typing import Any, Mapping, Self

import numpy as np

from .data.schema import FieldDataset
from .geometry.identity import geometry_fingerprint


_BATCH_VALUES = 1 << 20


def _sample_slices(count: int, field_shape: tuple[int, ...]):
    # Bound temporary work by field size; a single field remains the minimum unit.
    step = max(1, _BATCH_VALUES // int(np.prod(field_shape)))
    for start in range(0, count, step):
        yield slice(start, min(start + step, count))


def _support(data: FieldDataset) -> tuple[np.ndarray, np.ndarray]:
    spatial = data.fields.shape[1:-1]
    weights = data.geometry.integration_weights()
    area = np.ones(spatial) if weights is None else np.asarray(weights, dtype=float).reshape(spatial)
    if not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError("Conformal support requires finite nonnegative geometry weights")
    support = area > 0
    if data.geometry.mask is not None:
        support &= np.asarray(data.geometry.mask, dtype=bool).reshape(spatial)
    if not support.any():
        raise ValueError("Conformal support needs positive physical measure")
    return support, area


def _weights(values, count: int, label: str) -> np.ndarray:
    from .evaluation.fields import normalized_weights

    return normalized_weights(values, count, label)


def _residual_scores(predictions, truth, scales):
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.subtract(predictions, truth, dtype=np.float64)
        np.abs(scores, out=scores)
        scores /= scales
    if not np.isfinite(scores).all():
        raise ValueError("Calibration residual scores must be finite")
    return scores


def _require_complete_truth(data, support, *, indices=None):
    for index in range(len(data)) if indices is None else indices:
        if not data._sample_mask(index)[support].all():
            raise ValueError("Conformal calibration/coverage requires complete truth on positive-measure support")


def _calibration_rank(count, coverage):
    # Decimal input intent avoids a spurious extra rank at e.g. 100 * 0.07.
    rank = int((Decimal(count + 1) * Decimal(str(coverage))).to_integral_value(rounding=ROUND_CEILING))
    if rank > count:
        raise ValueError(
            f"Insufficient calibration samples ({count}) for finite {coverage:g} coverage: "
            f"ceil((n+1)*coverage)={rank} exceeds n; reserve more calibration samples or lower coverage"
        )
    return rank


class FieldCalibrator:
    """Symmetric field bands calibrated on samples excluded from every model fit.

    Simultaneous scores maximize over the fixed positive-measure field support.
    Pointwise scores calibrate each fixed spatial/channel coordinate separately.
    Coverage is marginal under exchangeable calibration and future samples; this
    class does not establish exchangeability for groups, time series, or OOD data.
    """

    def __init__(self, *, coverage: float = 0.9, scope: str = "simultaneous", normalization: str = "none"):
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage < 1:
            raise ValueError("coverage must be a finite number strictly between zero and one")
        if scope not in {"simultaneous", "pointwise"}:
            raise ValueError("scope must be simultaneous or pointwise")
        if normalization not in {"none", "channel"}:
            raise ValueError("normalization must be none or channel")
        self.coverage, self.scope, self.normalization = float(coverage), scope, normalization
        self._fitted = False

    def _validate_data(self, data: FieldDataset, *, complete: bool) -> None:
        if (
            tuple(data.fields.shape[1:]) != self.field_shape
            or geometry_fingerprint(data.geometry) != self.geometry_sha256
            or tuple(data.components) != self.components
            or data.field_units != self.field_units
            or data.field_kind != self.field_kind
        ):
            raise ValueError(
                "Calibration and prediction reference must use the same field geometry/components/units and kind"
            )
        if complete:
            _require_complete_truth(data, self.support)

    def _predictions(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions)
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != len(self.field_shape) + 1
            or tuple(values.shape[1:]) != self.field_shape
            or not len(values)
        ):
            raise ValueError("Predictions must be a real nonempty batch matching the calibrated field shape")
        for selected in _sample_slices(len(values), self.field_shape):
            if not np.isfinite(values[selected][:, self.support, :]).all():
                raise ValueError("Predictions must be finite on calibrated support")
        return values

    def _channel_scales(self, training: FieldDataset) -> np.ndarray:
        count = 0
        mean, squares = np.zeros(self.field_shape[-1]), np.zeros(self.field_shape[-1])
        with np.errstate(over="ignore", invalid="ignore"):
            for selected in _sample_slices(len(training), self.field_shape):
                observed = self.support
                if training.observation_mask is not None:
                    mask = training.observation_mask
                    if mask.ndim == training.fields.ndim - 1:
                        mask = mask[selected]
                    observed = mask & observed
                observed = np.broadcast_to(observed, training.fields[selected].shape[:-1])
                values = np.asarray(training.fields[selected], dtype=np.float64)[observed]
                if not len(values):
                    continue
                batch_mean = values.mean(axis=0)
                batch_squares = np.sum((values - batch_mean) ** 2, axis=0)
                if not count:
                    count, mean, squares = len(values), batch_mean, batch_squares
                    continue
                total = count + len(values)
                difference = batch_mean - mean
                squares += batch_squares + difference**2 * (count * len(values) / total)
                mean += difference * (len(values) / total)
                count = total
            scales = np.sqrt(squares / count) if count else np.ones(self.field_shape[-1])
        if not count:
            raise ValueError("Channel normalization needs observed proper training values")
        if not np.isfinite(scales).all():
            raise ValueError("Proper training channel normalization is not finite")
        return scales

    def _prepare_fit(self, calibration: FieldDataset, training: FieldDataset | None) -> None:
        self._fitted = False
        self.field_shape = tuple(calibration.fields.shape[1:])
        self.field_kind = calibration.field_kind
        self.geometry_sha256 = geometry_fingerprint(calibration.geometry)
        self.components, self.field_units = tuple(calibration.components), tuple(calibration.field_units or ())
        self.support, self.area = _support(calibration)
        self._validate_data(calibration, complete=True)
        self.channel_scales = np.ones(self.field_shape[-1])
        self.constant_channels: list[int] = []
        self.training_count = 0 if training is None else len(training)
        if training is not None:
            self._validate_data(training, complete=False)
            if set(training.sample_ids) & set(calibration.sample_ids):
                raise ValueError("Proper training and calibration sample IDs must be disjoint")
        if self.normalization == "channel":
            if training is None:
                raise ValueError("Channel normalization requires independent proper training data")
            self.channel_scales = self._channel_scales(training)
            self.constant_channels = np.flatnonzero(self.channel_scales == 0).tolist()
            self.channel_scales[self.channel_scales == 0] = 1.0
        self.calibration_count = len(calibration)
        self.rank = _calibration_rank(self.calibration_count, self.coverage)

    def _calibration_batches(self, calibration, predictions):
        start = 0
        for batch in predictions:
            values = self._predictions(batch)
            if start + len(values) > len(calibration):
                raise ValueError("Calibration predictions must have one row per calibration sample")
            for selected in _sample_slices(len(values), self.field_shape):
                stop = start + len(values[selected])
                yield slice(start, stop), values[selected]
                start = stop
        if start != len(calibration):
            raise ValueError("Calibration predictions must have one row per calibration sample")

    def fit(self, calibration: FieldDataset, predictions: np.ndarray, *, training: FieldDataset | None = None) -> Self:
        if self.scope == "simultaneous":
            return self.fit_batches(calibration, (predictions,), training=training)
        self._prepare_fit(calibration, training)
        values = self._predictions(predictions)
        if len(values) != len(calibration):
            raise ValueError("Calibration predictions must have one row per calibration sample")
        # With an existing array, visit coordinate blocks without retaining a
        # second sample-by-support score array. Streams instead retain one tail.
        support_indices = np.flatnonzero(self.support)
        channels = self.field_shape[-1]
        step = max(1, _BATCH_VALUES // (len(calibration) * channels))
        self.threshold = np.empty((len(support_indices), channels))
        for start in range(0, len(support_indices), step):
            selected = slice(start, min(start + step, len(support_indices)))
            coordinates = np.unravel_index(support_indices[selected], self.support.shape)
            index = (slice(None), *coordinates, slice(None))
            scores = _residual_scores(values[index], calibration.fields[index], self.channel_scales)
            scores.partition(self.rank - 1, axis=0)
            self.threshold[selected] = scores[self.rank - 1]
        return self._complete_fit()

    def fit_batches(
        self,
        calibration: FieldDataset,
        predictions: Iterable[np.ndarray],
        *,
        training: FieldDataset | None = None,
    ) -> Self:
        """Fit exact bands from one prediction per calibration sample, in order.

        Simultaneous calibration retains one score per sample. Pointwise
        calibration retains min(r, n-r+1) scores per coordinate for rank r and
        n samples; storage still grows with sample count and fixed support.
        """
        self._prepare_fit(calibration, training)
        batches = self._calibration_batches(calibration, predictions)
        if self.scope == "simultaneous":
            scores = np.empty(len(calibration))
            for selected, values in batches:
                residual = _residual_scores(
                    values[:, self.support, :],
                    calibration.fields[selected][:, self.support, :],
                    self.channel_scales,
                )
                scores[selected] = residual.max(axis=(1, 2))
            scores.partition(self.rank - 1, axis=0)
            self.threshold = scores[self.rank - 1].copy()
        else:
            self.threshold = self._pointwise_threshold(calibration, batches)
        return self._complete_fit()

    def _complete_fit(self) -> Self:
        self.half_width = np.full(self.field_shape, np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            self.half_width[self.support, :] = self.threshold * self.channel_scales
        if not np.isfinite(self.half_width[self.support, :]).all():
            raise ValueError("Calibrated field widths must be finite")
        self._fitted = True
        return self

    def _pointwise_threshold(self, calibration, batches):
        # The r-th smallest is also the (n-r+1)-th largest. Negating the upper
        # tail lets both cases keep the smallest k scores using the same path.
        keep = min(self.rank, self.calibration_count - self.rank + 1)
        direction = 1 if keep == self.rank else -1
        buffer = None
        retained = 0
        for selected, values in batches:
            residual = _residual_scores(
                values[:, self.support, :], calibration.fields[selected][:, self.support, :], self.channel_scales
            )
            residual *= direction
            count = len(values)
            active = retained + count
            if buffer is None or buffer.shape[-1] < active:
                capacity = min(self.calibration_count, keep + count)
                previous = buffer
                buffer = np.empty((*residual.shape[1:], capacity))
                if previous is not None:
                    buffer[..., :retained] = previous[..., :retained]
                del previous
            buffer[..., retained:active] = np.moveaxis(residual, 0, -1)
            if active > keep:
                buffer[..., :active].partition(keep - 1, axis=-1)
            retained = min(keep, active)
        assert buffer is not None
        return direction * buffer[..., :retained].max(axis=-1)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("FieldCalibrator must be fitted on independent calibration data first")

    def intervals(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        values = self._predictions(predictions)
        lower, upper = np.empty(values.shape), np.empty(values.shape)
        for selected in _sample_slices(len(values), self.field_shape):
            lower[selected], upper[selected] = self._intervals(values[selected])
        return lower, upper

    def _intervals(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            lower, upper = values - self.half_width, values + self.half_width
        if not np.isfinite(lower[:, self.support, :]).all() or not np.isfinite(upper[:, self.support, :]).all():
            raise ValueError("Prediction interval endpoints overflowed on calibrated support")
        return lower, upper

    def summary(self) -> dict[str, Any]:
        self._require_fitted()
        threshold = np.asarray(self.threshold)
        return {
            "method": "split_conformal",
            "target": "grid_fixed_support",
            "scope": self.scope,
            "coverage": self.coverage,
            "normalization": self.normalization,
            "channel_scales": self.channel_scales.tolist(),
            "constant_training_channels_using_unit_scale": self.constant_channels,
            "proper_training_count_for_normalization": self.training_count,
            "calibration_count": self.calibration_count,
            "order_statistic_rank": self.rank,
            "threshold": float(threshold) if threshold.ndim == 0 else None,
            "threshold_min": float(threshold.min()),
            "threshold_max": float(threshold.max()),
            "support_points": int(self.support.sum()),
            "field_shape": list(self.field_shape),
            "field_kind": getattr(self, "field_kind", None),
            "field_components": list(self.components),
            "field_units": list(self.field_units),
            "geometry_sha256": self.geometry_sha256,
            "band_interpretation": "prediction_band_not_standard_deviation_or_gaussian_posterior",
            "guarantee_condition": "marginal_coverage_if_calibration_and_future_samples_are_exchangeable_and_model_is_fixed",
            "sample_exchangeability_verified": False,
            "support": "complete_fixed_positive_geometry_measure",
        }

    def evaluate(
        self, reference: FieldDataset, predictions: np.ndarray, options: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_fitted()
        self._validate_data(reference, complete=True)
        values = self._predictions(predictions)
        if len(values) != len(reference):
            raise ValueError("Coverage predictions must have one row per reference sample")
        options = options or {}
        column = options.get("sample_weight_column")
        if column is not None and column not in reference.sample_metadata:
            raise ValueError("sample_weight_column must name reference sample metadata")
        sample_weights = _weights(
            np.ones(len(reference)) if column is None else reference.sample_metadata[column],
            len(reference),
            "Sample weights",
        )
        channels = self.field_shape[-1]
        channel_weights = _weights(options.get("channel_weights", np.ones(channels)), channels, "Channel weights")
        area = _weights(self.area[self.support], int(self.support.sum()), "Geometry weights")
        point_coverage = np.empty((len(reference), channels))
        sample_inside = np.empty(len(reference), dtype=bool)
        for selected in _sample_slices(len(reference), self.field_shape):
            lower, upper = self._intervals(values[selected])
            truth = np.asarray(reference.fields[selected], dtype=np.float64)[:, self.support, :]
            inside = (lower[:, self.support, :] <= truth) & (truth <= upper[:, self.support, :])
            point_coverage[selected] = np.sum(inside * area[None, :, None], axis=1)
            sample_inside[selected] = inside.all(axis=(1, 2))
        channel_coverage = sample_weights @ point_coverage
        channel_width = area @ (2 * self.half_width[self.support, :])
        mixed_units = len(set(self.field_units)) > 1
        return {
            "target": "grid_fixed_support",
            "scope": self.scope,
            "coverage_target": self.coverage,
            "sample_count": len(reference),
            "sample_simultaneous_coverage": float(sample_inside.mean()),
            "weighted_sample_simultaneous_coverage": float(sample_weights @ sample_inside),
            "pointwise_coverage": float(channel_coverage @ channel_weights),
            "channel_pointwise_coverage": channel_coverage.tolist(),
            "sample_simultaneous_inside": sample_inside.tolist(),
            "sample_channel_pointwise_coverage": point_coverage.tolist(),
            "sample_pointwise_coverage": (point_coverage @ channel_weights).tolist(),
            "channel_mean_width": channel_width.tolist(),
            "mean_width": None if mixed_units else float(channel_width @ channel_weights),
            "mean_scaled_width": float((channel_width / self.channel_scales) @ channel_weights),
            "field_units": list(self.field_units),
            "coverage_weighting": "simultaneous:equal_samples; pointwise:sample_weights_normalized_area_channel_weights",
        }
