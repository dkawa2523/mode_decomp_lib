"""Conditions to real coefficient targets to a physical field."""

from copy import deepcopy
from collections.abc import Iterator
from typing import Any, Self, cast

import numpy as np

from .catalog import build_regressor
from .representation import FieldRepresentation
from .data.schema import FieldDataset
from .policy import candidate_config
from .uncertainty import FieldCalibrator


_QUERY_WORK_BYTES = 8 * 1024 * 1024
_MAX_QUERY_BATCH = 256


class FieldSurrogate:
    def __init__(self, candidate: dict[str, Any], *, seed: int = 42):
        self.candidate = candidate_config({"id": "surrogate", **deepcopy(candidate)}, seed)
        self._fitted = False
        self.calibrator: FieldCalibrator | None = None
        self.calibration_sample_ids: tuple[str, ...] = ()

    def fit(self, dataset: FieldDataset) -> Self:
        self._fitted = False
        self.calibrator = None
        self.calibration_sample_ids = ()
        if dataset.conditions is None:
            raise ValueError("Surrogate training requires conditions")
        self.representation = FieldRepresentation(self.candidate["representation"])
        targets = self.representation.fit_transform(dataset)
        self.regressor = None
        self.prediction_kind = "constant_field" if targets.shape[1] == 0 else "regression"
        if targets.shape[1]:
            self.regressor = build_regressor(self.candidate["regression"])
            self.regressor.fit(dataset.conditions, targets)
        self.condition_columns = tuple(dataset.condition_columns)
        self.fit_sample_ids = tuple(str(x) for x in dataset.sample_ids)
        self._fitted = True
        return self

    def _condition_values(self, conditions: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Surrogate must be fitted first")
        values = np.asarray(conditions)
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != 2
            or values.shape[1] != len(self.condition_columns)
            or not len(values)
        ):
            raise ValueError("Conditions must be a finite matrix with the fitted column order")
        return values

    def predict_coefficients(self, conditions: np.ndarray) -> np.ndarray:
        values = self._condition_values(conditions)
        size = self._query_batch_size()
        output = None
        for start in range(0, len(values), size):
            targets = self._predict_coefficient_batch(values[start : start + size])
            if len(targets) == len(values):
                return targets
            if output is None:
                output = np.empty((len(values), targets.shape[1]), dtype=targets.dtype)
            output[start : start + len(targets)] = targets
        assert output is not None
        return output

    def _predict_coefficient_batch(self, values: np.ndarray) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError("Conditions must be a finite matrix with the fitted column order")
        targets = (
            np.empty((len(values), 0), dtype=float)
            if self.regressor is None
            else np.asarray(self.regressor.predict(values))
        )
        if targets.ndim != 2 or targets.shape != (len(values), self.representation.target_dim):
            raise ValueError("Regressor output does not match the fitted coefficient space")
        return targets

    def predict(self, conditions: np.ndarray) -> np.ndarray:
        return self._predict_arrays(conditions, intervals=False)[0]

    def _query_batch_size(self) -> int:
        # Share the query schedule across coefficients, fields and optional bands
        # so enabling bands cannot change float32 regressor accumulation order.
        field_values = int(np.prod(self.representation.field_shape))
        sample_bytes = 8 * (field_values * 3 + self.representation.target_dim + self.representation.packed_dim)
        return min(_MAX_QUERY_BATCH, max(1, _QUERY_WORK_BYTES // max(1, sample_bytes)))

    def iter_predict(
        self, conditions: np.ndarray, *, batch_size: int | None = None, intervals: bool = False
    ) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield fields in input order, optionally with their saved calibrated bounds.

        Only the current query batch's coefficients, fields and bands are held.
        The automatic batch size limits intermediate arrays; a single large field
        and the fitted regressor's state remain in memory.
        """
        values = self._condition_values(conditions)
        if intervals and self.calibrator is None:
            raise ValueError("This model has no calibrated field intervals; enable uncertainty during training")
        if batch_size is None:
            batch_size = self._query_batch_size()
        elif isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for start in range(0, len(values), batch_size):
            prediction = self.representation.inverse_transform(
                self.predict_coefficients(values[start : start + batch_size])
            )
            if intervals:
                assert self.calibrator is not None
                lower, upper = self.calibrator.intervals(prediction)
                yield prediction, lower, upper
            else:
                yield prediction

    def _predict_arrays(self, conditions: np.ndarray, *, intervals: bool) -> tuple[np.ndarray, ...]:
        values = self._condition_values(conditions)
        arrays = None
        start = 0
        for batch in self.iter_predict(values, intervals=intervals):
            parts = batch if isinstance(batch, tuple) else (batch,)
            if len(parts[0]) == len(values):
                return parts
            if arrays is None:
                arrays = tuple(np.empty((len(values), *part.shape[1:]), dtype=part.dtype) for part in parts)
            stop = start + len(parts[0])
            for output, part in zip(arrays, parts, strict=True):
                output[start:stop] = part
            start = stop
        assert arrays is not None
        return arrays

    def calibrate(
        self,
        calibration: FieldDataset,
        *,
        training: FieldDataset | None = None,
        coverage: float = 0.9,
        scope: str = "simultaneous",
        normalization: str = "channel",
    ) -> Self:
        """Calibrate field bands on samples unused by fitting and model selection."""
        if not self._fitted:
            raise ValueError("Surrogate must be fitted before calibration")
        if set(self.fit_sample_ids).intersection(str(x) for x in calibration.sample_ids):
            raise ValueError("Calibration sample IDs must be disjoint from model fitting IDs")
        if calibration.conditions is None:
            raise ValueError("Calibration requires conditions and reference fields")
        # Enforce the fitted geometry, components and units at this public API boundary.
        self.representation._inputs(calibration)
        self.calibrator = FieldCalibrator(coverage=coverage, scope=scope, normalization=normalization).fit_batches(
            calibration, cast(Iterator[np.ndarray], self.iter_predict(calibration.conditions)), training=training
        )
        self.calibration_sample_ids = tuple(str(value) for value in calibration.sample_ids)
        return self

    def predict_interval(self, conditions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the point prediction and calibrated lower/upper field bounds."""
        prediction, lower, upper = self._predict_arrays(conditions, intervals=True)
        return prediction, lower, upper
