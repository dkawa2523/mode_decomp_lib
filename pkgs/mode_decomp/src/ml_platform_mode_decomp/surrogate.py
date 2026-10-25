"""Input features to real coefficient targets to a physical field."""

from copy import deepcopy
from collections.abc import Iterator
from typing import Any, Self, cast

import numpy as np

from .catalog import build_regressor
from .representation import FieldRepresentation
from .data.schema import ConditionBatch, FieldDataset
from .inputs import CoefficientBatch, align_inputs
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

    def fit(self, dataset: FieldDataset, *, inputs: FieldDataset | CoefficientBatch | None = None) -> Self:
        self._fitted = False
        self.calibrator = None
        self.calibration_sample_ids = ()
        self.input_representation = None
        self.input_model_sha256 = None
        self.condition_columns = ()
        inputs = align_inputs(inputs, dataset)
        input_config = self.candidate.get("input_representation")
        if isinstance(inputs, FieldDataset):
            if input_config is None:
                raise ValueError("Field inputs require input_representation")
            self.input_kind = "field"
            self.input_representation = FieldRepresentation(input_config)
            features = self.input_representation.fit_transform(inputs)
        elif isinstance(inputs, CoefficientBatch):
            if input_config is not None:
                raise ValueError("Coefficient inputs already have a fixed representation; omit input_representation")
            self.input_kind = "coefficients"
            self.input_representation = deepcopy(inputs.representation)
            self.input_model_sha256 = inputs.model_sha256
            features = inputs.coefficients
        else:
            if input_config is not None:
                raise ValueError("input_representation requires source field inputs")
            if dataset.conditions is None:
                raise ValueError("Surrogate training requires conditions or explicit inputs")
            self.input_kind = "conditions"
            self.condition_columns = tuple(dataset.condition_columns)
            features = dataset.conditions
        self.input_dimension = features.shape[1]
        self.representation = FieldRepresentation(self.candidate["representation"])
        targets = self.representation.fit_transform(dataset)
        self.regressor = None
        self.constant_targets = None
        self.prediction_kind = "constant_field" if targets.shape[1] == 0 else "regression"
        if targets.shape[1]:
            if self.input_dimension == 0:
                self.prediction_kind = "constant_coefficients"
                self.constant_targets = targets.mean(axis=0)
            else:
                self.regressor = build_regressor(self.candidate["regression"])
                self.regressor.fit(features, targets)
        self.fit_sample_ids = tuple(str(x) for x in dataset.sample_ids)
        self._fitted = True
        return self

    def _condition_values(self, conditions) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Surrogate must be fitted first")
        if isinstance(conditions, (ConditionBatch, FieldDataset)):
            if tuple(conditions.condition_columns) != self.condition_columns:
                raise ValueError("Input condition columns differ from the fitted column order")
            conditions = conditions.conditions
        values = np.asarray(conditions)
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != 2
            or values.shape[1] != len(self.condition_columns)
            or not len(values)
        ):
            raise ValueError("Conditions must be a finite matrix with the fitted column order")
        return values

    def _query_inputs(self, inputs):
        if not self._fitted:
            raise ValueError("Surrogate must be fitted first")
        kind = getattr(self, "input_kind", "conditions")
        if kind == "conditions":
            return self._condition_values(inputs)
        if kind == "field":
            if not isinstance(inputs, FieldDataset):
                raise ValueError("This surrogate requires source fields as a FieldDataset")
        else:
            if not isinstance(inputs, CoefficientBatch):
                raise ValueError("This surrogate requires source coefficients with their saved representation")
            if inputs.model_sha256 != self.input_model_sha256:
                raise ValueError("Input coefficients belong to a different fitted representation")
            if inputs.coefficients.shape[1] != self.input_dimension:
                raise ValueError("Input coefficient dimension differs from the fitted representation")
        if not len(inputs):
            raise ValueError("Prediction requires at least one input sample")
        return inputs

    def _iter_coefficient_batches(self, inputs, *, batch_size: int | None = None) -> Iterator[np.ndarray]:
        values = self._query_inputs(inputs)
        if batch_size is None:
            batch_size = self._query_batch_size()
        elif isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for start in range(0, len(values), batch_size):
            section = slice(start, start + batch_size)
            if isinstance(values, FieldDataset):
                assert self.input_representation is not None
                features = self.input_representation.transform(values._subset_validated(section))
            elif isinstance(values, CoefficientBatch):
                features = values.coefficients[section]
            else:
                features = values[section]
            yield self._predict_coefficient_batch(features)

    def predict_coefficients(self, conditions) -> np.ndarray:
        """Return output Z coordinates for the model's fitted input kind."""
        values = self._query_inputs(conditions)
        output = None
        start = 0
        for targets in self._iter_coefficient_batches(values):
            if len(targets) == len(values):
                return targets
            if output is None:
                output = np.empty((len(values), targets.shape[1]), dtype=targets.dtype)
            output[start : start + len(targets)] = targets
            start += len(targets)
        assert output is not None
        return output

    def _predict_coefficient_batch(self, values: np.ndarray) -> np.ndarray:
        if not np.isfinite(values).all():
            raise ValueError("Input features must be a finite matrix")
        constant = getattr(self, "constant_targets", None)
        if self.representation.target_dim == 0:
            targets = np.empty((len(values), 0), dtype=float)
        elif constant is not None:
            targets = np.broadcast_to(constant, (len(values), self.representation.target_dim)).copy()
        else:
            assert self.regressor is not None
            targets = np.asarray(self.regressor.predict(values))
        if (
            targets.ndim != 2
            or targets.shape != (len(values), self.representation.target_dim)
            or np.iscomplexobj(targets)
            or not np.isfinite(targets).all()
        ):
            raise ValueError("Regressor output must be finite and match the fitted coefficient space")
        return targets

    def predict(self, conditions) -> np.ndarray:
        return self._predict_arrays(conditions, intervals=False)[0]

    def _query_batch_size(self) -> int:
        # Share the query schedule across coefficients, fields and optional bands
        # so enabling bands cannot change float32 regressor accumulation order.
        field_values = int(np.prod(self.representation.field_shape))
        sample_bytes = 8 * (field_values * 3 + self.representation.target_dim + self.representation.packed_dim)
        input_representation = getattr(self, "input_representation", None)
        if input_representation is not None:
            sample_bytes += 8 * (int(np.prod(input_representation.field_shape)) * 3 + input_representation.target_dim)
        return min(_MAX_QUERY_BATCH, max(1, _QUERY_WORK_BYTES // max(1, sample_bytes)))

    def iter_predict(
        self, conditions, *, batch_size: int | None = None, intervals: bool = False
    ) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield fields in input order, optionally with their saved calibrated bounds.

        Only the current query batch's coefficients, fields and bands are held.
        The automatic batch size limits intermediate arrays; a single large field
        and the fitted regressor's state remain in memory.
        """
        if intervals and self.calibrator is None:
            raise ValueError("This model has no calibrated field intervals; enable uncertainty during training")
        for coefficients in self._iter_coefficient_batches(conditions, batch_size=batch_size):
            prediction = self.representation.inverse_transform(coefficients)
            if intervals:
                assert self.calibrator is not None
                lower, upper = self.calibrator.intervals(prediction)
                yield prediction, lower, upper
            else:
                yield prediction

    def _predict_arrays(self, conditions, *, intervals: bool) -> tuple[np.ndarray, ...]:
        values = self._query_inputs(conditions)
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
        inputs: FieldDataset | CoefficientBatch | None = None,
    ) -> Self:
        """Calibrate field bands on samples unused by fitting and model selection."""
        if not self._fitted:
            raise ValueError("Surrogate must be fitted before calibration")
        if set(self.fit_sample_ids).intersection(str(x) for x in calibration.sample_ids):
            raise ValueError("Calibration sample IDs must be disjoint from model fitting IDs")
        if getattr(self, "input_kind", "conditions") != "conditions" and inputs is None:
            raise ValueError("Field or coefficient models require explicit calibration inputs")
        inputs = align_inputs(inputs, calibration)
        query = calibration if inputs is None else inputs
        # Enforce the fitted geometry, components and units at this public API boundary.
        self.representation._inputs(calibration)
        self.calibrator = FieldCalibrator(coverage=coverage, scope=scope, normalization=normalization).fit_batches(
            calibration, cast(Iterator[np.ndarray], self.iter_predict(query)), training=training
        )
        self.calibration_sample_ids = tuple(str(value) for value in calibration.sample_ids)
        return self

    def predict_interval(self, conditions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the point prediction and calibrated lower/upper field bounds."""
        prediction, lower, upper = self._predict_arrays(conditions, intervals=True)
        return prediction, lower, upper
