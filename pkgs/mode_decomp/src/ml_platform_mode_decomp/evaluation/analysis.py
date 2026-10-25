"""Interpret field errors without comparing incompatible coefficient spaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..catalog import decomposition_role
from .fields import EvaluationReport, PreparedReference, mean_field_prediction, prepare_reference
from .objectives import PreparedObjective, prepare_objective

if TYPE_CHECKING:
    from ..data.schema import FieldDataset
    from ..representation import FieldRepresentation
    from ..surrogate import FieldSurrogate


def representation_size(model: FieldRepresentation) -> dict:
    ambient_dimension = int(np.prod(model.field_shape))
    input_dimension = ambient_dimension
    if model.geometry.mask is not None:
        input_dimension = int(np.count_nonzero(model.geometry.mask)) * model.field_shape[-1]
    codec = model.codec.coeff_meta(model.raw_meta, np.broadcast_to(0.0, (model.packed_dim,))).get("codec", {})
    return {
        "field_dimension": input_dimension,
        "ambient_field_dimension": ambient_dimension,
        "packed_dimension": model.packed_dim,
        "target_dimension": model.target_dim,
        "coefficient_to_field_ratio": model.target_dim / input_dimension,
        "decomposition": model.config["decomposition"]["name"],
        "method_role": decomposition_role(model.config["decomposition"]["name"]),
        "codec": codec,
        "independent_real_coordinates": codec.get("independent_real_coordinates"),
        "basis_inner_product": model.raw_meta.get(
            "inner_product", model.raw_meta.get("projection_inner_product", "method_specific")
        ),
        "decomposition_diagnostics": model.raw_meta.get("diagnostics", {}),
        "coefficient_comparison": "A/Z belong to this fitted representation; compare methods in field space",
        "decomposition_error_scope": "field preprocessing + decomposition + codec, before coefficient postprocessing",
        "baseline_training_aggregation": "unweighted per-location mean of observed training values",
    }


def analyze_representation(
    model: FieldRepresentation,
    training: FieldDataset,
    reference: FieldDataset,
    options=None,
    *,
    prepared=None,
    retain_coefficients=False,
    retain_fields=True,
) -> EvaluationReport:
    """Analyze reconstruction errors; optional fields preserve the numerical API's detailed output."""
    started = perf_counter()
    packed = model.transform(reference, space="a")
    targets = model.post.transform(packed)
    encode_seconds = perf_counter() - started
    # The extra pre-postprocessing inverse is diagnostic work, not prediction.
    decomposed = None if model.config["coeff_post"]["name"] == "none" else model.inverse_transform(packed, space="a")
    if not retain_coefficients:
        packed = None
    started = perf_counter()
    reconstructed = model.inverse_transform(targets)
    reconstruct_seconds = encode_seconds + perf_counter() - started
    del targets
    if decomposed is None:
        decomposed = reconstructed
    prepared = prepared or prepare_objective(reference, options, training=training)
    report = prepared.score(reconstructed)
    decomposition_mse, decomposition_samples = prepared.loss(decomposed)
    post_mse, post_samples = prepared.loss(reconstructed, truth=decomposed)
    report.metrics.update(
        decomposition_rmse=float(np.sqrt(decomposition_mse)),
        representation_rmse=report.metrics[f"{getattr(prepared, 'metric_prefix', 'field')}_rmse"],
        coefficient_processing_rmse=float(np.sqrt(post_mse)),
    )
    report.per_sample["decomposition_mse"] = decomposition_samples
    report.per_sample["representation_mse"] = report.per_sample["mse"]
    report.per_sample["coefficient_processing_mse"] = post_samples
    report.summary.update(representation_size(model))
    report.summary["analysis_kind"] = "representation"
    report.summary["prediction_seconds"] = reconstruct_seconds
    if retain_fields:
        report.fields.update(decomposition=decomposed, representation=reconstructed)
    if retain_coefficients:
        report.packed_coefficients = packed
    return report


def analyze_surrogate(
    model: FieldSurrogate,
    training: FieldDataset,
    reference: FieldDataset,
    options=None,
    *,
    prepared=None,
    retain_coefficients=False,
    retain_fields=True,
    inputs=None,
) -> EvaluationReport:
    query = _surrogate_query(inputs, reference)
    prepared = prepared or prepare_objective(reference, options, training=training)
    represented = analyze_representation(
        model.representation, training, reference, options, prepared=prepared, retain_fields=retain_fields
    )
    started = perf_counter()
    targets = model.predict_coefficients(query)
    predicted = model.representation.inverse_transform(targets)
    prediction_seconds = perf_counter() - started
    report = prepared.score(predicted)
    regression_mse, regression_samples = prepared.loss(predicted, truth=represented.predictions)
    cross, cross_per_sample = prepared.inner_product(
        represented.predictions,
        predicted,
        first_reference=reference.fields,
        second_reference=represented.predictions,
    )
    report.metrics.update(
        decomposition_rmse=represented.metrics["decomposition_rmse"],
        representation_rmse=represented.metrics["representation_rmse"],
        coefficient_processing_rmse=represented.metrics["coefficient_processing_rmse"],
        regression_rmse=float(np.sqrt(regression_mse)),
        error_cross_term_mse=2 * cross,
    )
    for column in ("decomposition_mse", "representation_mse", "coefficient_processing_mse"):
        report.per_sample[column] = represented.per_sample[column]
    report.per_sample["regression_mse"] = regression_samples
    report.per_sample["error_cross_term_mse"] = 2 * cross_per_sample
    report.summary.update(representation_size(model.representation))
    report.summary.update(
        analysis_kind="surrogate",
        error_identity="total_mse = representation_mse + regression_mse + error_cross_term_mse",
        regression_method=model.candidate["regression"]["name"],
        prediction_seconds=prediction_seconds,
    )
    _input_evaluation_scope(report, inputs, model)
    report.fields.update(represented.fields)
    if retain_coefficients:
        report.packed_coefficients = model.representation.post.inverse_transform(targets)
    return report


def observation_mask(
    reference: FieldDataset, fraction: float | None = None, seed: int | None = None, *, pattern="shared", explicit=None
) -> np.ndarray:
    if explicit is not None:
        values = np.asarray(explicit)
        if (
            values.shape not in (reference.fields.shape[1:-1], reference.fields.shape[:-1])
            or values.dtype.kind not in "bui"
            or not np.isin(values, (0, 1)).all()
        ):
            raise ValueError("analysis.observation_mask must contain boolean/0-1 spatial or sample-by-spatial values")
        observed = np.broadcast_to(values.astype(bool), reference.effective_mask.shape)
        if np.any(observed & ~reference.effective_mask):
            raise ValueError("Explicit observations must be inside known reference support")
        area = reference.geometry.integration_weights()
        known = reference.effective_mask
        if area is not None:
            positive = np.asarray(area).reshape(reference.fields.shape[1:-1]) > 0
            if np.any(observed & ~positive):
                raise ValueError("Explicit observations require positive geometry measure")
            known = known & positive
        axes = tuple(range(1, observed.ndim))
        if not observed.any(axis=axes).all() or not (known & ~observed).any(axis=axes).all():
            raise ValueError("Every imputation reference sample needs observed and hidden known locations")
        return observed
    if fraction is None or seed is None or not 0 < fraction < 1 or pattern not in {"shared", "per_sample"}:
        raise ValueError("Random imputation needs observed_fraction in (0,1), a seed, and shared/per_sample pattern")
    known = np.array(reference.effective_mask, copy=True)
    area = reference.geometry.integration_weights()
    if area is not None:
        known &= np.asarray(area).reshape(reference.fields.shape[1:-1]) > 0
    rng = np.random.default_rng(seed)
    observed = np.zeros_like(known)
    masks = [known.all(axis=0)] if pattern == "shared" else list(known)
    for index, mask in enumerate(masks):
        valid = np.flatnonzero(mask.reshape(-1))
        if len(valid) < 2:
            raise ValueError("Artificial hiding requires at least two known locations per observation pattern")
        count = min(max(1, round(len(valid) * fraction)), len(valid) - 1)
        selected = rng.choice(valid, size=count, replace=False)
        if pattern == "shared":
            observed.reshape(len(reference), -1)[:, selected] = True
        else:
            observed[index].reshape(-1)[selected] = True
    return observed


def load_observation_mask(data, dataset, path):
    """Align an optional raw-array-row mask once, before any split/subset."""
    from pathlib import Path
    from ..data.input_spec import field_input_spec, read_input_spec
    from ..data.loading import _indices, _path, _sample_table

    root = Path(data["local_path"]).resolve()
    paths = []
    values = np.load(_path(root, path, "analysis.observation_mask", paths), allow_pickle=False)
    if values.dtype.kind not in "bui" or not np.isin(values, (0, 1)).all():
        raise ValueError("analysis.observation_mask must contain boolean or 0/1 values")
    spatial = dataset.fields.shape[1:-1]
    if values.shape == spatial:
        return np.broadcast_to(values.astype(bool), dataset.fields.shape[:-1])
    if values.shape != dataset.fields.shape[:-1]:
        raise ValueError("analysis.observation_mask must be spatial or raw-array-row by spatial")
    _, specification = field_input_spec(data)
    manifest = read_input_spec(root, specification, paths)
    samples, _, index_column = _sample_table(root, manifest, paths, count=len(dataset))
    return values[_indices(samples, index_column, len(dataset))].astype(bool, copy=False)


def _input_evaluation_scope(report, inputs, model):
    report.summary.update(
        prediction_input_kind=getattr(model, "input_kind", "conditions"),
        prediction_input_dimension=getattr(model, "input_dimension", len(model.condition_columns)),
    )
    if inputs is not None:
        report.summary["input_representation_fit_scope"] = (
            "fixed_input_representation" if hasattr(inputs, "coefficients") else "training_partition"
        )


def _surrogate_query(inputs, reference):
    if inputs is not None:
        from ..inputs import align_inputs

        return align_inputs(inputs, reference)
    if reference.conditions is None:
        raise ValueError("Surrogate evaluation requires reference conditions or paired inputs")
    return reference.conditions


def score_surrogate(model, training, reference, options=None, *, prepared=None, inputs=None):
    """Predict once and score; detailed representation inverses are opt-in."""
    query = _surrogate_query(inputs, reference)
    prepared = prepared or prepare_objective(reference, options, training=training)
    started = perf_counter()
    predictions = model.predict(query)
    prediction_seconds = perf_counter() - started
    report = prepared.score(predictions)
    report.summary.update(analysis_kind="surrogate_score", prediction_seconds=prediction_seconds)
    _input_evaluation_scope(report, inputs, model)
    return report


def score_representation(model, training, reference, options=None, *, prepared=None):
    prepared = prepared or prepare_objective(reference, options, training=training)
    started = perf_counter()
    predictions = model.reconstruct(reference)
    prediction_seconds = perf_counter() - started
    report = prepared.score(predictions)
    report.summary.update(analysis_kind="representation_score", prediction_seconds=prediction_seconds)
    return report


def mode_reconstruction_curve(representation, coefficients, settings, prepared, *, full_predictions=None):
    """Evaluate selected A-coordinate reconstructions on the fixed main objective."""
    indices = settings["indices"]
    origin = np.zeros(representation.packed_dim)
    metric = f"{getattr(prepared, 'metric_prefix', 'field')}_rmse"
    rows = []
    for count in settings["counts"]:
        prediction = representation.reconstruct_modes(coefficients, indices[:count], space="a", reference=origin)
        report = prepared.score(prediction)
        rows.append(
            {
                "selected_count": count,
                "full_dimension": representation.packed_dim,
                "reconstruction": "selected_a_coordinates",
                metric: report.metrics[metric],
            }
        )
    prediction = (
        representation.inverse_transform(coefficients, space="a") if full_predictions is None else full_predictions
    )
    report = prepared.score(prediction)
    rows.append(
        {
            "selected_count": representation.packed_dim,
            "full_dimension": representation.packed_dim,
            "reconstruction": "all_a_coordinates",
            metric: report.metrics[metric],
        }
    )
    table = pd.DataFrame(rows)
    return table


@dataclass
class ImputationReference:
    input_data: FieldDataset
    observed: np.ndarray
    hidden: PreparedObjective
    full: PreparedReference
    seen: PreparedReference


def prepare_imputation_reference(training, reference, observed, options=None):
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != reference.effective_mask.shape or np.any(observed & ~reference.effective_mask):
        raise ValueError("Input observations must be a subset of the known reference support")
    # Remove hidden truth once, before any candidate can inspect its input storage.
    input_data = replace(
        reference, fields=np.where(observed[..., None], reference.fields, np.nan), observation_mask=observed
    )
    baseline = mean_field_prediction(training, reference)
    options = dict(options or {})
    if options.get("reference", "hidden") != "hidden":
        raise ValueError("Imputation evaluation requires metrics.reference: hidden")
    loss_options = {**(options or {}), "names": []}
    return ImputationReference(
        input_data,
        observed,
        prepare_objective(
            reference,
            {**(options or {}), "reference": "hidden"},
            evaluation_mask=reference.effective_mask & ~observed,
            baseline=baseline,
        ),
        prepare_reference(reference, loss_options, baseline=baseline),
        prepare_reference(reference, loss_options, evaluation_mask=observed, baseline=baseline),
    )


def analyze_imputation(
    model: FieldRepresentation,
    training: FieldDataset,
    reference: FieldDataset,
    observed,
    options=None,
    *,
    prepared=None,
    retain_coefficients=False,
):
    prepared = prepared or prepare_imputation_reference(training, reference, observed, options)
    started = perf_counter()
    packed = model.transform(prepared.input_data, space="a")
    targets = model.post.transform(packed)
    predictions = model.inverse_transform(targets)
    prediction_seconds = perf_counter() - started
    report = prepared.hidden.score(predictions)
    full = prepared.full.score(predictions)
    seen = prepared.seen.score(predictions)
    report.metrics.update(full_field_rmse=full.metrics["field_rmse"], observed_field_rmse=seen.metrics["field_rmse"])
    report.per_sample["full_mse"] = full.per_sample["mse"]
    report.per_sample["observed_mse"] = seen.per_sample["mse"]
    report.summary.update(representation_size(model))
    report.summary.update(
        analysis_kind="imputation",
        primary_support="artificially_hidden_known_reference",
        prediction_seconds=prediction_seconds,
    )
    report.fields.update(observed_mask=prepared.observed, reference=reference.fields)
    if retain_coefficients:
        report.packed_coefficients = packed
    return report
