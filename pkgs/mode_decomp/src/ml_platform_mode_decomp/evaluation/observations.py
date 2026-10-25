"""Original-observation evaluation on fixed reverse stencils and support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np

from ..data.scattered import ScatteredGridMap, _mask, point_scores
from ..data.scattered_import import load_scattered_observations
from ..data.schema import FieldDataset
from ..data.loading import load_field_dataset
from .fields import normalized_weights


def _same_array(first, second):
    return first is second or (first is not None and second is not None and np.array_equal(first, second))


def _aggregate_points(reports: list[dict[str, Any]], key: str, sample_weights) -> dict[str, Any]:
    samples = [report[key] for report in reports]
    usable = np.asarray([sample["rmse"] is not None for sample in samples])
    scored = [sample for sample, valid in zip(samples, usable, strict=True) if valid]
    weights = np.asarray(sample_weights, dtype=float)
    active = weights[usable]
    scored_weight = active.sum()
    return {
        "rmse": np.sqrt(
            np.average(np.asarray([sample["rmse"] for sample in scored]) ** 2, axis=0, weights=active)
        ).tolist()
        if scored_weight > 0
        else None,
        "mae": np.average([sample["mae"] for sample in scored], axis=0, weights=active).tolist()
        if scored_weight > 0
        else None,
        "coverage": float(np.average([sample["coverage"] for sample in samples], weights=weights)),
        "known_points": sum(sample["known_points"] for sample in samples),
        "scored_points": sum(sample["scored_points"] for sample in samples),
        "scored_samples": len(scored),
        "total_samples": len(samples),
        "sample_aggregation": "normalized_sample_weights_of_point_mse",
        "point_weighting": samples[0]["weighting"],
    }


@dataclass(frozen=True)
class _PointReference:
    sample_id: str
    truth: np.ndarray
    observed: np.ndarray
    point_weights: np.ndarray | None
    vertices: np.ndarray
    weights: np.ndarray
    support: np.ndarray
    interpolation: dict[str, Any]


@dataclass(frozen=True)
class _PreparedScatteredEvaluation:
    field_shape: tuple[int, ...]
    samples: tuple[_PointReference, ...]

    def evaluate(self, predictions: Any, *, sample_weights=None) -> dict[str, Any]:
        values = np.asarray(predictions)
        if values.shape != (len(self.samples), *self.field_shape) or values.dtype.kind not in "fiu":
            raise ValueError("predicted_grid must be a real array matching the selected field shape.")
        sample_weights = normalized_weights(
            np.ones(len(self.samples)) if sample_weights is None else sample_weights,
            len(self.samples),
            "Sample weights",
        )
        reports = []
        for weight, grid, sample in zip(sample_weights, values, self.samples, strict=True):
            if weight == 0:
                reports.append(
                    {
                        "sample_id": sample.sample_id,
                        "observation": {**sample.interpolation, "rmse": None, "mae": None},
                        "interpolation": sample.interpolation,
                    }
                )
                continue
            stencils = grid.reshape(-1, grid.shape[-1])[sample.vertices]
            required = sample.weights > 1e-13
            scored = sample.support & sample.observed
            if sample.point_weights is not None:
                scored = scored & (sample.point_weights > 0)
            if not np.isfinite(stencils).all(where=(scored[:, None] & required)[..., None]):
                raise ValueError("Predictions must be finite on the fixed scattered scoring support.")
            predicted = np.einsum("qv,qvc->qc", sample.weights, np.where(required[..., None], stencils, 0.0))
            predicted[~sample.support] = np.nan
            reports.append(
                {
                    "sample_id": sample.sample_id,
                    "observation": point_scores(
                        sample.truth,
                        predicted,
                        sample.support,
                        observation_mask=sample.observed,
                        point_weights=sample.point_weights,
                    ),
                    "interpolation": sample.interpolation,
                }
            )
        return {
            "observation": _aggregate_points(reports, "observation", sample_weights),
            "interpolation": _aggregate_points(reports, "interpolation", sample_weights),
            "samples": reports,
            "note": "Field, observation, and interpolation RMSE are different measures and are not additive.",
        }


def prepare_scattered_evaluation(
    root: str | Path,
    reference: FieldDataset,
    *,
    source_manifest: str | Mapping[str, Any] = "field_dataset.yaml",
    grid_mask: Any = None,
) -> _PreparedScatteredEvaluation:
    """Prepare original-point stencils/support and roundtrip scores once per run.

    Only compact reverse stencils are retained, not per-sample Delaunay objects.
    Prediction values never determine the scoring population.
    """
    original = load_scattered_observations(root, source_manifest=source_manifest)
    source_index = {str(value): index for index, value in enumerate(original["sample_ids"])}
    explicit_weights = reference.manifest["scattered"]["weighting"] == "explicit_point_weights"
    supplied = None if grid_mask is None else np.asarray(grid_mask)
    if supplied is not None and supplied.shape not in (reference.fields.shape[1:-1], reference.fields.shape[:-1]):
        raise ValueError("grid_mask must match the spatial or sample-by-spatial shape.")
    samples = []
    previous_points = previous_triangles = mapping = None
    for row, sample_id in enumerate(reference.sample_ids):
        if str(sample_id) not in source_index:
            raise ValueError("Reference sample IDs must exist in preserved scattered observations.")
        index = source_index[str(sample_id)]
        start, stop = original["offsets"][index : index + 2]
        tri_start, tri_stop = original["simplex_offsets"][index : index + 2]
        points, truth = original["points"][start:stop], original["values"][start:stop]
        triangles = original["simplices"][tri_start:tri_stop]
        if (
            mapping is None
            or not _same_array(points, previous_points)
            or not _same_array(triangles, previous_triangles)
        ):
            mapping = ScatteredGridMap.build(
                points,
                reference.geometry.coords["x"][0],
                reference.geometry.coords["y"][:, 0],
                domain_mask=reference.geometry.mask,
                simplex_support=triangles,
            )
            previous_points, previous_triangles = points, triangles
        valid = reference._sample_mask(row)
        if supplied is not None:
            candidate_mask = supplied[row] if supplied.ndim == 3 else supplied
            valid = valid & _mask(candidate_mask.reshape(-1), candidate_mask.size, "grid_mask").reshape(
                candidate_mask.shape
            )
        observed = original["observed"][start:stop]
        interpolated, support = mapping.to_points(reference.fields[row], valid, observation_mask=observed)
        weights = original["weights"][start:stop] if explicit_weights else None
        samples.append(
            _PointReference(
                str(sample_id),
                truth,
                observed,
                weights,
                mapping.grid_to_point.vertices,
                mapping.grid_to_point.weights,
                support,
                point_scores(truth, interpolated, support, observation_mask=observed, point_weights=weights),
            )
        )
    return _PreparedScatteredEvaluation(tuple(reference.fields.shape[1:]), tuple(samples))


def evaluate_scattered_prediction(
    root: str | Path,
    predicted_grid: Any,
    *,
    sample_ids: Any = None,
    grid_mask: Any = None,
    source_manifest: str | Mapping[str, Any] = "field_dataset.yaml",
    field_options: Any = None,
) -> dict[str, Any]:
    """Report canonical field, original-observation, and interpolation loss."""
    from .fields import prepare_reference

    dataset = load_field_dataset(root, source_manifest=source_manifest)
    if sample_ids is not None:
        by_id = {str(value): index for index, value in enumerate(dataset.sample_ids)}
        requested = np.asarray(sample_ids).astype(str).tolist()
        if len(requested) != len(set(requested)) or any(value not in by_id for value in requested):
            raise ValueError("Prediction sample_ids must uniquely identify samples in the imported dataset.")
        dataset = dataset._subset_validated(np.asarray([by_id[value] for value in requested]))
    prepared = prepare_scattered_evaluation(root, dataset, source_manifest=source_manifest, grid_mask=grid_mask)
    field = prepare_reference(dataset, field_options)
    return {
        "field": field.score(predicted_grid).metrics,
        **prepared.evaluate(predicted_grid, sample_weights=field.measure.sample_weights),
    }
