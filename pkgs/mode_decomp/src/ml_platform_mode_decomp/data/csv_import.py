"""One-way import of explicitly indexed CSV fields into the canonical NPY format."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import yaml

from ._sources import _geometry_files
from ._sources import (
    _array,
    _columns,
    _conditions,
    _indices,
    _mapping,
    _path,
    _resolve_geometry,
    _sample_table,
    _table,
)
from .schema import ConditionBatch, FieldDataset, _field_units
from .input_spec import read_input_spec
from ._sources import _field_shape, _numbers


def _axis_positions(values: np.ndarray, expected: np.ndarray, tolerance: float, label: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"CSV {label} coordinates must be finite.")
    if len(expected) > 1 and 2 * tolerance >= np.min(np.diff(expected)):
        raise ValueError("coordinate_atol must be less than half the smallest axis spacing.")
    right = np.clip(np.searchsorted(expected, values), 0, len(expected) - 1)
    left = np.maximum(right - 1, 0)
    positions = np.where(np.abs(values - expected[left]) < np.abs(values - expected[right]), left, right)
    if np.any(np.abs(values - expected[positions]) > tolerance):
        raise ValueError(f"CSV {label} coordinate is outside the declared grid; interpolation is not performed.")
    return positions


def _read_field(path: Path, config: dict[str, Any], geometry: Any, shape: tuple, tolerance: float) -> np.ndarray:
    frame = _table(path)
    coordinate_columns = _mapping(config.get("coordinates"), "fields.coordinates")
    if geometry.name == "mesh":
        column = coordinate_columns.get("vertex_index")
        if not isinstance(column, str) or not column:
            raise ValueError("Mesh CSV requires fields.coordinates.vertex_index.")
        vertices = _numbers(frame, [column], path)[:, 0]
        if not np.isfinite(vertices).all() or not np.equal(vertices, np.floor(vertices)).all():
            raise ValueError("Mesh vertex_index must contain finite zero-based integers.")
        positions = vertices.astype(np.int64)
    else:
        sphere = geometry.name == "sphere_grid"
        x, y = ("lon", "lat") if sphere else ("x", "y")
        if any(not isinstance(coordinate_columns.get(axis), str) or not coordinate_columns[axis] for axis in (x, y)):
            raise ValueError(f"Grid CSV requires fields.coordinates.{x} and .{y}.")
        values = _numbers(frame, [coordinate_columns[x], coordinate_columns[y]], path)
        suffix = "_deg" if sphere and geometry.meta["angle_unit"] == "deg" else ""
        xx, yy = geometry.coords[x + suffix], geometry.coords[y + suffix]
        xi = _axis_positions(values[:, 0], xx[0], tolerance, x)
        yi = _axis_positions(values[:, 1], yy[:, 0], tolerance, y)
        positions = yi * shape[1] + xi
    count = int(np.prod(shape[:-1]))
    if np.any((positions < 0) | (positions >= count)):
        raise ValueError("CSV vertex_index is outside the declared mesh.")
    if len(np.unique(positions)) != len(positions):
        raise ValueError(f"Duplicate field coordinates in CSV: {path}")
    if len(positions) != count:
        raise ValueError(f"Missing field coordinates: every declared grid point or mesh vertex is required: {path}")
    result = np.empty((count, shape[-1]), dtype=np.float64)
    result[positions] = _numbers(frame, config["columns"], path)
    return result.reshape(shape)


def import_csv_field_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    source_manifest: str | Mapping[str, Any],
) -> Path:
    """Import fixed-grid/mesh CSV files, with no interpolation or domain inference.

    Sample rows use the normal ID/array_index contract. Components follow the
    explicit fields.components / fields.columns pairing, never CSV column order.
    Failed imports leave no partial destination and never modify the source.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source:
        raise ValueError("Import destination must be separate from the source dataset.")
    if destination.exists():
        raise FileExistsError(f"Import destination already exists: {destination}")
    paths: list[Path] = []
    manifest = read_input_spec(source, source_manifest, paths)
    if manifest.get("schema") != "field_csv_import_v1":
        raise ValueError("CSV import requires schema: field_csv_import_v1.")
    fields = _mapping(manifest.get("fields"), "fields")
    components = _columns(fields.get("components"), "fields.components")
    columns = _columns(fields.get("columns"), "fields.columns")
    if len(columns) != len(components):
        raise ValueError("fields.columns must name one CSV value column per component in order.")
    if fields.get("kind") not in {"scalar", "vector", "multichannel"} or (
        fields["kind"] == "scalar" and len(components) != 1
    ):
        raise ValueError("fields.kind must be scalar (C=1), vector, or multichannel.")
    if fields.get("path"):
        if manifest.get("samples") is not None or fields.get("path_column"):
            raise ValueError("Choose a single fields.path or samples with fields.path_column.")
        samples, id_column, index_column = _sample_table(source, manifest, paths, count=1)
        fields["path_column"] = "__field_csv__"
        samples["__field_csv__"] = str(fields["path"])
    else:
        samples, id_column, index_column = _sample_table(source, manifest, paths)
    indices = _indices(samples, index_column, len(samples))
    path_column = fields.get("path_column")
    if not isinstance(path_column, str) or path_column not in samples:
        raise ValueError("fields.path_column must name the sample-table column containing each field CSV path.")
    if path_column in {id_column, index_column}:
        raise ValueError("The field path column must be separate from sample ID and array_index.")
    conditions, condition_columns = _conditions(source, manifest.get("conditions"), samples, id_column, indices, paths)
    if conditions is not None:
        ConditionBatch(conditions, samples[id_column].to_numpy(), condition_columns)
    geometry_config = _mapping(manifest.get("geometry"), "geometry")
    shape = _field_shape(source, geometry_config, str(fields.get("layout", "")), len(components), paths)
    masks = _mapping(manifest.get("masks", {}), "masks")
    domain = _array(_path(source, masks["domain"], "masks.domain", paths)) if masks.get("domain") else None
    if domain is not None and (domain.dtype.kind not in "bui" or not np.isin(domain, (0, 1)).all()):
        raise ValueError("masks.domain must contain boolean or 0/1 values.")
    geometry = _resolve_geometry(source, geometry_config, shape, paths, domain)
    observation = (
        _array(_path(source, masks["observation"], "masks.observation", paths)) if masks.get("observation") else None
    )
    if observation is not None and observation.shape not in (shape[:-1], (len(samples), *shape[:-1])):
        raise ValueError("masks.observation must be spatial or sample-by-spatial.")
    tolerance = float(fields.get("coordinate_atol", 0.0))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("fields.coordinate_atol must be finite and non-negative.")
    output: dict[str, Any] = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv", "id_column": id_column, "index_column": index_column},
        "fields": {key: fields[key] for key in ("layout", "kind", "components")},
        "geometry": geometry_config,
        "imported_from": {"format": "field_csv_import_v1", "coordinate_atol": tolerance},
    }
    output["fields"]["path"] = "field.npy"
    if "units" in fields:
        output["fields"]["units"] = list(_field_units(fields["units"], len(components)))
    sample_output = samples.drop(columns=[path_column]).copy()
    if conditions is not None:
        output["conditions"] = {"format": "table", "columns": list(condition_columns)}
        for index, column in enumerate(condition_columns):
            if column in {id_column, index_column, path_column}:
                raise ValueError("Condition columns must be separate from ID, index, and field path columns.")
            sample_output[column] = conditions[:, index]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".field_import_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with (staging / "field.npy").open("wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle, {"descr": np.dtype(np.float64).str, "fortran_order": False, "shape": (len(samples), *shape)}
            )
            for row in np.argsort(indices):
                path = _path(source, samples.iloc[row][path_column], "field CSV", paths)
                array = _read_field(path, fields, geometry, shape, tolerance)
                mask = observation
                if observation is not None and observation.ndim == len(shape):
                    mask = observation[indices[row]]
                FieldDataset(
                    array[None],
                    geometry,
                    np.asarray([samples.iloc[row][id_column]]),
                    observation_mask=mask,
                    components=components,
                )
                handle.write(array.tobytes(order="C"))
        reserved = {"field.npy", "samples.csv", "field_dataset.yaml", "domain_mask.npy", "observation_mask.npy"}
        for reference in set(_geometry_files(geometry_config)):
            original = _path(source, reference, "geometry path", paths)
            relative = original.relative_to(source)
            if relative.as_posix() in reserved:
                raise ValueError(f"Geometry path conflicts with generated output: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        for role in ("domain", "observation"):
            if masks.get(role):
                original = _path(source, masks[role], f"masks.{role}", paths)
                output.setdefault("masks", {})[role] = f"{role}_mask.npy"
                shutil.copyfile(original, staging / f"{role}_mask.npy")
        sample_output.to_csv(staging / "samples.csv", index=False)
        (staging / "field_dataset.yaml").write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
        staging.rename(destination)
    return destination / "field_dataset.yaml"
