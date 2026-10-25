"""Explicit scattered observation import and evaluation against preserved points."""

from __future__ import annotations

import shutil
from contextlib import ExitStack
import tempfile
from pathlib import Path
from mmap import mmap
from typing import Any, Protocol, cast
from collections.abc import Mapping

import numpy as np
import yaml

from ._sources import _field_shape, _numbers
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
from .scattered import ScatteredGridMap, _mask
from .schema import ConditionBatch, _field_units
from .input_spec import read_input_spec


class _MemmapOwner(Protocol):
    _mmap: mmap


def _sample_array(array: np.ndarray, index: int, shared_ndim: int, count: int, label: str) -> np.ndarray:
    if array.ndim == shared_ndim:
        return array
    if array.ndim == shared_ndim + 1 and len(array) == count:
        return array[index]
    raise ValueError(f"{label} must be shared or have one leading row per sample.")


def _same_array(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    return left is right or (left is not None and right is not None and np.array_equal(left, right))


def _source_observations(root, config, samples, indices, channels, paths):
    count = len(samples)
    arrays = {}
    for key in ("coordinates", "values", "mask", "weights", "simplex_support"):
        if config.get("format") == "npy" and config.get(key):
            arrays[key] = _array(_path(root, config[key], f"observations.{key}", paths))
        elif key == "simplex_support" and config.get(key):
            arrays[key] = _array(_path(root, config[key], "observations.simplex_support", paths))
    if config.get("format") == "npy":
        if "coordinates" not in arrays or "values" not in arrays:
            raise ValueError("NPY observations require coordinates and values paths.")
        if arrays["values"].ndim != 3 or arrays["values"].shape[0] != count or arrays["values"].shape[-1] != channels:
            raise ValueError("observations.values must have shape (N,M,C) matching samples and components.")
    elif config.get("format") != "csv":
        raise ValueError("observations.format must explicitly be npy or csv.")
    for row in np.argsort(indices):
        index = indices[row]
        if config["format"] == "npy":
            points = _sample_array(arrays["coordinates"], index, 2, count, "observations.coordinates")
            values = arrays["values"][index]
            observed = _sample_array(arrays["mask"], index, 1, count, "observations.mask") if "mask" in arrays else None
            weights = (
                _sample_array(arrays["weights"], index, 1, count, "observations.weights")
                if "weights" in arrays
                else None
            )
        else:
            path_column = config.get("path_column")
            if path_column not in samples:
                raise ValueError("observations.path_column must name a sample-table column.")
            path = _path(root, samples.iloc[row][path_column], "point CSV", paths)
            frame = _table(path)
            coordinate_columns = _columns(config.get("coordinates"), "observations.coordinates")
            value_columns = _columns(config.get("columns"), "observations.columns")
            if len(coordinate_columns) != 2 or len(value_columns) != channels:
                raise ValueError("CSV observations need two coordinate columns and one value column per component.")
            points, values = _numbers(frame, coordinate_columns, path), _numbers(frame, value_columns, path)
            observed = _numbers(frame, [config["mask_column"]], path)[:, 0] if config.get("mask_column") else None
            if observed is not None:
                if not np.isin(observed, (0, 1)).all():
                    raise ValueError("CSV observation mask must contain 0/1 values.")
                observed = observed.astype(bool)
            weights = _numbers(frame, [config["weight_column"]], path)[:, 0] if config.get("weight_column") else None
        support = (
            _sample_array(arrays["simplex_support"], index, 2, count, "observations.simplex_support")
            if "simplex_support" in arrays
            else None
        )
        yield row, int(index), points, values, observed, weights, support


def import_scattered_field_dataset(
    source: str | Path, destination: str | Path, *, source_manifest: str | Mapping[str, Any]
) -> Path:
    """Interpolate Cartesian 2-D observations once, retaining original point values.

    Only linear / outside: mask is supported. Annular and arbitrary domains require
    explicitly allowed Delaunay triangles; a point cloud cannot identify its holes.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source:
        raise ValueError("Import destination must be separate from the source dataset.")
    if destination.exists():
        raise FileExistsError(f"Import destination already exists: {destination}")
    paths: list[Path] = []
    manifest = read_input_spec(source, source_manifest, paths)
    if manifest.get("schema") != "scattered_field_import_v1":
        raise ValueError("Expected schema: scattered_field_import_v1.")
    interpolation = {
        "method": "linear",
        "outside": "mask",
        **_mapping(manifest.get("interpolation", {}), "interpolation"),
    }
    if interpolation != {"method": "linear", "outside": "mask"}:
        raise ValueError("Supported interpolation is exactly {method: linear, outside: mask}.")
    fields = _mapping(manifest.get("fields"), "fields")
    components = _columns(fields.get("components"), "fields.components")
    if (
        fields.get("layout") != "NHWC"
        or fields.get("kind") not in {"scalar", "vector", "multichannel"}
        or (fields["kind"] == "scalar" and len(components) != 1)
    ):
        raise ValueError("Scattered import requires NHWC scalar (C=1), vector, or multichannel fields.")
    geometry_config = _mapping(manifest.get("geometry"), "geometry")
    if geometry_config.get("kind", geometry_config.get("name")) not in {
        "rectangle",
        "disk",
        "annulus",
        "mask",
        "arbitrary_mask",
    }:
        raise ValueError("Scattered import supports only explicit Cartesian 2-D grid geometry.")
    shape = _field_shape(source, geometry_config, "NHWC", len(components), paths)
    masks = _mapping(manifest.get("masks", {}), "masks")
    if set(masks) - {"domain"}:
        raise ValueError("Scattered masks accepts only domain; target observation support is computed from the points.")
    domain = _array(_path(source, masks["domain"], "masks.domain", paths)) if masks.get("domain") else None
    if domain is not None and (domain.dtype.kind not in "bui" or not np.isin(domain, (0, 1)).all()):
        raise ValueError("masks.domain must contain boolean or 0/1 values.")
    geometry = _resolve_geometry(source, geometry_config, shape, paths, domain)
    observations = _mapping(manifest.get("observations"), "observations")
    if geometry.name in {"annulus", "mask", "arbitrary_mask"} and not observations.get("simplex_support"):
        raise ValueError("Annular/arbitrary domains require observations.simplex_support to avoid bridging holes.")
    count = None
    if observations.get("format") == "npy":
        count = len(_array(_path(source, observations.get("values"), "observations.values", paths)))
    elif manifest.get("samples") is None and observations.get("path"):
        manifest = {**manifest, "samples": None}
        count = 1
    samples, id_column, index_column = _sample_table(source, manifest, paths, count=count)
    if observations.get("format") == "csv" and observations.get("path"):
        if observations.get("path_column"):
            raise ValueError("Choose observations.path or observations.path_column.")
        observations["path_column"] = "__point_csv__"
        samples["__point_csv__"] = str(observations["path"])
    indices = _indices(samples, index_column, len(samples))
    conditions, condition_columns = _conditions(source, manifest.get("conditions"), samples, id_column, indices, paths)
    if conditions is not None:
        ConditionBatch(conditions, samples[id_column].to_numpy(), condition_columns)
    output: dict[str, Any] = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv", "id_column": id_column, "index_column": index_column},
        "fields": {"path": "field.npy", "layout": "NHWC", "kind": fields["kind"], "components": list(components)},
        "geometry": geometry_config,
        "masks": {"observation": "observation_mask.npy"},
        "scattered": {
            "path": "scattered/index.yaml",
            **interpolation,
            "weighting": "explicit_point_weights"
            if observations.get("weights") or observations.get("weight_column")
            else "point_mean",
        },
    }
    if "units" in fields:
        output["fields"]["units"] = list(_field_units(fields["units"], len(components)))
    table = samples.copy()
    if observations.get("format") == "csv":
        if observations.get("path_column") in {id_column, index_column, *condition_columns}:
            raise ValueError("Point CSV paths must be separate from ID, array_index, and condition columns.")
        table = table.drop(columns=[observations["path_column"]])
    if conditions is not None:
        output["conditions"] = {"format": "table", "columns": list(condition_columns)}
        for channel, column in enumerate(condition_columns):
            if column in {id_column, index_column}:
                raise ValueError("Condition columns must be separate from ID and array_index.")
            table[column] = conditions[:, channel]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scattered_import_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        store = staging / "scattered"
        store.mkdir()
        converted = np.lib.format.open_memmap(
            staging / "field.npy", mode="w+", dtype=np.float64, shape=(len(samples), *shape)
        )
        supported = np.lib.format.open_memmap(
            staging / "observation_mask.npy", mode="w+", dtype=bool, shape=(len(samples), *shape[:-1])
        )
        offsets, triangle_offsets = [0], [0]
        point_count = triangle_count = 0
        previous_points = previous_support = mapping = None
        with ExitStack() as stack:
            # NumPy's memmap stub omits its mmap owner; close it before Windows
            # moves the staging directory or cleans up a failed import.
            stack.callback(cast(_MemmapOwner, converted)._mmap.close)
            stack.callback(cast(_MemmapOwner, supported)._mmap.close)
            names = ("points", "values", "observed", "weights", "simplices")
            streams = {name: stack.enter_context((store / f".{name}.raw").open("wb")) for name in names}
            for _row, index, points, values, observed, weights, support in _source_observations(
                source, observations, samples, indices, len(components), paths
            ):
                # A one-entry geometry cache covers the common shared-point input
                # without retaining all per-sample triangulations.
                if (
                    mapping is None
                    or not _same_array(points, previous_points)
                    or not _same_array(support, previous_support)
                ):
                    mapping = ScatteredGridMap.build(
                        points,
                        geometry.coords["x"][0],
                        geometry.coords["y"][:, 0],
                        domain_mask=geometry.mask,
                        simplex_support=support,
                    )
                    previous_points = np.array(points, copy=True)
                    previous_support = None if support is None else np.array(support, copy=True)
                observed = _mask(observed, len(points), "observations.mask")
                values = np.asarray(values)
                if values.dtype.kind not in "fiu" or values.shape != (len(points), len(components)):
                    raise ValueError("Observation values must be real numeric (M,C) arrays.")
                if not np.isfinite(values).all(where=observed[:, None]):
                    raise ValueError("Known observations must be finite.")
                weights = np.ones(len(points)) if weights is None else np.asarray(weights)
                if (
                    weights.dtype.kind not in "fiu"
                    or weights.shape != (len(points),)
                    or not np.isfinite(weights).all()
                    or np.any(weights < 0)
                ):
                    raise ValueError("point_weights must be finite, non-negative real numeric values with shape (M,).")
                if not np.any(observed & (weights > 0)):
                    raise ValueError("Every sample requires at least one observed point with positive weight.")
                array, target_support = mapping.forward(values, observed)
                if not target_support.any():
                    raise ValueError("Every sample requires at least one supported target grid location.")
                converted[index], supported[index] = array, target_support
                triangles = mapping.point_to_grid.simplices[mapping.point_to_grid.simplex_support]
                for name, item, dtype in (
                    ("points", points, np.float64),
                    ("values", values, np.float64),
                    ("observed", observed, np.bool_),
                    ("weights", weights, np.float64),
                    ("simplices", triangles, np.int64),
                ):
                    streams[name].write(np.asarray(item, dtype=dtype).tobytes(order="C"))
                point_count += len(points)
                triangle_count += len(triangles)
                offsets.append(point_count)
                triangle_offsets.append(triangle_count)
            converted.flush()
            supported.flush()
        del converted, supported
        for name, shape_item, dtype in (
            ("points", (point_count, 2), np.float64),
            ("values", (point_count, len(components)), np.float64),
            ("observed", (point_count,), np.bool_),
            ("weights", (point_count,), np.float64),
            ("simplices", (triangle_count, 3), np.int64),
        ):
            temporary_array = store / f".{name}.raw"
            with (store / f"{name}.npy").open("wb") as handle, temporary_array.open("rb") as raw:
                np.lib.format.write_array_header_2_0(
                    handle, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": shape_item}
                )
                shutil.copyfileobj(raw, handle)
            temporary_array.unlink()
        ordered_ids = samples.iloc[np.argsort(indices)][id_column].astype(str).to_numpy(dtype=str)
        for name, item in (("sample_ids", ordered_ids), ("offsets", offsets), ("simplex_offsets", triangle_offsets)):
            np.save(store / f"{name}.npy", item, allow_pickle=False)
        (store / "index.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "scattered_observations_v1",
                    "arrays": {name: f"{name}.npy" for name in (*names, "sample_ids", "offsets", "simplex_offsets")},
                }
            ),
            encoding="utf-8",
        )
        for reference in set(_geometry_files(geometry_config)):
            original = _path(source, reference, "geometry path", paths)
            relative = original.relative_to(source)
            if relative.parts[0] == "scattered" or relative.as_posix() in {
                "field.npy",
                "observation_mask.npy",
                "samples.csv",
                "field_dataset.yaml",
                "domain_mask.npy",
                "scattered/index.yaml",
            }:
                raise ValueError(f"Geometry reference conflicts with generated output: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        if masks.get("domain"):
            shutil.copyfile(_path(source, masks["domain"], "masks.domain", paths), staging / "domain_mask.npy")
            output["masks"]["domain"] = "domain_mask.npy"
        table.to_csv(staging / "samples.csv", index=False)
        (staging / "field_dataset.yaml").write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
        staging.rename(destination)
    return destination / "field_dataset.yaml"


def load_scattered_observations(
    root: str | Path, *, source_manifest: str | Mapping[str, Any] = "field_dataset.yaml"
) -> dict[str, np.ndarray]:
    """Load original observations, indexed by sample_ids and packed point offsets."""
    root = Path(root).resolve()
    paths: list[Path] = []
    manifest = read_input_spec(root, source_manifest, paths)
    config = _mapping(manifest.get("scattered"), "scattered")
    path = _path(root, config.get("path"), "scattered.path", paths)
    if path.suffix.lower() == ".npz":
        # Historical imported datasets remain readable at this input boundary.
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}
    from ml_platform_core.config import load_yaml

    index = load_yaml(path)
    if index.get("schema") != "scattered_observations_v1":
        raise ValueError("Expected schema: scattered_observations_v1.")
    return {
        name: _array(_path(path.parent, reference, f"scattered.{name}", paths))
        for name, reference in _mapping(index.get("arrays"), "scattered arrays").items()
    }


# Historical input-module imports remain available at this boundary. The
# numerical implementation belongs to evaluation, not the dataset converter.
def prepare_scattered_evaluation(*args, **kwargs):
    from ..evaluation.observations import prepare_scattered_evaluation as prepare

    return prepare(*args, **kwargs)


def evaluate_scattered_prediction(*args, **kwargs):
    from ..evaluation.observations import evaluate_scattered_prediction as evaluate

    return evaluate(*args, **kwargs)
