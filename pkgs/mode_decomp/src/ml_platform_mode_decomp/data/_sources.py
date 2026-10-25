"""Shared file, table and explicit geometry adapters for every input format."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import csv

import numpy as np
import pandas as pd
from ml_platform_core.io import read_table
from ml_platform_core.paths import resolve_path_within
from ..geometry import DomainSpec, build_domain_spec


def _path(root: Path, name: Any, label: str, paths: list[Path]) -> Path:
    path = resolve_path_within(root, name, setting=label, root_description="the Dataset root")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    paths.append(path)
    return path


def _mapping(value: Any, label: str, *, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if keys is not None and (unknown := set(value) - keys):
        raise ValueError(f"Unknown {label} settings: {sorted(unknown, key=str)}")
    return dict(value)


def _array(path: Path, *, mmap: bool = True) -> np.ndarray:
    array = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"Expected an NPY array: {path}")
    if array.dtype.hasobject:
        raise ValueError(f"Object arrays are not supported: {path}")
    return array


def _binary_mask(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "bui" or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label} must contain boolean or 0/1 values.")
    return array.astype(bool, copy=False)


def _table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if not header or any(not name for name in header) or len(header) != len(set(header)):
            raise ValueError(f"Table needs non-empty unique column names: {path}")
        # IDs remain strings (including leading zeroes) and numeric conversion is explicit below.
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return read_table(path)


def _columns(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must explicitly name the condition columns in order.")
    columns = tuple(value)
    if any(not isinstance(name, str) or not name.strip() for name in columns) or len(columns) != len(set(columns)):
        raise ValueError(f"{label} must contain non-empty unique names.")
    return columns


def _sample_table(
    root: Path, manifest: dict[str, Any], paths: list[Path], *, count: int | None = None
) -> tuple[pd.DataFrame, str, str]:
    if manifest.get("samples") is None and count is not None:
        return (
            pd.DataFrame(
                {"sample_id": [f"sample_{index:06d}" for index in range(count)], "array_index": np.arange(count)}
            ),
            "sample_id",
            "array_index",
        )
    samples = _mapping(manifest.get("samples"), "samples", keys={"path", "id_column", "index_column"})
    frame = _table(_path(root, samples.get("path"), "samples.path", paths))
    id_column = samples.get("id_column", "sample_id")
    index_column = samples.get("index_column", "array_index")
    if any(not isinstance(name, str) or not name.strip() for name in (id_column, index_column)):
        raise ValueError("Sample ID and index column names must be non-empty strings.")
    if id_column not in frame or frame.empty:
        raise ValueError("samples must contain a non-empty ID column.")
    ids = frame[id_column].astype(str)
    if frame[id_column].isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Sample IDs must be non-empty and unique.")
    if index_column not in frame and "index_column" not in samples:
        frame[index_column] = np.arange(len(frame))
    return frame, id_column, index_column


def _indices(frame: pd.DataFrame, column: str, count: int) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"samples is missing array index column {column!r}.")
    numeric = np.asarray(pd.to_numeric(frame[column], errors="raise"), dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("array_index must contain finite integers.")
    indices = numeric.astype(np.int64)
    if len(indices) != count or not np.array_equal(np.sort(indices), np.arange(count)):
        raise ValueError("array_index must reference every array row exactly once, without duplicates.")
    return indices


def _conditions(
    root: Path,
    config: Any,
    samples: pd.DataFrame,
    id_column: str,
    indices: np.ndarray,
    paths: list[Path],
    expected_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    if config is None:
        return None, ()
    config = _mapping(config, "conditions")
    format_name = config.get("format", "table")
    _mapping(
        config,
        "conditions",
        keys={"format", "path", "columns"} if format_name == "npy" else {"format", "path", "columns", "id_column"},
    )
    defaults = expected_columns
    values = None
    if format_name == "npy":
        values = _array(_path(root, config.get("path"), "conditions.path", paths))
        if values.ndim != 2 or len(values) != len(samples):
            raise ValueError("NPY conditions must have shape (N,P) matching samples.")
        if defaults is None:
            defaults = [f"condition_{index}" for index in range(values.shape[1])]
    columns = _columns(config.get("columns", defaults), "conditions.columns")
    if expected_columns is not None and columns != tuple(expected_columns):
        raise ValueError("Condition manifest columns disagree with the trained model schema.")
    if values is not None:
        return (values if np.array_equal(indices, np.arange(len(values))) else values[indices]), columns
    if format_name not in {"table", "csv"}:
        raise ValueError("conditions.format must be table or npy.")
    frame = samples
    if config.get("path"):
        frame = _table(_path(root, config["path"], "conditions.path", paths))
        condition_id = config.get("id_column", id_column)
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError("conditions.id_column must be a non-empty string.")
        if condition_id not in frame or frame[condition_id].duplicated().any():
            raise ValueError("Condition table needs unique sample IDs.")
        frame = frame.set_index(condition_id)
        sample_ids = samples[id_column].astype(str).tolist()
        if set(frame.index.astype(str)) != set(sample_ids):
            raise ValueError("Condition table IDs must exactly match samples.")
        frame.index = frame.index.astype(str)
        frame = frame.loc[sample_ids]
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ValueError(f"Missing condition columns: {missing}.")
    try:
        values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Condition columns must contain finite real numeric values.") from exc
    return values, columns


def _resolve_geometry(
    root: Path,
    config: Any,
    field_shape: tuple[int, ...],
    paths: list[Path],
    domain_mask: np.ndarray | None,
) -> DomainSpec:
    geometry = _mapping(config, "geometry")
    if "kind" in geometry and "name" in geometry:
        raise ValueError("Specify geometry.kind or geometry.name, not both.")
    geometry["name"] = geometry.pop("kind", geometry.get("name", ""))
    axes = geometry.get("axes", {})
    if not isinstance(axes, Mapping):
        raise ValueError("geometry.axes must be a mapping.")
    if "axes" in geometry or geometry["name"] != "mesh":
        geometry["axes"] = {}
    for name, raw_axis in axes.items():
        axis = _mapping(raw_axis, f"geometry.axes.{name}")
        if axis.get("path"):
            if any(key in axis for key in ("start", "step", "count", "values")):
                raise ValueError("Axis path and generated values cannot be combined.")
            axis["values"] = _array(_path(root, axis.pop("path"), f"geometry.axes.{name}.path", paths))
        geometry["axes"][name] = axis
    if geometry["name"] == "mesh":
        mesh = _mapping(geometry.pop("mesh", {}), "geometry.mesh", keys={"path"})
        if "mesh_path" in geometry and mesh.get("path") is not None:
            raise ValueError("Specify geometry.mesh.path or geometry.mesh_path, not both.")
        mesh_path = mesh.get("path", geometry.pop("mesh_path", None))
        if mesh_path:
            if any(key in geometry for key in ("vertices", "faces")):
                raise ValueError("geometry.mesh.path cannot be combined with vertices/faces.")
            with np.load(_path(root, mesh_path, "geometry.mesh.path", paths), allow_pickle=False) as arrays:
                geometry["vertices"], geometry["faces"] = arrays["vertices"], arrays["faces"]
                embedded = arrays["vertex_mask"] if "vertex_mask" in arrays else arrays.get("mask")
        else:
            embedded = None
        if {"vertex_mask_path", "mask_path"} <= set(geometry) or {"vertex_mask", "mask"} <= set(geometry):
            raise ValueError("Specify one mesh mask array or path, not duplicate aliases.")
        mask_path = geometry.pop("vertex_mask_path", geometry.pop("mask_path", None))
        mask_inline = geometry.pop("vertex_mask", geometry.pop("mask", None))
        if mask_path is not None and mask_inline is not None:
            raise ValueError("Provide a geometry mask array or path, not both.")
        mask = _array(_path(root, mask_path, "geometry.vertex_mask_path", paths)) if mask_path else mask_inline
        if mask is not None:
            mask = _binary_mask(mask, "geometry.vertex_mask").reshape(-1)
        for value in (embedded, domain_mask):
            if value is not None:
                value = _binary_mask(value, "geometry vertex mask").reshape(-1)
                mask = value if mask is None else mask & value
        if mask is not None:
            geometry["vertex_mask"] = np.asarray(mask).reshape(-1)
        shape = (field_shape[0], 1, field_shape[-1])
    else:
        shape = field_shape
        if geometry["name"] in {"mask", "arbitrary_mask"}:
            mask_path = geometry.pop("mask_path", None)
            mask_inline = geometry.pop("mask", None)
            source = geometry.pop("mask_source", None)
            if source == "dataset":
                raise ValueError("mask_source=dataset is ambiguous; use masks.domain.")
            if source not in {None, "file", "inline"}:
                raise ValueError("geometry.mask_source must be inline or file.")
            if mask_path is not None and mask_inline is not None:
                raise ValueError("Provide a geometry mask array or path, not both.")
            if source == "file" and not mask_path:
                raise ValueError("geometry.mask_source: file requires geometry.mask_path.")
            if source == "inline" and mask_inline is None:
                raise ValueError("geometry.mask_source: inline requires geometry.mask.")
            mask = _array(_path(root, mask_path, "geometry.mask_path", paths)) if mask_path else mask_inline
            if mask is None:
                mask = domain_mask
            if mask is None:
                raise ValueError("arbitrary_mask geometry requires an explicit fixed masks.domain.")
            geometry["mask"] = mask
    result = build_domain_spec(geometry, shape)
    if domain_mask is not None:
        spatial = field_shape[:-1]
        if domain_mask.shape != spatial:
            raise ValueError("masks.domain must match the fixed spatial field shape.")
        mask = domain_mask if result.mask is None else np.asarray(result.mask).reshape(spatial) & domain_mask
        result = replace(result, mask=mask)
    return result


def _axis_count(root: Path, axis: Any, label: str, paths: list[Path]) -> int:
    axis = _mapping(axis, label)
    if axis.get("path"):
        return len(_array(_path(root, axis["path"], label, paths)))
    if "values" in axis:
        return len(axis["values"])
    count = axis.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"{label} requires explicit coordinates or an integer count >= 2.")
    return count


def _field_shape(root: Path, geometry: dict[str, Any], layout: str, channels: int, paths: list[Path]) -> tuple:
    kind = geometry.get("kind", geometry.get("name"))
    if not isinstance(kind, str) or not kind:
        raise ValueError("CSV import requires an explicit geometry.kind.")
    if kind == "mesh":
        if layout != "NVC":
            raise ValueError("Mesh CSV import requires fields.layout: NVC.")
        mesh = _mapping(geometry.get("mesh"), "geometry.mesh")
        path = _path(root, mesh.get("path"), "geometry.mesh.path", paths)
        with np.load(path, allow_pickle=False) as arrays:
            return (len(arrays["vertices"]), channels)
    if layout != "NHWC":
        raise ValueError("Grid CSV import requires fields.layout: NHWC.")
    axes = _mapping(geometry.get("axes"), "geometry.axes")
    x, y = ("lon", "lat") if kind == "sphere_grid" else ("x", "y")
    return (
        _axis_count(root, axes.get(y), f"geometry.axes.{y}", paths),
        _axis_count(root, axes.get(x), f"geometry.axes.{x}", paths),
        channels,
    )


def _numbers(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], path: Path) -> np.ndarray:
    if any(column not in frame for column in columns):
        raise ValueError(f"Field CSV is missing declared columns {columns}: {path}")
    try:
        return frame[list(columns)].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Field CSV must contain numeric coordinates and values: {path}") from exc


def _geometry_files(geometry: Mapping[str, Any]) -> list[str]:
    references = [axis["path"] for axis in geometry.get("axes", {}).values() if "path" in axis]
    mesh_path = geometry.get("mesh", {}).get("path", geometry.get("mesh_path"))
    if mesh_path:
        references.append(mesh_path)
    for name in ("vertex_mask_path", "mask_path"):
        if geometry.get(name):
            references.append(geometry[name])
    return references
