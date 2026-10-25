"""One-way, explicit import of the former NPY directory format."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from ml_platform_core.config import deep_merge
from ml_platform_core.paths import resolve_path_within
from ._sources import _geometry_files


def _geometry(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(override, Mapping):
        raise ValueError("Explicit geometry must be a mapping.")
    override = dict(override)
    if "name" in override and "kind" not in override:
        override["kind"] = override.pop("name")
    merged = deep_merge(base, override)
    axes = override.get("axes", {})
    if not isinstance(axes, Mapping):
        raise ValueError("Explicit geometry.axes must be a mapping.")
    for name, axis in axes.items():
        if not isinstance(axis, Mapping):
            raise ValueError(f"Explicit geometry.axes.{name} must be a mapping.")
        if "values" in axis or "path" in axis:
            merged["axes"][name] = dict(axis)
        elif any(key in axis for key in ("start", "step", "count")):
            merged["axes"][name].pop("values", None)
            merged["axes"][name].pop("path", None)
    return merged


def import_legacy_npy_dataset(
    source: str | Path,
    destination: str | Path,
    *,
    condition_columns: Sequence[str] | None = None,
    geometry_override: Mapping[str, Any] | None = None,
) -> Path:
    """Copy legacy numeric files unchanged and add explicit IDs, axes, and mask roles.

    Existing destinations are rejected. A variable legacy mask cannot define one
    fixed arbitrary domain; that ambiguity requires an explicitly supplied domain.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination == source:
        raise ValueError("Import destination must be separate from the source dataset.")
    if destination.exists():
        raise FileExistsError(f"Import destination already exists: {destination}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    fields = np.load(source / "field.npy", allow_pickle=False, mmap_mode="r")
    conditions = np.load(source / "cond.npy", allow_pickle=False, mmap_mode="r")
    if fields.ndim != 4 or fields.shape[-1] not in (1, 2) or conditions.ndim != 2 or len(conditions) != len(fields):
        raise ValueError("Legacy import requires NHWC fields and matching (N,P) conditions.")
    columns = list(condition_columns or [f"x{index + 1}" for index in range(conditions.shape[1])])
    if len(columns) != conditions.shape[1] or len(set(columns)) != len(columns):
        raise ValueError("condition_columns must uniquely name all condition dimensions.")
    old_grid, old_domain = manifest["grid"], dict(manifest["domain"])
    domain_name = old_domain.pop("type", old_domain.pop("name", ""))
    domain_axes = old_domain.pop("axes", None)
    height, width = fields.shape[1:3]
    if (height, width) != (int(old_grid["H"]), int(old_grid["W"])):
        raise ValueError("Legacy field shape disagrees with manifest grid.")
    axes = {}
    for name, count, key in (("x", width, "x_range"), ("y", height, "y_range")):
        lower, upper = old_grid[key]
        axes[name] = {"start": float(lower), "step": float(upper - lower) / max(count - 1, 1), "count": count}
    for name in ("mask_source", "mask_path", "mask_file", "mask"):
        old_domain.pop(name, None)
    geometry = {"kind": domain_name, **old_domain}
    if domain_name == "sphere_grid":
        geometry["axes"] = {"lon": axes["x"], "lat": axes["y"]}
    else:
        geometry["axes"] = axes
    if domain_axes is not None:
        geometry = _geometry(geometry, {"axes": domain_axes})
    if "geometry" in manifest:
        geometry = _geometry(geometry, manifest["geometry"])
    if geometry_override is not None:
        geometry = _geometry(geometry, geometry_override)
    output: dict[str, Any] = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv", "id_column": "sample_id", "index_column": "array_index"},
        "conditions": {"format": "npy", "path": "cond.npy", "columns": columns},
        "fields": {
            "path": "field.npy",
            "layout": "NHWC",
            "kind": "scalar" if fields.shape[-1] == 1 else "vector",
            "components": ["value"] if fields.shape[-1] == 1 else ["fx", "fy"],
        },
        "geometry": geometry,
        "imported_from": {
            "format": "legacy_npy_dir",
            "condition_names": "provided" if condition_columns else "positional",
            "geometry_source": "override"
            if geometry_override is not None
            else "manifest"
            if "geometry" in manifest or domain_axes is not None
            else "legacy_ranges",
        },
    }
    mask_path = source / "mask.npy"
    domain_mask = None
    if mask_path.exists():
        mask = np.load(mask_path, allow_pickle=False, mmap_mode="r")
        if mask.shape not in ((height, width), fields.shape[:-1]):
            raise ValueError("Legacy mask shape must be spatial or sample-by-spatial.")
        if geometry["kind"] in {"mask", "arbitrary_mask"}:
            if mask.ndim == 3 and not np.all(mask == mask[0]):
                raise ValueError("A variable observation mask cannot be inferred as a fixed arbitrary domain.")
            domain_mask = mask[0] if mask.ndim == 3 else mask
    geometry_files = [
        resolve_path_within(source, name, setting="geometry path", root_description="the source Dataset root")
        for name in _geometry_files(geometry)
    ]
    if any(not path.is_file() for path in geometry_files):
        raise FileNotFoundError("A referenced geometry file is missing from the source Dataset.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".legacy_import_", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name in ("cond.npy", "field.npy"):
            shutil.copyfile(source / name, staging / name)
        for original in geometry_files:
            target = staging / original.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        if mask_path.exists():
            shutil.copyfile(mask_path, staging / "observation_mask.npy")
            output["masks"] = {"observation": "observation_mask.npy"}
        if domain_mask is not None:
            np.save(staging / "domain_mask.npy", domain_mask, allow_pickle=False)
            output["masks"]["domain"] = "domain_mask.npy"
        with (staging / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "array_index"])
            writer.writerows((f"npy_{index:04d}", index) for index in range(len(fields)))
        manifest_path = staging / "field_dataset.yaml"
        manifest_path.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding="utf-8")
        staging.rename(destination)
    return destination / "field_dataset.yaml"
