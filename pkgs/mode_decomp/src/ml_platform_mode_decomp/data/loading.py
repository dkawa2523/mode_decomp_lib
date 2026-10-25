"""Explicit field manifests and condition-only input; all references use the Dataset root."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml_platform_core.artifacts import file_hash
from ml_platform_core.config import load_yaml

from .schema import ConditionBatch, FieldDataset, _array_geometry, _field_units
from .input_spec import field_input_spec, read_input_spec, validate_data_settings
from ._sources import _path, _mapping, _array, _table, _columns, _sample_table, _indices, _conditions, _resolve_geometry


def _revision(root: Path, paths: list[Path], manifest: dict[str, Any], dataset_id: Any) -> str:
    files = {}
    for path in sorted(set(paths)):
        files[path.relative_to(root).as_posix()] = file_hash(path)
    value: dict[str, Any] = {"files": files, "manifest": manifest}
    if dataset_id:
        value["clearml_dataset_id"] = str(dataset_id)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _config_root(value: Mapping[str, Any] | str | Path, *, condition_only=False) -> tuple[dict[str, Any], Path]:
    if isinstance(value, Mapping):
        config = _mapping(value.get("data", value), "data")
        validate_data_settings(config, condition_only=condition_only)
        local_path = config.get("local_path", config.get("path"))
        if not local_path:
            raise ValueError("data.local_path is required after runtime Dataset resolution.")
        return config, Path(local_path).expanduser().resolve()
    return {}, Path(value).expanduser().resolve()


def load_field_dataset(
    cfg_or_root: Mapping[str, Any] | str | Path,
    source_manifest: str | Mapping[str, Any] | None = None,
    *,
    _source_paths: list[Path] | None = None,
) -> FieldDataset:
    config, root = _config_root(cfg_or_root)
    if root.is_file():
        if source_manifest is not None or config.get("source_manifest"):
            raise ValueError("Pass a Dataset root plus source_manifest, or a manifest file, not both.")
        source_manifest, root = root.name, root.parent
    paths: list[Path] = []
    if source_manifest is not None:
        if config.get("source_manifest") is not None or any(
            key in config
            for key in ("fields", "geometry", "samples", "conditions", "masks", "observations", "interpolation")
        ):
            raise ValueError("Pass one inline or referenced input specification, not both.")
        format_name, reference = config.get("format", "npy"), source_manifest
    else:
        format_name, reference = field_input_spec(config)
    if format_name != "npy":
        raise ValueError(
            "load_field_dataset accepts npy input; convert CSV/scattered input through the import workflow."
        )
    manifest = read_input_spec(root, reference, paths)
    if manifest.get("schema") != "field_dataset_v1":
        raise ValueError("Expected schema: field_dataset_v1; legacy input requires explicit import.")
    if manifest.get("scattered"):
        scattered = _mapping(manifest["scattered"], "scattered")
        scattered_path = _path(root, scattered.get("path"), "scattered.path", paths)
        if scattered_path.suffix.lower() != ".npz":
            scattered_index = load_yaml(scattered_path)
            if scattered_index.get("schema") != "scattered_observations_v1":
                raise ValueError("Expected schema: scattered_observations_v1.")
            for name, reference in _mapping(scattered_index.get("arrays"), "scattered arrays").items():
                array_path = _path(scattered_path.parent, reference, f"scattered.{name}", paths)
                if not array_path.is_relative_to(root):
                    raise ValueError("Scattered arrays must stay within the Dataset root.")
    fields_config = _mapping(manifest.get("fields"), "fields")
    fields = _array(_path(root, fields_config.get("path"), "fields.path", paths))
    layout = fields_config.setdefault("layout", "NHWC")
    dimensions = {"NHWC": 4, "NVC": 3, "HWC": 3, "VC": 2}
    if layout not in dimensions or fields.ndim != dimensions[layout]:
        raise ValueError("fields.layout must explicitly match NHWC, NVC, HWC, or VC.")
    if layout in {"HWC", "VC"}:
        fields = fields[None, ...]
    kind = fields_config.setdefault("kind", "scalar" if fields.shape[-1] == 1 else "multichannel")
    if kind not in {"scalar", "vector", "multichannel"} or (kind == "scalar" and fields.shape[-1] != 1):
        raise ValueError("fields.kind must be scalar (C=1), vector, or multichannel.")
    components = _columns(
        fields_config.setdefault("components", [f"component_{index}" for index in range(fields.shape[-1])]),
        "fields.components",
    )
    manifest["fields"] = fields_config
    units = _field_units(fields_config.get("units"), fields.shape[-1])
    samples, id_column, index_column = _sample_table(root, manifest, paths, count=len(fields))
    indices = _indices(samples, index_column, len(fields))
    ordered_rows = slice(None) if np.array_equal(indices, np.arange(len(fields))) else indices
    conditions, columns = _conditions(root, manifest.get("conditions"), samples, id_column, indices, paths)
    masks = _mapping(manifest.get("masks", {}), "masks")
    domain_mask = None
    if masks.get("domain"):
        domain_mask = _array(_path(root, masks["domain"], "masks.domain", paths))
        if domain_mask.dtype.kind not in "bui" or not np.isin(domain_mask, (0, 1)).all():
            raise ValueError("masks.domain must contain boolean or 0/1 values.")
        domain_mask = domain_mask.astype(bool, copy=False)
    if manifest.get("geometry") is None:
        if layout in {"NVC", "VC"}:
            raise ValueError("Mesh NVC/VC input requires explicit geometry.")
        geometry = _array_geometry(fields.shape[1:], None, None)
        if domain_mask is not None:
            if domain_mask.shape != fields.shape[1:-1]:
                raise ValueError("masks.domain must match the fixed spatial field shape.")
            geometry = replace(geometry, mask=domain_mask)
    else:
        geometry = _resolve_geometry(root, manifest.get("geometry"), fields.shape[1:], paths, domain_mask)
    if (layout in {"NVC", "VC"}) != (geometry.name == "mesh"):
        raise ValueError("NVC/VC fields require mesh geometry; grid geometry requires NHWC/HWC.")
    observation = None
    if masks.get("observation"):
        observation = _array(_path(root, masks["observation"], "masks.observation", paths))
        if observation.shape == fields.shape[:-1]:
            observation = observation[ordered_rows]
    metadata = {str(name): samples[name].to_numpy() for name in samples if name not in {id_column, index_column}}
    if _source_paths is not None:
        _source_paths.extend(paths)
    return FieldDataset(
        fields=fields[ordered_rows],
        geometry=geometry,
        sample_ids=samples[id_column].astype(str).to_numpy(),
        conditions=conditions,
        condition_columns=columns,
        observation_mask=observation,
        sample_metadata=metadata,
        revision=_revision(root, paths, manifest, config.get("clearml_dataset_id")),
        manifest=manifest,
        components=components,
        field_kind=kind,
        field_units=units,
    )


def load_condition_batch(
    cfg_or_file: Mapping[str, Any] | str | Path,
    condition_columns: Sequence[str] | None = None,
) -> ConditionBatch:
    config, root = _config_root(cfg_or_file, condition_only=True)
    paths: list[Path] = []
    expected_columns = None if condition_columns is None else tuple(condition_columns)
    if config.get("condition_columns") is not None:
        declared = _columns(config["condition_columns"], "data.condition_columns")
        if expected_columns is not None and declared != expected_columns:
            raise ValueError("data.condition_columns disagrees with the trained model column order.")
        expected_columns = declared
    if config.get("source_manifest") is not None:
        if config.get("dataset_file"):
            raise ValueError("Choose dataset_file or source_manifest for inference, not both.")
        if "id_column" in config:
            raise ValueError("data.id_column is not used with source_manifest; set samples.id_column in the manifest.")
        manifest = read_input_spec(root, config["source_manifest"], paths)
        if manifest.get("schema") != "condition_batch_v1":
            raise ValueError("Condition-only manifest must use schema: condition_batch_v1.")
        samples, id_column, index_column = _sample_table(root, manifest, paths)
        indices = _indices(samples, index_column, len(samples))
        values, columns = _conditions(
            root, manifest.get("conditions"), samples, id_column, indices, paths, expected_columns
        )
        if values is None:
            raise ValueError("Condition manifest requires conditions.")
        return ConditionBatch(values, samples[id_column].astype(str).to_numpy(), columns)
    if root.is_file() and config.get("dataset_file") is not None:
        raise ValueError("Specify a condition file as local_path or dataset_file, not both.")
    path = root if root.is_file() else _path(root, config.get("dataset_file"), "dataset_file", paths)
    columns = _columns(expected_columns, "condition_columns")
    if path.suffix.lower() == ".npy":
        if "id_column" in config:
            raise ValueError(
                "data.id_column is available for CSV/table condition inputs; NPY uses positional sample IDs."
            )
        values = _array(path)
        ids = np.asarray([f"sample_{index:06d}" for index in range(len(values))])
        return ConditionBatch(values, ids, columns)
    frame = _table(path)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing condition columns: {missing}.")
    try:
        values = frame[list(columns)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inference conditions must be numeric.") from exc
    id_column = config.get("id_column", "sample_id")
    if not isinstance(id_column, str) or not id_column.strip():
        raise ValueError("data.id_column must be a non-empty string.")
    if "id_column" in config and id_column not in frame:
        raise ValueError(f"Configured ID column {id_column!r} is missing from inference conditions.")
    ids = (
        frame[id_column].astype(str).to_numpy()
        if id_column in frame
        else np.asarray([f"sample_{i:06d}" for i in range(len(frame))])
    )
    metadata = {str(name): frame[name].to_numpy() for name in frame if name not in {*columns, id_column}}
    return ConditionBatch(values, ids, columns, metadata)
