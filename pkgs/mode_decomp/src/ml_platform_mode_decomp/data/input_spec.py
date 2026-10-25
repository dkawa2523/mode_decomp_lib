"""Resolve one field input description, inline or explicitly referenced."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ml_platform_core.config import load_yaml
from ml_platform_core.paths import resolve_path_within


_SCHEMAS = {
    "npy": "field_dataset_v1",
    "csv": "field_csv_import_v1",
    "scattered": "scattered_field_import_v1",
    "legacy-npy": "legacy_npy_dir",
}
_SPEC_KEYS = {
    "samples",
    "conditions",
    "fields",
    "geometry",
    "masks",
    "observations",
    "interpolation",
    "scattered",
    "imported_from",
    "generation",
    "schema",
}


def _known_keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} settings: {sorted(unknown, key=str)}")
    return dict(value)


def validate_data_settings(data: Mapping[str, Any], *, condition_only: bool = False) -> None:
    roots = {"local_path", "path", "source_manifest"}
    if condition_only:
        allowed = roots | {"dataset_file", "condition_columns", "id_column", "input"}
        if data.get("input") is not None:
            allowed = {"local_path", "input"}
    else:
        allowed = roots | _SPEC_KEYS | {"format", "clearml_dataset_id", "input"}
        if data.get("format") == "legacy-npy":
            allowed.add("condition_columns")
    _known_keys(data, allowed, "data")
    if "local_path" in data and "path" in data:
        raise ValueError("Specify data.local_path or data.path, not both.")


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError("Input specification must be a mapping.")
    schema = spec.get("schema")
    roots = {"schema", "samples", "conditions"}
    field_keys = {"layout", "kind", "components", "units"}
    if schema == "field_dataset_v1":
        roots |= {"fields", "geometry", "masks", "scattered", "imported_from", "generation"}
        field_keys.add("path")
    elif schema == "field_csv_import_v1":
        roots |= {"fields", "geometry", "masks"}
        field_keys |= {"path", "path_column", "columns", "coordinates", "coordinate_atol"}
    elif schema == "scattered_field_import_v1":
        roots |= {"fields", "geometry", "masks", "observations", "interpolation"}
        observations = spec.get("observations", {})
        if not isinstance(observations, Mapping):
            raise ValueError("observations must be a mapping.")
        _known_keys(
            observations,
            {"format", "coordinates", "values", "mask", "weights", "simplex_support"}
            if observations.get("format") == "npy"
            else {
                "format",
                "coordinates",
                "columns",
                "path",
                "path_column",
                "mask_column",
                "weight_column",
                "simplex_support",
            },
            "observations",
        )
    elif schema != "condition_batch_v1":
        return dict(spec)  # The selected reader reports an unsupported schema.
    validated = _known_keys(spec, roots, "input specification")
    if "fields" in spec:
        _known_keys(spec["fields"], field_keys, "fields")
    for key, allowed in (
        ("samples", {"path", "id_column", "index_column"}),
        ("masks", {"domain", "observation"}),
    ):
        if spec.get(key) is not None:
            _known_keys(spec[key], allowed, key)
    return validated


def field_input_spec(data: Mapping[str, Any]) -> tuple[str, str | dict[str, Any]]:
    """Return the selected format and one manifest mapping or external path.

    The run's data section owns the same fields as a standalone manifest; users
    need not introduce another manifest file or schema key for local work.
    """
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping.")
    validate_data_settings(data)
    format_name = data.get("format", "npy")
    if format_name not in _SCHEMAS:
        raise ValueError("data.format must be npy, csv, scattered, or legacy-npy.")
    inline = {key: deepcopy(data[key]) for key in _SPEC_KEYS if key in data}
    external = data.get("source_manifest")
    if external is not None:
        if not isinstance(external, (str, Path)) or not str(external).strip():
            raise ValueError("data.source_manifest must be a non-empty path; use direct data fields for inline input.")
        if inline:
            raise ValueError("Choose inline data fields or data.source_manifest; do not provide both.")
        return format_name, str(external)
    if inline:
        expected = _SCHEMAS[format_name]
        if "schema" in inline and inline["schema"] != expected:
            raise ValueError(f"Inline data schema disagrees with data.format: {format_name}.")
        inline["schema"] = expected
        return format_name, inline
    return format_name, {"npy": "field_dataset.yaml", "legacy-npy": "manifest.json"}.get(format_name, "import.yaml")


def read_input_spec(root: Path, reference: str | Mapping[str, Any], paths: list[Path]) -> dict[str, Any]:
    """Read external metadata or copy inline metadata without temporary files."""
    if isinstance(reference, Mapping):
        return _validate_spec(deepcopy(dict(reference)))
    path = resolve_path_within(root, reference, setting="source_manifest", root_description="the Dataset root")
    if not path.is_file():
        raise FileNotFoundError(f"source_manifest not found: {path}")
    paths.append(path)
    return _validate_spec(load_yaml(path))
