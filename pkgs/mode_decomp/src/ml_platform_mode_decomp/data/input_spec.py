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


def field_input_spec(data: Mapping[str, Any]) -> tuple[str, str | dict[str, Any]]:
    """Return the selected format and one manifest mapping or external path.

    The run's data section owns the same fields as a standalone manifest; users
    need not introduce another manifest file or schema key for local work.
    """
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping.")
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
        return deepcopy(dict(reference))
    path = resolve_path_within(root, reference, setting="source_manifest", root_description="the Dataset root")
    if not path.is_file():
        raise FileNotFoundError(f"source_manifest not found: {path}")
    paths.append(path)
    return load_yaml(path)
