"""Convert explicitly selected field formats once before any numerical workflow."""

from copy import deepcopy

from pathlib import Path
import shutil

from ml_platform_core.io import read_json, write_json
from ml_platform_core.paths import resolve_path_within

from .data import import_csv_field_dataset, import_legacy_npy_dataset
from .data.input_spec import field_input_spec
from .data.scattered_import import import_scattered_field_dataset


def convert_field_input(data, destination, *, format_name=None, condition_columns=None, geometry=None):
    if format_name is not None and data.get("format", format_name) != format_name:
        raise ValueError("data.format and import.format must select the same input format")
    settings = {**data, **({"format": format_name} if format_name is not None else {})}
    selected, reference = field_input_spec(settings)
    source = settings["local_path"]
    if selected == "legacy-npy":
        if isinstance(reference, str) and reference != "manifest.json":
            raise ValueError("legacy-npy reads the source manifest.json; use data.geometry for coordinate overrides")
        if isinstance(reference, dict) and set(reference) - {"schema", "geometry", "conditions"}:
            raise ValueError("legacy-npy supports geometry and condition-column overrides; use npy for new inputs")
        return import_legacy_npy_dataset(
            source,
            destination,
            condition_columns=condition_columns
            or settings.get("condition_columns")
            or settings.get("conditions", {}).get("columns"),
            geometry_override=geometry or settings.get("geometry"),
        )
    importers = {"csv": import_csv_field_dataset, "scattered": import_scattered_field_dataset}
    if selected not in importers:
        raise ValueError("Input conversion requires data.format: csv, scattered, or legacy-npy")
    return importers[selected](source, destination, source_manifest=reference)


def prepare_field_input(cfg, result):
    if cfg["task"] in {"mode_decomp_infer", "mode_decomp_import"} or cfg.get("stage_inputs", {}).get(
        "canonical_dataset"
    ):
        return cfg
    selected, reference = field_input_spec(cfg["data"])
    if selected == "npy":
        source = cfg["data"].get("local_path")
        if source and Path(source).is_file():
            if cfg["data"].get("source_manifest") or isinstance(reference, dict):
                raise ValueError("Pass a Dataset root plus source_manifest, or a manifest file, not both.")
            manifest = Path(source).expanduser().resolve()
            prepared = deepcopy(cfg)
            prepared["data"].update(local_path=str(manifest.parent), source_manifest=manifest.name)
            return prepared
        return cfg
    if not cfg["data"].get("local_path"):
        raise ValueError("Input conversion requires data.local_path")
    manifest = convert_field_input(cfg["data"], result.run_dir / "input")
    result.artifacts["input_dataset_manifest"] = manifest
    prepared = deepcopy(cfg)
    prepared.setdefault("_meta", {})["input_conversion"] = {
        "source": deepcopy(cfg["data"]),
        "dataset_manifest": str(manifest),
    }
    mask = prepared.get("analysis", {}).get("observation_mask")
    if mask is not None:
        source_mask = resolve_path_within(
            Path(cfg["data"]["local_path"]).expanduser().resolve(),
            mask,
            setting="analysis.observation_mask",
            root_description="the original Dataset root",
        )
        # Keep original array row order; load_observation_mask aligns it exactly once.
        target = manifest.parent / "analysis_observation_mask.npy"
        shutil.copyfile(source_mask, target)
        prepared["analysis"]["observation_mask"] = target.name
    prepared["data"] = {"local_path": str(manifest.parent), "source_manifest": manifest.name}
    return prepared


def snapshot_dataset(data, destination):
    """Copy the canonical input's referenced files only, preserving revision identity."""
    from .data.loading import load_field_dataset

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Canonical snapshot already exists: {destination}")
    paths = []
    dataset = load_field_dataset(data, _source_paths=paths)
    return _copy_snapshot(data, destination, paths, dataset.revision)


def _copy_snapshot(data, destination, paths, revision):
    """Write a snapshot from this operation's already loaded source references."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Canonical snapshot already exists: {destination}")
    source = Path(data.get("local_path", data.get("path"))).expanduser().resolve()
    source_file = source.name if source.is_file() else None
    if source_file:
        source = source.parent
    destination.mkdir(parents=True)
    try:
        payload = destination / "dataset"
        payload.mkdir()
        for path in sorted(set(paths)):
            relative = path.relative_to(source)
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        settings = deepcopy(data)
        settings["local_path"] = payload.name
        settings.pop("path", None)
        if source_file:
            settings["source_manifest"] = source_file
        write_json(
            {"schema": "canonical_stage_input_v1", "data": settings, "dataset_revision": revision},
            destination / "stage_input.json",
        )
    except (OSError, ValueError, TypeError):
        shutil.rmtree(destination)
        raise
    return destination


def stage_dataset_config(reference):
    root = Path(reference).expanduser().resolve()
    record = read_json(root / "stage_input.json")
    if record.get("schema") != "canonical_stage_input_v1":
        raise ValueError("Expected a canonical Dataset snapshot from prepare_fields")
    settings = record["data"]
    if settings.get("local_path") is not None:
        root = resolve_path_within(
            root,
            settings["local_path"],
            setting="canonical Dataset data.local_path",
            root_description="the canonical snapshot",
        )
    return {**settings, "local_path": str(root)}, record["dataset_revision"]
