"""Convert explicitly selected field formats once before any numerical workflow."""

from copy import deepcopy
from dataclasses import replace

from pathlib import Path
import shutil

from ml_platform_core.artifacts import file_hash
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
    reference = cfg.get("stage_inputs", {}).get("canonical_dataset")
    if reference is not None:
        data, _ = stage_dataset_config(reference)
        prepared = deepcopy(cfg)
        prepared["data"] = data
        return prepared
    prepared = _prepare_target_input(cfg, result)
    source = cfg["data"].get("input")
    if source is None or source.get("kind") != "field":
        return prepared
    source_data = {key: value for key, value in source.items() if key != "kind"}
    source_data.setdefault("local_path", cfg["data"].get("local_path"))
    source_cfg = {"task": "mode_decomp_pipeline", "data": source_data}
    source_result = replace(result, run_dir=result.run_dir / "source", artifacts={})
    source_prepared = _prepare_target_input(source_cfg, source_result)
    if source_prepared is not source_cfg:
        prepared = deepcopy(prepared)
        prepared["data"]["input"] = {"kind": "field", **source_prepared["data"]}
        if "input_conversion" in source_prepared.get("_meta", {}):
            prepared.setdefault("_meta", {})["source_input_conversion"] = source_prepared["_meta"]["input_conversion"]
        result.artifacts.update({f"source_{key}": value for key, value in source_result.artifacts.items()})
    return prepared


def _prepare_target_input(cfg, result):
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
    if "input" in cfg["data"]:
        source = deepcopy(cfg["data"]["input"])
        source.setdefault("local_path", cfg["data"]["local_path"])
        prepared["data"]["input"] = source
    return prepared


def snapshot_dataset(data, destination):
    """Copy the canonical input's referenced files only, preserving revision identity."""
    from .data.loading import load_field_dataset

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Canonical snapshot already exists: {destination}")
    paths = []
    dataset = load_field_dataset(data, _source_paths=paths)
    from .inputs import load_surrogate_inputs

    input_paths = []
    inputs = load_surrogate_inputs(data, target=dataset, _source_paths=input_paths)
    return _copy_snapshot(
        data,
        destination,
        paths,
        dataset.revision,
        input_paths=input_paths,
        input_revision=None if inputs is None else inputs.revision,
    )


def _copy_references(source, destination, paths):
    """Keep referenced field paths and therefore their Dataset revision unchanged."""
    for path in sorted(set(paths)):
        relative = Path(path).resolve().relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _snapshot_source(data, destination, paths):
    from .inputs import normalize_input_settings
    from .model_artifact import _info_path

    settings = normalize_input_settings(data)
    assert settings is not None
    source = Path(settings["local_path"])
    root_file = source.name if source.is_file() else None
    if root_file:
        source = source.parent
    payload = destination / "source"
    payload.mkdir()
    settings["local_path"] = payload.name
    integrity = {}
    if settings["kind"] == "field":
        _copy_references(source, payload, paths)
        if root_file:
            settings["source_manifest"] = root_file
    else:
        coefficients = resolve_path_within(
            source, settings["path"], setting="data.input.path", root_description="the coefficient input root"
        )
        model = Path(settings["model_artifact"])
        info = Path(settings.get("info_path") or _info_path(model))
        for name, original in (("coefficients.npz", coefficients), ("model.joblib", model), ("model_info.json", info)):
            target = payload / name
            shutil.copyfile(original, target)
            if name != "coefficients.npz":
                integrity[target.relative_to(destination).as_posix()] = file_hash(target)
        settings.update(path="coefficients.npz", model_artifact="model.joblib", info_path="model_info.json")
    return settings, integrity


def _copy_snapshot(data, destination, paths, revision, *, input_paths=None, input_revision=None):
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
        _copy_references(source, payload, paths)
        settings = deepcopy(data)
        settings["local_path"] = payload.name
        settings.pop("path", None)
        if source_file:
            settings["source_manifest"] = source_file
        record = {"schema": "canonical_stage_input_v1", "data": settings, "dataset_revision": revision}
        if data.get("input") is not None:
            if not input_revision or input_paths is None:
                raise ValueError("Source snapshots require prepared source references and input_revision")
            settings["input"], integrity = _snapshot_source(data, destination, input_paths)
            record.update(schema="canonical_stage_input_v2", input_revision=input_revision, input_integrity=integrity)
        write_json(record, destination / "stage_input.json")
    except (OSError, ValueError, TypeError):
        shutil.rmtree(destination)
        raise
    return destination


def stage_dataset_config(reference):
    root = Path(reference).expanduser().resolve()
    record = read_json(root / "stage_input.json")
    if record.get("schema") not in {"canonical_stage_input_v1", "canonical_stage_input_v2"}:
        raise ValueError("Expected a canonical Dataset snapshot from prepare_fields")
    settings = deepcopy(record["data"])
    if record["schema"] == "canonical_stage_input_v1" and settings.get("input") is not None:
        raise ValueError("Legacy canonical snapshots cannot declare an unverified source input")
    target = root
    if settings.get("local_path") is not None:
        target = resolve_path_within(
            root,
            settings["local_path"],
            setting="canonical Dataset data.local_path",
            root_description="the canonical snapshot",
        )
    if record["schema"] == "canonical_stage_input_v2":
        source = settings.get("input")
        if (
            not isinstance(source, dict)
            or not isinstance(record.get("input_revision"), str)
            or not record["input_revision"]
        ):
            raise ValueError("Canonical source snapshot requires input settings and input_revision")
        source_root = source.get("local_path")
        if not isinstance(source_root, str) or not source_root.strip():
            raise ValueError("Canonical source snapshot requires a non-empty relative local_path")
        source["local_path"] = str(
            resolve_path_within(
                root,
                source_root,
                setting="canonical source local_path",
                root_description="the canonical snapshot",
            )
        )
        if source.get("kind") == "coefficients":
            for key in ("path", "model_artifact", "info_path"):
                reference = source.get(key)
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"Canonical coefficient snapshot requires a non-empty relative {key}")
                resolved = resolve_path_within(
                    Path(source["local_path"]),
                    reference,
                    setting=f"canonical source {key}",
                    root_description="the canonical source",
                )
                if key != "path":
                    source[key] = str(resolved)
        integrity = record.get("input_integrity")
        if not isinstance(integrity, dict):
            raise ValueError("Canonical source snapshot requires integrity records")
        if source.get("kind") == "coefficients" and not {
            Path(source[key]).relative_to(root).as_posix() for key in ("model_artifact", "info_path")
        }.issubset(integrity):
            raise ValueError("Canonical coefficient snapshot requires model and sidecar integrity records")
        for name, expected in integrity.items():
            path = resolve_path_within(
                root, name, setting="canonical source integrity", root_description="the canonical snapshot"
            )
            if file_hash(path) != expected:
                raise ValueError("Canonical source model content differs from its prepared integrity record")
    return {**settings, "local_path": str(target)}, record["dataset_revision"]
