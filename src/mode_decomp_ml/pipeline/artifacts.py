"""Artifact layout helpers for flat run directories."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

try:  # optional dependency for config conversion
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback when OmegaConf is missing
    OmegaConf = None

from mode_decomp_ml.pipeline.utils import ensure_dir, read_json, resolve_path, write_json

RUN_YAML = "run.yaml"
MANIFEST_JSON = "manifest_run.json"
METRICS_JSON = "metrics.json"
PREDS_NPZ = "preds.npz"


def _to_container(obj: Any) -> Any:
    if OmegaConf is not None:
        try:
            if OmegaConf.is_config(obj):
                return OmegaConf.to_container(obj, resolve=True)
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(k): _to_container(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_container(v) for v in obj]
    return obj


def _resolve_run_dir(run_dir: str | Path) -> Path:
    return resolve_path(str(run_dir))


def ensure_standard_dirs(run_dir: str | Path) -> dict[str, Path]:
    run_root = _resolve_run_dir(run_dir)
    model_dir = ensure_dir(run_root / "model")
    states_dir = ensure_dir(run_root / "states")
    figures_dir = ensure_dir(run_root / "figures")
    tables_dir = ensure_dir(run_root / "tables")
    return {
        "run_dir": run_root,
        "model_dir": model_dir,
        "states_dir": states_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
    }


class ArtifactWriter:
    """Writes artifacts into the flat run layout."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = _resolve_run_dir(run_dir)
        self.model_dir = self.run_dir / "model"
        self.states_dir = self.run_dir / "states"
        self.figures_dir = self.run_dir / "figures"
        self.tables_dir = self.run_dir / "tables"

    def ensure_layout(self) -> None:
        ensure_dir(self.model_dir)
        ensure_dir(self.states_dir)
        ensure_dir(self.figures_dir)
        ensure_dir(self.tables_dir)

    def write_run_yaml(self, cfg: Mapping[str, Any], *, overwrite: bool = False) -> Path:
        path = self.run_dir / RUN_YAML
        if path.exists() and not overwrite:
            return path
        payload = _to_container(cfg)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        return path

    def write_manifest(
        self,
        *,
        meta: Mapping[str, Any] | None = None,
        dataset_meta: Mapping[str, Any] | None = None,
        preds_meta: Mapping[str, Any] | None = None,
        steps: Sequence[Mapping[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path:
        manifest: dict[str, Any] = {}
        if meta is not None:
            manifest["meta"] = dict(meta)
        if dataset_meta is not None:
            manifest["dataset_meta"] = dict(dataset_meta)
        if preds_meta is not None:
            manifest["preds_meta"] = dict(preds_meta)
        if steps is not None:
            manifest["steps"] = [dict(step) for step in steps]
        if extra:
            for key, value in extra.items():
                manifest[key] = value
        return write_json(self.run_dir / MANIFEST_JSON, manifest)

    def write_metrics(self, metrics: Mapping[str, Any]) -> Path:
        return write_json(self.run_dir / METRICS_JSON, metrics)

    def write_preds(self, arrays: Mapping[str, Any]) -> Path:
        payload = {name: np.asarray(value) for name, value in arrays.items() if value is not None}
        path = self.run_dir / PREDS_NPZ
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        return path


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    run_root = _resolve_run_dir(run_dir)
    path = run_root / MANIFEST_JSON
    if path.exists():
        return read_json(path)
    legacy = run_root / "meta.json"
    if legacy.exists():
        return read_json(legacy)
    return {}


def read_dataset_meta(run_dir: str | Path) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    dataset_meta = manifest.get("dataset_meta")
    if isinstance(dataset_meta, Mapping):
        return dict(dataset_meta)
    run_root = _resolve_run_dir(run_dir)
    legacy = run_root / "artifacts" / "dataset_meta.json"
    if legacy.exists():
        return read_json(legacy)
    return {}


def read_preds_meta(run_dir: str | Path) -> dict[str, Any]:
    manifest = read_manifest(run_dir)
    preds_meta = manifest.get("preds_meta")
    if isinstance(preds_meta, Mapping):
        return dict(preds_meta)
    run_root = _resolve_run_dir(run_dir)
    legacy = run_root / "preds" / "preds_meta.json"
    if legacy.exists():
        return read_json(legacy)
    return {}


def read_field_std_meta(run_dir: str | Path) -> dict[str, Any]:
    preds_meta = read_preds_meta(run_dir)
    field_std_meta = preds_meta.get("field_std_meta") if isinstance(preds_meta, Mapping) else None
    if isinstance(field_std_meta, Mapping):
        return dict(field_std_meta)
    run_root = _resolve_run_dir(run_dir)
    legacy = run_root / "preds" / "field_std_meta.json"
    if legacy.exists():
        return read_json(legacy)
    return {}


def load_preds_npz(run_dir: str | Path) -> dict[str, np.ndarray]:
    run_root = _resolve_run_dir(run_dir)
    path = run_root / PREDS_NPZ
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def load_coeff_predictions(
    run_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, Mapping[str, Any]]:
    run_root = _resolve_run_dir(run_dir)
    arrays = load_preds_npz(run_root)
    if arrays:
        coeff = arrays.get("coeff_mean")
        if coeff is None:
            coeff = arrays.get("coeff")
        if coeff is None:
            raise ValueError("preds.npz missing coeff/coeff_mean")
        coeff_std = arrays.get("coeff_std")
        return coeff, coeff_std, read_preds_meta(run_root)

    pred_root = run_root / "preds"
    coeff_path = pred_root / "coeff_mean.npy"
    if not coeff_path.exists():
        coeff_path = pred_root / "coeff.npy"
    coeff = np.load(coeff_path)
    std_path = pred_root / "coeff_std.npy"
    coeff_std = np.load(std_path) if std_path.exists() else None
    meta_path = pred_root / "preds_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return coeff, coeff_std, meta


def load_field_predictions(run_dir: str | Path) -> np.ndarray:
    run_root = _resolve_run_dir(run_dir)
    arrays = load_preds_npz(run_root)
    if arrays:
        field = arrays.get("field")
        if field is None:
            raise ValueError("preds.npz missing field")
        return field
    return np.load(run_root / "preds" / "field.npy")


def load_field_std(run_dir: str | Path) -> tuple[np.ndarray | None, Mapping[str, Any]]:
    run_root = _resolve_run_dir(run_dir)
    arrays = load_preds_npz(run_root)
    if arrays and "field_std" in arrays:
        return arrays["field_std"], read_field_std_meta(run_root)
    std_path = run_root / "preds" / "field_std.npy"
    if not std_path.exists():
        return None, {}
    std = np.load(std_path)
    meta_path = run_root / "preds" / "field_std_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return std, meta
