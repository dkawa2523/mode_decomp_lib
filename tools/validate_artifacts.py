#!/usr/bin/env python3
"""Artifact validator for run dirs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency through hydra
    yaml = None

CONFIG_CANDIDATES = (
    Path("run.yaml"),
    Path(".hydra/config.yaml"),
    Path("hydra/config.yaml"),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_config(run_dir: Path) -> Path | None:
    for rel in CONFIG_CANDIDATES:
        path = run_dir / rel
        if path.exists():
            return path
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Hydra configs.")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _task_name(cfg: Mapping[str, Any]) -> str:
    task_cfg = cfg.get("task")
    if isinstance(task_cfg, str):
        return task_cfg
    if isinstance(task_cfg, Mapping):
        name = task_cfg.get("name")
        if name:
            return str(name)
    return "unknown"


def _task_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    task_cfg = cfg.get("task")
    return task_cfg if isinstance(task_cfg, Mapping) else {}


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root / path


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest_run.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    legacy = run_dir / "meta.json"
    if legacy.exists():
        try:
            with legacy.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _check_meta(run_dir: Path, errors: list[str]) -> None:
    manifest = _load_manifest(run_dir)
    if not manifest:
        errors.append("manifest_run.json missing (or unreadable)")
        return
    meta = manifest.get("meta") if isinstance(manifest, Mapping) else None
    if not isinstance(meta, Mapping):
        meta = manifest
    seed = meta.get("seed")
    git = meta.get("git") if isinstance(meta, Mapping) else None
    commit = git.get("commit") if isinstance(git, Mapping) else None
    if seed is None:
        errors.append("manifest meta missing seed")
    if not commit:
        errors.append("manifest meta missing git.commit")


def _check_metrics(run_dir: Path, errors: list[str]) -> None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir / "metrics" / "metrics.json"
    if not metrics_path.exists():
        errors.append(f"metrics.json missing ({metrics_path})")


def _check_model_dir(run_dir: Path, errors: list[str], label: str) -> None:
    model_dir = run_dir / "model"
    if not model_dir.exists():
        model_dir = run_dir / "artifacts" / "model"
    if not model_dir.exists():
        errors.append(f"{label}: model dir missing ({model_dir})")
        return
    if not any(model_dir.iterdir()):
        errors.append(f"{label}: model dir empty ({model_dir})")


def _check_pred_files(
    run_dir: Path,
    names: Sequence[str],
    errors: list[str],
    label: str,
) -> None:
    preds_npz = run_dir / "preds.npz"
    if preds_npz.exists():
        return
    preds_dir = run_dir / "preds"
    if not preds_dir.exists():
        errors.append(f"{label}: preds dir missing ({preds_dir})")
        return
    for name in names:
        if (preds_dir / name).exists():
            return
    errors.append(f"{label}: missing preds file ({', '.join(names)}) in {preds_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1:
        print("Usage: python tools/validate_artifacts.py <run_dir>", file=sys.stderr)
        return 2

    run_dir = Path(args[0]).expanduser().resolve()
    errors: list[str] = []
    if not run_dir.exists():
        errors.append(f"run dir does not exist ({run_dir})")

    cfg: dict[str, Any] = {}
    config_path = _find_config(run_dir)
    if config_path is None:
        errors.append("hydra config missing (.hydra/config.yaml or hydra/config.yaml)")
    else:
        try:
            cfg = _load_yaml(config_path)
        except Exception as exc:
            errors.append(f"hydra config unreadable ({config_path}): {exc}")
            cfg = {}

    # CONTRACT: config + meta are required to keep runs reproducible.
    _check_meta(run_dir, errors)

    task_name = _task_name(cfg) if cfg else "unknown"
    task_cfg = _task_cfg(cfg) if cfg else {}
    project_root = _project_root()

    if task_name == "eval":
        _check_metrics(run_dir, errors)

    if task_name == "train":
        _check_model_dir(run_dir, errors, "train")
    elif task_name == "predict":
        _check_pred_files(run_dir, ("coeff.npy", "coeff_mean.npy"), errors, "predict")
    elif task_name == "reconstruct":
        _check_pred_files(run_dir, ("field.npy",), errors, "reconstruct")

    train_run_dir = task_cfg.get("train_run_dir")
    if train_run_dir:
        train_dir = _resolve_path(project_root, train_run_dir)
        _check_model_dir(train_dir, errors, "train_run_dir")

    predict_run_dir = task_cfg.get("predict_run_dir")
    if predict_run_dir:
        pred_dir = _resolve_path(project_root, predict_run_dir)
        _check_pred_files(pred_dir, ("coeff.npy", "coeff_mean.npy"), errors, "predict_run_dir")

    reconstruct_run_dir = task_cfg.get("reconstruct_run_dir")
    if reconstruct_run_dir:
        recon_dir = _resolve_path(project_root, reconstruct_run_dir)
        _check_pred_files(recon_dir, ("field.npy",), errors, "reconstruct_run_dir")

    if errors:
        print("[FAIL] Artifact validation failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("[PASS] Artifact validation passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
