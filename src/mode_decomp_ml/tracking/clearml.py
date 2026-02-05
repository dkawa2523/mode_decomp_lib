"""ClearML integration (optional)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:  # optional dependency for config conversion
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback when OmegaConf is missing
    OmegaConf = None

from mode_decomp_ml.pipeline import cfg_get, resolve_path, task_name


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


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags] if tags.strip() else []
    if isinstance(tags, Sequence):
        return [str(tag) for tag in tags if str(tag).strip()]
    return [str(tags)]


def _upload_json(task: Any, name: str, path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    task.upload_artifact(name, artifact_object=payload)


def _upload_config_yaml(task: Any, run_dir: Path) -> None:
    config_path = run_dir / "configuration" / "run.yaml"
    if not config_path.exists():
        config_path = run_dir / "configuration" / "resolved.yaml"
    if not config_path.exists():
        return
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    task.upload_artifact("config_yaml", artifact_object=payload)


def _upload_model(task: Any, run_dir: Path) -> None:
    model_dir = run_dir / "model"
    if not model_dir.exists():
        model_dir = run_dir / "artifacts" / "model"
        if not model_dir.exists():
            return
    for filename in ("model.pkl", "model.pth", "model.pt"):
        path = model_dir / filename
        if path.exists():
            task.upload_artifact("model", artifact_object=str(path))
            return


def _upload_plots(task: Any, run_dir: Path) -> None:
    viz_dir = run_dir / "plots"
    if not viz_dir.exists():
        return
    for path in sorted(viz_dir.rglob("*.png")):
        rel = str(path.relative_to(run_dir)).replace("/", "_")
        task.upload_artifact(f"plot_{rel}", artifact_object=str(path))


def maybe_log_run(cfg: Mapping[str, Any], run_dir: str | Path) -> None:
    clearml_cfg = cfg_get(cfg, "clearml", None)
    enabled = bool(cfg_get(clearml_cfg, "enabled", False))
    # CONTRACT: ClearML logging only runs when enabled.
    if not enabled:
        return
    try:
        from clearml import Task
    except Exception as exc:
        raise RuntimeError("clearml.enabled=true but clearml is not installed") from exc

    run_dir_path = resolve_path(run_dir)
    project = str(cfg_get(clearml_cfg, "project", "mode-decomp-ml"))
    name = str(cfg_get(clearml_cfg, "task_name", "")).strip()
    if not name:
        name = f"{task_name(cfg)}:{run_dir_path.name}"

    task = Task.init(project_name=project, task_name=name)
    tags = _normalize_tags(cfg_get(clearml_cfg, "tags", None))
    if tags:
        task.add_tags(tags)

    task.connect(_to_container(cfg), name="config")
    _upload_config_yaml(task, run_dir_path)
    metrics_path = run_dir_path / "outputs" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = run_dir_path / "metrics" / "metrics.json"
    _upload_json(task, "metrics", metrics_path)
    _upload_json(task, "manifest_run", run_dir_path / "outputs" / "manifest_run.json")
    _upload_model(task, run_dir_path)
    _upload_plots(task, run_dir_path)
