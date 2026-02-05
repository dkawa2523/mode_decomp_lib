"""Run decomposition/preprocessing/train/inference for a single project directory."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, Mapping):
        raise ValueError("run.yaml must be a mapping")
    return dict(data)


def _project_name(cfg: Mapping[str, Any], override: str | None, fallback: str) -> str:
    if override:
        return override
    output_cfg = cfg.get("output", {})
    if isinstance(output_cfg, Mapping):
        name = output_cfg.get("name", None)
        if name:
            return str(name).strip()
    return fallback


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _task_run_dir(project_dir: Path, task: str) -> Path:
    return project_dir / task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a project suite under one output directory.")
    parser.add_argument("--config", required=True, help="Path to base run.yaml")
    parser.add_argument("--project", default=None, help="Project name override")
    parser.add_argument("--steps", default="decomposition,preprocessing,train,inference")
    args = parser.parse_args(argv)

    base_path = Path(args.config).expanduser()
    if not base_path.is_absolute():
        base_path = (Path.cwd() / base_path).resolve()

    base_cfg = _load_yaml(base_path)
    project = _project_name(base_cfg, args.project, fallback=base_path.stem)
    output_cfg = dict(base_cfg.get("output", {}) or {})
    output_root = str(output_cfg.get("root", "runs")).strip() or "runs"
    output_cfg["name"] = project
    base_cfg["output"] = output_cfg

    project_dir = PROJECT_ROOT / output_root / project

    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    if not steps:
        raise ValueError("steps must be non-empty")

    if "decomposition" in steps:
        decomposition_cfg = dict(base_cfg)
        decomposition_cfg["task"] = {"name": "decomposition"}
        decomposition_path = PROJECT_ROOT / "work" / f"run_{project}_decomposition.yaml"
        _write_yaml(decomposition_path, decomposition_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(decomposition_path)])

    decomposition_dir = _task_run_dir(project_dir, "decomposition")

    if "preprocessing" in steps:
        preprocessing_cfg = dict(base_cfg)
        preprocessing_cfg["task"] = {"name": "preprocessing", "decomposition_run_dir": str(decomposition_dir)}
        preprocessing_path = PROJECT_ROOT / "work" / f"run_{project}_preprocessing.yaml"
        _write_yaml(preprocessing_path, preprocessing_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(preprocessing_path)])

    preprocessing_dir = _task_run_dir(project_dir, "preprocessing")

    if "train" in steps:
        train_cfg = dict(base_cfg)
        train_cfg["task"] = {"name": "train", "preprocessing_run_dir": str(preprocessing_dir)}
        train_path = PROJECT_ROOT / "work" / f"run_{project}_train.yaml"
        _write_yaml(train_path, train_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(train_path)])

    train_dir = _task_run_dir(project_dir, "train")

    if "inference" in steps:
        inference_cfg = dict(base_cfg)
        inference_cfg["task"] = {
            "name": "inference",
            "decomposition_run_dir": str(decomposition_dir),
            "preprocessing_run_dir": str(preprocessing_dir),
            "train_run_dir": str(train_dir),
        }
        inference_path = PROJECT_ROOT / "work" / f"run_{project}_inference.yaml"
        _write_yaml(inference_path, inference_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(inference_path)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
