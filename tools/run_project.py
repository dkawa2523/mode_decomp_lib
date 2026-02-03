"""Run train/predict/reconstruct/eval/viz for a single project directory."""
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
        project = output_cfg.get("project", None)
        if project:
            return str(project).strip()
        tag = output_cfg.get("tag", None)
        if tag:
            return str(tag).strip()
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
    parser.add_argument("--steps", default="train,predict,reconstruct,eval,viz")
    args = parser.parse_args(argv)

    base_path = Path(args.config).expanduser()
    if not base_path.is_absolute():
        base_path = (Path.cwd() / base_path).resolve()

    base_cfg = _load_yaml(base_path)
    project = _project_name(base_cfg, args.project, fallback=base_path.stem)
    output_cfg = dict(base_cfg.get("output", {}) or {})
    output_root = str(output_cfg.get("root", "runs")).strip() or "runs"
    output_cfg["project"] = project
    base_cfg["output"] = output_cfg

    project_dir = PROJECT_ROOT / output_root / project

    steps = [step.strip() for step in args.steps.split(",") if step.strip()]
    if not steps:
        raise ValueError("steps must be non-empty")

    # train
    if "train" in steps:
        train_cfg = dict(base_cfg)
        train_cfg["task"] = "train"
        train_path = PROJECT_ROOT / "work" / f"run_{project}_train.yaml"
        _write_yaml(train_path, train_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(train_path)])

    train_dir = _task_run_dir(project_dir, "train")

    if "predict" in steps:
        predict_cfg = dict(base_cfg)
        predict_cfg["task"] = "predict"
        task_params = dict(predict_cfg.get("task_params", {}) or {})
        task_params["train_run_dir"] = str(train_dir)
        predict_cfg["task_params"] = task_params
        predict_path = PROJECT_ROOT / "work" / f"run_{project}_predict.yaml"
        _write_yaml(predict_path, predict_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(predict_path)])

    predict_dir = _task_run_dir(project_dir, "predict")

    if "reconstruct" in steps:
        recon_cfg = dict(base_cfg)
        recon_cfg["task"] = "reconstruct"
        task_params = dict(recon_cfg.get("task_params", {}) or {})
        task_params["train_run_dir"] = str(train_dir)
        task_params["predict_run_dir"] = str(predict_dir)
        recon_cfg["task_params"] = task_params
        recon_path = PROJECT_ROOT / "work" / f"run_{project}_reconstruct.yaml"
        _write_yaml(recon_path, recon_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(recon_path)])

    reconstruct_dir = _task_run_dir(project_dir, "reconstruct")

    if "eval" in steps:
        eval_cfg = dict(base_cfg)
        eval_cfg["task"] = "eval"
        task_params = dict(eval_cfg.get("task_params", {}) or {})
        task_params["train_run_dir"] = str(train_dir)
        task_params["predict_run_dir"] = str(predict_dir)
        task_params["reconstruct_run_dir"] = str(reconstruct_dir)
        eval_cfg["task_params"] = task_params
        eval_path = PROJECT_ROOT / "work" / f"run_{project}_eval.yaml"
        _write_yaml(eval_path, eval_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(eval_path)])

    if "viz" in steps:
        viz_cfg = dict(base_cfg)
        viz_cfg["task"] = "viz"
        task_params = dict(viz_cfg.get("task_params", {}) or {})
        task_params["train_run_dir"] = str(train_dir)
        task_params["predict_run_dir"] = str(predict_dir)
        task_params["reconstruct_run_dir"] = str(reconstruct_dir)
        viz_cfg["task_params"] = task_params
        viz_path = PROJECT_ROOT / "work" / f"run_{project}_viz.yaml"
        _write_yaml(viz_path, viz_cfg)
        _run(["python3", "-m", "mode_decomp_ml.run", "--config", str(viz_path)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
