"""run.yaml entrypoint adapter (non-Hydra)."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

from mode_decomp_ml.cli import run as hydra_cli
from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.pipeline import RunDirManager, resolve_path

_LOGGER = logging.getLogger(__name__)


def _configure_logging(cfg: Mapping[str, Any]) -> None:
    if logging.getLogger().handlers:
        return
    level = logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging_cfg = cfg_get(cfg, "logging", None)
    if isinstance(logging_cfg, Mapping):
        level_name = str(logging_cfg.get("level", "")).upper()
        if level_name:
            level = getattr(logging, level_name, level)
        fmt = str(logging_cfg.get("format", fmt)) or fmt
    logging.basicConfig(level=level, format=fmt)

_TASK_ALIASES = {
    "bench": "benchmark",
    "benchmark": "benchmark",
    "doctor": "doctor",
    "eval": "eval",
    "evaluate": "eval",
    "leaderboard": "leaderboard",
    "predict": "predict",
    "preprocess": "preprocess",
    "reconstruct": "reconstruct",
    "train": "train",
    "viz": "viz",
    "visualize": "viz",
}


def _load_yaml(path: Path) -> MutableMapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("run.yaml must be a mapping")
    return dict(data)


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value


def _normalize_task(task_value: Any) -> str:
    if task_value is None:
        raise ValueError("task is required in run.yaml")
    task = str(task_value).strip().lower()
    if not task:
        raise ValueError("task must be non-empty in run.yaml")
    return _TASK_ALIASES.get(task, task)


def _normalize_dataset(dataset_value: Any) -> tuple[str | None, Mapping[str, Any]]:
    if dataset_value is None:
        raise ValueError("dataset is required in run.yaml")
    if isinstance(dataset_value, str):
        return "npy_dir", {"root": dataset_value}
    if isinstance(dataset_value, Mapping):
        dataset_cfg = dict(dataset_value)
        name = dataset_cfg.get("name")
        if name:
            return str(name), dataset_cfg
        if "root" in dataset_cfg:
            return "npy_dir", dataset_cfg
        return None, dataset_cfg
    raise ValueError("dataset must be a string or mapping in run.yaml")


def _normalize_pipeline(pipeline_value: Any) -> Mapping[str, Any]:
    if pipeline_value is None:
        return {}
    if not isinstance(pipeline_value, Mapping):
        raise ValueError("pipeline must be a mapping in run.yaml")
    return dict(pipeline_value)


def _normalize_output(output_value: Any) -> Mapping[str, Any]:
    if output_value is None:
        return {}
    if not isinstance(output_value, Mapping):
        raise ValueError("output must be a mapping in run.yaml")
    return dict(output_value)


def _normalize_params(params_value: Any) -> Mapping[str, Any]:
    if params_value is None:
        return {}
    if not isinstance(params_value, Mapping):
        raise ValueError("params must be a mapping in run.yaml")
    return dict(params_value)


def _write_snapshot(run_dir: Path, cfg: Mapping[str, Any]) -> None:
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    with (hydra_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)


def _build_config(run_cfg: Mapping[str, Any]) -> tuple[MutableMapping[str, Any], str, Mapping[str, Any]]:
    task_name = _normalize_task(run_cfg.get("task"))
    dataset_group, dataset_updates = _normalize_dataset(run_cfg.get("dataset"))
    pipeline_cfg = _normalize_pipeline(run_cfg.get("pipeline"))
    output_cfg = _normalize_output(run_cfg.get("output"))
    params_cfg = _normalize_params(run_cfg.get("params"))
    task_params = _normalize_params(run_cfg.get("task_params"))

    overrides = [f"task={task_name}"]
    if dataset_group:
        overrides.append(f"dataset={dataset_group}")

    decomposer = pipeline_cfg.get("decomposer") or pipeline_cfg.get("decompose")
    codec = pipeline_cfg.get("codec")
    coeff_post = pipeline_cfg.get("coeff_post")
    model = pipeline_cfg.get("model")

    if decomposer:
        overrides.append(f"decompose={decomposer}")
    if codec:
        overrides.append(f"codec={codec}")
    if coeff_post:
        overrides.append(f"coeff_post={coeff_post}")
    if model:
        overrides.append(f"model={model}")

    cfg = hydra_cli._compose_config(overrides)

    if dataset_updates:
        _deep_update(cfg.setdefault("dataset", {}), dataset_updates)

    seed = run_cfg.get("seed", None)
    if seed is not None:
        cfg["seed"] = seed

    output_root = str(output_cfg.get("root", "runs")).strip() or "runs"
    tag = str(output_cfg.get("tag", "default")).strip() or "default"
    project = output_cfg.get("project", None)
    cfg["output_dir"] = output_root
    cfg["tag"] = tag
    run_id = str(output_cfg.get("run_id", "")).strip()
    if run_id:
        cfg["run_id"] = run_id
    if project is not None:
        project_name = str(project).strip()
        if project_name:
            cfg["project_dir"] = str(Path(output_root) / project_name)

    if pipeline_cfg:
        cfg["pipeline"] = dict(pipeline_cfg)
    if task_params:
        _deep_update(cfg.setdefault("task", {}), task_params)
    if params_cfg:
        _deep_update(cfg, params_cfg)

    cfg = hydra_cli._resolve_interpolations(cfg)
    run_dir = RunDirManager(cfg).ensure()

    return cfg, str(run_dir), {
        "task": task_name,
        "decomposer": decomposer,
        "coeff_post": coeff_post,
        "model": model,
        "codec": codec,
        "tag": tag,
        "run_id": cfg.get("run_id"),
    }


def _print_dry_run(summary: Mapping[str, Any], run_dir: str, cfg: Mapping[str, Any]) -> None:
    _LOGGER.info("Dry run summary:")
    _LOGGER.info("  task: %s", summary.get("task"))
    if summary.get("decomposer"):
        _LOGGER.info("  decomposer: %s", summary.get("decomposer"))
    if summary.get("coeff_post"):
        _LOGGER.info("  coeff_post: %s", summary.get("coeff_post"))
    if summary.get("model"):
        _LOGGER.info("  model: %s", summary.get("model"))
    if summary.get("codec"):
        _LOGGER.info("  codec: %s", summary.get("codec"))
    if summary.get("run_id"):
        _LOGGER.info("  run_id: %s", summary.get("run_id"))
    _LOGGER.info("  run_dir: %s", run_dir)
    _LOGGER.info("Resolved config:\n%s", yaml.safe_dump(dict(cfg), sort_keys=False))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run entrypoint using run.yaml")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to run.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_cfg_path = Path(args.config).expanduser()
    if not run_cfg_path.is_absolute():
        run_cfg_path = (Path.cwd() / run_cfg_path).resolve()
    run_cfg = _load_yaml(run_cfg_path)

    cfg, run_dir, summary = _build_config(run_cfg)
    _configure_logging(cfg)

    if args.dry_run:
        _print_dry_run(summary, run_dir, cfg)
        return 0

    run_dir_path = resolve_path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    run_yaml_out = run_dir_path / "run.yaml"
    with run_yaml_out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(run_cfg, fh, sort_keys=False)
    _write_snapshot(run_dir_path, cfg)

    os.chdir(run_dir_path)

    task_name = hydra_cli._task_name(cfg)
    module = hydra_cli._load_task_module(task_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"processes.{task_name} has no main()")
    return int(hydra_cli._call_main(module.main, cfg))


if __name__ == "__main__":
    raise SystemExit(main())
