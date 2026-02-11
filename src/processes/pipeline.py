"""Process entrypoint: pipeline."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    PROJECT_ROOT,
    StepRecorder,
    artifact_ref,
    cfg_get,
    resolve_run_dir,
)
from mode_decomp_ml.pipeline.process_base import finalize_run, init_run
from processes import decomposition as decomposition_process
from processes import preprocessing as preprocessing_process
from processes import train as train_process

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None

CONFIG_DIR = PROJECT_ROOT / "configs"


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for pipeline")
    return task_cfg


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


def _deep_update(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            dst[key] = _deep_update(dict(dst[key]), value)
        else:
            dst[key] = value
    return dst


def _resolve_group_config_path(group: str, name: str) -> Path:
    if name.startswith("/"):
        return CONFIG_DIR / f"{name.lstrip('/')}.yaml"
    return CONFIG_DIR / group / f"{name}.yaml"


def _load_yaml_with_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    defaults = cfg.pop("defaults", [])
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            if item.strip() == "_self_":
                continue
            if item.startswith("/"):
                sub_path = CONFIG_DIR / f"{item.lstrip('/')}.yaml"
            else:
                sub_path = CONFIG_DIR / f"{item}.yaml"
            sub_cfg = _load_yaml_with_defaults(sub_path)
            merged = _deep_update(merged, sub_cfg)
            continue
        if isinstance(item, Mapping):
            group, name = next(iter(item.items()))
            sub_path = _resolve_group_config_path(str(group), str(name))
            sub_cfg = _load_yaml_with_defaults(sub_path)
            merged = _deep_update(merged, sub_cfg)
            continue
    merged = _deep_update(merged, cfg)
    return merged


def _load_group_cfg(group: str, name: str) -> dict[str, Any]:
    path = _resolve_group_config_path(group, name)
    return _load_yaml_with_defaults(path)


def _resolve_seed_tokens(value: Any, *, seed: int | None) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _resolve_seed_tokens(v, seed=seed) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_seed_tokens(v, seed=seed) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_seed_tokens(v, seed=seed) for v in value)
    if isinstance(value, str):
        if value.strip() == "${seed}":
            return seed
        if seed is None:
            return value
        return value.replace("${seed}", str(seed))
    return value


def _normalize_list(value: Any, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _output_root(cfg: Mapping[str, Any]) -> Path:
    output_cfg = cfg_get(cfg, "output", {}) or {}
    root = str(cfg_get(output_cfg, "root", "runs")).strip() or "runs"
    return Path(root)


def _output_name(cfg: Mapping[str, Any]) -> str:
    output_cfg = cfg_get(cfg, "output", {}) or {}
    return str(cfg_get(output_cfg, "name", "default")).strip() or "default"


def _combo_name(base: str, *parts: str) -> str:
    suffix = "__".join(part for part in parts if part)
    return f"{base}__{suffix}" if suffix else base


def _write_leaderboard(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path,
    output_md: Path | None = None,
    sort_by: str | None = None,
) -> None:
    if not rows:
        return
    if sort_by and sort_by in rows[0]:
        rows.sort(key=lambda r: r.get(sort_by) if r.get(sort_by) is not None else float("inf"))
    columns = list({key for row in rows for key in row.keys()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with output_md.open("w", encoding="utf-8") as fh:
            fh.write("| " + " | ".join(columns) + " |\n")
            fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
            for row in rows:
                fh.write("| " + " | ".join(str(row.get(key, "")) for key in columns) + " |\n")


def _plot_leaderboard(rows: list[dict[str, Any]], *, metric: str, output_path: Path, label_key: str) -> None:
    if not rows:
        return
    labels = [str(row.get(label_key, "")) for row in rows]
    values = [row.get(metric) for row in rows]
    values = [float(v) if v is not None else float("nan") for v in values]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    ax.bar(range(len(values)), values, color="#4C78A8", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _required_components(energy_cumsum: Any, *, threshold: float) -> int | None:
    if energy_cumsum is None:
        return None
    try:
        values = np.asarray(energy_cumsum, dtype=float).reshape(-1)
    except Exception:
        return None
    if values.size == 0:
        return None
    meets = np.where(values >= float(threshold))[0]
    if meets.size == 0:
        return int(values.size)
    return int(meets[0] + 1)


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("pipeline requires config from the Hydra entrypoint")
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    continue_on_error = bool(cfg_get(task_cfg, "continue_on_error", False))

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    init_run(writer=writer, steps=steps, cfg=cfg, run_dir=run_dir, clean=True, snapshot=True)

    base_cfg = _to_container(cfg)
    seed_value = cfg_get(base_cfg, "seed", None)
    base_name = _output_name(cfg)
    output_root = _output_root(cfg)
    stages = _normalize_list(cfg_get(task_cfg, "stages", None), ["decomposition", "preprocessing", "train"])
    decompose_list = _normalize_list(cfg_get(task_cfg, "decompose_list", None), [cfg_get(cfg, "decompose", {}).get("name", "fft2")])
    coeff_post_list = _normalize_list(cfg_get(task_cfg, "coeff_post_list", None), [cfg_get(cfg, "coeff_post", {}).get("name", "none")])
    model_list = _normalize_list(cfg_get(task_cfg, "model_list", None), [cfg_get(cfg, "model", {}).get("name", "ridge")])
    energy_threshold = float(cfg_get(task_cfg, "energy_threshold", 0.9))

    decompositions: list[dict[str, Any]] = []
    trainings: list[dict[str, Any]] = []

    with steps.step(
        "run_pipeline",
        outputs=[artifact_ref("outputs/tables/pipeline_manifest.json", kind="table")],
    ):
        for decompose_name in decompose_list:
            decompose_cfg = _load_group_cfg("decompose", decompose_name)
            decompose_cfg = _deep_update(decompose_cfg, cfg_get(base_cfg, "decompose", {}) or {})
            decompose_cfg = _resolve_seed_tokens(decompose_cfg, seed=seed_value)
            combo_base = _combo_name(base_name, f"decomp-{decompose_name}")
            decomp_run_dir = output_root / combo_base / "decomposition"

            if "decomposition" in stages:
                decomp_cfg = copy.deepcopy(base_cfg)
                decomp_cfg["task"] = {"name": "decomposition"}
                decomp_cfg["run_dir"] = str(decomp_run_dir)
                decomp_cfg["output"] = {"root": str(output_root), "name": combo_base}
                decomp_cfg["decompose"] = _to_container(decompose_cfg)
                decomp_status = "ok"
                decomp_error = None
                decomp_cached = False
                try:
                    decomposition_process.main(decomp_cfg)
                except Exception as exc:
                    decomp_status = "failed"
                    decomp_error = f"{type(exc).__name__}: {exc}"
                    if not continue_on_error:
                        raise
            else:
                if not decomp_run_dir.exists():
                    raise ValueError(
                        "pipeline stages skip decomposition but run dir is missing: "
                        f"{decomp_run_dir}"
                    )
                # Reuse existing decomposition artifacts.
                decomp_status = "ok"
                decomp_error = None
                decomp_cached = True

            decompositions.append(
                {
                    "name": combo_base,
                    "decompose": decompose_name,
                    "decomposition_run_dir": str(decomp_run_dir),
                    "status": decomp_status,
                    "error": decomp_error,
                    "cached": decomp_cached,
                }
            )

            if decomp_status != "ok":
                # Can't proceed to preprocessing/train without decomposition artifacts.
                continue

            if "preprocessing" not in stages and "train" not in stages:
                continue

            for coeff_post_name in coeff_post_list:
                coeff_post_cfg = _load_group_cfg("coeff_post", coeff_post_name)
                coeff_post_cfg = _deep_update(coeff_post_cfg, cfg_get(base_cfg, "coeff_post", {}) or {})
                coeff_post_cfg = _resolve_seed_tokens(coeff_post_cfg, seed=seed_value)
                preproc_name = _combo_name(combo_base, f"post-{coeff_post_name}")
                preproc_run_dir = output_root / preproc_name / "preprocessing"

                if "preprocessing" in stages:
                    preproc_cfg = copy.deepcopy(base_cfg)
                    preproc_cfg["task"] = {"name": "preprocessing", "decomposition_run_dir": str(decomp_run_dir)}
                    preproc_cfg["run_dir"] = str(preproc_run_dir)
                    preproc_cfg["output"] = {"root": str(output_root), "name": preproc_name}
                    preproc_cfg["coeff_post"] = _to_container(coeff_post_cfg)
                    try:
                        preprocessing_process.main(preproc_cfg)
                    except Exception as exc:
                        if not continue_on_error:
                            raise
                        # Skip training if preprocessing failed.
                        trainings.append(
                            {
                                "name": preproc_name,
                                "decompose": decompose_name,
                                "coeff_post": coeff_post_name,
                                "model": None,
                                "train_run_dir": None,
                                "status": "skipped",
                                "error": f"preprocessing failed: {type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                else:
                    if "train" in stages and not preproc_run_dir.exists():
                        raise ValueError(
                            "pipeline stages skip preprocessing but run dir is missing: "
                            f"{preproc_run_dir}"
                        )

                if "train" not in stages:
                    continue

                for model_name in model_list:
                    model_cfg = _load_group_cfg("model", model_name)
                    model_cfg = _deep_update(model_cfg, cfg_get(base_cfg, "model", {}) or {})
                    model_cfg = _resolve_seed_tokens(model_cfg, seed=seed_value)
                    model_run_name = _combo_name(preproc_name, f"model-{model_name}")
                    train_run_dir = output_root / model_run_name / "train"
                    train_cfg = copy.deepcopy(base_cfg)
                    train_cfg["task"] = {"name": "train", "preprocessing_run_dir": str(preproc_run_dir)}
                    train_cfg["run_dir"] = str(train_run_dir)
                    train_cfg["output"] = {"root": str(output_root), "name": model_run_name}
                    train_cfg["model"] = _to_container(model_cfg)
                    train_status = "ok"
                    train_error = None
                    try:
                        train_process.main(train_cfg)
                    except Exception as exc:
                        train_status = "failed"
                        train_error = f"{type(exc).__name__}: {exc}"
                        if not continue_on_error:
                            raise
                    trainings.append(
                        {
                            "name": model_run_name,
                            "decompose": decompose_name,
                            "coeff_post": coeff_post_name,
                            "model": model_name,
                            "train_run_dir": str(train_run_dir),
                            "status": train_status,
                            "error": train_error,
                        }
                    )

    writer.tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = writer.tables_dir / "pipeline_manifest.json"
    manifest_path.write_text(
        yaml.safe_dump(
            {"decompositions": decompositions, "trainings": trainings},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    decomp_rows = []
    for row in decompositions:
        metrics_path = Path(row["decomposition_run_dir"]) / "outputs" / "metrics.json"
        if metrics_path.exists():
            metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8")) or {}
        else:
            metrics = {}
        # Keep leaderboards small and CSV-friendly. Large arrays (e.g. energy_cumsum)
        # are still available in each run's outputs/metrics.json.
        energy_cumsum = metrics.pop("energy_cumsum", None)
        record = dict(row)
        record.update(metrics)
        required = _required_components(metrics.get("energy_cumsum"), threshold=energy_threshold)
        # If we removed energy_cumsum from metrics, compute required from the original payload.
        if required is None and energy_cumsum is not None:
            required = _required_components(energy_cumsum, threshold=energy_threshold)
        if required is not None:
            record["energy_threshold"] = energy_threshold
            record["n_components_required"] = required
        if energy_cumsum is not None:
            try:
                values = np.asarray(energy_cumsum, dtype=float).reshape(-1)
                record["energy_cumsum_len"] = int(values.size)
                record["energy_cumsum_last"] = float(values[-1]) if values.size else None
            except Exception:
                record["energy_cumsum_len"] = None
                record["energy_cumsum_last"] = None
        decomp_rows.append(record)
    _write_leaderboard(
        decomp_rows,
        output_csv=writer.tables_dir / "leaderboard_decomposition.csv",
        output_md=writer.tables_dir / "leaderboard_decomposition.md",
        sort_by=cfg_get(task_cfg, "decomposition_sort_by", None),
    )
    metric_name = cfg_get(task_cfg, "decomposition_sort_by", None)
    if metric_name:
        _plot_leaderboard(
            decomp_rows,
            metric=str(metric_name),
            output_path=writer.plots_dir / "leaderboard_decomposition.png",
            label_key="decompose",
        )

    train_rows = []
    for row in trainings:
        train_run_dir = row.get("train_run_dir")
        metrics = {}
        if train_run_dir:
            metrics_path = Path(train_run_dir) / "outputs" / "metrics.json"
            if metrics_path.exists():
                metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8")) or {}
        record = dict(row)
        record.update(metrics)
        train_rows.append(record)
    _write_leaderboard(
        train_rows,
        output_csv=writer.tables_dir / "leaderboard_train.csv",
        output_md=writer.tables_dir / "leaderboard_train.md",
        sort_by=cfg_get(task_cfg, "train_sort_by", None),
    )
    metric_name = cfg_get(task_cfg, "train_sort_by", None)
    if metric_name:
        _plot_leaderboard(
            train_rows,
            metric=str(metric_name),
            output_path=writer.plots_dir / "leaderboard_train.png",
            label_key="model",
        )

    finalize_run(writer=writer, steps=steps, cfg=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
