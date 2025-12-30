"""Process entrypoint: benchmark."""
from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.pipeline import (
    PROJECT_ROOT,
    build_dataset_meta,
    build_meta,
    cfg_get,
    ensure_dir,
    require_cfg_keys,
    resolve_run_dir,
    split_indices,
    write_json,
)
from processes import eval as eval_process
from processes import predict as predict_process
from processes import reconstruct as reconstruct_process
from processes import train as train_process

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional dependency
    OmegaConf = None

CONFIG_DIR = PROJECT_ROOT / "configs"


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for benchmark")
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


def _load_group_cfg(group: str, name: str) -> dict[str, Any]:
    path = CONFIG_DIR / group / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalize_list(value: Any, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _resolve_domain_cfg(cfg: Mapping[str, Any], task_cfg: Mapping[str, Any]) -> dict[str, Any]:
    override = cfg_get(task_cfg, "domain", None)
    if override is None:
        return _to_container(cfg_get(cfg, "domain", {}))
    if isinstance(override, str):
        return _load_group_cfg("domain", override)
    if isinstance(override, Mapping):
        return _to_container(override)
    raise ValueError("task.domain must be a string or mapping")


def _prepare_decompose_cfg(decompose_cfg: Mapping[str, Any], domain_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg_out = _to_container(decompose_cfg)
    method = str(cfg_get(cfg_out, "name", "")).strip()
    domain_name = str(cfg_get(domain_cfg, "name", "")).strip()
    if method == "zernike" and domain_name != "disk":
        raise ValueError("benchmark requires disk domain for zernike")
    if domain_name == "disk" and method in {"fft2", "dct2"}:
        # REVIEW: disk domain requires zero-fill policy for FFT/DCT.
        cfg_out["disk_policy"] = "mask_zero_fill"
    return cfg_out


def _prepare_model_cfg(model_cfg: Mapping[str, Any], coeff_post_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg_out = _to_container(model_cfg)
    coeff_name = str(cfg_get(coeff_post_cfg, "name", "")).strip().lower()
    cfg_out["target_space"] = "z" if coeff_name and coeff_name != "none" else "a"
    return cfg_out


def _write_config_snapshot(run_dir: Path, cfg: Mapping[str, Any]) -> None:
    hydra_dir = ensure_dir(run_dir / ".hydra")
    # CONTRACT: benchmark writes config snapshots for leaderboard traceability.
    with (hydra_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_container(cfg), fh, sort_keys=False)


def _build_process_cfg(
    base_cfg: Mapping[str, Any],
    *,
    task_cfg: Mapping[str, Any],
    run_dir: Path,
    domain_cfg: Mapping[str, Any],
    decompose_cfg: Mapping[str, Any],
    coeff_post_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg_out = copy.deepcopy(_to_container(base_cfg))
    cfg_out["task"] = dict(task_cfg)
    cfg_out["run_dir"] = str(run_dir)
    cfg_out["domain"] = _to_container(domain_cfg)
    cfg_out["decompose"] = _to_container(decompose_cfg)
    cfg_out["coeff_post"] = _to_container(coeff_post_cfg)
    cfg_out["model"] = _to_container(model_cfg)
    return cfg_out


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("benchmark requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "output_dir", "dataset", "split", "domain", "model", "eval"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    base_cfg = _to_container(cfg)
    decompose_list = _normalize_list(cfg_get(task_cfg, "decompose_list", None), ["fft2", "zernike"])
    coeff_post_list = _normalize_list(cfg_get(task_cfg, "coeff_post_list", None), ["none", "pca"])
    domain_cfg = _resolve_domain_cfg(cfg, task_cfg)
    model_cfg_base = _to_container(cfg_get(cfg, "model", {}))

    runs: list[dict[str, Any]] = []
    for idx, (decompose_name, coeff_post_name) in enumerate(
        itertools.product(decompose_list, coeff_post_list)
    ):
        combo_name = f"run_{idx:03d}__{decompose_name}__{coeff_post_name}"
        combo_dir = run_dir / combo_name
        train_dir = combo_dir / "train"
        predict_dir = combo_dir / "predict"
        reconstruct_dir = combo_dir / "reconstruct"
        eval_dir = combo_dir / "eval"

        decompose_cfg = _prepare_decompose_cfg(
            _load_group_cfg("decompose", decompose_name),
            domain_cfg,
        )
        coeff_post_cfg = _load_group_cfg("coeff_post", coeff_post_name)
        model_cfg = _prepare_model_cfg(model_cfg_base, coeff_post_cfg)

        train_cfg = _build_process_cfg(
            base_cfg,
            task_cfg={"name": "train"},
            run_dir=train_dir,
            domain_cfg=domain_cfg,
            decompose_cfg=decompose_cfg,
            coeff_post_cfg=coeff_post_cfg,
            model_cfg=model_cfg,
        )
        train_process.main(train_cfg)

        predict_cfg = _build_process_cfg(
            base_cfg,
            task_cfg={"name": "predict", "train_run_dir": str(train_dir)},
            run_dir=predict_dir,
            domain_cfg=domain_cfg,
            decompose_cfg=decompose_cfg,
            coeff_post_cfg=coeff_post_cfg,
            model_cfg=model_cfg,
        )
        predict_process.main(predict_cfg)

        reconstruct_cfg = _build_process_cfg(
            base_cfg,
            task_cfg={
                "name": "reconstruct",
                "train_run_dir": str(train_dir),
                "predict_run_dir": str(predict_dir),
            },
            run_dir=reconstruct_dir,
            domain_cfg=domain_cfg,
            decompose_cfg=decompose_cfg,
            coeff_post_cfg=coeff_post_cfg,
            model_cfg=model_cfg,
        )
        reconstruct_process.main(reconstruct_cfg)

        eval_cfg = _build_process_cfg(
            base_cfg,
            task_cfg={
                "name": "eval",
                "train_run_dir": str(train_dir),
                "predict_run_dir": str(predict_dir),
                "reconstruct_run_dir": str(reconstruct_dir),
            },
            run_dir=eval_dir,
            domain_cfg=domain_cfg,
            decompose_cfg=decompose_cfg,
            coeff_post_cfg=coeff_post_cfg,
            model_cfg=model_cfg,
        )
        eval_process.main(eval_cfg)
        _write_config_snapshot(eval_dir, eval_cfg)

        runs.append(
            {
                "name": combo_name,
                "decompose": decompose_name,
                "coeff_post": coeff_post_name,
                "train_run_dir": str(train_dir),
                "predict_run_dir": str(predict_dir),
                "reconstruct_run_dir": str(reconstruct_dir),
                "eval_run_dir": str(eval_dir),
            }
        )

    # CONTRACT: benchmark records run directories for leaderboard collection.
    write_json(run_dir / "benchmark_manifest.json", {"runs": runs})

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    split_cfg = cfg_get(cfg, "split")
    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
    split_meta = split_indices(split_cfg, len(dataset), seed)
    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    write_json(run_dir / "artifacts" / "dataset_meta.json", dataset_meta)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    write_json(run_dir / "meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
