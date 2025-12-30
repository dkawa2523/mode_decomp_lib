"""Pipeline helpers for process entrypoints."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:  # optional at runtime; used for config conversion
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback when OmegaConf is missing
    OmegaConf = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def task_name(cfg: Mapping[str, Any]) -> str:
    task_cfg = cfg_get(cfg, "task", None)
    if task_cfg is None:
        raise ValueError("task config is missing")
    if isinstance(task_cfg, str):
        return task_cfg
    name = cfg_get(task_cfg, "name", None)
    if not name:
        raise ValueError("task.name is missing")
    return str(name)


def require_cfg_keys(cfg: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if cfg_get(cfg, key, None) is None]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_run_dir(cfg: Mapping[str, Any]) -> Path:
    run_dir = cfg_get(cfg, "run_dir", None)
    if run_dir is None or str(run_dir).strip() == "":
        raise ValueError("run_dir is required in config")
    return resolve_path(str(run_dir))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_fallback)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dataset_to_arrays(
    dataset: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[dict[str, Any]]]:
    conds: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    masks: list[np.ndarray | None] = []
    metas: list[dict[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        conds.append(np.asarray(sample.cond))
        fields.append(np.asarray(sample.field))
        masks.append(None if sample.mask is None else np.asarray(sample.mask))
        metas.append(dict(sample.meta))
    cond_arr = np.stack(conds, axis=0)
    field_arr = np.stack(fields, axis=0)
    if all(mask is None for mask in masks):
        mask_arr = None
    elif any(mask is None for mask in masks):
        raise ValueError("mask must be present for all samples or none")
    else:
        mask_arr = np.stack([mask for mask in masks if mask is not None], axis=0)
    return cond_arr, field_arr, mask_arr, metas


def combine_masks(
    mask_batch: np.ndarray | None,
    domain_mask: np.ndarray | None,
    *,
    spatial_shape: tuple[int, int],
    n_samples: int,
) -> np.ndarray | None:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if domain_mask is None and mask_batch is None:
        return None

    domain_arr: np.ndarray | None = None
    if domain_mask is not None:
        domain_arr = np.asarray(domain_mask).astype(bool)
        if domain_arr.ndim == 1:
            if spatial_shape[1] != 1 or domain_arr.shape[0] != spatial_shape[0]:
                raise ValueError("domain_mask 1D shape does not match field spatial shape")
            domain_arr = domain_arr[:, None]
        if domain_arr.ndim != 2:
            raise ValueError("domain_mask must be 1D or 2D")
        if domain_arr.shape != spatial_shape:
            raise ValueError("domain_mask shape does not match field spatial shape")

    if mask_batch is None:
        if domain_arr is None:
            return None
        return np.broadcast_to(domain_arr, (n_samples, *spatial_shape)).copy()

    mask_arr = np.asarray(mask_batch).astype(bool)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    if mask_arr.ndim != 3:
        raise ValueError("mask_batch must be 2D or 3D")
    if mask_arr.shape[0] != n_samples:
        raise ValueError("mask_batch sample count does not match n_samples")
    if mask_arr.shape[1:] != spatial_shape:
        raise ValueError("mask_batch spatial shape does not match field spatial shape")

    if domain_arr is None:
        return mask_arr
    return mask_arr & domain_arr[None, ...]


def split_indices(
    split_cfg: Mapping[str, Any],
    n_samples: int,
    seed: int | None,
) -> dict[str, Any]:
    name = str(cfg_get(split_cfg, "name", "")).strip().lower()
    if not name:
        raise ValueError("split.name is required")
    if name != "all":
        raise NotImplementedError(f"split.name={name} is not supported yet")
    idx = np.arange(n_samples, dtype=int)
    # CONTRACT: split info is stored for reproducibility.
    return {"name": name, "train_idx": idx.tolist(), "seed": seed}


def _to_container(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is None:
        return {}
    if (
        OmegaConf is not None
        and hasattr(OmegaConf, "to_container")
        and getattr(OmegaConf, "is_config", lambda _: False)(cfg)
    ):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, Mapping):
        return {str(k): _to_container(v) if isinstance(v, Mapping) else v for k, v in cfg.items()}
    return {}


def build_dataset_meta(
    dataset: Any,
    dataset_cfg: Mapping[str, Any],
    split_meta: Mapping[str, Any],
    domain_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    sample_ids = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        sample_ids.append(str(sample.meta.get("sample_id", f"sample_{idx:04d}")))
    digest = hashlib.sha1()
    digest.update(json.dumps({"sample_ids": sample_ids}, sort_keys=True).encode("utf-8"))
    dataset_hash = digest.hexdigest()
    # CONTRACT: dataset hash uses sample IDs for versioning.
    return {
        "name": getattr(dataset, "name", "unknown"),
        "num_samples": int(len(dataset)),
        "sample_ids": sample_ids,
        "dataset_hash": dataset_hash,
        "split": dict(split_meta),
        "dataset_cfg": _to_container(dataset_cfg),
        "domain_cfg": _to_container(domain_cfg),
    }


def _git_cmd(args: Iterable[str]) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return output.decode("utf-8").strip() or None


def _git_info() -> dict[str, Any]:
    commit = _git_cmd(["rev-parse", "HEAD"])
    branch = _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _git_cmd(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
    }


def _extract_upstream_artifacts(cfg: Mapping[str, Any]) -> dict[str, Any]:
    task_cfg = cfg_get(cfg, "task", None)
    if task_cfg is None:
        return {}
    upstream: dict[str, Any] = {}
    for key in ("train_run_dir", "predict_run_dir", "reconstruct_run_dir", "eval_run_dir"):
        value = cfg_get(task_cfg, key, None)
        if value is None or str(value).strip() == "":
            continue
        upstream[key] = str(resolve_path(value))
    return upstream


def build_meta(cfg: Mapping[str, Any], dataset_hash: str | None = None) -> dict[str, Any]:
    packages: dict[str, Any] = {"numpy": np.__version__}
    try:
        import scipy

        packages["scipy"] = scipy.__version__
    except Exception:
        packages["scipy"] = None
    try:
        import sklearn

        packages["scikit_learn"] = sklearn.__version__
    except Exception:
        packages["scikit_learn"] = None
    meta = {
        "task": task_name(cfg),
        "run_dir": str(resolve_run_dir(cfg)),
        "output_dir": cfg_get(cfg, "output_dir", None),
        "seed": cfg_get(cfg, "seed", None),
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "dataset_hash": dataset_hash,
        "python": sys.version,
        "packages": packages,
        "git": _git_info(),
    }
    upstream = _extract_upstream_artifacts(cfg)
    if upstream:
        # CONTRACT: record upstream artifacts for lineage tracking.
        meta["upstream_artifacts"] = upstream
    return meta
