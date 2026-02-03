"""Pipeline helpers for process entrypoints."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.data.manifest import load_manifest, manifest_domain_cfg
from mode_decomp_ml.domain.sphere_grid import fill_sphere_grid_ranges

try:  # optional at runtime; used for config conversion
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback when OmegaConf is missing
    OmegaConf = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def resolve_domain_cfg(
    dataset_cfg: Mapping[str, Any],
    domain_cfg: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    root = cfg_get(dataset_cfg, "root", None)
    if root is not None:
        manifest = load_manifest(Path(str(root)))
        if manifest is not None:
            return manifest_domain_cfg(manifest, root), manifest
    if domain_cfg is None:
        raise ValueError("domain config is required when manifest.json is missing")
    return fill_sphere_grid_ranges(domain_cfg), None


def resolve_path(path_value: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_run_dir(cfg: Mapping[str, Any]) -> Path:
    run_dir = cfg_get(cfg, "run_dir", None)
    if (
        run_dir is None
        or str(run_dir).strip() == ""
        or _has_interpolation(run_dir)
    ):
        if isinstance(cfg, MutableMapping):
            return RunDirManager(cfg).ensure()
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


def load_coeff_meta(run_dir: str | Path) -> dict[str, Any]:
    run_root = resolve_path(str(run_dir))
    candidates = (
        run_root / "states" / "coeff_meta.json",
        run_root / "states" / "coeff_codec" / "coeff_meta.json",
        run_root / "states" / "decomposer" / "coeff_meta.json",
        run_root / "artifacts" / "decomposer" / "coeff_meta.json",
    )
    for path in candidates:
        if path.exists():
            return read_json(path)
    raise FileNotFoundError(f"coeff_meta.json not found under {run_root}")


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


def _short_hash(payload: Mapping[str, Any]) -> str:
    data = json.dumps(_to_container(payload), sort_keys=True, default=_json_fallback).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:6]


def make_run_id(cfg: Mapping[str, Any]) -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{_short_hash(cfg)}"


def _has_interpolation(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return "${" in value and "}" in value


def _sanitize_run_id_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    filtered = {k: v for k, v in cfg.items() if k not in {"run_id", "run_dir"}}
    return filtered


class RunDirManager:
    """Resolve run_dir from output_dir/tag/run_id with stable defaults."""

    def __init__(self, cfg: MutableMapping[str, Any]) -> None:
        self._cfg = cfg

    def _task_name(self) -> str:
        task_cfg = cfg_get(self._cfg, "task", None)
        if isinstance(task_cfg, str):
            name = task_cfg
        elif isinstance(task_cfg, Mapping):
            name = task_cfg.get("name", None)
        else:
            name = None
        return str(name).strip() if name else "task"

    def ensure(self) -> Path:
        run_dir_value = cfg_get(self._cfg, "run_dir", None)
        if (
            run_dir_value is not None
            and str(run_dir_value).strip() != ""
            and not _has_interpolation(run_dir_value)
        ):
            return resolve_path(str(run_dir_value))

        project_dir = cfg_get(self._cfg, "project_dir", None)
        if project_dir is not None and str(project_dir).strip() != "" and not _has_interpolation(project_dir):
            project_path = Path(str(project_dir))
            task_name = self._task_name()
            run_dir = project_path / task_name
            self._cfg["run_dir"] = run_dir.as_posix()
            hydra_cfg = self._cfg.get("hydra")
            if isinstance(hydra_cfg, MutableMapping):
                hydra_cfg = dict(hydra_cfg)
                hydra_cfg.setdefault("run", {})
                if isinstance(hydra_cfg.get("run"), MutableMapping):
                    hydra_cfg["run"] = dict(hydra_cfg["run"])
                    hydra_cfg["run"]["dir"] = run_dir.as_posix()
                self._cfg["hydra"] = hydra_cfg
            return resolve_path(run_dir)

        output_dir = str(cfg_get(self._cfg, "output_dir", "runs")).strip() or "runs"
        tag = str(cfg_get(self._cfg, "tag", "default")).strip() or "default"
        run_id = str(cfg_get(self._cfg, "run_id", "")).strip()
        if not run_id or _has_interpolation(run_id):
            run_id = make_run_id(_sanitize_run_id_cfg(self._cfg))
            self._cfg["run_id"] = run_id
        run_dir = Path(output_dir) / tag / run_id
        self._cfg["output_dir"] = output_dir
        self._cfg["tag"] = tag
        self._cfg["run_dir"] = run_dir.as_posix()
        hydra_cfg = self._cfg.get("hydra")
        if isinstance(hydra_cfg, MutableMapping):
            hydra_cfg = dict(hydra_cfg)
            hydra_cfg.setdefault("run", {})
            if isinstance(hydra_cfg.get("run"), MutableMapping):
                hydra_cfg["run"] = dict(hydra_cfg["run"])
                hydra_cfg["run"]["dir"] = run_dir.as_posix()
            self._cfg["hydra"] = hydra_cfg
        return resolve_path(run_dir)


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
    meta = {
        "name": getattr(dataset, "name", "unknown"),
        "num_samples": int(len(dataset)),
        "sample_ids": sample_ids,
        "dataset_hash": dataset_hash,
        "split": dict(split_meta),
        "dataset_cfg": _to_container(dataset_cfg),
        "domain_cfg": _to_container(domain_cfg),
    }
    manifest = getattr(dataset, "manifest", None)
    if manifest:
        meta["manifest"] = manifest
    field_kind = getattr(dataset, "field_kind", None)
    if field_kind:
        meta["field_kind"] = field_kind
    grid = getattr(dataset, "grid", None)
    if grid:
        meta["grid"] = grid
    mask_generated = getattr(dataset, "mask_generated", None)
    if mask_generated is not None:
        meta["mask_generated"] = bool(mask_generated)
    return meta


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
        "run_id": cfg_get(cfg, "run_id", None),
        "tag": cfg_get(cfg, "tag", None),
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
