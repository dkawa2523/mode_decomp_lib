"""Process entrypoint: predict."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.models import BaseRegressor
from mode_decomp_ml.pipeline import (
    build_dataset_meta,
    build_meta,
    cfg_get,
    dataset_to_arrays,
    ensure_dir,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    split_indices,
    write_json,
)


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for predict")
    return task_cfg


def _load_model(train_run_dir: str) -> BaseRegressor:
    model_path = resolve_path(train_run_dir) / "artifacts" / "model" / "model.pkl"
    return BaseRegressor.load_state(model_path)


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("predict requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "domain", "split"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    if train_run_dir is None or str(train_run_dir).strip() == "":
        raise ValueError("task.train_run_dir is required for predict")

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    split_cfg = cfg_get(cfg, "split")
    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
    conds, _, _, _ = dataset_to_arrays(dataset)

    model = _load_model(str(train_run_dir))
    coeff_pred, coeff_std = model.predict_with_std(conds)
    coeff_pred = np.asarray(coeff_pred)
    coeff_std = None if coeff_std is None else np.asarray(coeff_std)

    preds_dir = ensure_dir(run_dir / "preds")
    # CONTRACT: predictions are persisted under preds/coeff.npy and coeff_mean.npy.
    np.save(preds_dir / "coeff.npy", coeff_pred)
    np.save(preds_dir / "coeff_mean.npy", coeff_pred)
    if coeff_std is not None:
        if coeff_std.shape != coeff_pred.shape:
            raise ValueError("coeff_std shape does not match coeff_mean")
        np.save(preds_dir / "coeff_std.npy", coeff_std)
    preds_meta = {
        "target_space": getattr(model, "target_space", None),
        "num_samples": int(coeff_pred.shape[0]),
        "coeff_dim": int(coeff_pred.shape[1]) if coeff_pred.ndim == 2 else int(coeff_pred.shape[-1]),
        "train_run_dir": str(resolve_path(train_run_dir)),
        "coeff_std": coeff_std is not None,
    }
    write_json(preds_dir / "preds_meta.json", preds_meta)

    split_meta = split_indices(split_cfg, len(dataset), seed)
    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    write_json(run_dir / "artifacts" / "dataset_meta.json", dataset_meta)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    write_json(run_dir / "meta.json", meta)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
