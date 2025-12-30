"""Process entrypoint: reconstruct."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mode_decomp_ml.coeff_post import BaseCoeffPost
from mode_decomp_ml.decompose import BaseDecomposer
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.models import BaseRegressor
from mode_decomp_ml.pipeline import (
    build_meta,
    cfg_get,
    ensure_dir,
    read_json,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    write_json,
)


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for reconstruct")
    return task_cfg


def _load_train_artifacts(train_run_dir: str) -> tuple[BaseDecomposer, BaseCoeffPost, BaseRegressor]:
    train_root = resolve_path(train_run_dir)
    decomposer = BaseDecomposer.load_state(train_root / "artifacts" / "decomposer" / "state.pkl")
    coeff_post = BaseCoeffPost.load_state(train_root / "artifacts" / "coeff_post" / "state.pkl")
    model = BaseRegressor.load_state(train_root / "artifacts" / "model" / "model.pkl")
    return decomposer, coeff_post, model


def _load_pred_coeff(predict_run_dir: str) -> tuple[np.ndarray, np.ndarray | None, Mapping[str, Any]]:
    pred_root = resolve_path(predict_run_dir)
    coeff_path = pred_root / "preds" / "coeff_mean.npy"
    if not coeff_path.exists():
        coeff_path = pred_root / "preds" / "coeff.npy"
    coeff = np.load(coeff_path)
    std_path = pred_root / "preds" / "coeff_std.npy"
    coeff_std = np.load(std_path) if std_path.exists() else None
    meta_path = pred_root / "preds" / "preds_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return coeff, coeff_std, meta


def _select_case_indices(n_samples: int, num_cases: int) -> list[int]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if num_cases <= 0:
        raise ValueError("uncertainty.num_cases must be positive when enabled")
    if num_cases >= n_samples:
        return list(range(n_samples))
    indices = np.linspace(0, n_samples - 1, num_cases, dtype=int)
    return indices.tolist()


def _mc_field_std(
    coeff_mean: np.ndarray,
    coeff_std: np.ndarray,
    *,
    case_indices: Sequence[int],
    num_samples: int,
    target_space: str,
    coeff_post: BaseCoeffPost,
    decomposer: BaseDecomposer,
    domain_spec: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("uncertainty.num_mc_samples must be positive when enabled")
    field_std_cases = []
    for idx in case_indices:
        mean = np.asarray(coeff_mean[idx])
        std = np.asarray(coeff_std[idx])
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("coeff_mean/std per sample must be 1D")
        if mean.shape != std.shape:
            raise ValueError("coeff_mean/std dimension mismatch")
        std = np.clip(std, 0.0, None)
        # REVIEW: MC assumes independent Gaussian coeffs per sample.
        coeff_samples = rng.normal(loc=mean, scale=std, size=(num_samples, mean.shape[0]))
        if target_space == "z":
            coeff_samples = coeff_post.inverse_transform(coeff_samples)
        fields = []
        for sample in coeff_samples:
            fields.append(decomposer.inverse_transform(sample, domain_spec=domain_spec))
        field_stack = np.stack(fields, axis=0)
        field_std_cases.append(field_stack.std(axis=0))
    return np.stack(field_std_cases, axis=0)


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("reconstruct requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["run_dir", "domain"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    predict_run_dir = cfg_get(task_cfg, "predict_run_dir", None)
    if not train_run_dir or not str(train_run_dir).strip():
        raise ValueError("task.train_run_dir is required for reconstruct")
    if not predict_run_dir or not str(predict_run_dir).strip():
        raise ValueError("task.predict_run_dir is required for reconstruct")

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    domain_cfg = cfg_get(cfg, "domain")
    coeff_pred, coeff_std, preds_meta = _load_pred_coeff(str(predict_run_dir))
    decomposer, coeff_post, model = _load_train_artifacts(str(train_run_dir))

    target_space = preds_meta.get("target_space") or getattr(model, "target_space", None)
    if target_space not in {"a", "z"}:
        raise ValueError("target_space must be 'a' or 'z' for reconstruction")

    coeff_meta = read_json(resolve_path(train_run_dir) / "artifacts" / "decomposer" / "coeff_meta.json")
    field_shape = tuple(coeff_meta.get("field_shape", []))
    if not field_shape:
        raise ValueError("coeff_meta.field_shape is required for reconstruction")
    # REVIEW: domain spec is reconstructed from coeff_meta field shape.
    domain_spec = build_domain_spec(domain_cfg, field_shape)

    coeff_a = coeff_post.inverse_transform(coeff_pred) if target_space == "z" else coeff_pred
    fields = []
    for idx in range(coeff_a.shape[0]):
        fields.append(decomposer.inverse_transform(coeff_a[idx], domain_spec=domain_spec))
    field_hat = np.stack(fields, axis=0)

    preds_dir = ensure_dir(run_dir / "preds")
    # CONTRACT: reconstructed fields are persisted under preds/field.npy.
    np.save(preds_dir / "field.npy", field_hat)
    uncertainty_cfg = cfg_get(cfg, "uncertainty", {}) or {}
    uncertainty_enabled = bool(cfg_get(uncertainty_cfg, "enabled", False))
    if uncertainty_enabled:
        if coeff_std is None:
            raise ValueError("uncertainty enabled but coeff_std is missing from predict run")
        coeff_std = np.asarray(coeff_std)
        if coeff_std.ndim == 1:
            coeff_std = coeff_std[:, None]
        if coeff_std.shape != coeff_pred.shape:
            raise ValueError("coeff_std shape does not match coeff_mean")
        num_mc_samples = int(cfg_get(uncertainty_cfg, "num_mc_samples", 32))
        num_cases = int(cfg_get(uncertainty_cfg, "num_cases", 3))
        case_indices = _select_case_indices(coeff_pred.shape[0], num_cases)
        seed = cfg_get(cfg, "seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        field_std = _mc_field_std(
            coeff_pred,
            coeff_std,
            case_indices=case_indices,
            num_samples=num_mc_samples,
            target_space=target_space,
            coeff_post=coeff_post,
            decomposer=decomposer,
            domain_spec=domain_spec,
            rng=rng,
        )
        # CONTRACT: MC uncertainty maps are persisted under preds/field_std.npy.
        np.save(preds_dir / "field_std.npy", field_std)
        write_json(
            preds_dir / "field_std_meta.json",
            {
                "case_indices": [int(i) for i in case_indices],
                "num_mc_samples": int(num_mc_samples),
                "target_space": target_space,
                "predict_run_dir": str(resolve_path(predict_run_dir)),
            },
        )
    preds_meta_out = {
        "target_space": target_space,
        "num_samples": int(field_hat.shape[0]),
        "field_shape": [int(x) for x in field_hat.shape[1:]],
        "train_run_dir": str(resolve_path(train_run_dir)),
        "predict_run_dir": str(resolve_path(predict_run_dir)),
    }
    write_json(preds_dir / "preds_meta.json", preds_meta_out)

    meta = build_meta(cfg)
    write_json(run_dir / "meta.json", meta)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
