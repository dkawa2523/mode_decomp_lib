"""Process entrypoint: viz."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mode_decomp_ml.coeff_post import BaseCoeffPost
from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.decompose import BaseDecomposer
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.models import BaseRegressor
from mode_decomp_ml.pipeline import (
    build_dataset_meta,
    build_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    ensure_dir,
    read_json,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    split_indices,
    write_json,
)
from mode_decomp_ml.tracking import maybe_log_run
from mode_decomp_ml.viz import (
    coeff_energy_spectrum,
    plot_coeff_spectrum,
    plot_error_map,
    plot_field_grid,
    plot_uncertainty_map,
)


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for viz")
    return task_cfg


def _load_train_artifacts(train_run_dir: str) -> tuple[BaseDecomposer, BaseCoeffPost, BaseRegressor]:
    train_root = resolve_path(train_run_dir)
    decomposer = BaseDecomposer.load_state(train_root / "artifacts" / "decomposer" / "state.pkl")
    coeff_post = BaseCoeffPost.load_state(train_root / "artifacts" / "coeff_post" / "state.pkl")
    model = BaseRegressor.load_state(train_root / "artifacts" / "model" / "model.pkl")
    return decomposer, coeff_post, model


def _load_pred_coeff(predict_run_dir: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    pred_root = resolve_path(predict_run_dir)
    coeff_path = pred_root / "preds" / "coeff_mean.npy"
    if not coeff_path.exists():
        coeff_path = pred_root / "preds" / "coeff.npy"
    coeff = np.load(coeff_path)
    meta_path = pred_root / "preds" / "preds_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return coeff, meta


def _load_recon_field(reconstruct_run_dir: str) -> np.ndarray:
    recon_root = resolve_path(reconstruct_run_dir)
    return np.load(recon_root / "preds" / "field.npy")


def _load_field_std(reconstruct_run_dir: str) -> tuple[np.ndarray | None, Mapping[str, Any]]:
    recon_root = resolve_path(reconstruct_run_dir)
    std_path = recon_root / "preds" / "field_std.npy"
    if not std_path.exists():
        return None, {}
    std = np.load(std_path)
    meta_path = recon_root / "preds" / "field_std_meta.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    return std, meta


def _normalize_k_list(
    k_list: Sequence[int],
    *,
    total_coeff: int,
    coeff_meta: Mapping[str, Any] | None,
) -> list[int]:
    if total_coeff <= 0:
        raise ValueError("total_coeff must be positive")
    complex_format = str(coeff_meta.get("complex_format", "")) if coeff_meta else ""
    coeff_shape = coeff_meta.get("coeff_shape") if coeff_meta else None
    needs_even = complex_format == "real_imag" and isinstance(coeff_shape, list) and coeff_shape[-1] == 2
    max_coeff = total_coeff
    if needs_even and max_coeff % 2 != 0:
        max_coeff -= 1
    out: list[int] = []
    for value in k_list:
        k = int(value)
        if k <= 0:
            continue
        if needs_even and k % 2 != 0:
            k += 1
        k = min(k, max_coeff)
        if k not in out:
            out.append(k)
    if not out:
        out.append(max_coeff)
    return out


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("viz requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "domain", "split", "viz"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    predict_run_dir = cfg_get(task_cfg, "predict_run_dir", None)
    reconstruct_run_dir = cfg_get(task_cfg, "reconstruct_run_dir", None)
    if not train_run_dir or not str(train_run_dir).strip():
        raise ValueError("task.train_run_dir is required for viz")
    if not predict_run_dir or not str(predict_run_dir).strip():
        raise ValueError("task.predict_run_dir is required for viz")
    if not reconstruct_run_dir or not str(reconstruct_run_dir).strip():
        raise ValueError("task.reconstruct_run_dir is required for viz")

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    split_cfg = cfg_get(cfg, "split")
    viz_cfg = cfg_get(cfg, "viz", {}) or {}

    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
    _, fields_true_all, masks_all, _ = dataset_to_arrays(dataset)
    split_meta = split_indices(split_cfg, len(dataset), seed)
    eval_idx = np.asarray(split_meta["train_idx"], dtype=int)
    fields_true = fields_true_all[eval_idx]
    masks = masks_all[eval_idx] if masks_all is not None else None

    domain_name = str(cfg_get(domain_cfg, "name", "")).strip().lower()
    needs_mask = domain_name in {"disk", "mask", "arbitrary_mask", "mesh"}
    if needs_mask:
        domain_spec = build_domain_spec(domain_cfg, fields_true.shape[1:])
        masks = combine_masks(
            masks,
            domain_spec.mask,
            spatial_shape=fields_true.shape[1:3],
            n_samples=fields_true.shape[0],
        )

    field_pred_full = _load_recon_field(str(reconstruct_run_dir))
    field_pred = field_pred_full[eval_idx]
    if field_pred.shape != fields_true.shape:
        raise ValueError("field_hat shape does not match dataset field shape")

    sample_index = int(cfg_get(viz_cfg, "sample_index", 0))
    if sample_index < 0 or sample_index >= fields_true.shape[0]:
        raise ValueError("viz.sample_index is out of range")

    field_true_sample = fields_true[sample_index]
    field_pred_sample = field_pred[sample_index]
    mask_sample = masks[sample_index] if masks is not None else None

    coeff_pred, preds_meta = _load_pred_coeff(str(predict_run_dir))
    coeff_pred = coeff_pred[eval_idx]
    decomposer, coeff_post, model = _load_train_artifacts(str(train_run_dir))
    target_space = preds_meta.get("target_space") or getattr(model, "target_space", None)
    if target_space not in {"a", "z"}:
        raise ValueError("target_space must be 'a' or 'z' for viz")
    coeff_a = coeff_post.inverse_transform(coeff_pred) if target_space == "z" else coeff_pred

    # REVIEW: coeff_meta defines coefficient ordering and spectrum semantics.
    coeff_meta = read_json(resolve_path(train_run_dir) / "artifacts" / "decomposer" / "coeff_meta.json")
    coeff_shape = coeff_meta.get("coeff_shape")
    if isinstance(coeff_shape, list):
        expected = int(np.prod(coeff_shape))
        if coeff_a.shape[1] != expected:
            raise ValueError("coeff_a dimension does not match coeff_meta")

    field_shape = tuple(coeff_meta.get("field_shape", []))
    if not field_shape:
        raise ValueError("coeff_meta.field_shape is required for viz")
    domain_spec = build_domain_spec(domain_cfg, field_shape)

    spectrum = coeff_energy_spectrum(coeff_a, coeff_meta)
    spectrum_scale = str(cfg_get(viz_cfg, "spectrum_scale", "log")).strip().lower()

    k_list = cfg_get(viz_cfg, "k_list", None)
    if k_list is None:
        k_list = [1, 2, 4, 8, 16]
    k_list = _normalize_k_list(k_list, total_coeff=coeff_a.shape[1], coeff_meta=coeff_meta)

    coeff_sample = coeff_a[sample_index]
    recon_fields = []
    for k in k_list:
        coeff_trunc = coeff_sample.copy()
        if k < coeff_trunc.size:
            coeff_trunc[k:] = 0.0
        # REVIEW: sequential recon uses flat coeff order for comparability.
        recon_fields.append(decomposer.inverse_transform(coeff_trunc, domain_spec=domain_spec))

    viz_root = ensure_dir(run_dir / "viz")
    sample_dir = ensure_dir(viz_root / f"sample_{sample_index:04d}")
    # CONTRACT: fixed filenames enable cross-run comparisons.
    plot_field_grid(
        sample_dir / "field_compare.png",
        [field_true_sample, field_pred_sample],
        ["true", "pred"],
        mask=mask_sample,
        suptitle="field_true vs field_hat",
    )
    plot_error_map(sample_dir / "error_map.png", field_true_sample, field_pred_sample, mask=mask_sample)
    plot_field_grid(
        sample_dir / "recon_sequence.png",
        recon_fields,
        [f"k={k}" for k in k_list],
        mask=mask_sample,
        suptitle="sequential reconstruction",
    )
    plot_coeff_spectrum(viz_root / "coeff_spectrum.png", spectrum, scale=spectrum_scale)

    uncertainty_cfg = cfg_get(cfg, "uncertainty", {}) or {}
    uncertainty_enabled = bool(cfg_get(uncertainty_cfg, "enabled", False))
    if uncertainty_enabled:
        field_std, std_meta = _load_field_std(str(reconstruct_run_dir))
        if field_std is None:
            raise ValueError("uncertainty enabled but field_std is missing from reconstruct run")
        case_indices = std_meta.get("case_indices")
        if not isinstance(case_indices, list) or not case_indices:
            raise ValueError("field_std_meta.case_indices is required for uncertainty viz")
        if field_std.shape[0] != len(case_indices):
            raise ValueError("field_std batch size does not match case_indices")
        for offset, case_idx in enumerate(case_indices):
            idx = int(case_idx)
            if idx < 0 or idx >= fields_true_all.shape[0]:
                raise ValueError("field_std case index is out of dataset bounds")
            mask_case = masks_all[idx] if masks_all is not None else None
            # CONTRACT: uncertainty maps are persisted under viz/uncertainty_map_<id>.png.
            plot_uncertainty_map(
                viz_root / f"uncertainty_map_{idx:04d}.png",
                field_std[offset],
                mask=mask_case,
            )

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    write_json(run_dir / "artifacts" / "dataset_meta.json", dataset_meta)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    write_json(run_dir / "meta.json", meta)
    maybe_log_run(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
