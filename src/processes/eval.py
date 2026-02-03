"""Process entrypoint: eval."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import compute_metrics
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    build_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    load_coeff_meta,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.pipeline.artifacts import load_coeff_predictions, load_field_predictions
from mode_decomp_ml.pipeline.loaders import load_preprocess_state_from_run, load_train_artifacts
from mode_decomp_ml.tracking import maybe_log_run


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for eval")
    return task_cfg


def _load_pred_coeff(predict_run_dir: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    coeff, _, meta = load_coeff_predictions(predict_run_dir)
    return coeff, meta


def _grid_spacing_from_domain(domain_spec) -> tuple[float, float]:
    x = domain_spec.coords.get("x")
    y = domain_spec.coords.get("y")
    if x is None or y is None:
        raise ValueError("domain_spec must include x/y for div/curl metrics")
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    if dx <= 0 or dy <= 0:
        raise ValueError("domain_spec spacing must be positive for div/curl metrics")
    return dx, dy


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("eval requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "split", "eval"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    predict_run_dir = cfg_get(task_cfg, "predict_run_dir", None)
    reconstruct_run_dir = cfg_get(task_cfg, "reconstruct_run_dir", None)
    if not train_run_dir or not str(train_run_dir).strip():
        raise ValueError("task.train_run_dir is required for eval")
    if not predict_run_dir or not str(predict_run_dir).strip():
        raise ValueError("task.predict_run_dir is required for eval")
    if not reconstruct_run_dir or not str(reconstruct_run_dir).strip():
        raise ValueError("task.reconstruct_run_dir is required for eval")

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    with steps.step(
        "init_run",
        outputs=[artifact_ref("run.yaml", kind="config")],
    ):
        writer.ensure_layout()
        writer.write_run_yaml(cfg)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, domain_cfg)
    split_cfg = cfg_get(cfg, "split")
    eval_cfg = cfg_get(cfg, "eval")
    metrics_list = list(cfg_get(eval_cfg, "metrics", []))
    if not metrics_list:
        raise ValueError("eval.metrics must be configured")

    with steps.step(
        "build_dataset",
        inputs=[artifact_ref("run.yaml", kind="config")],
    ) as step:
        dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
        _, fields_true, masks, _ = dataset_to_arrays(dataset)
        split_meta = split_indices(split_cfg, len(dataset), seed)
        eval_idx = np.asarray(split_meta["train_idx"], dtype=int)
        fields_true = fields_true[eval_idx]
        masks = masks[eval_idx] if masks is not None else None
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(dataset))

    needs_divcurl = any(name in {"div_rmse", "curl_rmse"} for name in metrics_list)
    domain_spec = None
    domain_name = str(cfg_get(domain_cfg, "name", "")).strip().lower()
    needs_mask = domain_name in {"disk", "mask", "arbitrary_mask", "mesh"}
    if needs_mask or needs_divcurl:
        domain_spec = build_domain_spec(domain_cfg, fields_true.shape[1:])
        if needs_mask:
            masks = combine_masks(
                masks,
                domain_spec.mask,
                spatial_shape=fields_true.shape[1:3],
                n_samples=fields_true.shape[0],
            )

    with steps.step(
        "load_predictions",
        inputs=[artifact_ref(Path(str(reconstruct_run_dir)) / "preds.npz", kind="preds")],
    ):
        field_pred = load_field_predictions(str(reconstruct_run_dir))
        field_pred = field_pred[eval_idx]

    if field_pred.shape != fields_true.shape:
        raise ValueError("field_hat shape does not match dataset field shape")

    coeff_true_a = None
    coeff_pred_a = None
    coeff_true_z = None
    coeff_pred_z = None
    coeff_meta = None
    grid_spacing = None
    if needs_divcurl:
        if domain_spec is None:
            domain_spec = build_domain_spec(domain_cfg, fields_true.shape[1:])
        grid_spacing = _grid_spacing_from_domain(domain_spec)
    needs_coeff = any(name.startswith("coeff_") or name == "energy_cumsum" for name in metrics_list)
    if needs_coeff:
        preprocess = load_preprocess_state_from_run(str(train_run_dir))
        fields_coeff, masks_coeff = preprocess.transform(fields_true, masks)
        coeff_pred, preds_meta = _load_pred_coeff(str(predict_run_dir))
        coeff_pred = coeff_pred[eval_idx]
        decomposer, coeff_post, codec, model = load_train_artifacts(str(train_run_dir))
        target_space = preds_meta.get("target_space") or getattr(model, "target_space", None)
        if target_space not in {"a", "z"}:
            raise ValueError("target_space must be 'a' or 'z' for coeff metrics")
        # REVIEW: coeff metrics use coeff_meta-derived domain for consistency.
        coeff_meta = load_coeff_meta(str(train_run_dir))
        raw_meta = coeff_meta.get("raw_meta") if isinstance(coeff_meta, Mapping) else None
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else coeff_meta
        field_shape = tuple(coeff_meta.get("field_shape", []))
        if not field_shape:
            raise ValueError("coeff_meta.field_shape is required for coeff metrics")
        domain_spec = build_domain_spec(domain_cfg, field_shape)
        coeffs = []
        for idx in range(fields_coeff.shape[0]):
            raw_coeff = decomposer.transform(
                fields_coeff[idx],
                mask=None if masks_coeff is None else masks_coeff[idx],
                domain_spec=domain_spec,
            )
            coeffs.append(codec.encode(raw_coeff, raw_meta))
        coeff_true_a = np.stack(coeffs, axis=0)
        coeff_true_z = coeff_post.transform(coeff_true_a)
        if target_space == "a":
            coeff_pred_a = coeff_pred
            coeff_pred_z = coeff_post.transform(coeff_pred_a)
        else:
            coeff_pred_z = coeff_pred
            coeff_pred_a = coeff_post.inverse_transform(coeff_pred_z)
        if coeff_true_a.shape != coeff_pred_a.shape:
            raise ValueError("coeff_true_a shape does not match coeff_pred_a shape")
        if coeff_true_z.shape != coeff_pred_z.shape:
            raise ValueError("coeff_true_z shape does not match coeff_pred_z shape")

    with steps.step(
        "compute_metrics",
        outputs=[artifact_ref("metrics.json", kind="metrics")],
    ):
        metrics = compute_metrics(
            metrics_list,
            field_true=fields_true,
            field_pred=field_pred,
            mask=masks,
            coeff_true_a=coeff_true_a,
            coeff_pred_a=coeff_pred_a,
            coeff_true_z=coeff_true_z,
            coeff_pred_z=coeff_pred_z,
            coeff_meta=coeff_meta,
            grid_spacing=grid_spacing,
        )

        # CONTRACT: metrics are persisted under metrics.json.
        writer.write_metrics(metrics)

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    writer.write_manifest(meta=meta, dataset_meta=dataset_meta, steps=steps.to_list())
    maybe_log_run(cfg, run_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
