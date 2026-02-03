"""Process entrypoint: predict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    build_meta,
    cfg_get,
    dataset_to_arrays,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_path,
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.pipeline.loaders import load_model_state


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for predict")
    return task_cfg


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("predict requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "split"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    if train_run_dir is None or str(train_run_dir).strip() == "":
        raise ValueError("task.train_run_dir is required for predict")

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
    with steps.step(
        "build_dataset",
        inputs=[artifact_ref("run.yaml", kind="config")],
    ) as step:
        dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
        conds, _, _, _ = dataset_to_arrays(dataset)
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(dataset))

    with steps.step(
        "load_model",
        inputs=[artifact_ref(Path(str(train_run_dir)) / "model" / "model.pkl", kind="model")],
    ):
        model = load_model_state(str(train_run_dir))

    with steps.step(
        "predict_coeff",
        outputs=[artifact_ref("preds.npz", kind="preds")],
    ):
        coeff_pred, coeff_std = model.predict_with_std(conds)
        coeff_pred = np.asarray(coeff_pred)
        coeff_std = None if coeff_std is None else np.asarray(coeff_std)

        preds_payload = {"coeff": coeff_pred, "coeff_mean": coeff_pred}
        if coeff_std is not None:
            if coeff_std.shape != coeff_pred.shape:
                raise ValueError("coeff_std shape does not match coeff_mean")
            preds_payload["coeff_std"] = coeff_std
        # CONTRACT: predictions are persisted under preds.npz.
        writer.write_preds(preds_payload)
        preds_meta = {
            "target_space": getattr(model, "target_space", None),
            "num_samples": int(coeff_pred.shape[0]),
            "coeff_dim": int(coeff_pred.shape[1]) if coeff_pred.ndim == 2 else int(coeff_pred.shape[-1]),
            "train_run_dir": str(resolve_path(train_run_dir)),
            "coeff_std": coeff_std is not None,
        }

    split_meta = split_indices(split_cfg, len(dataset), seed)
    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    writer.write_manifest(
        meta=meta,
        dataset_meta=dataset_meta,
        preds_meta=preds_meta,
        steps=steps.to_list(),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
