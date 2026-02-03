"""Process entrypoint: train."""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.plugins.coeff_post import build_coeff_post
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.data import FieldSample, build_dataset
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.models import build_regressor
from mode_decomp_ml.preprocess import build_preprocess
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
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.tracking import maybe_log_run


def _require_task_config(task_cfg: Mapping[str, Any]) -> None:
    if task_cfg is None:
        raise ValueError("task config is required for train")


class _ArrayDataset:
    def __init__(
        self,
        conds: np.ndarray,
        fields: np.ndarray,
        masks: np.ndarray | None,
        metas: list[dict[str, Any]],
        *,
        name: str,
    ) -> None:
        self._conds = conds
        self._fields = fields
        self._masks = masks
        self._metas = metas
        self.name = name

    def __len__(self) -> int:
        return int(self._conds.shape[0])

    def __getitem__(self, index: int) -> FieldSample:
        idx = int(index)
        return FieldSample(
            cond=self._conds[idx],
            field=self._fields[idx],
            mask=None if self._masks is None else self._masks[idx],
            meta=self._metas[idx],
        )


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("train requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "decompose", "codec", "coeff_post", "model", "split"])
    _require_task_config(cfg_get(cfg, "task", None))

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
        conds, fields, masks, metas = dataset_to_arrays(dataset)
        split_meta = split_indices(split_cfg, len(dataset), seed)
        train_idx = np.asarray(split_meta["train_idx"], dtype=int)
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(dataset))

    conds = conds[train_idx]
    fields = fields[train_idx]
    masks = masks[train_idx] if masks is not None else None
    metas_train = [metas[idx] for idx in train_idx.tolist()]

    with steps.step(
        "preprocess_fit",
        outputs=[artifact_ref("states/preprocess/state.pkl", kind="state")],
    ):
        preprocess = build_preprocess(cfg_get(cfg, "preprocess"))
        preprocess.fit(fields, masks, split="train")
        fields, masks = preprocess.transform(fields, masks)
        preprocess.save_state(run_dir)

    # REVIEW: domain spec derives from training field shape for reproducible basis.
    domain_spec = build_domain_spec(domain_cfg, fields.shape[1:])
    decomposer = build_decomposer(cfg_get(cfg, "decompose"))
    codec = build_coeff_codec(cfg_get(cfg, "codec"))
    # CONTRACT: data-driven decomposers must fit on the training split only.
    fit_time_sec: float | None = None
    with steps.step(
        "fit_decomposer",
        outputs=[artifact_ref("states/decomposer/state.pkl", kind="state")],
    ) as step:
        start = time.perf_counter()
        decomposer.fit(
            dataset=_ArrayDataset(conds, fields, masks, metas_train, name=dataset.name),
            domain_spec=domain_spec,
        )
        fit_time_sec = float(time.perf_counter() - start)
        step["meta"]["fit_time_sec"] = fit_time_sec

    coeffs = []
    raw_meta: Mapping[str, Any] | None = None
    with steps.step(
        "encode_coeffs",
        outputs=[
            artifact_ref("states/decomposer/coeff_meta.json", kind="metadata"),
            artifact_ref("states/decomposer/state.pkl", kind="state"),
            artifact_ref("states/coeff_meta.json", kind="metadata"),
            artifact_ref("states/coeff_codec/coeff_meta.json", kind="metadata"),
            artifact_ref("states/coeff_codec/state.pkl", kind="state"),
        ],
    ):
        for idx in range(fields.shape[0]):
            raw_coeff = decomposer.transform(
                fields[idx],
                mask=None if masks is None else masks[idx],
                domain_spec=domain_spec,
            )
            if raw_meta is None:
                raw_meta = decomposer.coeff_meta()
            coeffs.append(codec.encode(raw_coeff, raw_meta))
        A = np.stack(coeffs, axis=0)
        if raw_meta is None:
            raise ValueError("coeff_meta missing after decomposer.transform")

        decomposer.save_coeff_meta(run_dir)
        decomposer.save_state(run_dir)
        codec.save_coeff_meta(run_dir, raw_meta, A[0])
        codec.save_state(run_dir)

    coeff_post = build_coeff_post(cfg_get(cfg, "coeff_post"))
    model = build_regressor(cfg_get(cfg, "model"))
    # CONTRACT: target_space=a cannot be paired with non-identity coeff_post.
    if model.target_space == "a" and coeff_post.name != "none":
        raise ValueError("model.target_space=a requires coeff_post.name=none")

    with steps.step(
        "fit_coeff_post",
        outputs=[artifact_ref("states/coeff_post/state.pkl", kind="state")],
    ):
        coeff_post.fit(A, split="train")
        coeff_post.save_state(run_dir)

    with steps.step(
        "fit_model",
        outputs=[artifact_ref("model/model.pkl", kind="model")],
    ):
        targets = coeff_post.transform(A) if model.target_space == "z" else A
        model.fit(conds, targets)
        model.save_state(run_dir)

    if fit_time_sec is not None:
        with steps.step(
            "write_metrics",
            outputs=[artifact_ref("metrics.json", kind="metrics")],
        ):
            writer.write_metrics({"fit_time_sec": fit_time_sec})

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    writer.write_manifest(meta=meta, dataset_meta=dataset_meta, steps=steps.to_list())
    maybe_log_run(cfg, run_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
