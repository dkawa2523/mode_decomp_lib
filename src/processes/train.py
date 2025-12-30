"""Process entrypoint: train."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.coeff_post import build_coeff_post
from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.decompose import build_decomposer
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.models import build_regressor
from mode_decomp_ml.pipeline import (
    build_dataset_meta,
    build_meta,
    cfg_get,
    dataset_to_arrays,
    ensure_dir,
    require_cfg_keys,
    resolve_run_dir,
    split_indices,
    write_json,
)
from mode_decomp_ml.tracking import maybe_log_run


def _require_task_config(task_cfg: Mapping[str, Any]) -> None:
    if task_cfg is None:
        raise ValueError("task config is required for train")


class _SubsetDataset:
    def __init__(self, dataset: Any, indices: np.ndarray) -> None:
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=int)
        self.name = getattr(dataset, "name", "unknown")

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, index: int) -> Any:
        return self._dataset[int(self._indices[index])]


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("train requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "domain", "decompose", "coeff_post", "model", "split"])
    _require_task_config(cfg_get(cfg, "task", None))

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    split_cfg = cfg_get(cfg, "split")

    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
    conds, fields, masks, _ = dataset_to_arrays(dataset)
    split_meta = split_indices(split_cfg, len(dataset), seed)
    train_idx = np.asarray(split_meta["train_idx"], dtype=int)

    conds = conds[train_idx]
    fields = fields[train_idx]
    masks = masks[train_idx] if masks is not None else None

    # REVIEW: domain spec derives from training field shape for reproducible basis.
    domain_spec = build_domain_spec(domain_cfg, fields.shape[1:])
    decomposer = build_decomposer(cfg_get(cfg, "decompose"))
    # CONTRACT: data-driven decomposers must fit on the training split only.
    decomposer.fit(dataset=_SubsetDataset(dataset, train_idx), domain_spec=domain_spec)

    coeffs = []
    for idx in range(fields.shape[0]):
        coeffs.append(
            decomposer.transform(
                fields[idx],
                mask=None if masks is None else masks[idx],
                domain_spec=domain_spec,
            )
        )
    A = np.stack(coeffs, axis=0)

    decomposer.save_coeff_meta(run_dir)
    decomposer.save_state(run_dir)

    coeff_post = build_coeff_post(cfg_get(cfg, "coeff_post"))
    model = build_regressor(cfg_get(cfg, "model"))
    # CONTRACT: target_space=a cannot be paired with non-identity coeff_post.
    if model.target_space == "a" and coeff_post.name != "none":
        raise ValueError("model.target_space=a requires coeff_post.name=none")

    coeff_post.fit(A, split="train")
    coeff_post.save_state(run_dir)

    targets = coeff_post.transform(A) if model.target_space == "z" else A
    model.fit(conds, targets)
    model.save_state(run_dir)

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    write_json(run_dir / "artifacts" / "dataset_meta.json", dataset_meta)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    write_json(run_dir / "meta.json", meta)
    maybe_log_run(cfg, run_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
