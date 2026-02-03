"""Process entrypoint: doctor."""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.pipeline import cfg_get, resolve_domain_cfg


_LOGGER = logging.getLogger(__name__)


def _require_keys(cfg: Mapping[str, object], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")


def _shape(value: object) -> str:
    if value is None:
        return "None"
    shape = getattr(value, "shape", None)
    return str(tuple(shape)) if shape is not None else "unknown"


def main(cfg: Mapping[str, object] | None = None) -> int:
    # CONTRACT: doctor fails fast if core config is missing.
    if cfg is None:
        raise ValueError("doctor requires config from the Hydra entrypoint")
    _require_keys(cfg, ["seed", "run_dir", "output_dir", "task", "dataset"])

    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, domain_cfg)
    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=cfg.get("seed"))
    sample = dataset[0]

    # REVIEW: log dataset schema shapes for smoke validation.
    _LOGGER.info("dataset=%s", getattr(dataset, "name", "unknown"))
    _LOGGER.info("cond shape: %s", _shape(sample.cond))
    _LOGGER.info("field shape: %s", _shape(sample.field))
    _LOGGER.info("mask shape: %s", _shape(sample.mask))
    if getattr(dataset, "mask_generated", False):
        _LOGGER.info("mask generated: full_true")
    _LOGGER.info("meta keys: %s", sorted(sample.meta.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
