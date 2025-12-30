"""Process entrypoint: doctor."""
from __future__ import annotations

from typing import Iterable, Mapping

from mode_decomp_ml.data import build_dataset


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
    _require_keys(cfg, ["seed", "run_dir", "output_dir", "task", "dataset", "domain"])

    dataset = build_dataset(cfg["dataset"], domain_cfg=cfg.get("domain"), seed=cfg.get("seed"))
    sample = dataset[0]

    # REVIEW: log dataset schema shapes for smoke validation.
    print(f"dataset={getattr(dataset, 'name', 'unknown')}")
    print(f"cond shape: {_shape(sample.cond)}")
    print(f"field shape: {_shape(sample.field)}")
    print(f"mask shape: {_shape(sample.mask)}")
    print(f"meta keys: {sorted(sample.meta.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
