from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mode_decomp_ml.data import build_dataset


def test_synthetic_dataset_shapes() -> None:
    cfg = {
        "name": "synthetic",
        "num_samples": 2,
        "cond_dim": 4,
        "height": 8,
        "width": 6,
        "channels": 2,
        "mask_policy": "require",
        "mask_mode": "full",
    }
    domain_cfg = {"name": "rectangle"}
    dataset = build_dataset(cfg, domain_cfg=domain_cfg, seed=123)

    sample = dataset[0]
    assert sample.cond.shape == (4,)
    assert sample.field.shape == (8, 6, 2)
    assert sample.mask is not None
    assert sample.mask.shape == (8, 6)
    assert sample.meta["domain"] == "rectangle"


def test_npy_dir_dataset_loads(tmp_path: Path) -> None:
    cond = np.ones((2, 3), dtype=np.float32)
    field = np.zeros((2, 4, 5, 1), dtype=np.float32)
    mask = np.ones((2, 4, 5), dtype=bool)

    np.save(tmp_path / "cond.npy", cond)
    np.save(tmp_path / "field.npy", field)
    np.save(tmp_path / "mask.npy", mask)

    cfg = {
        "name": "npy_dir",
        "root": str(tmp_path),
        "mask_policy": "require",
    }
    dataset = build_dataset(cfg, domain_cfg={"name": "rectangle"}, seed=0)

    sample = dataset[1]
    assert sample.cond.shape == (3,)
    assert sample.field.shape == (4, 5, 1)
    assert sample.mask is not None
    assert sample.mask.shape == (4, 5)


def test_npy_dir_requires_mask(tmp_path: Path) -> None:
    cond = np.ones((1, 2), dtype=np.float32)
    field = np.zeros((1, 3, 3, 1), dtype=np.float32)

    np.save(tmp_path / "cond.npy", cond)
    np.save(tmp_path / "field.npy", field)

    cfg = {
        "name": "npy_dir",
        "root": str(tmp_path),
        "mask_policy": "require",
    }

    with pytest.raises(FileNotFoundError):
        build_dataset(cfg, domain_cfg={"name": "rectangle"}, seed=0)
