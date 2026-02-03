from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.pipeline import resolve_domain_cfg


def _write_manifest(root: Path, payload: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_arrays(root: Path, *, height: int, width: int, channels: int, n_samples: int = 2) -> None:
    cond = np.zeros((n_samples, 3), dtype=np.float32)
    field = np.zeros((n_samples, height, width, channels), dtype=np.float32)
    np.save(root / "cond.npy", cond)
    np.save(root / "field.npy", field)


def test_manifest_disk_domain_inference(tmp_path: Path) -> None:
    height, width = 8, 6
    _write_arrays(tmp_path, height=height, width=width, channels=1)
    _write_manifest(
        tmp_path,
        {
            "field_kind": "scalar",
            "grid": {"H": height, "W": width, "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
            "domain": {"type": "disk", "center": [0.5, 0.5], "radius": 0.4},
        },
    )

    dataset_cfg = {"name": "npy_dir", "root": str(tmp_path), "mask_policy": "allow_none"}
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, None)
    assert domain_cfg["name"] == "disk"

    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=0)
    sample = dataset[0]
    assert sample.mask is not None
    assert sample.mask.shape == (height, width)
    assert sample.mask.all()


def test_manifest_rectangle_domain_inference(tmp_path: Path) -> None:
    height, width = 5, 7
    _write_arrays(tmp_path, height=height, width=width, channels=1)
    _write_manifest(
        tmp_path,
        {
            "field_kind": "scalar",
            "grid": {"H": height, "W": width, "x_range": [-1.0, 2.0], "y_range": [0.0, 3.0]},
            "domain": {"type": "rectangle"},
        },
    )

    dataset_cfg = {"name": "npy_dir", "root": str(tmp_path), "mask_policy": "allow_none"}
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, None)
    assert domain_cfg["name"] == "rectangle"
    assert domain_cfg["x_range"] == (-1.0, 2.0)
    assert domain_cfg["y_range"] == (0.0, 3.0)


def test_manifest_annulus_domain_inference(tmp_path: Path) -> None:
    height, width = 6, 6
    _write_arrays(tmp_path, height=height, width=width, channels=1)
    _write_manifest(
        tmp_path,
        {
            "field_kind": "scalar",
            "grid": {"H": height, "W": width, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
            "domain": {"type": "annulus", "center": [0.0, 0.0], "r_inner": 0.2, "r_outer": 0.5},
        },
    )

    dataset_cfg = {"name": "npy_dir", "root": str(tmp_path), "mask_policy": "allow_none"}
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, None)
    assert domain_cfg["name"] == "annulus"
    assert domain_cfg["r_inner"] == 0.2
    assert domain_cfg["r_outer"] == 0.5


def test_manifest_sphere_grid_domain_inference(tmp_path: Path) -> None:
    height, width = 6, 12
    _write_arrays(tmp_path, height=height, width=width, channels=1)
    _write_manifest(
        tmp_path,
        {
            "field_kind": "scalar",
            "grid": {"H": height, "W": width, "x_range": [0.0, 360.0], "y_range": [-90.0, 90.0]},
            "domain": {"type": "sphere_grid", "radius": 1.0, "angle_unit": "deg"},
        },
    )

    dataset_cfg = {"name": "npy_dir", "root": str(tmp_path), "mask_policy": "allow_none"}
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, None)
    assert domain_cfg["name"] == "sphere_grid"
    assert domain_cfg["radius"] == 1.0


def test_manifest_arbitrary_mask_defaults_to_dataset(tmp_path: Path) -> None:
    height, width = 4, 4
    _write_arrays(tmp_path, height=height, width=width, channels=1)
    _write_manifest(
        tmp_path,
        {
            "field_kind": "scalar",
            "grid": {"H": height, "W": width, "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
            "domain": {"type": "arbitrary_mask"},
        },
    )

    dataset_cfg = {"name": "npy_dir", "root": str(tmp_path), "mask_policy": "allow_none"}
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, None)
    assert domain_cfg["name"] == "arbitrary_mask"
    assert domain_cfg["mask_source"] == "dataset"

    dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=0)
    sample = dataset[0]
    assert sample.mask is not None
    assert sample.mask.shape == (height, width)
    assert sample.mask.all()
