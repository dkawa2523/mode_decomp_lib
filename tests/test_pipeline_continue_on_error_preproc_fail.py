from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _write_npy_dir_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 4
    h = 16
    w = 16
    cond = np.ones((n, 1), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(xx / 3.0) + np.cos(yy / 5.0)).astype(np.float32)
    field = np.stack([(1.0 + 0.05 * i) + base for i in range(n)], axis=0).astype(np.float32)
    field = field[..., None]  # (N,H,W,1)
    np.save(root / "cond.npy", cond)
    np.save(root / "field.npy", field)
    manifest = {
        "field_kind": "scalar",
        "grid": {"H": h, "W": w, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        "domain": {"type": "rectangle"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_pipeline_continue_on_error_preprocessing_failure_writes_leaderboard(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_npy_dir_dataset(dataset_root)

    # Lazy import pattern consistent with tools/bench/run_benchmark_v1.py.
    import sys

    project_root = Path(__file__).resolve().parents[1]
    for p in (str(project_root / "src"), str(project_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from processes import pipeline as pipeline_process

    run_root = tmp_path / "runs"
    run_dir = run_root / "pipeline"
    output_root = run_root

    # Intentionally make preprocessing fail: pca n_components must be > 0.
    cfg = {
        "seed": 0,
        "task": {
            "name": "pipeline",
            "stages": ["decomposition", "preprocessing"],
            "decompose_list": ["dct2"],
            "coeff_post_list": ["pca"],
            "model_list": ["ridge"],
            "continue_on_error": True,
            "train_sort_by": "val_rmse",
        },
        "run_dir": str(run_dir),
        "output": {"root": str(output_root), "name": "pipeline"},
        "dataset": {"name": "npy_dir", "root": str(dataset_root), "mask_policy": "allow_none"},
        "domain": {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        "split": {"name": "all"},
        "preprocess": {"name": "basic"},
        "eval": {"name": "basic", "metrics": ["field_rmse"]},
        "codec": {"name": "none"},
        "coeff_post": {"name": "pca", "n_components": 0},
        "model": {"name": "ridge"},
        "train": {"name": "basic"},
        "viz": {"name": "basic"},
    }

    ret = pipeline_process.main(cfg)
    assert ret == 0

    train_csv = run_dir / "outputs" / "tables" / "leaderboard_train.csv"
    assert train_csv.exists()
    rows = _read_csv(train_csv)
    assert rows, "expected at least one skipped row due to preprocessing failure"
    assert any(r.get("status") == "skipped" for r in rows)
    assert any("preprocessing failed:" in (r.get("error") or "") for r in rows)

