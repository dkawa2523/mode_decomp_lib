from __future__ import annotations

from pathlib import Path

import numpy as np


def test_decomposition_field_r2k_supports_offset_residual(tmp_path: Path) -> None:
    from mode_decomp_ml.pipeline import read_json
    from processes.decomposition import main as decomp_main

    n, h, w, c = 2, 8, 8, 1
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pat = np.sin(np.pi * xx) + 0.5 * np.cos(np.pi * yy)
    pat = (pat - float(np.mean(pat))).astype(np.float32)
    field = np.zeros((n, h, w, c), dtype=np.float32)
    for i in range(n):
        field[i, :, :, 0] = 10.0 + 0.5 * pat
    cond = np.zeros((n, 2), dtype=np.float32)
    mask = np.ones((n, h, w), dtype=bool)

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    np.save(dataset_root / "cond.npy", cond)
    np.save(dataset_root / "field.npy", field)
    np.save(dataset_root / "mask.npy", mask)

    run_dir = tmp_path / "run"
    cfg = {
        "seed": 0,
        "run_dir": str(run_dir),
        "task": {"name": "decomposition"},
        "dataset": {"name": "npy_dir", "root": str(dataset_root), "mask_policy": "require"},
        "split": {"name": "all"},
        "domain": {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        "decompose": {"name": "dct2", "disk_policy": "error"},
        "codec": {"name": "auto_codec_v1"},
        "offset_split": {
            "enabled": True,
            "f_offset": 5.0,
            "max_samples": 2,
            "seed": 0,
            "min_residual_rms": 1e-8,
            "offset_def": "mean_per_channel",
            "agg": "median",
        },
        "preprocess": {"name": "none"},
        "eval": {"metrics": ["field_rmse", "coeff_rmse_a", "energy_cumsum"]},
        "viz": {
            "sample_index": 0,
            "max_samples": 1,
            "coeff_diag": {"enabled": False},
            "data_driven": {"enabled": False},
            "spatial_stats": {"enabled": False},
            "validity": {"enabled": False},
        },
    }

    assert decomp_main(cfg) == 0
    metrics = read_json(run_dir / "outputs" / "metrics.json")
    assert "field_r2_k1" in metrics
    assert "field_r2_k4" in metrics
    assert metrics.get("offset_split_enabled") is True
