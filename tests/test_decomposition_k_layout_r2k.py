from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def test_decomposition_field_r2k_supports_k_layout(tmp_path: Path) -> None:
    from mode_decomp_ml.pipeline import read_json
    from processes.decomposition import main as decomp_main

    n, h, w, c = 6, 8, 8, 1
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pat = np.sin(np.pi * xx) + 0.5 * np.cos(np.pi * yy)
    pat = (pat - float(np.mean(pat))).astype(np.float32)
    pat /= float(np.std(pat) + 1e-8)

    cond = np.zeros((n, 2), dtype=np.float32)
    field = np.zeros((n, h, w, c), dtype=np.float32)
    mask = np.ones((n, h, w), dtype=bool)
    for i in range(n):
        offset = 5.0 + 0.2 * float(i)
        amp = 0.6 + 0.1 * float(i)
        cond[i, 0] = offset
        cond[i, 1] = amp
        field[i, :, :, 0] = offset + amp * pat

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
        "decompose": {
            "name": "autoencoder",
            "latent_dim": 6,
            "hidden_channels": [4],
            "epochs": 8,
            "batch_size": 2,
            "lr": 1.0e-2,
            "weight_decay": 0.0,
            "mask_policy": "error",
            "device": "cpu",
            "seed": 0,
        },
        "codec": {"name": "none"},
        "preprocess": {"name": "none"},
        "eval": {"metrics": ["field_rmse", "field_r2", "energy_cumsum"]},
        "viz": {
            "sample_index": 0,
            "max_samples": 1,
            "coeff_diag": {"enabled": False},
            "data_driven": {"enabled": False},
            "spatial_stats": {"enabled": False},
            "validity": {"enabled": False},
            "key_dashboard": {"enabled": False},
        },
    }

    assert decomp_main(cfg) == 0
    metrics = read_json(run_dir / "outputs" / "metrics.json")
    assert "field_r2_k1" in metrics
    assert "field_r2_k4" in metrics

