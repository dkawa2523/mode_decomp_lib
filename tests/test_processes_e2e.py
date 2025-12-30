from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processes import eval as eval_process
from processes import predict as predict_process
from processes import reconstruct as reconstruct_process
from processes import train as train_process
from processes import viz as viz_process


def _base_cfg(run_dir: Path, task_name: str) -> dict:
    output_dir = run_dir.parent.parent
    return {
        "seed": 123,
        "output_dir": str(output_dir),
        "run_dir": str(run_dir),
        "task": {"name": task_name},
        "dataset": {
            "name": "synthetic",
            "num_samples": 6,
            "cond_dim": 3,
            "height": 16,
            "width": 16,
            "channels": 1,
            "mask_policy": "require",
            "mask_mode": "full",
            "mask_radius": 0.45,
        },
        "split": {"name": "all"},
        "domain": {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "decompose": {"name": "dct2", "disk_policy": "error"},
        "coeff_post": {"name": "none"},
        "model": {
            "name": "ridge",
            "target_space": "a",
            "alpha": 1.0,
            "cond_scaler": "standardize",
            "seed": 123,
        },
        "eval": {"name": "basic", "metrics": ["field_rmse", "coeff_rmse_a", "coeff_rmse_z", "energy_cumsum"]},
        "viz": {"name": "basic", "sample_index": 0, "k_list": [1, 2, 4]},
    }


def test_process_pipeline_e2e(tmp_path: Path) -> None:
    train_dir = tmp_path / "outputs" / "train" / "run"
    predict_dir = tmp_path / "outputs" / "predict" / "run"
    reconstruct_dir = tmp_path / "outputs" / "reconstruct" / "run"
    eval_dir = tmp_path / "outputs" / "eval" / "run"

    train_cfg = _base_cfg(train_dir, "train")
    assert train_process.main(train_cfg) == 0

    predict_cfg = _base_cfg(predict_dir, "predict")
    predict_cfg["task"]["train_run_dir"] = str(train_dir)
    assert predict_process.main(predict_cfg) == 0

    reconstruct_cfg = _base_cfg(reconstruct_dir, "reconstruct")
    reconstruct_cfg["task"]["train_run_dir"] = str(train_dir)
    reconstruct_cfg["task"]["predict_run_dir"] = str(predict_dir)
    assert reconstruct_process.main(reconstruct_cfg) == 0

    eval_cfg = _base_cfg(eval_dir, "eval")
    eval_cfg["task"]["train_run_dir"] = str(train_dir)
    eval_cfg["task"]["predict_run_dir"] = str(predict_dir)
    eval_cfg["task"]["reconstruct_run_dir"] = str(reconstruct_dir)
    assert eval_process.main(eval_cfg) == 0

    viz_dir = tmp_path / "outputs" / "viz" / "run"
    viz_cfg = _base_cfg(viz_dir, "viz")
    viz_cfg["task"]["train_run_dir"] = str(train_dir)
    viz_cfg["task"]["predict_run_dir"] = str(predict_dir)
    viz_cfg["task"]["reconstruct_run_dir"] = str(reconstruct_dir)
    assert viz_process.main(viz_cfg) == 0

    assert (train_dir / "artifacts" / "model" / "model.pkl").exists()
    assert (train_dir / "artifacts" / "coeff_post" / "state.pkl").exists()
    assert (train_dir / "artifacts" / "decomposer" / "coeff_meta.json").exists()
    assert (train_dir / "meta.json").exists()

    coeff_pred = np.load(predict_dir / "preds" / "coeff.npy")
    assert coeff_pred.ndim == 2
    assert (predict_dir / "preds" / "preds_meta.json").exists()
    assert (predict_dir / "meta.json").exists()
    predict_meta = json.loads((predict_dir / "meta.json").read_text(encoding="utf-8"))
    assert predict_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)

    field_pred = np.load(reconstruct_dir / "preds" / "field.npy")
    assert field_pred.ndim == 4
    assert (reconstruct_dir / "preds" / "preds_meta.json").exists()
    assert (reconstruct_dir / "meta.json").exists()
    reconstruct_meta = json.loads((reconstruct_dir / "meta.json").read_text(encoding="utf-8"))
    assert reconstruct_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert reconstruct_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)

    metrics_path = eval_dir / "metrics" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "field_rmse" in metrics
    assert "coeff_rmse_a" in metrics
    assert "coeff_rmse_z" in metrics
    assert "energy_cumsum" in metrics
    assert len(metrics["energy_cumsum"]) > 0
    eval_meta = json.loads((eval_dir / "meta.json").read_text(encoding="utf-8"))
    assert eval_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert eval_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)
    assert eval_meta["upstream_artifacts"]["reconstruct_run_dir"] == str(reconstruct_dir)

    viz_root = viz_dir / "viz"
    sample_dir = viz_root / "sample_0000"
    for name in ("field_compare.png", "error_map.png", "recon_sequence.png"):
        path = sample_dir / name
        assert path.exists()
        assert path.stat().st_size > 0
    spectrum_path = viz_root / "coeff_spectrum.png"
    assert spectrum_path.exists()
    assert spectrum_path.stat().st_size > 0
    viz_meta = json.loads((viz_dir / "meta.json").read_text(encoding="utf-8"))
    assert viz_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert viz_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)
    assert viz_meta["upstream_artifacts"]["reconstruct_run_dir"] == str(reconstruct_dir)
