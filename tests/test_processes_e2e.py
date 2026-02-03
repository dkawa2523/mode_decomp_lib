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
    tag = run_dir.parent.name
    return {
        "seed": 123,
        "output_dir": str(output_dir),
        "tag": tag,
        "run_id": run_dir.name,
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
        "preprocess": {"name": "none"},
        "decompose": {"name": "dct2", "disk_policy": "error"},
        "codec": {"name": "none"},
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
    output_root = tmp_path / "runs"
    tag = "e2e"
    train_dir = output_root / tag / "train"
    predict_dir = output_root / tag / "predict"
    reconstruct_dir = output_root / tag / "reconstruct"
    eval_dir = output_root / tag / "eval"

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

    viz_dir = output_root / tag / "viz"
    viz_cfg = _base_cfg(viz_dir, "viz")
    viz_cfg["task"]["train_run_dir"] = str(train_dir)
    viz_cfg["task"]["predict_run_dir"] = str(predict_dir)
    viz_cfg["task"]["reconstruct_run_dir"] = str(reconstruct_dir)
    assert viz_process.main(viz_cfg) == 0

    assert (train_dir / "model" / "model.pkl").exists()
    assert (train_dir / "states" / "coeff_post" / "state.pkl").exists()
    assert (train_dir / "states" / "decomposer" / "coeff_meta.json").exists()
    assert (train_dir / "states" / "coeff_codec" / "state.pkl").exists()
    assert (train_dir / "states" / "coeff_meta.json").exists()
    assert (train_dir / "manifest_run.json").exists()
    train_manifest = json.loads((train_dir / "manifest_run.json").read_text(encoding="utf-8"))
    assert "steps" in train_manifest
    assert len(train_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in train_manifest["steps"])
    train_metrics_path = train_dir / "metrics.json"
    assert train_metrics_path.exists()
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    assert "fit_time_sec" in train_metrics
    assert train_metrics["fit_time_sec"] >= 0.0

    with np.load(predict_dir / "preds.npz") as data:
        coeff_pred = data["coeff"]
    assert coeff_pred.ndim == 2
    predict_manifest = json.loads((predict_dir / "manifest_run.json").read_text(encoding="utf-8"))
    predict_meta = predict_manifest["meta"]
    assert predict_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert "steps" in predict_manifest
    assert len(predict_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in predict_manifest["steps"])

    with np.load(reconstruct_dir / "preds.npz") as data:
        field_pred = data["field"]
    assert field_pred.ndim == 4
    reconstruct_manifest = json.loads((reconstruct_dir / "manifest_run.json").read_text(encoding="utf-8"))
    reconstruct_meta = reconstruct_manifest["meta"]
    assert reconstruct_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert reconstruct_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)
    assert "steps" in reconstruct_manifest
    assert len(reconstruct_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in reconstruct_manifest["steps"])

    metrics_path = eval_dir / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "field_rmse" in metrics
    assert "coeff_rmse_a" in metrics
    assert "coeff_rmse_z" in metrics
    assert "energy_cumsum" in metrics
    assert len(metrics["energy_cumsum"]) > 0
    eval_manifest = json.loads((eval_dir / "manifest_run.json").read_text(encoding="utf-8"))
    eval_meta = eval_manifest["meta"]
    assert eval_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert eval_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)
    assert eval_meta["upstream_artifacts"]["reconstruct_run_dir"] == str(reconstruct_dir)
    assert "steps" in eval_manifest
    assert len(eval_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in eval_manifest["steps"])

    viz_root = viz_dir / "figures"
    sample_dir = viz_root / "sample_0000"
    for name in ("field_compare.png", "error_map.png", "recon_sequence.png"):
        path = sample_dir / name
        assert path.exists()
        assert path.stat().st_size > 0
    spectrum_path = viz_root / "coeff_spectrum.png"
    assert spectrum_path.exists()
    assert spectrum_path.stat().st_size > 0
    for name in ("coeff_hist.png", "coeff_topk_energy.png"):
        path = viz_root / name
        assert path.exists()
        assert path.stat().st_size > 0
    viz_manifest = json.loads((viz_dir / "manifest_run.json").read_text(encoding="utf-8"))
    viz_meta = viz_manifest["meta"]
    assert viz_meta["upstream_artifacts"]["train_run_dir"] == str(train_dir)
    assert viz_meta["upstream_artifacts"]["predict_run_dir"] == str(predict_dir)
    assert viz_meta["upstream_artifacts"]["reconstruct_run_dir"] == str(reconstruct_dir)
    assert "steps" in viz_manifest
    assert len(viz_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in viz_manifest["steps"])
