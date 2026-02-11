from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from processes import decomposition as decomposition_process
from processes import inference as inference_process
from processes import preprocessing as preprocessing_process
from processes import train as train_process


def _base_cfg(run_dir: Path, task_name: str) -> dict:
    output_dir = run_dir.parent.parent
    name = run_dir.parent.name
    return {
        "seed": 123,
        "output": {"root": str(output_dir), "name": name},
        "run_dir": str(run_dir),
        "task": {"name": task_name},
        "dataset": {
            "name": "synthetic",
            "num_samples": 6,
            "cond_dim": 3,
            "feature_columns": ["x1", "x2", "x3"],
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
        "eval": {"name": "basic", "metrics": ["field_rmse", "coeff_rmse_a", "energy_cumsum"]},
        "viz": {"name": "basic", "sample_index": 0, "k_list": [1, 2, 4]},
        "inference": {"mode": "single", "values": {"x1": 0.1, "x2": 0.2, "x3": 0.3}},
    }


def test_process_pipeline_e2e(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    name = "e2e"
    decomposition_dir = output_root / name / "decomposition"
    preprocessing_dir = output_root / name / "preprocessing"
    train_dir = output_root / name / "train"
    inference_dir = output_root / name / "inference"

    decomposition_cfg = _base_cfg(decomposition_dir, "decomposition")
    assert decomposition_process.main(decomposition_cfg) == 0

    preprocessing_cfg = _base_cfg(preprocessing_dir, "preprocessing")
    preprocessing_cfg["task"]["decomposition_run_dir"] = str(decomposition_dir)
    assert preprocessing_process.main(preprocessing_cfg) == 0

    train_cfg = _base_cfg(train_dir, "train")
    train_cfg["task"]["preprocessing_run_dir"] = str(preprocessing_dir)
    assert train_process.main(train_cfg) == 0

    inference_cfg = _base_cfg(inference_dir, "inference")
    inference_cfg["task"]["decomposition_run_dir"] = str(decomposition_dir)
    inference_cfg["task"]["preprocessing_run_dir"] = str(preprocessing_dir)
    inference_cfg["task"]["train_run_dir"] = str(train_dir)
    assert inference_process.main(inference_cfg) == 0

    assert (decomposition_dir / "configuration" / "run.yaml").exists()
    assert (decomposition_dir / "outputs" / "coeffs.npz").exists()
    assert (decomposition_dir / "outputs" / "preds.npz").exists()
    assert (decomposition_dir / "outputs" / "metrics.json").exists()
    assert (decomposition_dir / "outputs" / "states" / "decomposer" / "state.pkl").exists()
    assert (decomposition_dir / "outputs" / "states" / "coeff_codec" / "state.pkl").exists()
    assert (decomposition_dir / "outputs" / "states" / "preprocess" / "state.pkl").exists()
    decomp_manifest = json.loads(
        (decomposition_dir / "outputs" / "manifest_run.json").read_text(encoding="utf-8")
    )
    assert "steps" in decomp_manifest
    assert len(decomp_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in decomp_manifest["steps"])

    assert (preprocessing_dir / "outputs" / "coeffs.npz").exists()
    assert (preprocessing_dir / "outputs" / "states" / "coeff_post" / "state.pkl").exists()
    preprocess_manifest = json.loads(
        (preprocessing_dir / "outputs" / "manifest_run.json").read_text(encoding="utf-8")
    )
    assert "steps" in preprocess_manifest
    assert len(preprocess_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in preprocess_manifest["steps"])

    assert (train_dir / "model" / "model.pkl").exists()
    train_manifest = json.loads((train_dir / "outputs" / "manifest_run.json").read_text(encoding="utf-8"))
    assert "steps" in train_manifest
    assert len(train_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in train_manifest["steps"])
    train_metrics_path = train_dir / "outputs" / "metrics.json"
    assert train_metrics_path.exists()
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    assert "fit_time_sec" in train_metrics
    assert train_metrics["fit_time_sec"] >= 0.0
    # Field-space metrics should be present when viz.field_eval is enabled (default).
    assert ("val_field_r2" in train_metrics) or ("train_field_r2" in train_metrics)

    with np.load(inference_dir / "outputs" / "preds.npz") as data:
        coeff_pred = data["coeff"]
        field_pred = data["field"]
    assert coeff_pred.ndim == 2
    assert field_pred.ndim == 4
    infer_manifest = json.loads((inference_dir / "outputs" / "manifest_run.json").read_text(encoding="utf-8"))
    infer_meta = infer_manifest["meta"]
    upstream = infer_meta["upstream_artifacts"]
    assert upstream["decomposition_run_dir"] == str(decomposition_dir)
    assert upstream["preprocessing_run_dir"] == str(preprocessing_dir)
    assert upstream["train_run_dir"] == str(train_dir)
    assert "steps" in infer_manifest
    assert len(infer_manifest["steps"]) > 0
    assert all(step["status"] == "ok" for step in infer_manifest["steps"])
    assert (inference_dir / "plots" / "field_pred_0000.png").exists()
