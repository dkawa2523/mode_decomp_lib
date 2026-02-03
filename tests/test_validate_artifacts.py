from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


def _load_validator():
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path) -> None:
    payload = {
        "meta": {
            "seed": 123,
            "git": {"commit": "deadbeef"},
            "task": "eval",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_artifacts_eval_pass(tmp_path: Path) -> None:
    validator = _load_validator()

    train_dir = tmp_path / "train"
    (train_dir / "model").mkdir(parents=True)
    (train_dir / "model" / "model.pkl").write_bytes(b"0")

    predict_dir = tmp_path / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    (predict_dir / "preds.npz").write_bytes(b"0")

    reconstruct_dir = tmp_path / "reconstruct"
    reconstruct_dir.mkdir(parents=True, exist_ok=True)
    (reconstruct_dir / "preds.npz").write_bytes(b"0")

    run_dir = tmp_path / "eval"
    (run_dir / ".hydra").mkdir(parents=True)
    config = {
        "task": {
            "name": "eval",
            "train_run_dir": str(train_dir),
            "predict_run_dir": str(predict_dir),
            "reconstruct_run_dir": str(reconstruct_dir),
        }
    }
    (run_dir / ".hydra" / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    _write_manifest(run_dir / "manifest_run.json")

    assert validator.main([str(run_dir)]) == 0
