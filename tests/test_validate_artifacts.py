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
            "task": "decomposition",
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_artifacts_decomposition_pass(tmp_path: Path) -> None:
    validator = _load_validator()

    run_dir = tmp_path / "decomposition"
    (run_dir / "configuration").mkdir(parents=True)
    (run_dir / "outputs").mkdir(parents=True)
    config = {
        "task": {
            "name": "decomposition",
        }
    }
    (run_dir / "configuration" / "run.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "outputs" / "preds.npz").write_bytes(b"0")
    (run_dir / "outputs" / "coeffs.npz").write_bytes(b"0")
    _write_manifest(run_dir / "outputs" / "manifest_run.json")

    assert validator.main([str(run_dir)]) == 0
