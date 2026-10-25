"""Installed-wheel acceptance: public entry points, shipped examples and fresh inference."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from verify_extensions import verify_extensions


def _verify_source_tree(installed: Path, source: Path) -> int:
    """Dev acceptance check: a wheel must contain exactly the intended Python sources."""
    installed_files = {path.relative_to(installed): path for path in installed.rglob("*.py") if path.is_file()}
    source_files = {path.relative_to(source): path for path in source.rglob("*.py") if path.is_file()}
    if installed_files.keys() != source_files.keys():
        extra = sorted(str(path) for path in installed_files.keys() - source_files.keys())
        missing = sorted(str(path) for path in source_files.keys() - installed_files.keys())
        raise ValueError(f"Wheel Python source file set differs: extra={extra}, missing={missing}")
    changed = [
        str(path)
        for path in sorted(source_files)
        if installed_files[path].read_bytes() != source_files[path].read_bytes()
    ]
    if changed:
        raise ValueError(f"Wheel Python source bytes differ: {changed}")
    return len(source_files)


def _verify_analysis(training: Path, workspace: Path):
    from ml_platform_mode_decomp import run_task

    results = {}
    for kind in ("representation", "surrogate", "imputation"):
        decomposition = {"name": "gappy_pod", "params": {"n_modes": 2}} if kind == "imputation" else {"name": "dct2"}
        result = run_task(
            {
                "task": "mode_decomp_analysis",
                "data": {"local_path": str(training)},
                "runtime": {"output_dir": str(workspace / "analysis" / kind)},
                "analysis": {"kind": kind, "split_seeds": [11], "model_seeds": [42], "bootstrap_samples": 0},
                "model": {"candidates": [{"id": "example", "representation": {"decomposition": decomposition}}]},
                "report": {"enabled": False},
            }
        )
        summary = json.loads(result.artifacts["analysis_summary"].read_text(encoding="utf-8"))
        if not summary["complete"] or not result.tables["comparison"].is_file():
            raise ValueError(f"Installed {kind} analysis did not complete")
        results[kind] = str(result.run_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, help="A new directory for this acceptance")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = Path(args.work_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import ml_platform_mode_decomp as package
    from ml_platform_mode_decomp import load_config, run_task

    installed = Path(package.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Use the installed wheel interpreter, not an editable installation")
    source_files = _verify_source_tree(installed.parent, root / "pkgs/mode_decomp/src/ml_platform_mode_decomp")
    for name in ("clearml", "ml_platform_spdml", "ml_platform_tabular", "mode_decomp_ml"):
        if importlib.util.find_spec(name) is not None:
            raise ValueError(f"The standalone acceptance must not contain {name}")
    training = workspace / "training_input"
    subprocess.run(
        [sys.executable, str(root / "examples/create_tiny_dataset.py"), "--output", str(training)],
        check=True,
        capture_output=True,
        text=True,
        cwd=workspace,
    )
    models = {}
    for name, filename in (("fixed", "mode_decomp_pipeline.yaml"), ("cv", "mode_decomp_search.yaml")):
        cfg = load_config(
            root / "config/tasks" / filename,
            overrides={
                "data": {"local_path": str(training)},
                "runtime": {"output_dir": str(workspace / name)},
            },
        )
        cfg["model"]["candidates"] = cfg["model"]["candidates"][:1]
        if name == "cv":
            cfg["tuning"].update(n_trials=1, search_spaces={})
            cfg["tuning"]["cv"]["n_splits"] = 2
        result = run_task(cfg)
        summary = json.loads(result.artifacts["result_summary"].read_text(encoding="utf-8"))
        scope = "fixed_configuration" if name == "fixed" else "inner_cv"
        if summary["selection_scope"] != scope or not np.isfinite(result.metrics["field_rmse"]):
            raise ValueError(f"Installed {name} pipeline returned the wrong selection scope")
        models[name] = result
    analyses = _verify_analysis(training, workspace)
    training.rename(workspace / "hidden_training_input")
    conditions = workspace / "new.csv"
    conditions.write_text("sample_id,x2,x1\nnew,0.5,0.2\n", encoding="utf-8")
    for name, model in models.items():
        suffix = "npz" if name == "fixed" else "npy"
        command = [
            sys.executable,
            "-m",
            "ml_platform_mode_decomp",
            str(root / "config/tasks/mode_decomp_infer.yaml"),
            "--set",
            f"data.local_path={conditions}",
            "--set",
            f"model.artifact_path={model.artifacts['model']}",
            "--set",
            f"runtime.output_dir={workspace / 'predictions' / name}",
            "--set",
            f"output.prediction_name=predictions.{suffix}",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, cwd=workspace)
        inference = json.loads(completed.stdout)
        if suffix == "npz":
            with np.load(inference["artifacts"]["predictions"], allow_pickle=False) as saved:
                fields, ids = saved["predictions"], saved["sample_ids"]
        else:
            fields = np.load(inference["artifacts"]["predictions"], mmap_mode="r", allow_pickle=False)
            ids = np.load(inference["artifacts"]["prediction_sample_ids"], allow_pickle=False)
            metadata = json.loads(Path(inference["artifacts"]["prediction_metadata"]).read_text(encoding="utf-8"))
            if metadata["arrays"]["predictions"] != "predictions.npy":
                raise ValueError("NPY metadata does not identify the saved prediction array")
        if fields.shape != (1, 10, 12, 1) or not np.isfinite(fields).all():
            raise ValueError(f"Invalid installed {name} model prediction")
        if ids.tolist() != ["new"]:
            raise ValueError("Condition CSV IDs were not preserved")
    report = {
        "success": True,
        "source_files": source_files,
        "source_match": True,
        "sdk_packages_absent": True,
        "training_hidden_before_fresh_process_inference": True,
        "analysis": analyses,
        "pipeline_scopes": ["fixed_configuration", "inner_cv"],
        "prediction_formats": ["npz", "npy"],
        "extensions": verify_extensions(root, workspace),
        "runtime": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "ml-platform-core", "ml-platform-mode-decomp")
        },
    }
    (workspace / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
