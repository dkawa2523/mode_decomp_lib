"""Accept one-file local execution and three shipped extensions on the wheel interpreter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], workspace: Path) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, cwd=workspace)
    if completed.returncode:
        raise ValueError(f"Extension command failed ({completed.returncode}): {command}\n{completed.stderr[-4000:]}")
    return completed.stdout


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_standalone(root: Path, workspace: Path) -> dict:
    import numpy as np
    import yaml
    from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

    source = workspace / "extension_scalar"
    source.mkdir()
    xx, yy = np.meshgrid(np.arange(5), np.arange(4))
    fields = (np.sin(xx) + np.cos(yy))[None, ..., None].astype(np.float32)
    np.save(source / "field.npy", fields, allow_pickle=False)
    config = {
        "task": "mode_decomp_representation",
        "data": {
            "local_path": source.name,
            "fields": {"path": "field.npy"},
        },
        "representation": {"decomposition": {"name": "fft2"}},
        "runtime": {"output_dir": "scalar_runs"},
    }
    config_path = source / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    payload = json.loads(_run([sys.executable, "-m", "ml_platform_mode_decomp", str(config_path)], workspace))
    summary = _read(Path(payload["artifacts"]["analysis_summary"]))
    if summary["metric_scope"] != "fitted_input_reconstruction" or summary["fit_sample_ids"] != ["sample_000000"]:
        raise ValueError("One-file scalar execution changed the direct-decomposition scope or generated IDs")
    with np.load(payload["artifacts"]["reconstructed_fields"], allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["reconstruction"], fields, atol=1e-6)
    data = FieldDataset.from_arrays(fields)
    model = FieldRepresentation({"decomposition": {"name": "fft2"}}).fit(data)
    np.testing.assert_allclose(model.inverse_transform(model.transform(data)), fields, atol=1e-6)
    if data.fields.dtype != fields.dtype:
        raise ValueError("The array-only entry point changed the source dtype")
    np.testing.assert_array_equal(data.fields, fields)
    if list(Path(payload["run_dir"]).rglob("split.json")) or (source / "field_dataset.yaml").exists():
        raise ValueError("Direct scalar decomposition must not need a Dataset manifest or holdout split")
    return {
        "run_dir": payload["run_dir"],
        "sample_count": 1,
        "field_shape": list(fields.shape[1:]),
        "field_rmse": payload["metrics"]["field_rmse"],
        "single_positional_yaml_without_profile_verified": True,
        "array_api_dtype_and_reconstruction_verified": True,
        "limit": "The direct fitted-input reconstruction does not measure unknown-sample performance.",
    }


def _verify_uncertainty(root: Path, workspace: Path) -> dict:
    import numpy as np
    from ml_platform_mode_decomp import load_config, run_task
    from ml_platform_mode_decomp.data import load_field_dataset
    from ml_platform_mode_decomp.model_artifact import load_model

    training, output = workspace / "extension_nonlinear", workspace / "uncertainty_runs"
    _run([sys.executable, str(root / "examples/create_analysis_dataset.py"), "--output", str(training)], workspace)
    cfg = load_config(
        root / "config/tasks/mode_decomp_uncertainty.yaml",
        overrides={"data": {"local_path": str(training)}, "runtime": {"output_dir": str(output)}},
    )
    result = run_task(cfg)
    split = _read(next(output.rglob("split.json")))
    proper, calibration, test = (set(split[key]) for key in ("train_ids", "calibration_ids", "test_ids"))
    if proper & calibration or proper & test or calibration & test or len(proper | calibration | test) != 120:
        raise ValueError("Uncertainty did not preserve the three disjoint nonlinear-example partitions")
    model = load_model(result.artifacts["model"], result.artifacts["model_info"])
    if (
        set(model.fit_sample_ids) != proper
        or set(model.representation.fit_sample_ids) != proper
        or set(model.calibration_sample_ids) != calibration
        or model.calibrator is None
    ):
        raise ValueError("The saved uncertainty model used calibration/test samples for fitting or lost its bands")
    for path in output.rglob("candidate_result.json"):
        candidate = _read(path)
        if not set(candidate["fit_ids"]) <= proper:
            raise ValueError("Uncertainty candidate selection fitted outside proper training")
    data = load_field_dataset(cfg["data"])
    reference = data.subset(split["test_indices"])
    if reference.conditions is None:
        raise ValueError("The uncertainty example requires condition inputs")
    expected = model.predict_interval(reference.conditions)
    with np.load(result.artifacts["predictions"], allow_pickle=False) as saved:
        for key, values in zip(("predictions", "lower", "upper"), expected, strict=True):
            np.testing.assert_array_equal(saved[key], values)
        included = (reference.fields >= saved["lower"]) & (reference.fields <= saved["upper"])
        empirical_coverage = float(included[:, model.calibrator.support, :].all(axis=(1, 2)).mean())
    if result.metrics["interval_sample_simultaneous_coverage"] != empirical_coverage:
        raise ValueError("Saved prediction bands and reported empirical field coverage disagree")

    conditions = np.asarray([[0.2, 0.5], [-0.4, 0.1], [0.0, -0.3]])
    condition_path = workspace / "extension_conditions.csv"
    with condition_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *reversed(model.condition_columns)])
        writer.writerows((f"new_{index}", *reversed(row)) for index, row in enumerate(conditions))
    inferred_values = model.predict_interval(conditions)
    # The loader may mmap field.npy; release its views before renaming on Windows.
    del reference, data
    training.rename(workspace / "hidden_extension_nonlinear")
    inference = {}
    for enabled in (True, False):
        payload = json.loads(
            _run(
                [
                    sys.executable,
                    "-m",
                    "ml_platform_mode_decomp",
                    str(root / "config/tasks/mode_decomp_uncertainty_infer.yaml"),
                    "--set",
                    f"data.local_path={condition_path}",
                    "--set",
                    f"model.artifact_path={result.artifacts['model']}",
                    "--set",
                    f"runtime.output_dir={workspace / 'uncertainty_inference' / str(enabled).lower()}",
                    "--set",
                    f"uncertainty.enabled={str(enabled).lower()}",
                ],
                workspace,
            )
        )
        path = Path(payload["artifacts"]["predictions"])
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["predictions"], inferred_values[0])
            if saved["sample_ids"].tolist() != [f"new_{index}" for index in range(3)]:
                raise ValueError("Uncertainty inference changed the condition sample IDs")
            if enabled:
                for key, values in zip(("lower", "upper"), inferred_values[1:], strict=True):
                    np.testing.assert_array_equal(saved[key], values)
                for key in ("method", "coverage", "scope", "normalization"):
                    if saved[f"interval_{key}"].item() != model.calibrator.summary()[key]:
                        raise ValueError(f"Inference did not preserve the saved interval {key}")
            elif any(key in saved for key in ("lower", "upper", "interval_method")):
                raise ValueError("Disabling uncertainty must emit point predictions only")
        inference["enabled" if enabled else "disabled"] = str(path)
    return {
        "run_dir": str(result.run_dir),
        "proper_training_count": len(proper),
        "calibration_count": len(calibration),
        "test_count": len(test),
        "coverage_target": model.calibrator.coverage,
        "empirical_simultaneous_coverage": empirical_coverage,
        "mean_width": result.metrics["interval_mean_width"],
        "field_rmse": result.metrics["field_rmse"],
        "inference": inference,
        "training_input_hidden_before_fresh_process_inference": True,
        "limit": "Marginal coverage requires exchangeable calibration/future samples; test coverage is empirical.",
    }


def _verify_scattered(root: Path, workspace: Path) -> dict:
    import numpy as np
    from ml_platform_mode_decomp import load_config
    from ml_platform_mode_decomp.data import load_scattered_observations

    source = workspace / "extension_scattered_source"
    _run(
        [
            sys.executable,
            str(root / "examples/create_scattered_dataset.py"),
            "--output",
            str(source),
            "--verify",
        ],
        workspace,
    )
    verification = workspace / "scattered_verification.json"
    report = _read(verification)
    normalized = Path(report["normalized_dataset"])
    cfg = load_config(root / "config/tasks/mode_decomp_scattered_analysis.yaml")
    rows = report["candidate_point_scores"]
    expected = {
        (candidate["id"], seed) for candidate in cfg["model"]["candidates"] for seed in cfg["analysis"]["split_seeds"]
    }
    if len(rows) != len(expected) or {(row["candidate_id"], row["split_seed"]) for row in rows} != expected:
        raise ValueError("Scattered acceptance omitted a shipped candidate/holdout comparison")
    for row in rows:
        values = [row["field_rmse"], *row["observation_rmse"], *row["interpolation_rmse"]]
        if not np.isfinite(values).all() or any(value <= 0 for value in values) or not 0 < row["point_coverage"] < 1:
            raise ValueError("Scattered example lost its distinct nonzero errors or partial point coverage")
    if not len(load_scattered_observations(normalized)["values"]):
        raise ValueError("Scattered import did not retain the original observations")
    return {
        "verification": str(verification),
        "normalized_dataset": str(normalized),
        "candidate_point_scores": rows,
        "single_yaml_raw_import_and_analysis_verified": True,
        "limit": "Toy interpolation and observation errors are different measures; no universal error floor is claimed.",
    }


def _verify_hodge(root: Path, workspace: Path) -> dict:
    import numpy as np
    import pandas as pd
    from ml_platform_mode_decomp import load_config, run_task
    from ml_platform_mode_decomp.data import load_field_dataset

    source = workspace / "extension_hodge_fields"
    _run([sys.executable, str(root / "examples/create_hodge_dataset.py"), "--output", str(source)], workspace)
    cfg = load_config(
        root / "config/tasks/mode_decomp_hodge.yaml",
        overrides={"data": {"local_path": str(source)}, "runtime": {"output_dir": str(workspace / "hodge_runs")}},
    )
    result = run_task(cfg)
    summary = _read(result.artifacts["analysis_summary"])
    if summary["metric_scope"] != "fitted_input_reconstruction":
        raise ValueError("The shipped Hodge example must directly decompose its supplied fields")
    dataset = load_field_dataset(cfg["data"])
    sample_index = {str(sample_id): index for index, sample_id in enumerate(dataset.sample_ids)}
    report = summary.get("component_analysis", {})
    if report.get("status") != "ok":
        raise ValueError("The shipped Hodge YAML did not enable physical component export")
    paths = {key: result.run_dir / value for key, value in report["outputs"].items()}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Hodge component output files are missing")
    metadata = _read(paths["component_summary"])
    energy = pd.read_csv(paths["component_energy"])
    if metadata["native_parts"] != ["exact", "coexact", "harmonic"] or metadata["harmonic_dimension"] != 1:
        raise ValueError("The annulus example lost its native Hodge parts or one-hole harmonic space")
    with np.load(paths["component_fields"], allow_pickle=False) as saved:
        ids = saved["sample_ids"].tolist()
        if ids != energy.sample_id.tolist() or len(ids) != metadata["sample_count"] or not ids:
            raise ValueError("Hodge component arrays and energy table have different sample alignment")
        if ids != dataset.sample_ids.tolist() or ids[0] != "sample_000000":
            raise ValueError("The Hodge example did not decompose every field using automatically generated IDs")
        edges = [saved[f"edge_{name}"] for name in metadata["native_parts"]]
        np.testing.assert_allclose(sum(edges), saved["edge_sampled"], atol=1e-8, rtol=1e-8)
        for name, values in zip(metadata["native_parts"], edges, strict=True):
            np.testing.assert_allclose(
                (saved["edge_weights"] * values**2).sum(axis=1), energy[f"{name}_edge_energy"], atol=1e-8
            )
        np.testing.assert_allclose(energy.input_edge_energy, energy.sum_component_edge_energy, atol=1e-8)
        lifted = sum(saved[f"node_{name}_edge_lift"] for name in metadata["native_parts"])
        nodes = lifted + saved["node_sampling_residual"]
        truth = dataset.fields[[sample_index[sample_id] for sample_id in ids]]
        np.testing.assert_allclose(nodes[:, saved["domain_mask"]], truth[:, saved["domain_mask"]], atol=1e-8)
    return {
        "run_dir": str(result.run_dir),
        "sample_count": len(ids),
        "harmonic_dimension": 1,
        "outputs": {key: str(value) for key, value in paths.items()},
        "native_energy_and_four_node_parts_verified": True,
        "limit": "Orthogonality and energy apply to native edge cochains; node lifts need the sampling residual.",
    }


def verify_extensions(root: Path, workspace: Path) -> dict:
    """Run actual example generators and shipped Tasks without changing repository inputs."""
    from ml_platform_core.result import RunError

    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, verify in (
        ("standalone", _verify_standalone),
        ("uncertainty", _verify_uncertainty),
        ("scattered", _verify_scattered),
        ("hodge", _verify_hodge),
    ):
        try:
            reports[name] = verify(root, workspace)
        except (AssertionError, RunError, OSError, KeyError, StopIteration, ValueError) as exc:
            raise ValueError(f"{name} wheel extension acceptance failed: {exc}") from exc
    return reports
