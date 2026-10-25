"""Independent acceptance checks for fixed-candidate field studies."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.io import read_json
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.evaluation import analyze_imputation, observation_mask
from ml_platform_mode_decomp.model_artifact import load_model


def _input(path, *, conditions=True, masked=False):
    path.mkdir()
    rng = np.random.default_rng(120)
    inputs = rng.normal(size=(16, 2))
    x, y = np.meshgrid(np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 4))
    fields = inputs[:, 0, None, None] * x + inputs[:, 1, None, None] * y
    fields = fields[..., None] + 0.02 * rng.normal(size=(16, 4, 4, 1))
    np.save(path / "field.npy", fields)
    pd.DataFrame(
        {
            "sample_id": [f"s{i:02d}" for i in range(16)],
            "array_index": np.arange(16),
            "x1": inputs[:, 0],
            "x2": inputs[:, 1],
            "experiment": np.repeat(np.arange(8), 2),
            "measured_at": [f"2026-01-{i + 1:02d}" for i in range(16)],
        }
    ).to_csv(path / "samples.csv", index=False)
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {"path": "field.npy", "layout": "NHWC", "kind": "scalar", "components": ["signal"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {"x": {"start": -1.0, "step": 2 / 3, "count": 4}, "y": {"start": -1.0, "step": 2 / 3, "count": 4}},
        },
    }
    if conditions:
        manifest["conditions"] = {"format": "table", "columns": ["x1", "x2"]}
    if masked:
        mask = np.ones((16, 4, 4), dtype=bool)
        mask[:, 0, 0] = False
        np.save(path / "observed.npy", mask)
        manifest["masks"] = {"observation": "observed.npy"}
    (path / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _settings(root, output, *, kind="surrogate"):
    return {
        "task": "mode_decomp_analysis",
        "run": {"name": "study", "seed": 29},
        "data": {"local_path": str(root)},
        "runtime": {"output_dir": str(output)},
        "split": {"method": "random", "valid_size": 0.25},
        "analysis": {
            "kind": kind,
            "split_seeds": [17],
            "model_seeds": [23],
            **({"mask_seed": 31, "observed_fraction": 0.5} if kind == "imputation" else {}),
            "bootstrap_samples": 40,
        },
        "model": {
            "candidates": [
                {"id": "one_mode", "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}}},
                {
                    "id": "three_modes",
                    "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 3}}},
                },
            ]
        },
    }


def _tables(result):
    return {key: pd.read_csv(value) for key, value in result.tables.items()}


@pytest.mark.parametrize("kind", ["representation", "surrogate", "imputation"])
def test_analysis_kinds_write_shared_holdout_reports_without_publication(tmp_path, kind):
    root = tmp_path / "input"
    _input(root, conditions=kind == "surrogate")
    cfg = _settings(root, tmp_path / "outputs", kind=kind)
    cfg["analysis"]["modes"] = {"enabled": True, "indices": [0]}
    if kind == "imputation":
        cfg["model"]["candidates"] = [
            {
                "id": "gappy",
                "representation": {"decomposition": {"name": "gappy_graph_fourier", "params": {"n_modes": 4}}},
            }
        ]
    result = run_task(cfg)
    tables = _tables(result)
    assert result.extra["model_publications"] == []
    assert "model" not in result.artifacts and not list(result.run_dir.rglob("model.joblib"))
    assert result.artifacts["analysis_report"].read_text(encoding="utf-8").startswith("<!doctype html>")
    summary = read_json(result.artifacts["analysis_summary"])
    assert summary["complete"] and summary["analysis_kind"] == kind
    protocol = read_json(result.artifacts["protocol"])["repetitions"][0]
    assert protocol["split"]["nested"] is False
    assert set(protocol["split"]["train_ids"]).isdisjoint(protocol["split"]["test_ids"])
    assert "selection_indices" not in protocol["split"]
    for _, samples in tables["per_sample"].groupby("candidate_id"):
        assert set(samples.sample_id) == set(protocol["split"]["test_ids"])
        assert len(samples) == len(protocol["split"]["test_ids"])
    assert set(tables["comparison"].status) == {"ok"}
    assert all(
        item["mode_analysis"]["status"] == "ok" for item in summary["details"] if item["decomposition"] != "baseline"
    )
    if kind == "imputation":
        assert summary["metric_scope"] == "heldout_hidden_reference"
        assert (tables["comparison"].observed_field_rmse.dropna() >= 0).all()
    elif kind == "surrogate":
        assert {"baseline_mean", "baseline_linear"} <= set(tables["comparison"].candidate_id)
        assert "regression_rmse" in tables["comparison"]
    else:
        assert "baseline_linear" not in set(tables["comparison"].candidate_id)
        assert "representation_rmse" in tables["comparison"]


def test_analysis_splits_and_component_seeds_are_independent(tmp_path):
    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "first", kind="representation")
    cfg["analysis"].update(split_seeds=[5, 6], model_seeds=[7, 11], bootstrap_samples=0)
    for candidate in cfg["model"]["candidates"]:
        candidate["representation"]["decomposition"]["name"] = "pod"
        candidate["representation"]["decomposition"]["params"]["solver"] = "randomized"
    cfg["model"]["candidates"][1]["representation"]["decomposition"]["params"]["seed"] = 41
    first = run_task(cfg)
    protocol = read_json(first.artifacts["protocol"])["repetitions"]
    assert len(protocol) == 4
    for split_seed in (5, 6):
        repeated = [item for item in protocol if item["split_seed"] == split_seed]
        assert len({item["split"]["fingerprint"] for item in repeated}) == 1
        for item in repeated:
            assert item["split"]["seed"] == split_seed
            candidates = {c["id"]: c for c in item["candidates"]}
            assert candidates["one_mode"]["representation"]["decomposition"]["params"]["seed"] == item["model_seed"]
            assert candidates["one_mode"]["regression"]["params"]["seed"] == item["model_seed"]
            assert candidates["three_modes"]["representation"]["decomposition"]["params"]["seed"] == 41
    assert protocol[0]["split"]["test_ids"] != protocol[2]["split"]["test_ids"]
    cfg["run"]["seed"] = 101
    cfg["runtime"]["output_dir"] = str(tmp_path / "second")
    second = run_task(cfg)
    assert read_json(second.artifacts["protocol"])["repetitions"] == protocol
    sort = ["split_seed", "model_seed", "candidate_id", "sample_id"]
    left, right = _tables(first)["per_sample"], _tables(second)["per_sample"]
    pd.testing.assert_frame_equal(
        left.sort_values(sort).reset_index(drop=True), right.sort_values(sort).reset_index(drop=True)
    )


def test_imputation_never_uses_values_outside_input_observation_mask(tmp_path):
    root = tmp_path / "input"
    _input(root, conditions=False, masked=True)
    cfg = _settings(root, tmp_path / "outputs", kind="imputation")
    cfg["analysis"]["save_models"] = True
    cfg["model"]["candidates"] = [
        {"id": "gappy", "representation": {"decomposition": {"name": "gappy_graph_fourier", "params": {"n_modes": 4}}}}
    ]
    result = run_task(cfg)
    assert result.extra["model_publications"] == []
    summary = read_json(result.artifacts["analysis_summary"])
    details = next(item for item in summary["details"] if item["candidate_id"] == "gappy")
    model = load_model(result.run_dir / details["saved_model"], result.run_dir / details["saved_model_info"])
    split = read_json(result.artifacts["protocol"])["repetitions"][0]["split"]
    dataset = load_field_dataset(root)
    training, reference = dataset.subset(split["train_indices"]), dataset.subset(split["test_indices"])
    observed = observation_mask(reference, 0.5, 31 + 17)
    original = analyze_imputation(model, training, reference, observed)
    unknown_values = reference.fields.copy()
    unknown_values[~reference.effective_mask] = 1e12
    unknown = analyze_imputation(model, training, replace(reference, fields=unknown_values), observed)
    np.testing.assert_array_equal(unknown.predictions, original.predictions)
    assert unknown.metrics == original.metrics
    hidden_values = reference.fields.copy()
    hidden_values[reference.effective_mask & ~observed] += 100.0
    hidden = analyze_imputation(model, training, replace(reference, fields=hidden_values), observed)
    np.testing.assert_array_equal(hidden.predictions, original.predictions)
    assert hidden.metrics["hidden_rmse"] > original.metrics["hidden_rmse"] + 50.0


def test_paired_comparison_matches_sample_losses_and_group_resampling(tmp_path):
    root = tmp_path / "input"
    _input(root)
    cfg = _settings(root, tmp_path / "outputs")
    cfg["split"] = {"method": "group", "group_column": "experiment", "valid_size": 0.5}
    result = run_task(cfg)
    tables = _tables(result)
    samples, paired = tables["per_sample"], tables["paired_comparisons"]
    pair = paired[(paired.candidate_a == "one_mode") & (paired.candidate_b == "three_modes")].iloc[0]
    left = samples[samples.candidate_id == "one_mode"].set_index("sample_id").sort_index()
    right = samples[samples.candidate_id == "three_modes"].set_index("sample_id").sort_index()
    assert left.index.equals(right.index)
    expected = np.sqrt(left.sample_weight @ left.mse) - np.sqrt(right.sample_weight @ right.mse)
    assert pair.rmse_a_minus_b == pytest.approx(expected)
    assert pair.resampling_unit == "group" and pair.independent_units == 4
    assert np.isfinite(pair.ci_low) and pair.ci_low <= pair.ci_high
    assert pair.interval_scope == "conditional_on_fitted_models_and_this_test_split"
    summary = read_json(result.artifacts["analysis_summary"])
    assert "unadjusted" in summary["uncertainty"]["multiple_comparisons"]


@pytest.mark.parametrize("allow_failures", [False, True])
def test_failed_candidate_leaves_partial_analysis_and_never_publishes(tmp_path, allow_failures):
    root = tmp_path / "input"
    _input(root)
    cfg = _settings(root, tmp_path / "outputs")
    cfg["analysis"]["allow_failures"] = allow_failures
    cfg["model"]["candidates"].append(
        {"id": "broken", "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 999}}}}
    )
    if allow_failures:
        result = run_task(cfg)
    else:
        with pytest.raises(RunError, match="Some predeclared candidates failed") as captured:
            run_task(cfg)
        result = captured.value.result
    assert result.extra["model_publications"] == []
    assert result.extra["analysis_complete"] is False
    assert result.artifacts["analysis_report"].is_file()
    summary = read_json(result.artifacts["analysis_summary"])
    assert not summary["complete"] and summary["failures"][0]["candidate_id"] == "broken"
    table = _tables(result)["comparison"]
    assert set(table[table.candidate_id == "broken"].status) == {"failed"}
    assert {"one_mode", "three_modes"} <= set(table[table.status == "ok"].candidate_id)
    assert "Incomplete" in result.artifacts["analysis_report"].read_text(encoding="utf-8")


@pytest.mark.parametrize("allow_failures", [False, True])
def test_missing_optional_backend_keeps_partial_report(tmp_path, monkeypatch, allow_failures):
    import ml_platform_mode_decomp.model_artifact as artifact
    import ml_platform_mode_decomp.regressors.gbdt as gbdt

    root = tmp_path / "input"
    _input(root)
    cfg = _settings(root, tmp_path / "outputs")
    cfg["analysis"]["allow_failures"] = allow_failures
    cfg["model"]["candidates"].append({"id": "missing_xgb", "regression": {"name": "xgb"}})
    original_import, original_version = gbdt.import_optional, artifact.version

    def missing_import(name, *, component):
        if name == "xgboost":
            raise ImportError("xgboost deliberately unavailable in this acceptance check")
        return original_import(name, component=component)

    def missing_version(name):
        if name == "xgboost":
            raise PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(gbdt, "import_optional", missing_import)
    monkeypatch.setattr(artifact, "version", missing_version)
    if allow_failures:
        result = run_task(cfg)
    else:
        with pytest.raises(RunError, match="Some predeclared candidates failed") as captured:
            run_task(cfg)
        result = captured.value.result
    assert result.extra["analysis_complete"] is False
    assert result.extra["model_publications"] == []
    assert result.artifacts["analysis_report"].is_file()
    failures = read_json(result.artifacts["analysis_summary"])["failures"]
    assert failures[0]["candidate_id"] == "missing_xgb"


def test_time_analysis_does_not_claim_independent_sample_intervals(tmp_path):
    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "outputs", kind="representation")
    cfg["split"] = {"method": "time", "time_column": "measured_at", "valid_size": 0.25}
    result = run_task(cfg)
    summary = read_json(result.artifacts["analysis_summary"])
    assert summary["uncertainty"]["independent_unit"] == "not_assumed_for_time_series"
    protocol = read_json(result.artifacts["protocol"])["repetitions"][0]["split"]
    assert max(protocol["train_indices"]) < min(protocol["test_indices"])
    paired = _tables(result)["paired_comparisons"]
    assert set(paired.resampling_unit) == {"not_assumed_independent"}
    assert paired.ci_low.isna().all() and paired.ci_high.isna().all()


def test_unsaved_comparison_never_serializes_models(tmp_path, monkeypatch):
    import joblib

    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "outputs", kind="representation")
    cfg["report"] = {"enabled": False}

    def forbidden(*args, **kwargs):
        raise AssertionError("A comparison without save_models must not serialize")

    monkeypatch.setattr(joblib, "dump", forbidden)
    result = run_task(cfg)
    comparison = _tables(result)["comparison"]
    assert set(comparison.status) == {"ok"}
    assert comparison.model_bytes.isna().all()


def test_report_failure_keeps_numeric_artifacts_and_metrics(tmp_path, monkeypatch):
    import ml_platform_mode_decomp.analysis_reporting as reporting

    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "outputs", kind="representation")

    def fail(*args, **kwargs):
        raise OSError("disk write unavailable")

    monkeypatch.setattr(reporting, "write_analysis_report", fail)
    with pytest.raises(RunError, match="numeric artifacts remain available") as raised:
        run_task(cfg)
    result = raised.value.result
    assert result.tables["comparison"].is_file() and result.tables["per_sample"].is_file()
    assert result.metrics["successful_evaluations"] > 0
    assert not result.extra["analysis_complete"]
    summary = read_json(result.artifacts["analysis_summary"])
    assert not summary["complete"] and "disk write unavailable" in summary["report_error"]
