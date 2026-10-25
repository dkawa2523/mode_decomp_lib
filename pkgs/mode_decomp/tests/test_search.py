"""Training-boundary and numerical checks for common-fold CV and discrete search."""

from copy import deepcopy
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.io import read_json
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.data.splitting import cross_validation_folds
from ml_platform_mode_decomp.model_artifact import load_model
from ml_platform_mode_decomp.policy import resolve_config
from ml_platform_mode_decomp.stage_inputs import load_candidate_results, load_stage_dataset
from ml_platform_mode_decomp.surrogate import FieldSurrogate
from ml_platform_mode_decomp.training.candidate_training import selection_options
from ml_platform_mode_decomp.training.search import search_candidate, trial_candidates


def write_input(root: Path, count=24):
    root.mkdir()
    conditions = np.random.default_rng(18).normal(size=(count, 2))
    x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    fields = np.einsum("nk,khw->nhw", conditions, np.stack((x, y)))[..., None]
    np.save(root / "field.npy", fields)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "a", "b", "group", "time", "role", "weight"])
        writer.writerows(
            (
                f"s{i:02d}",
                i,
                *value,
                f"g{i // 3}",
                f"2026-01-{i // 2 + 1:02d}",
                "test" if i >= count - 6 else "train",
                i + 1,
            )
            for i, value in enumerate(conditions)
        )
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "conditions": {"format": "table", "columns": ["a", "b"]},
        "fields": {"path": "field.npy", "layout": "NHWC", "kind": "scalar", "components": ["signal"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {"x": {"start": -1, "step": 1, "count": 3}, "y": {"start": -1, "step": 1, "count": 3}},
        },
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def settings(root: Path, output: Path):
    return {
        "task": "mode_decomp_pipeline",
        "run": {"seed": 27},
        "runtime": {"output_dir": str(output)},
        "data": {"local_path": str(root)},
        "split": {"method": "fixed", "valid_filter_column": "role", "valid_filter_value": "test"},
        "model": {
            "candidates": [
                {"id": "pod", "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}}},
                {"id": "dct", "representation": {"decomposition": {"name": "dct2"}}},
            ]
        },
        "tuning": {
            "enabled": True,
            "method": "grid",
            "n_trials": 2,
            "seed": 51,
            "cv": {"method": "random", "n_splits": 3, "seed": 19},
            "search_spaces": {key: {"regression.params.alpha": [0.0, 100.0]} for key in ("pod", "dct")},
        },
    }


def payloads(output):
    return {p["candidate_id"]: p for path in output.rglob("candidate_result.json") if (p := read_json(path))}


def test_cv_pipeline_shares_folds_fresh_fit_and_refits_only_selected_parameters(tmp_path, monkeypatch):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root)
    calls, fitted_models = [], []
    original = FieldSurrogate.fit

    def fit(self, training):
        calls.append(tuple(training.sample_ids))
        fitted_models.append(self)
        return original(self, training)

    monkeypatch.setattr(FieldSurrogate, "fit", fit)
    result = run_task(settings(root, output))
    split = read_json(next(output.rglob("split.json")))
    assert not split["nested"] and "selection_indices" not in split
    folds = [read_json(path)["folds"] for path in output.rglob("cv_folds.json")]
    assert len(folds) == 2 and folds[0] == folds[1]
    assert len(calls) == 2 * 2 * 3 + 1
    assert len({id(model) for model in fitted_models}) == len(calls)
    for fold in folds[0]:
        assert set(fold["fit_ids"]).isdisjoint(fold["validation_ids"])
        assert set(fold["fit_ids"] + fold["validation_ids"]) == set(split["train_ids"])
        assert set(fold["fit_ids"] + fold["validation_ids"]).isdisjoint(split["test_ids"])
    assert calls[:6] == calls[6:12]
    assert all(set(ids) <= set(split["train_ids"]) for ids in calls)
    assert calls[-1] == tuple(split["train_ids"])
    for table in output.rglob("search_trials.csv"):
        trials = pd.read_csv(table)
        assert len(trials) == 2 and set(trials.status) == {"ok"}
    selected = payloads(output)[result.extra["best_candidate"]]
    assert selected["candidate"]["regression"]["params"]["alpha"] == 0
    assert selected["metric_scope"] == "inner_cv"
    model = load_model(result.artifacts["model"])
    assert model.candidate == selected["candidate"]
    assert result.metrics["field_rmse"] < 1e-12
    summary = read_json(result.artifacts["result_summary"])
    assert summary["selection_scope"] == "inner_cv" and summary["metric_scope"] == "final_test"


def test_outer_test_fields_cannot_change_search_scores_or_best_parameters(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_input(first)
    shutil.copytree(first, second)
    values = np.load(second / "field.npy")
    values[-6:] += 10000
    np.save(second / "field.npy", values)
    outcomes = []
    for index, root in enumerate((first, second)):
        output = tmp_path / f"run{index}"
        result = run_task(settings(root, output))
        outcomes.append((result, payloads(output)))
    for key in outcomes[0][1]:
        left, right = outcomes[0][1][key], outcomes[1][1][key]
        assert left["candidate"] == right["candidate"]
        assert left["search"] == right["search"]
        assert left["metrics"] == right["metrics"]
    assert outcomes[0][0].metrics["field_rmse"] < 1e-12
    assert outcomes[1][0].metrics["field_rmse"] > 9000


@pytest.mark.parametrize("method", ["group", "time"])
def test_cv_respects_group_and_forward_time_boundaries(tmp_path, method):
    root = tmp_path / "input"
    write_input(root)
    dataset = load_field_dataset(root)
    config = {"method": method, "n_splits": 3, "seed": 19, f"{method}_column": method}
    folds = cross_validation_folds(dataset, config)
    seen = []
    for fold in folds:
        fit, validation = fold["fit_indices"], fold["validation_indices"]
        assert set(fit).isdisjoint(validation)
        values = dataset.sample_metadata[method]
        if method == "group":
            assert set(values[fit]).isdisjoint(values[validation])
        else:
            assert max(values[fit]) < min(values[validation])
        seen.extend(validation)
    assert len(seen) == len(set(seen))
    assert len(seen) == (len(dataset) if method == "group" else 18)


def test_cv_pools_weighted_sample_losses_without_retaining_field_predictions(tmp_path):
    root = tmp_path / "input"
    write_input(root, count=23)
    dataset = load_field_dataset(root)
    raw = settings(root, tmp_path / "runs")
    raw["tuning"]["n_trials"] = 1
    raw["tuning"]["search_spaces"] = {"pod": {"regression.params.alpha": [100.0]}}
    raw["model"]["candidates"] = raw["model"]["candidates"][:1]
    raw["metrics"] = {"names": ["field_rmse"], "channel_scales": [2], "sample_weight_column": "weight"}
    cfg = resolve_config(raw)
    outcome = search_candidate(cfg, cfg["model"]["candidates"][0], dataset, selection_options(cfg))
    report = outcome.report
    assert report is not None
    indices = np.concatenate([fold["validation_indices"] for fold in outcome.folds])
    predictions = []
    for fold in outcome.folds:
        fit = dataset.subset(fold["fit_indices"])
        validation = dataset.subset(fold["validation_indices"])
        model = FieldSurrogate(outcome.candidate).fit(fit)
        predictions.append(model.predict(validation.conditions))
    errors = (np.concatenate(predictions) - dataset.fields[indices]) / 2
    area = dataset.geometry.integration_weights()
    assert area is not None
    spatial = area / area.sum()
    sample_mse = np.einsum("nhwc,hw->n", errors**2, spatial)
    weights = dataset.sample_metadata["weight"][indices].astype(float)
    weights /= weights.sum()
    expected = float(np.sqrt(weights @ sample_mse))
    assert report.metrics["field_rmse"] == pytest.approx(expected)
    np.testing.assert_allclose(report.per_sample.sample_weight, weights)
    assert report.predictions is None
    np.testing.assert_allclose(report.per_sample.mse, sample_mse)


def test_seeded_random_search_uses_unique_equal_budgets(tmp_path):
    raw = settings(tmp_path / "input", tmp_path / "runs")
    raw["tuning"].update(method="random", n_trials=3)
    raw["tuning"]["search_spaces"] = {
        key: {"regression.params.alpha": [0.01, 0.1, 1, 10, 100]} for key in ("pod", "dct")
    }
    cfg = resolve_config(raw)
    candidates = cfg["model"]["candidates"]
    first = trial_candidates(candidates[0], cfg["tuning"], 27)
    assert first == trial_candidates(candidates[0], cfg["tuning"], 27)
    assert len({json.dumps(c, sort_keys=True) for c in first}) == 3
    another = deepcopy(cfg["tuning"])
    another["seed"] += 1
    assert first != trial_candidates(candidates[0], another, 27)
    assert len(trial_candidates(candidates[1], cfg["tuning"], 27)) == 3


@pytest.mark.parametrize("outer", ["group", "time"])
def test_tuning_cannot_silently_replace_group_or_time_with_random_cv(tmp_path, outer):
    raw = settings(tmp_path / "input", tmp_path / "runs")
    raw["split"] = {"method": outer, f"{outer}_column": outer}
    with pytest.raises(ValueError, match="preserve"):
        resolve_config(raw)
    raw["tuning"]["cv"].update(method=outer, **{f"{outer}_column": "another_column"})
    with pytest.raises(ValueError, match="preserve"):
        resolve_config(raw)


@pytest.mark.parametrize("method", ["group", "time"])
def test_group_and_time_search_run_through_final_refit_with_truthful_fit_coverage(tmp_path, method):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root)
    raw = settings(root, output)
    raw["split"] = {"method": method, "valid_size": 0.25, f"{method}_column": method}
    raw["tuning"]["cv"]["method"] = method
    result = run_task(raw)
    split = read_json(next(output.rglob("split.json")))
    model = load_model(result.artifacts["model"])
    assert list(model.fit_sample_ids) == split["train_ids"]
    for path in output.rglob("candidate_result.json"):
        payload = read_json(path)
        actual_fit_ids = {
            sample_id for fold in read_json(path.parent / "cv_folds.json")["folds"] for sample_id in fold["fit_ids"]
        }
        assert set(payload["fit_ids"]) == actual_fit_ids
        assert actual_fit_ids <= set(split["train_ids"])
        assert actual_fit_ids.isdisjoint(split["test_ids"])
        if method == "time":
            assert actual_fit_ids < set(split["train_ids"])
        else:
            assert actual_fit_ids == set(split["train_ids"])


def test_fixed_outer_cv_and_budget_require_explicit_valid_settings(tmp_path):
    raw = settings(tmp_path / "input", tmp_path / "runs")
    del raw["tuning"]["cv"]["method"]
    with pytest.raises(ValueError, match="fixed outer"):
        resolve_config(raw)
    raw["tuning"]["cv"]["method"] = "random"
    raw["tuning"]["n_trials"] = 3
    with pytest.raises(ValueError, match="fewer unique"):
        resolve_config(raw)
    raw["tuning"]["n_trials"] = 2
    raw["tuning"]["search_spaces"]["pod"] = {"representation.decomposition.name": ["pod", "pod_svd"]}
    with pytest.raises(ValueError, match="component parameter"):
        resolve_config(raw)


def test_failed_trials_are_recorded_and_do_not_replace_a_valid_winner(tmp_path):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root)
    raw = settings(root, output)
    raw["model"]["candidates"] = raw["model"]["candidates"][:1]
    raw["tuning"]["search_spaces"] = {"pod": {"representation.decomposition.params.n_modes": [99, 2]}}
    result = run_task(raw)
    trials = pd.read_csv(next(output.rglob("search_trials.csv")))
    assert trials.status.tolist() == ["failed", "ok"]
    assert isinstance(trials.error.iloc[0], str)
    assert payloads(output)["pod"]["search"]["best_trial"] == 1
    assert result.extra["model_publications"]


def test_all_failed_trials_leave_partial_evidence_without_a_published_model(tmp_path):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root)
    raw = settings(root, output)
    raw["model"]["candidates"] = raw["model"]["candidates"][:1]
    raw["tuning"]["search_spaces"] = {"pod": {"representation.decomposition.params.n_modes": [99, 100]}}
    with pytest.raises(RunError, match="All tuning trials failed") as raised:
        run_task(raw)
    assert raised.value.result.tables["search_trials"].is_file()
    assert set(pd.read_csv(raised.value.result.tables["search_trials"]).status) == {"failed"}
    assert not raised.value.result.extra.get("model_publications")
    assert not list(output.rglob("model.joblib"))


def test_cv_rejects_missing_groups_and_inadequate_time_history(tmp_path):
    root = tmp_path / "input"
    write_input(root)
    dataset = load_field_dataset(root)
    metadata = {**dataset.sample_metadata, "group": np.asarray([None] * len(dataset), dtype=object)}
    with pytest.raises(ValueError, match="missing"):
        cross_validation_folds(
            replace(dataset, sample_metadata=metadata),
            {"method": "group", "n_splits": 3, "seed": 1, "group_column": "group"},
        )
    with pytest.raises(ValueError, match="timestamps"):
        cross_validation_folds(dataset, {"method": "time", "n_splits": 12, "seed": 1, "time_column": "time"})


def test_candidate_context_allows_stage_ids_but_rejects_another_release_or_workspace(tmp_path):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root)
    raw = settings(root, output)
    context = {
        "source_pipeline_task_id": "pipeline-1",
        "source_release_id": "release-1",
        "execution_release_id": "release-1",
        "runtime_id": "cpu-runtime",
        "workspace_key": "science",
        "evaluation_task_id": None,
    }
    raw["runtime"]["execution_context"] = context
    run_task(raw)
    final_cfg = resolve_config(raw)
    final_cfg["task"] = "mode_decomp_stage"
    final_cfg["run"]["stage"] = "finalize_surrogate"
    final_cfg["runtime"]["output_dir"] = str(tmp_path / "finalize")
    final_cfg["runtime"]["execution_context"]["evaluation_task_id"] = "finalize-task-42"
    final_cfg["stage_inputs"] = {
        "split": str(next(output.rglob("split.json"))),
        "candidate_results": [str(path) for path in output.rglob("candidate_result.json")],
    }
    final = run_task(final_cfg)
    assert read_json(final.artifacts["model_info"])["provenance"]["evaluation_task_id"] == "finalize-task-42"
    dataset, split = load_stage_dataset(final_cfg)
    for key in ("source_pipeline_task_id", "source_release_id", "execution_release_id", "runtime_id", "workspace_key"):
        different = deepcopy(final_cfg)
        different["runtime"]["execution_context"][key] = "another-context"
        with pytest.raises(ValueError, match="execution context"):
            load_candidate_results(different, dataset, split)
