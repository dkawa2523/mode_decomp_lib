"""Small end-to-end checks of the field workflow and its statistical boundaries."""

from __future__ import annotations

import csv
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.io import read_json, write_json
from ml_platform_core.result import RunError, RunResult

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import load_field_dataset, split_dataset
from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.evaluation import evaluate_fields
from ml_platform_mode_decomp.model_artifact import load_model
from ml_platform_mode_decomp.policy import resolve_config
from ml_platform_mode_decomp.representation import FieldRepresentation
from ml_platform_mode_decomp.stage_inputs import load_candidate_results, load_stage_dataset


def tiny_input(root: Path, *, mesh=False, channels=1, conditions=True):
    root.mkdir()
    rng = np.random.default_rng(15)
    values = rng.normal(size=(20, 2))
    if mesh:
        basis = np.asarray([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]).reshape(2, 4, 1)
    else:
        x, y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
        basis = np.stack((x, y))[..., None]
    fields = np.einsum("np,p...->n...", values, basis)
    fields = np.concatenate([fields * (index + 1) for index in range(channels)], axis=-1)
    np.save(root / "field.npy", fields)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "a", "b"])
        writer.writerows((f"sample_{index:02d}", index, *row) for index, row in enumerate(values))
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {
            "path": "field.npy",
            "layout": "NVC" if mesh else "NHWC",
            "kind": "scalar" if channels == 1 else "multichannel",
            "components": [f"signal_{index}" for index in range(channels)],
        },
        "geometry": {
            "kind": "rectangle",
            "axes": {"x": {"start": -1, "step": 2 / 3, "count": 4}, "y": {"start": -1, "step": 2 / 3, "count": 4}},
        },
    }
    if conditions:
        manifest["conditions"] = {"format": "table", "columns": ["a", "b"]}
    if mesh:
        np.savez(
            root / "mesh.npz", vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], faces=[[0, 1, 2], [0, 2, 3]]
        )
        manifest["geometry"] = {"kind": "mesh", "mesh": {"path": "mesh.npz"}}
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return fields, values


def settings(root: Path, output: Path, *, rank=2):
    return {
        "task": "mode_decomp_pipeline",
        "run": {"name": "tiny", "seed": 31},
        "runtime": {"output_dir": str(output)},
        "data": {"local_path": str(root), "source_manifest": "field_dataset.yaml"},
        "split": {"method": "random", "valid_size": 0.2, "selection_size": 0.25},
        "model": {
            "selection_metric": "field_rmse",
            "candidates": [
                {
                    "id": "pod_alpha0",
                    "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": rank}}},
                    "regression": {"name": "ridge", "params": {"alpha": 0.0}},
                },
                {
                    "id": "pod_alpha5",
                    "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": rank}}},
                    "regression": {"name": "ridge", "params": {"alpha": 5.0}},
                },
            ],
        },
        "output": {},
    }


def candidate_payloads(output: Path):
    return {
        payload["candidate_id"]: payload
        for path in output.rglob("candidate_result.json")
        if (payload := read_json(path))
    }


def test_pipeline_bundle_and_condition_only_inference(tmp_path):
    root = tmp_path / "input"
    _, values = tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    result = run_task(cfg)
    assert isinstance(result, RunResult)
    assert result.extra["metric_scope"] == "final_test"
    assert result.extra["model_publications"]
    assert {"field_rmse", "field_mae", "field_skill", "baseline_field_rmse"} <= result.metrics.keys()
    assert "representation_rmse" not in result.metrics  # Detailed decomposition is an explicit analysis.
    assert result.artifacts["analysis_summary"].is_file()
    assert read_json(result.artifacts["analysis_summary"])["reference_scope"] == "final_test"
    model = load_model(result.artifacts["model"], result.artifacts["model_info"])
    split_path = next((tmp_path / "training").rglob("split.json"))
    split = read_json(split_path)
    assert set(model.fit_sample_ids) == set(split["train_ids"])
    assert set(model.fit_sample_ids).isdisjoint(split["test_ids"])
    samples = pd.read_csv(result.tables["per_sample"])
    assert samples["sample_id"].tolist() == split["test_ids"]
    assert {"mse", "rmse", "mae"} <= set(samples)
    for path in (tmp_path / "training").rglob("candidate_result.json"):
        payload = read_json(path)
        assert payload["evaluation_identity"]["runtime"]["packages"]["ml-platform-mode-decomp"]
        assert read_json(path.parent / "analysis_summary.json")["reference_scope"] == "selection_holdout"
        sample_errors = pd.read_csv(path.parent / "per_sample.csv")
        assert sample_errors["sample_id"].tolist() == split["selection_ids"]
    for payload in candidate_payloads(tmp_path / "training").values():
        assert payload["fit_ids"] == split["fit_ids"]
        assert set(payload["fit_ids"]).isdisjoint(split["selection_ids"] + split["test_ids"])
    prediction_root = tmp_path / "new_conditions"
    prediction_root.mkdir()
    (prediction_root / "conditions.csv").write_text("sample_id,b,a,note\nnew,0.4,0.2,unused\n", encoding="utf-8")
    infer = run_task(
        {
            "task": "mode_decomp_infer",
            "runtime": {"output_dir": str(tmp_path / "inference")},
            "data": {"local_path": str(prediction_root), "dataset_file": "conditions.csv"},
            "model": {
                "artifact_path": str(result.artifacts["model"]),
                "info_path": str(result.artifacts["model_info"]),
            },
        }
    )
    with np.load(infer.artifacts["predictions"], allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["predictions"], model.predict([[0.2, 0.4]]), atol=1e-12)
        assert saved["sample_ids"].tolist() == ["new"]
    assert model.predict(values[:2]).shape == (2, 4, 4, 1)


def test_individual_stages_accept_raw_candidate_settings(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "prepare")
    cfg["task"] = "mode_decomp_stage"
    cfg["run"]["stage"] = "prepare_fields"
    prepared = run_task(cfg)
    inputs = {
        "canonical_dataset": str(prepared.artifacts["canonical_dataset"]),
        "split": str(prepared.artifacts["split"]),
    }
    results = []
    for raw_candidate in cfg["model"]["candidates"]:
        stage_cfg = deepcopy(cfg)
        stage_cfg["runtime"]["output_dir"] = str(tmp_path / raw_candidate["id"])
        stage_cfg["run"]["stage"] = "train_candidate"
        stage_cfg["candidate"] = raw_candidate
        stage_cfg["stage_inputs"] = inputs
        trained = run_task(stage_cfg)
        assert trained.extra["metric_scope"] == "selection_holdout"
        assert trained.extra["model_publications"] == []
        assert read_json(trained.artifacts["candidate_result"])["fit_ids"] == read_json(inputs["split"])["fit_ids"]
        results.append(str(trained.artifacts["candidate_result"]))
    final_cfg = deepcopy(cfg)
    final_cfg["runtime"]["output_dir"] = str(tmp_path / "finalize")
    final_cfg["run"]["stage"] = "finalize_surrogate"
    final_cfg["stage_inputs"] = {**inputs, "candidate_results": results}
    final = run_task(final_cfg)
    assert final.extra["metric_scope"] == "final_test"
    assert final.extra["model_publications"]
    assert set(load_model(final.artifacts["model"]).fit_sample_ids) == set(read_json(inputs["split"])["train_ids"])


def test_final_test_change_does_not_change_candidate_selection(tmp_path):
    root = tmp_path / "input"
    fields, _ = tiny_input(root)
    cfg = settings(root, tmp_path / "before")
    dataset = load_field_dataset(root)
    split = split_dataset(dataset, cfg["split"], cfg["run"]["seed"])
    first = run_task(cfg)
    fields[split["test_indices"]] += 1000
    changed_root = tmp_path / "changed_input"
    shutil.copytree(root, changed_root)
    np.save(changed_root / "field.npy", fields)
    second_cfg = deepcopy(cfg)
    second_cfg["data"]["local_path"] = str(changed_root)
    second_cfg["runtime"]["output_dir"] = str(tmp_path / "after")
    second = run_task(second_cfg)
    before, after = candidate_payloads(tmp_path / "before"), candidate_payloads(tmp_path / "after")
    assert before.keys() == after.keys()
    for candidate_id in before:
        assert before[candidate_id]["metrics"] == after[candidate_id]["metrics"]
        assert before[candidate_id]["fit_ids"] == after[candidate_id]["fit_ids"]
    assert first.extra["best_candidate"] == second.extra["best_candidate"]
    assert second.metrics["field_rmse"] > first.metrics["field_rmse"] + 100


def test_html_report_requires_an_analysis_task():
    with pytest.raises(ValueError, match="report.enabled is available for mode_decomp_analysis"):
        run_task({"task": "mode_decomp_pipeline", "report": {"enabled": True}})


@pytest.mark.parametrize("suffix", ["npz", "npy"])
def test_training_prediction_filename_uses_the_public_output_setting(tmp_path, suffix):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    cfg["output"]["prediction_name"] = f"field_predictions.{suffix}"
    result = run_task(cfg)
    assert result.artifacts["predictions"].name == f"field_predictions.{suffix}"
    if suffix == "npy":
        model, dataset = load_model(result.artifacts["model"]), load_field_dataset(root)
        split = read_json(next((tmp_path / "training").rglob("split.json")))
        values = np.load(result.artifacts["predictions"], mmap_mode="r", allow_pickle=False)
        np.testing.assert_allclose(values, model.predict(dataset.conditions[split["test_indices"]]), atol=1e-12)
        assert np.load(result.artifacts["prediction_sample_ids"], allow_pickle=False).tolist() == split["test_ids"]


def test_final_metric_uses_original_fields_with_truncated_representation(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training", rank=1)
    result = run_task(cfg)
    model = load_model(result.artifacts["model"])
    dataset = load_field_dataset(root)
    split = split_dataset(dataset, cfg["split"], cfg["run"]["seed"])
    test = dataset.subset(split["test_indices"])
    prediction = model.predict(test.conditions)
    original_error = evaluate_fields(test, prediction)["field_rmse"]
    reconstructed = replace(test, fields=model.representation.reconstruct(test))
    reconstructed_error = evaluate_fields(reconstructed, prediction)["field_rmse"]
    assert result.metrics["field_rmse"] == pytest.approx(original_error)
    assert original_error > reconstructed_error + 0.05


def test_representation_needs_no_conditions_and_refuses_different_geometry(tmp_path):
    root = tmp_path / "input"
    tiny_input(root, conditions=False)
    result = run_task(
        {
            "task": "mode_decomp_representation",
            "data": {"local_path": str(root)},
            "runtime": {"output_dir": str(tmp_path / "representation")},
            "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}},
        }
    )
    model = load_model(result.artifacts["model"])
    dataset = load_field_dataset(root)
    np.testing.assert_allclose(model.reconstruct(dataset), dataset.fields, atol=1e-10)
    shifted = replace(dataset.geometry, coords={**dataset.geometry.coords, "x": dataset.geometry.coords["x"] + 1})
    with pytest.raises(ValueError, match="geometry"):
        model.transform(replace(dataset, geometry=shifted))


@pytest.mark.parametrize("mask_mode", ["none", "shared", "batch"])
def test_mesh_pod_multichannel_uses_public_nvc_shape(tmp_path, mask_mode):
    root = tmp_path / "input"
    fields, _ = tiny_input(root, mesh=True, channels=3)
    dataset = load_field_dataset(root)
    if mask_mode != "none":
        shape = (4,) if mask_mode == "shared" else (len(dataset), 4)
        dataset = replace(dataset, observation_mask=np.ones(shape, dtype=bool))
    candidate = resolve_config(settings(root, tmp_path / "unused"))["model"]["candidates"][0]
    model = FieldRepresentation(candidate["representation"]).fit(dataset)
    np.testing.assert_allclose(model.reconstruct(dataset), fields, atol=1e-10)


def test_field_metric_handles_uint_and_masked_nan_without_invalid_arithmetic():
    geometry = DomainSpec(
        "rectangle", {"x": np.zeros((2, 2)), "y": np.zeros((2, 2))}, None, np.array([[1.0, 3.0], [0.0, 0.0]]), {}
    )
    from ml_platform_mode_decomp.data import FieldDataset

    truth = np.array([[[[3], [1]], [[0], [0]]]], dtype=np.uint8)
    dataset = FieldDataset(truth, geometry, np.asarray(["one"]))
    result = evaluate_fields(dataset, np.zeros_like(truth))
    assert result["field_rmse"] == pytest.approx(np.sqrt(3.0))
    assert result["field_mae"] == pytest.approx(1.5)
    masked = replace(
        dataset,
        fields=np.array([[[[3.0], [1.0]], [[np.nan], [np.nan]]]]),
        observation_mask=np.array([[1, 1], [0, 0]], dtype=bool),
    )
    with np.errstate(invalid="raise", over="raise"):
        assert evaluate_fields(masked, np.zeros_like(masked.fields))["field_rmse"] == result["field_rmse"]


def test_weighted_channel_metrics_keep_physical_channel_errors(tmp_path):
    root = tmp_path / "input"
    tiny_input(root, channels=3)
    dataset = load_field_dataset(root)
    errors = np.asarray([1.0, 2.0, 4.0])
    options = {"channel_scales": [1.0, 2.0, 2.0], "channel_weights": [1.0, 1.0, 2.0]}
    metrics = evaluate_fields(dataset, dataset.fields + errors, options)
    assert metrics["field_rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["field_mae"] == pytest.approx(1.5)
    for channel, expected in enumerate(errors):
        assert metrics[f"channel_{channel}_rmse"] == pytest.approx(expected)
        assert metrics[f"channel_{channel}_mae"] == pytest.approx(expected)


def test_public_field_api_rejects_complex_and_invalid_masks(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    dataset = load_field_dataset(root)
    candidate = resolve_config(settings(root, tmp_path / "unused"))["model"]["candidates"][0]
    model = FieldRepresentation(candidate["representation"]).fit(dataset)
    with pytest.raises(ValueError, match="real numeric"):
        model.transform(dataset.fields.astype(complex) + 1j)
    with pytest.raises(ValueError, match="0/1"):
        model.transform(dataset.fields, np.full(dataset.fields.shape[1:-1], 2))
    with pytest.raises(ValueError, match="real numeric"):
        evaluate_fields(dataset, dataset.fields.astype(complex))
    with pytest.raises(ValueError, match="finite z coefficients"):
        model.inverse_transform(model.transform(dataset).astype(complex))


def test_stage_results_reject_different_evaluation_fit_ids_and_split(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    run_task(cfg)
    resolved = resolve_config(cfg)
    split_path = next((tmp_path / "training").rglob("split.json"))
    resolved["stage_inputs"] = {
        "split": split_path,
        "candidate_results": list((tmp_path / "training").rglob("candidate_result.json")),
    }
    dataset, split = load_stage_dataset(resolved)
    first_path = resolved["stage_inputs"]["candidate_results"][0]
    payload = read_json(first_path)
    altered = deepcopy(payload)
    altered["evaluation"]["channel_scales"] = [2.0]
    write_json(altered, first_path)
    with pytest.raises(ValueError, match="another data"):
        load_candidate_results(resolved, dataset, split)
    payload["fit_ids"] = ["from_another_run"]
    write_json(payload, first_path)
    with pytest.raises(ValueError, match="another data"):
        load_candidate_results(resolved, dataset, split)
    broken = read_json(split_path)
    broken["test_indices"] = [0]
    write_json(broken, split_path)
    with pytest.raises(ValueError, match="Split artifact"):
        load_stage_dataset(resolved)


def test_training_failure_preserves_partial_result_and_disables_publication(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    cfg["model"]["candidates"][0]["representation"]["decomposition"]["params"]["n_modes"] = 999
    with pytest.raises(RunError) as captured:
        run_task(cfg)
    assert captured.value.__cause__ is not None
    assert captured.value.result.artifacts["config"].is_file()
    assert captured.value.result.extra["model_publications"] == []


def test_external_candidate_runtime_is_recorded_without_whole_dictionary_equality(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    result = run_task(cfg)
    resolved = resolve_config(cfg)
    split_path = next((tmp_path / "training").rglob("split.json"))
    resolved["stage_inputs"] = {
        "split": split_path,
        "candidate_results": list((tmp_path / "training").rglob("candidate_result.json")),
    }
    dataset, split = load_stage_dataset(resolved)
    assert len(load_candidate_results(resolved, dataset, split)) == 2
    path = resolved["stage_inputs"]["candidate_results"][0]
    original = read_json(path)
    changed = deepcopy(original)
    changed["evaluation_identity"]["runtime"]["packages"]["numpy"] = "0.0.old"
    write_json(changed, path)
    assert len(load_candidate_results(resolved, dataset, split)) == 2
    info = read_json(result.artifacts["model_info"])
    assert info["provenance"]["evaluation_identity"]["runtime"]["packages"]["ml-platform-mode-decomp"]


def _assert_report_loss_identity(metrics, samples):
    weights = samples["sample_weight"].to_numpy()
    assert weights.sum() == pytest.approx(1.0)
    aggregate_mse = float(weights @ samples["mse"].to_numpy())
    assert metrics["field_rmse"] ** 2 == pytest.approx(aggregate_mse, abs=1e-12)
    assert metrics["field_mae"] == pytest.approx(float(weights @ samples["mae"].to_numpy()), abs=1e-12)
    if "field_skill" in metrics:
        baseline_mse = float(weights @ samples["baseline_mse"].to_numpy())
        assert metrics["baseline_field_rmse"] ** 2 == pytest.approx(baseline_mse, abs=1e-12)
        assert metrics["field_skill"] == pytest.approx(1.0 - aggregate_mse / baseline_mse, abs=1e-12)


def test_candidate_and_final_reports_share_weighted_loss_and_baseline_skill(tmp_path):
    root = tmp_path / "input"
    tiny_input(root, channels=2)
    table = pd.read_csv(root / "samples.csv")
    table["importance"] = np.arange(1, len(table) + 1)
    table.to_csv(root / "samples.csv", index=False)
    cfg = settings(root, tmp_path / "training", rank=1)
    cfg["metrics"] = {
        "names": ["field_rmse", "field_mae"],
        "channel_scales": [1.0, 2.0],
        "channel_weights": [1.0, 3.0],
        "sample_weight_column": "importance",
    }
    cfg["model"]["selection_metric"] = "field_skill"
    result = run_task(cfg)
    final_samples = pd.read_csv(result.tables["per_sample"])
    _assert_report_loss_identity(result.metrics, final_samples)
    assert read_json(result.artifacts["analysis_summary"])["sample_weight_column"] == "importance"
    payloads = candidate_payloads(tmp_path / "training")
    for path in (tmp_path / "training").rglob("candidate_result.json"):
        payload = read_json(path)
        _assert_report_loss_identity(payload["metrics"], pd.read_csv(path.parent / "per_sample.csv"))
        assert read_json(path.parent / "analysis_summary.json")["reference_scope"] == "selection_holdout"
    expected = sorted(payloads.values(), key=lambda item: (-item["metrics"]["field_skill"], item["candidate_id"]))[0]
    assert result.extra["best_candidate"] == expected["candidate_id"]
    assert read_json(result.artifacts["analysis_summary"])["reference_scope"] == "final_test"
    dataset = load_field_dataset(root)
    split = read_json(next((tmp_path / "training").rglob("split.json")))
    training, reference = dataset.subset(split["train_indices"]), dataset.subset(split["test_indices"])
    baseline = np.broadcast_to(training.fields.mean(axis=0), reference.fields.shape)
    baseline_rmse = evaluate_fields(reference, baseline, cfg["metrics"])["field_rmse"]
    assert result.metrics["baseline_field_rmse"] == pytest.approx(baseline_rmse)


@pytest.mark.parametrize("missing_scope", ["selection", "final"])
def test_training_baseline_undefined_is_reported_without_inventing_skill(tmp_path, missing_scope):
    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    cfg["model"]["candidates"] = [
        {
            "id": "gappy",
            "representation": {
                "decomposition": {"name": "gappy_graph_fourier", "params": {"n_modes": 4, "solver": "dense"}}
            },
            "regression": {"name": "ridge"},
        }
    ]
    if missing_scope == "selection":
        cfg["model"]["candidates"].append({**deepcopy(cfg["model"]["candidates"][0]), "id": "gappy_other"})
    else:
        cfg["split"].pop("selection_size")
    dataset = load_field_dataset(root)
    split = split_dataset(dataset, cfg["split"], cfg["run"]["seed"], nested=missing_scope == "selection")
    mask = np.ones(dataset.fields.shape[:-1], dtype=bool)
    unseen = split["fit_indices"] if missing_scope == "selection" else split["train_indices"]
    mask[unseen, -1, -1] = False
    np.save(root / "observation_mask.npy", mask)
    manifest = yaml.safe_load((root / "field_dataset.yaml").read_text(encoding="utf-8"))
    manifest["masks"] = {"observation": "observation_mask.npy"}
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    result = run_task(cfg)
    if missing_scope == "selection":
        payload = next(iter(candidate_payloads(tmp_path / "training").values()))
        metrics = payload["metrics"]
        summary = read_json(
            next((tmp_path / "training").rglob("candidate_result.json")).parent / "analysis_summary.json"
        )
    else:
        metrics, summary = result.metrics, read_json(result.artifacts["analysis_summary"])
    assert "field_skill" not in metrics
    assert "baseline_field_rmse" not in metrics
    assert summary["baseline_status"] == "training_mean_undefined_at_evaluated_locations"
    assert np.isfinite(metrics["field_rmse"])
    if missing_scope == "selection":
        cfg["runtime"]["output_dir"] = str(tmp_path / "undefined_skill")
        cfg["model"]["selection_metric"] = "field_skill"
        with pytest.raises(RunError, match="undefined|must be finite"):
            run_task(cfg)


def test_surrogate_r2_uses_original_fields_when_representation_is_constant(tmp_path):
    root = tmp_path / "input"
    tiny_input(root)
    conditions = np.arange(1.0, 21.0)
    pattern = np.tile(np.asarray([-1.0, -1.0, 1.0, 1.0]), (4, 1))[..., None]
    np.save(root / "field.npy", conditions[:, None, None, None] * pattern[None])
    cfg = settings(root, tmp_path / "training")
    cfg["split"].pop("selection_size")
    cfg["metrics"] = {"names": ["field_rmse", "field_r2"]}
    cfg["model"]["candidates"] = [
        {
            "id": "dc_only",
            "representation": {"decomposition": {"name": "fft2_lowpass", "params": {"block_size": 1}}},
            "regression": {"name": "ridge"},
        }
    ]
    result = run_task(cfg)
    assert np.isfinite(result.metrics["field_rmse"])
    assert result.metrics["field_r2"] == pytest.approx(0.0)
    model = load_model(result.artifacts["model"])
    reference = load_field_dataset(root).subset(read_json(result.artifacts["split"])["test_indices"])
    np.testing.assert_allclose(model.representation.reconstruct(reference), 0, atol=1e-12)


def test_single_fixed_candidate_loads_once_and_fits_once_without_selection(tmp_path, monkeypatch):
    from ml_platform_mode_decomp.surrogate import FieldSurrogate
    from ml_platform_mode_decomp.training import preparation

    root = tmp_path / "input"
    tiny_input(root)
    cfg = settings(root, tmp_path / "training")
    cfg["model"]["candidates"] = cfg["model"]["candidates"][:1]
    cfg["split"].pop("selection_size")
    fitted, loaded = [], []
    fit, load = FieldSurrogate.fit, preparation.load_field_dataset

    def tracked_fit(self, training):
        fitted.append(list(training.sample_ids))
        return fit(self, training)

    def tracked_load(data):
        loaded.append(data)
        return load(data)

    monkeypatch.setattr(FieldSurrogate, "fit", tracked_fit)
    monkeypatch.setattr(preparation, "load_field_dataset", tracked_load)
    result = run_task(cfg)
    split = read_json(result.artifacts["split"])
    summary = read_json(result.artifacts["result_summary"])
    assert len(fitted) == len(loaded) == 1
    assert fitted[0] == split["train_ids"]
    assert set(fitted[0]).isdisjoint(split["test_ids"])
    assert not split["nested"] and "selection_indices" not in split
    assert summary["selection_scope"] == "fixed_configuration" and summary["selection_score"] is None
    assert "leaderboard" not in result.tables
