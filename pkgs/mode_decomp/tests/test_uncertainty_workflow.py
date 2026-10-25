"""Independent holdout, perturbation and persistence checks for calibrated field bands."""

from copy import deepcopy
import shutil

import numpy as np
import pandas as pd
import pytest
from ml_platform_core.io import read_json
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.model_artifact import load_model, save_model
from ml_platform_mode_decomp.policy import resolve_config
from ml_platform_mode_decomp.surrogate import FieldSurrogate

from test_search import payloads, settings, write_input


def calibrated_settings(root, output, *, tuned=True):
    cfg = settings(root, output)
    cfg["uncertainty"] = {
        "enabled": True,
        "coverage": 0.9,
        "scope": "simultaneous",
        "normalization": "channel",
        "calibration_size": 0.3,
        "calibration_method": "random",
    }
    if not tuned:
        cfg["tuning"] = {"enabled": False}
        cfg["split"]["selection_method"] = "random"
    return cfg


@pytest.fixture
def calibrated_run(tmp_path):
    root, output = tmp_path / "input", tmp_path / "training"
    write_input(root, count=48)
    cfg = calibrated_settings(root, output)
    result = run_task(cfg)
    split = read_json(next(output.rglob("split.json")))
    return root, output, cfg, result, split


@pytest.mark.parametrize("issue", ["too_few", "missing_calibration", "missing_test"])
def test_static_band_requirements_fail_before_any_candidate_fit(tmp_path, monkeypatch, issue):
    import yaml
    from ml_platform_mode_decomp.data import split_dataset

    root = tmp_path / "input"
    write_input(root, count=48)
    cfg = calibrated_settings(root, tmp_path / "runs", tuned=False)
    if issue == "too_few":
        cfg["uncertainty"]["calibration_size"] = 0.05
        message = "Insufficient calibration samples"
    else:
        resolved = resolve_config(cfg)
        dataset = load_field_dataset(root)
        split = split_dataset(
            dataset, resolved["split"], resolved["run"]["seed"], nested=True, calibration=resolved["uncertainty"]
        )
        role = "calibration_indices" if issue == "missing_calibration" else "test_indices"
        mask = np.ones(dataset.fields.shape[:-1], dtype=bool)
        mask[split[role][0], 0, 0] = False
        np.save(root / "observed.npy", mask)
        manifest_path = root / "field_dataset.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["masks"] = {"observation": "observed.npy"}
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        message = "requires complete truth"

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Static band requirements must be checked before fitting")

    monkeypatch.setattr(FieldSurrogate, "fit", unexpected_fit)
    with pytest.raises(RunError, match=message):
        run_task(cfg)


@pytest.mark.parametrize("tuned", [False, True])
def test_calibration_is_reserved_before_selection_and_final_model_is_never_refitted_afterward(
    tmp_path, monkeypatch, tuned
):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root, count=48)
    events = []
    fit, calibrate = FieldSurrogate.fit, FieldSurrogate.calibrate

    def tracked_fit(self, training):
        events.append(("fit", tuple(training.sample_ids)))
        return fit(self, training)

    def tracked_calibrate(self, calibration, **kwargs):
        events.append(("calibrate", tuple(calibration.sample_ids)))
        return calibrate(self, calibration, **kwargs)

    monkeypatch.setattr(FieldSurrogate, "fit", tracked_fit)
    monkeypatch.setattr(FieldSurrogate, "calibrate", tracked_calibrate)
    monkeypatch.setattr(FieldSurrogate, "_query_batch_size", lambda self: 2)
    result = run_task(calibrated_settings(root, output, tuned=tuned))
    split = read_json(next(output.rglob("split.json")))
    proper, calibration, test = (set(split[key]) for key in ("train_ids", "calibration_ids", "test_ids"))
    assert proper.isdisjoint(calibration) and proper.isdisjoint(test) and calibration.isdisjoint(test)
    assert proper | calibration == set(split["outer_train_ids"])
    assert all(set(ids) <= proper for operation, ids in events if operation == "fit")
    assert events[-2:] == [("fit", tuple(split["train_ids"])), ("calibrate", tuple(split["calibration_ids"]))]
    assert sum(operation == "calibrate" for operation, _ in events) == 1
    model = load_model(result.artifacts["model"])
    assert set(model.fit_sample_ids) == proper
    assert set(model.representation.fit_sample_ids) == proper
    assert set(model.calibration_sample_ids) == calibration
    if tuned:
        folds = [read_json(path)["folds"] for path in output.rglob("cv_folds.json")]
        assert len(folds) == 2 and folds[0] == folds[1]
        for fold in folds[0]:
            assert set(fold["fit_ids"] + fold["validation_ids"]) == proper
    else:
        assert set(split["fit_ids"] + split["selection_ids"]) == proper
    assert all(set(item["fit_ids"]) <= proper for item in payloads(output).values())
    assert read_json(result.artifacts["result_summary"])["calibration_ids"] == split["calibration_ids"]
    data = load_field_dataset(root)
    truth = data.fields[split["test_indices"]]
    expected = model.predict_interval(data.conditions[split["test_indices"]])
    with np.load(result.artifacts["predictions"], allow_pickle=False) as saved:
        for key, values in zip(("predictions", "lower", "upper"), expected, strict=True):
            np.testing.assert_array_equal(saved[key], values)
        included = (truth >= saved["lower"]) & (truth <= saved["upper"])
        assert (
            result.metrics["interval_sample_simultaneous_coverage"]
            == np.all(included.reshape(len(truth), -1), axis=1).mean()
        )


def test_changing_only_calibration_truth_changes_bands_but_not_search_or_point_predictions(calibrated_run, tmp_path):
    root, output, cfg, result, split = calibrated_run
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    values = np.load(changed / "field.npy")
    values[split["calibration_indices"]] += 10000
    np.save(changed / "field.npy", values)
    other_cfg = deepcopy(cfg)
    other_output = tmp_path / "other_runs"
    other_cfg["data"]["local_path"] = str(changed)
    other_cfg["runtime"]["output_dir"] = str(other_output)
    other = run_task(other_cfg)
    left, right = payloads(output), payloads(other_output)
    for candidate_id, original in left.items():
        assert original["candidate"] == right[candidate_id]["candidate"]
        assert original["metrics"] == right[candidate_id]["metrics"]
        assert original["search"] == right[candidate_id]["search"]
    model, other_model = (load_model(value.artifacts["model"]) for value in (result, other))
    assert model.candidate == other_model.candidate
    np.testing.assert_array_equal(model.calibrator.channel_scales, other_model.calibrator.channel_scales)
    with (
        np.load(result.artifacts["predictions"], allow_pickle=False) as before,
        np.load(other.artifacts["predictions"], allow_pickle=False) as after,
    ):
        np.testing.assert_array_equal(before["predictions"], after["predictions"])
        assert np.min(after["upper"] - after["lower"]) > 19000
        assert np.max(before["upper"] - before["lower"]) < 1e-10


def test_saved_calibration_reproduces_condition_only_inference_and_opt_out_is_point_only(calibrated_run, tmp_path):
    _, _, _, result, split = calibrated_run
    model = load_model(result.artifacts["model"])
    schema = read_json(result.artifacts["model_info"])["input_schema"]
    assert schema["field_uncertainty"]["coverage"] == 0.9
    assert model.calibrator.summary()["calibration_count"] == len(split["calibration_ids"])
    inputs = tmp_path / "new_conditions"
    inputs.mkdir()
    (inputs / "conditions.csv").write_text("sample_id,b,a\nnew,0.4,0.2\n", encoding="utf-8")
    cfg = {
        "task": "mode_decomp_infer",
        "data": {"local_path": str(inputs), "dataset_file": "conditions.csv"},
        "model": {"artifact_path": str(result.artifacts["model"])},
    }
    expected, lower, upper = model.predict_interval(np.asarray([[0.2, 0.4]]))
    for enabled in (True, False):
        inferred = run_task(
            {**cfg, "runtime": {"output_dir": str(tmp_path / f"infer_{enabled}")}, "uncertainty": {"enabled": enabled}}
        )
        with np.load(inferred.artifacts["predictions"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["predictions"], expected)
            if enabled:
                np.testing.assert_array_equal(saved["lower"], lower)
                np.testing.assert_array_equal(saved["upper"], upper)
                assert saved["interval_method"].item() == "split_conformal"
                assert saved["interval_scope"].item() == "simultaneous"
            else:
                assert "lower" not in saved and "upper" not in saved
    with pytest.raises(RunError, match="differs from the saved calibration"):
        run_task(
            {
                **cfg,
                "runtime": {"output_dir": str(tmp_path / "wrong_coverage")},
                "uncertainty": {"enabled": True, "coverage": 0.95},
            }
        )


def test_inference_does_not_invent_intervals_for_an_uncalibrated_model(tmp_path):
    root = tmp_path / "input"
    write_input(root)
    model = FieldSurrogate({"id": "uncalibrated"}).fit(load_field_dataset(root))
    artifact, _ = save_model(model, tmp_path / "model.joblib")
    inputs = tmp_path / "conditions.csv"
    inputs.write_text("sample_id,a,b\nnew,0,0\n", encoding="utf-8")
    with pytest.raises(RunError, match="calibrated on independent held-out fields"):
        run_task(
            {
                "task": "mode_decomp_infer",
                "runtime": {"output_dir": str(tmp_path / "infer")},
                "data": {"local_path": str(inputs)},
                "model": {"artifact_path": str(artifact)},
                "uncertainty": {"enabled": True},
            }
        )


@pytest.mark.parametrize("method", ["group", "time"])
def test_calibration_keeps_group_and_purged_time_boundaries_without_claiming_exchangeability(tmp_path, method):
    root, output = tmp_path / "input", tmp_path / "runs"
    write_input(root, count=60)
    cfg = calibrated_settings(root, output)
    cfg["split"] = {"method": method, "valid_size": 0.2, f"{method}_column": method}
    cfg["uncertainty"]["calibration_method"] = method
    cfg["tuning"]["cv"]["method"] = method
    if method == "time":
        cfg["split"]["gap_seconds"] = 172800
    result = run_task(cfg)
    split = read_json(next(output.rglob("split.json")))
    metadata = pd.read_csv(root / "samples.csv").set_index("sample_id")
    partitions = [metadata.loc[split[key], method] for key in ("train_ids", "calibration_ids", "test_ids")]
    if method == "group":
        assert set(partitions[0]).isdisjoint(partitions[1])
        assert set(partitions[0]).isdisjoint(partitions[2])
        assert set(partitions[1]).isdisjoint(partitions[2])
    else:
        proper, calibration, test = [pd.to_datetime(values, utc=True) for values in partitions]
        assert max(proper) + pd.Timedelta(days=2) <= min(calibration)
        assert max(calibration) + pd.Timedelta(days=2) <= min(test)
        assert split["calibration_purged_ids"] and split["outer_purged_ids"]
        for path in output.rglob("cv_folds.json"):
            for fold in read_json(path)["folds"]:
                assert (
                    pd.to_datetime(metadata.loc[fold["fit_ids"], "time"]).max() + pd.Timedelta(days=2)
                    <= pd.to_datetime(metadata.loc[fold["validation_ids"], "time"]).min()
                )
    calibration = read_json(result.artifacts["analysis_summary"])["field_uncertainty"]["calibration"]
    assert calibration["sample_exchangeability_verified"] is False
    assert "if_calibration_and_future_samples_are_exchangeable" in calibration["guarantee_condition"]


def test_surrogate_analysis_uses_shared_calibration_and_saves_empirical_coverage(tmp_path, monkeypatch):
    root = tmp_path / "input"
    write_input(root, count=48)
    cfg = calibrated_settings(root, tmp_path / "analysis", tuned=False)
    cfg.update(
        task="mode_decomp_analysis",
        split={"method": "random", "valid_size": 0.2},
        analysis={"kind": "surrogate", "split_seeds": [13], "model_seeds": [17, 19], "bootstrap_samples": 0},
    )
    fitting_ids = []
    fit = FieldSurrogate.fit

    def tracked_fit(self, training):
        fitting_ids.append(tuple(training.sample_ids))
        return fit(self, training)

    monkeypatch.setattr(FieldSurrogate, "fit", tracked_fit)
    result = run_task(cfg)
    repetitions = read_json(result.artifacts["protocol"])["repetitions"]
    assert len(repetitions) == 2 and repetitions[0]["split"] == repetitions[1]["split"]
    split = repetitions[0]["split"]
    assert set(split["train_ids"]).isdisjoint(split["calibration_ids"] + split["test_ids"])
    assert all(ids == tuple(split["train_ids"]) for ids in fitting_ids)
    comparison = pd.read_csv(result.tables["comparison"])
    candidates = comparison[~comparison.candidate_id.str.startswith("baseline_")]
    assert len(candidates) == 4 and candidates.interval_pointwise_coverage.notna().all()
    details = read_json(result.artifacts["analysis_summary"])["details"]
    for detail in details:
        if not detail["candidate_id"].startswith("baseline_"):
            calibration = detail["field_uncertainty"]["calibration"]
            assert calibration["calibration_count"] == len(split["calibration_ids"])
            assert calibration["sample_exchangeability_verified"] is False
    assert not result.extra["model_publications"] and not list(result.run_dir.rglob("model.joblib"))


@pytest.mark.parametrize("kind", ["representation", "imputation"])
def test_uncertainty_is_rejected_for_non_surrogate_analysis(tmp_path, kind):
    cfg = calibrated_settings(tmp_path / "input", tmp_path / "run", tuned=False)
    cfg.update(task="mode_decomp_analysis", analysis={"kind": kind})
    with pytest.raises(ValueError, match="surrogate training, analysis and inference"):
        resolve_config(cfg)


@pytest.mark.parametrize("method", ["group", "time"])
def test_calibration_cannot_weaken_an_outer_group_or_time_split(tmp_path, method):
    cfg = calibrated_settings(tmp_path / "input", tmp_path / "run", tuned=False)
    cfg["split"] = {"method": method, f"{method}_column": method}
    with pytest.raises(ValueError, match="preserve the outer group/time"):
        resolve_config(cfg)
