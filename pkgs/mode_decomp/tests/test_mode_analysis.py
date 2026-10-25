"""Field-space acceptance of optional mode views, including hidden-reference isolation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.evaluation import observation_mask


def _fixture(directory, *, conditions=True):
    directory.mkdir()
    rng = np.random.default_rng(71)
    inputs = rng.normal(size=(20, 2))
    y, x = np.meshgrid(np.arange(6), np.arange(8), indexing="ij")
    fields = inputs[:, 0, None, None] * np.cos(2 * np.pi * x / 8)
    fields += inputs[:, 1, None, None] * np.sin(2 * np.pi * y / 6)
    np.save(directory / "fields.npy", fields[..., None])
    pd.DataFrame(
        {
            "sample_id": [f"s{i:02d}" for i in range(20)],
            "array_index": np.arange(20),
            "x1": inputs[:, 0],
            "x2": inputs[:, 1],
            "importance": np.arange(1, 21),
        }
    ).to_csv(directory / "samples.csv", index=False)
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {
            "path": "fields.npy",
            "layout": "NHWC",
            "kind": "scalar",
            "components": ["temperature"],
            "units": ["K"],
        },
        "geometry": {
            "kind": "rectangle",
            "coordinate_units": {"x": "m", "y": "s"},
            "axes": {
                "x": {"start": 2.0, "step": 0.25, "count": 8, "periodic": True},
                "y": {"start": -1.0, "step": 0.5, "count": 6, "periodic": True},
            },
        },
    }
    if conditions:
        manifest["conditions"] = {"format": "table", "columns": ["x1", "x2"]}
    (directory / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _config(directory, output, kind):
    representation = {"decomposition": {"name": "fft2_lowpass", "params": {"block_size": 3}}}
    indices = [2, 0, 1]
    if kind == "imputation":
        representation = {"decomposition": {"name": "gappy_pod", "params": {"n_modes": 2, "reg_lambda": 0.0}}}
        indices = [1, 0]
    else:
        if kind == "surrogate":
            indices = [2, 0, 1, *range(3, 9)]
        representation["coeff_post"] = {"name": "pca", "params": {"n_components": 1 if kind == "representation" else 2}}
    candidate = {"id": "candidate", "representation": representation}
    if kind == "surrogate":
        candidate["regression"] = {"name": "ridge", "params": {"alpha": 0.7}}
    metric = "hidden_rmse" if kind == "imputation" else "field_rmse"
    settings = {
        "task": "mode_decomp_analysis",
        "run": {"name": "mode_acceptance", "seed": 29},
        "data": {"local_path": str(directory)},
        "runtime": {"output_dir": str(output)},
        "split": {"method": "random", "valid_size": 0.3},
        "metrics": {
            "reference": "hidden" if kind == "imputation" else "grid",
            "names": [metric],
            "sample_weight_column": "importance",
            "channel_scales": [2.0],
        },
        "analysis": {
            "kind": kind,
            "split_seeds": [17],
            "model_seeds": [23],
            "bootstrap_samples": 0,
            "modes": {"indices": indices, "cumulative_counts": [1, len(indices)]},
        },
        "model": {"candidates": [candidate]},
    }
    if kind == "imputation":
        settings["analysis"].update(observed_fraction=0.5, mask_seed=31)
    return settings


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _one(paths, prefix):
    return next(Path(value) for key, value in paths.items() if key.startswith(prefix))


def _mode_tables(paths):
    return pd.read_csv(_one(paths, "mode_metadata_")), pd.read_csv(_one(paths, "mode_reconstruction_"))


@pytest.mark.parametrize("kind", ["representation", "surrogate", "imputation"])
def test_optional_modes_keep_the_study_measure_and_meaning(tmp_path, kind):
    directory = tmp_path / "input"
    _fixture(directory, conditions=kind == "surrogate")
    config = _config(directory, tmp_path / "api", kind)
    result = run_task(config)
    descriptions, curve = _mode_tables(result.tables)
    assert descriptions["index"].tolist() == config["analysis"]["modes"]["indices"]
    assert set(descriptions.space) == {"a"}
    assert result.extra["model_publications"] == [] and not list(result.run_dir.rglob("*.joblib"))
    assert _one(result.artifacts, "mode_report_").is_file()
    full = curve[curve.reconstruction == "all_a_coordinates"].iloc[0]
    comparison = pd.read_csv(result.tables["comparison"]).set_index("candidate_id").loc["candidate"]
    if kind == "surrogate":
        # A comes from predicted Z, not from the held-out truth being encoded.
        assert comparison.field_rmse > 0.001
        assert full.field_rmse == pytest.approx(comparison.field_rmse, abs=1e-12)
        selected_all = curve[
            (curve.reconstruction == "selected_a_coordinates") & (curve.selected_count == full.full_dimension)
        ].iloc[0]
        assert selected_all.field_rmse == pytest.approx(comparison.field_rmse, abs=1e-12)
    elif kind == "representation":
        # A retains both harmonics; the one-dimensional target PCA is intentionally lossy.
        assert full.field_rmse < 1e-12
        assert comparison.field_rmse > 0.1
        assert full.field_rmse == pytest.approx(comparison.decomposition_rmse, abs=1e-12)
    else:
        assert "field_rmse" not in curve and "field_rmse" not in comparison.index
        assert full.hidden_rmse < 1e-12
        assert full.hidden_rmse == pytest.approx(comparison.hidden_rmse, abs=1e-12)
    if kind != "imputation":
        np.testing.assert_allclose(descriptions.frequency_x, descriptions.kx / (8 * 0.25))
        np.testing.assert_allclose(descriptions.frequency_y, descriptions.ky / (6 * 0.5))
    details = _read(result.artifacts["analysis_summary"])["details"]
    candidate = next(item for item in details if item["candidate_id"] == "candidate")
    expected_source = {
        "surrogate": "condition_predictions",
        "imputation": "input_observations_only",
        "representation": "heldout_fields",
    }[kind]
    assert candidate["mode_analysis"]["coefficient_source"] == expected_source


def test_imputation_mode_inputs_are_invariant_when_only_hidden_truth_changes(tmp_path, monkeypatch):
    import ml_platform_mode_decomp.analysis_task as analysis_task

    original_writer = analysis_task.write_mode_report
    inputs = []

    def capture(representation, coefficients, *args, **kwargs):
        inputs.append(np.array(coefficients, copy=True))
        return original_writer(representation, coefficients, *args, **kwargs)

    monkeypatch.setattr(analysis_task, "write_mode_report", capture)
    left, right = tmp_path / "left", tmp_path / "right"
    _fixture(left, conditions=False)
    _fixture(right, conditions=False)
    config = _config(left, tmp_path / "first", "imputation")
    first = run_task(config)
    split = _read(first.artifacts["protocol"])["repetitions"][0]["split"]
    reference = load_field_dataset(left).subset(split["test_indices"])
    observed = observation_mask(reference, 0.5, 31 + 17)
    changed = np.load(right / "fields.npy")
    for position, source in enumerate(split["test_indices"]):
        changed[source, ..., 0][~observed[position]] += 100.0
    np.save(right / "fields.npy", changed)
    config["data"]["local_path"] = str(right)
    config["runtime"]["output_dir"] = str(tmp_path / "second")
    second = run_task(config)
    assert len(inputs) == 2
    np.testing.assert_array_equal(inputs[0], inputs[1])
    _, first_curve = _mode_tables(first.tables)
    _, second_curve = _mode_tables(second.tables)
    first_full = first_curve[first_curve.reconstruction == "all_a_coordinates"].hidden_rmse.iloc[0]
    second_full = second_curve[second_curve.reconstruction == "all_a_coordinates"].hidden_rmse.iloc[0]
    assert first_full < 1e-12
    # Hidden truth changes the score, not the input coefficients. Channel scale is 2 K.
    assert second_full == pytest.approx(50.0, abs=1e-11)
    assert second.extra["metric_scope"] == "heldout_hidden_reference"


def test_nonlinear_synthesis_single_coordinate_is_reported_as_perturbation(tmp_path):
    directory = tmp_path / "input"
    _fixture(directory, conditions=False)
    config = _config(directory, tmp_path / "outputs", "representation")
    candidate = config["model"]["candidates"][0]
    candidate["representation"].pop("coeff_post")
    candidate["representation"]["codec"] = {"name": "auto", "params": {"complex_mode": "mag_phase"}}
    config["analysis"]["modes"] = {"indices": [0, 1], "cumulative_counts": [1, 2]}
    result = run_task(config)
    descriptions, _ = _mode_tables(result.tables)
    assert set(descriptions.interpretation) == {"perturbation_coordinate"}
    assert not descriptions.additive.any() and descriptions.reference_required.all()
    # The semantic distinction reaches the displayed coordinate captions, not only its CSV.
    assert "perturbation_coordinate" in _one(result.artifacts, "mode_report_").read_text(encoding="utf-8")
