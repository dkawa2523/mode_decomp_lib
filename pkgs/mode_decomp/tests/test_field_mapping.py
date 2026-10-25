"""Paired field mappings retain split, bundle and source-only inference contracts."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ml_platform_core.artifacts import file_hash

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate, load_config, load_model, run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.model_artifact import save_model


ROOT = Path(__file__).resolve().parents[3]


def _pair(parameters, *, prefix="case"):
    values = np.asarray(parameters)
    sx, sy = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 3))
    tx, ty = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4))
    source = 300 + values[:, None, None] * (1 + sx) + values[:, None, None] ** 2 * (2 + sy)
    target = 20 + values[:, None, None] * tx + 0.15 * values[:, None, None] ** 2 * ty
    ids = [f"{prefix}_{index}" for index in range(len(values))]
    return (
        FieldDataset.from_arrays(
            source[..., None],
            x=np.linspace(0, 3, 4),
            y=np.linspace(-1, 1, 3),
            sample_ids=ids,
            components=["temperature"],
            field_units=["K"],
        ),
        FieldDataset.from_arrays(
            target[..., None],
            x=np.linspace(10, 12, 5),
            y=np.linspace(0, 4, 4),
            sample_ids=ids,
            components=["stress"],
            field_units=["Pa"],
        ),
    )


@pytest.mark.parametrize(
    "source_method,target_method",
    [
        ({"name": "tucker", "params": {"ranks": [2, 2, 1]}}, {"name": "quadratic_manifold", "params": {"n_modes": 1}}),
        ({"name": "quadratic_manifold", "params": {"n_modes": 2}}, {"name": "tucker", "params": {"ranks": [2, 2, 1]}}),
    ],
    ids=["tucker_to_quadratic", "quadratic_to_tucker"],
)
def test_paired_array_api_coefficients_calibration_and_saved_model(tmp_path, source_method, target_method):
    source, target = _pair(np.linspace(-0.8, 0.8, 32))
    model = FieldSurrogate(
        {
            "input_representation": {"decomposition": source_method},
            "representation": {"decomposition": target_method},
            "regression": {"name": "ridge", "params": {"alpha": 1e-10, "solver": "svd"}},
        }
    ).fit(target, inputs=source.subset(np.arange(len(source))[::-1]))
    assert model.input_representation.fit_sample_ids == tuple(target.sample_ids)
    assert model.representation.fit_sample_ids == tuple(target.sample_ids)
    query, truth = _pair([-0.31, 0.57], prefix="new")
    coordinates = model.predict_coefficients(query)
    predicted = model.predict(query)
    np.testing.assert_allclose(predicted, truth.fields, atol=2e-8)
    np.testing.assert_allclose(model.representation.inverse_transform(coordinates), predicted, atol=1e-12)
    calibration_source, calibration_target = _pair(np.linspace(-0.75, 0.75, 16), prefix="calibration")
    with pytest.raises(ValueError, match="explicit calibration inputs"):
        model.calibrate(calibration_target, training=target)
    model.calibrate(calibration_target, inputs=calibration_source, training=target)
    path, info = save_model(model, tmp_path / "mapping.joblib")
    restored = load_model(path)
    bands = restored.predict_interval(query)
    np.testing.assert_array_equal(bands[0], predicted)
    assert all(value.shape == truth.fields.shape and np.isfinite(value).all() for value in bands)
    assert np.all(bands[1] <= bands[0]) and np.all(bands[0] <= bands[2])
    assert json.loads(info.read_text(encoding="utf-8"))["input_schema"]["schema_version"] == 4


@pytest.mark.parametrize("input_kind", ["field", "coefficients"])
def test_shipped_example_cv_and_source_only_fresh_process_inference(tmp_path, monkeypatch, input_kind):
    generated = tmp_path / "data"
    subprocess.run(
        [sys.executable, str(ROOT / "examples/create_field_mapping_dataset.py"), "--output", str(generated)],
        check=True,
        capture_output=True,
        text=True,
    )
    source_root, target_root = generated / "source", generated / "target"
    source = load_field_dataset({"local_path": str(source_root)})
    source_settings = {"kind": "field", "local_path": str(source_root)}
    query_source = load_field_dataset({"local_path": str(generated / "new_source")})
    query_settings = {"kind": "field", "local_path": str(generated / "new_source")}
    original_source_model = None
    if input_kind == "coefficients":
        frozen = FieldRepresentation({"decomposition": {"name": "tucker", "params": {"ranks": [3, 3, 1]}}})
        coefficients = frozen.fit_transform(source)
        original_source_model, _ = save_model(frozen, source_root / "model.joblib")
        identity = file_hash(original_source_model)
        for path, values, ids in (
            (source_root / "coefficients.npz", coefficients, source.sample_ids),
            (generated / "new_source/coefficients.npz", frozen.transform(query_source), query_source.sample_ids),
        ):
            np.savez_compressed(
                path, coefficients=values, sample_ids=ids, space=np.asarray("z"), model_sha256=np.asarray(identity)
            )
        source_settings = {
            "kind": "coefficients",
            "local_path": str(source_root),
            "path": "coefficients.npz",
            "model_artifact": "model.joblib",
        }
        query_settings = {
            "kind": "coefficients",
            "local_path": str(generated / "new_source"),
            "path": "coefficients.npz",
        }
    config = load_config(
        ROOT / "config/tasks/mode_decomp_field_mapping.yaml",
        overrides={
            "data": {"local_path": str(target_root), "input": source_settings},
            "runtime": {"output_dir": str(tmp_path / "runs")},
        },
    )
    config["data"]["input"] = source_settings
    if input_kind == "coefficients":
        config["model"]["candidates"][0].pop("input_representation")
    config["tuning"] = {"enabled": True, "n_trials": 1, "cv": {"n_splits": 2}}
    fit_calls = []
    original_fit = FieldRepresentation._fit

    def record_fit(self, dataset, **kwargs):
        if dataset.fields.shape[1:] == source.fields.shape[1:]:
            fit_calls.append(tuple(dataset.sample_ids))
        return original_fit(self, dataset, **kwargs)

    monkeypatch.setattr(FieldRepresentation, "_fit", record_fit)
    result = run_task(config)
    summary = json.loads(result.artifacts["analysis_summary"].read_text(encoding="utf-8"))
    scope = "training_partition" if input_kind == "field" else "fixed_input_representation"
    assert summary["input_representation_fit_scope"] == scope
    assert np.isfinite(result.metrics["field_rmse"]) and result.metrics["field_rmse"] < 0.1
    split = json.loads(result.artifacts["split"].read_text(encoding="utf-8"))
    if input_kind == "field":
        assert len(fit_calls) == 3  # Two independent CV fits and the selected final fit.
        assert all(set(ids) <= set(split["train_ids"]) for ids in fit_calls)
        assert fit_calls[-1] == tuple(split["train_ids"])
        assert all(len(ids) < len(fit_calls[-1]) for ids in fit_calls[:-1])
    else:
        assert not fit_calls  # The externally identified source representation is frozen.
    analysis = run_task(
        {
            **config,
            "task": "mode_decomp_analysis",
            "tuning": {"enabled": False},
            "analysis": {"kind": "surrogate", "split_seeds": [13], "model_seeds": [42], "bootstrap_samples": 0},
            "report": {"enabled": False},
            "runtime": {"output_dir": str(tmp_path / "analysis")},
        }
    )
    analysis_summary = json.loads(analysis.artifacts["analysis_summary"].read_text(encoding="utf-8"))
    details = [item for item in analysis_summary["details"] if item["candidate_id"] == "tucker_to_quadratic"]
    assert analysis_summary["complete"] and analysis.tables["comparison"].is_file()
    assert len(details) == 1 and details[0]["input_representation_fit_scope"] == scope
    restored = load_model(result.artifacts["model"])
    if original_source_model is not None:
        original_source_model.rename(source_root / "hidden_original_source.joblib")
        from ml_platform_mode_decomp.inputs import load_surrogate_inputs

        query = load_surrogate_inputs({"input": query_settings}, model=restored)
    else:
        query = query_source
    expected = restored.predict(query)
    inference = {
        "task": "mode_decomp_infer",
        "data": {"input": query_settings},
        "model": {"artifact_path": str(result.artifacts["model"])},
        "runtime": {"output_dir": str(tmp_path / "inference")},
    }
    import yaml

    path = tmp_path / "infer.yaml"
    path.write_text(yaml.safe_dump(inference), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ml_platform_mode_decomp", str(path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    payload = json.loads(completed.stdout)
    with np.load(payload["artifacts"]["predictions"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["predictions"], expected)
        assert saved["sample_ids"].tolist() == ["new_0", "new_1"]
        assert saved["field_units"].tolist() == ["Pa"]
