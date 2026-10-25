"""New steady-field decoders use the existing codec, workflow and model contracts."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate, load_model, run_task
from ml_platform_mode_decomp.catalog import decomposition_role
from ml_platform_mode_decomp.config import resolve_representation
from ml_platform_mode_decomp.evaluation import analyze_imputation
from ml_platform_mode_decomp.model_artifact import save_model


METHODS = [
    {"name": "tucker", "params": {"ranks": [2, 2, 2], "inner_product": "domain_weights"}},
    {"name": "quadratic_manifold", "params": {"n_modes": 2, "inner_product": "domain_weights"}},
]


def _fields(conditions):
    x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 6))
    first = x[..., None] * np.array([1.0, 0.5])
    second = y[..., None] * np.array([0.2, -1.0])
    baseline = (2 + x**2 + y)[..., None] * np.array([1.0, 2.0])
    return baseline + conditions[:, 0, None, None, None] * first + conditions[:, 1, None, None, None] * second


def _dataset():
    conditions = np.random.default_rng(27).normal(size=(30, 2))
    return FieldDataset.from_arrays(
        _fields(conditions),
        conditions,
        condition_columns=["load", "setting"],
        components=["first", "second"],
        field_units=["K", "K"],
    )


@pytest.mark.parametrize("decomposition", METHODS, ids=lambda value: value["name"])
def test_array_coefficients_modes_and_persisted_surrogate(tmp_path, decomposition):
    data = _dataset()
    settings = {"decomposition": decomposition, "coeff_post": {"name": "standardize"}}
    representation = FieldRepresentation(settings)
    coordinates = representation.fit_transform(data)
    metadata = deepcopy(representation.raw_meta)
    assert decomposition_role(decomposition["name"]) == "learned_representation"
    assert representation.layout.real_dimension == coordinates.shape[1]
    np.testing.assert_allclose(representation.inverse_transform(coordinates), data.fields, atol=2e-10)

    observed = np.zeros(data.fields.shape[1:-1], dtype=bool)
    observed[::2, ::2] = True
    query = replace(data, fields=np.where(observed[..., None], data.fields, np.nan), observation_mask=observed)
    np.testing.assert_allclose(representation.reconstruct(query), data.fields, atol=2e-9)
    assert representation.raw_meta == metadata

    row = representation.list_modes([0])[0]
    if decomposition["name"] == "tucker":
        assert row["additive"] and row["core_index"] == {"y": 0, "x": 0, "component": 0}
        assert "channel" not in row  # A component factor can mix physical channels.
    else:
        assert row["interpretation"] == "latent_coordinate" and not row["additive"]
        with pytest.raises(ValueError, match="reference"):
            representation.mode_fields([0])
        assert np.isfinite(representation.mode_fields([0], reference=np.zeros(representation.packed_dim))).all()

    path, info = save_model(representation, tmp_path / "representation.joblib")
    restored = load_model(path)
    np.testing.assert_allclose(restored.reconstruct(query), data.fields, atol=2e-9)
    assert json.loads(info.read_text(encoding="utf-8"))["required_packages"] == ["joblib", "numpy", "scipy"]

    surrogate = FieldSurrogate(
        {"representation": settings, "regression": {"name": "ridge", "params": {"alpha": 0.0}}}
    ).fit(data)
    conditions = np.array([[0.3, -0.5], [0.8, 0.2]])
    path, _ = save_model(surrogate, tmp_path / "surrogate.joblib")
    prediction = load_model(path).predict(conditions)
    np.testing.assert_allclose(prediction, _fields(conditions), atol=2e-9)


@pytest.mark.parametrize("decomposition", METHODS, ids=lambda value: value["name"])
def test_hidden_reference_evaluation_uses_only_supplied_observations(decomposition):
    data = _dataset()
    training, reference = data.subset(slice(0, 22)), data.subset(slice(22, None))
    representation = FieldRepresentation({"decomposition": decomposition}).fit(training)
    observed = np.zeros_like(reference.effective_mask)
    observed[:, ::2, ::2] = True
    report = analyze_imputation(representation, training, reference, observed)
    assert report.summary["primary_support"] == "artificially_hidden_known_reference"
    assert report.metrics["hidden_rmse"] < 2e-9
    assert report.metrics["observed_field_rmse"] < 2e-9


@pytest.mark.parametrize("decomposition", METHODS, ids=lambda value: value["name"])
def test_direct_and_training_tasks_accept_inline_method_config(tmp_path, decomposition):
    data = _dataset()
    np.save(tmp_path / "fields.npy", data.fields)
    np.save(tmp_path / "conditions.npy", data.conditions)
    inputs = {
        "local_path": str(tmp_path),
        "fields": {"path": "fields.npy", "layout": "NHWC", "components": ["first", "second"]},
        "conditions": {"path": "conditions.npy", "format": "npy", "columns": ["load", "setting"]},
    }
    config = {
        "task": "mode_decomp_representation",
        "data": inputs,
        "representation": {"decomposition": decomposition},
        "runtime": {"output_dir": str(tmp_path / "runs")},
        "analysis": {"modes": {"enabled": True, "indices": [0, 1]}},
        "report": {"enabled": True},
    }
    direct = run_task(config)
    assert direct.metrics["field_rmse"] < 2e-9
    assert direct.artifacts["mode_report"].is_file()
    candidate = {
        "id": decomposition["name"],
        "representation": {"decomposition": decomposition},
        "regression": {"name": "ridge", "params": {"alpha": 0.0}},
    }
    trained = run_task(
        {
            "task": "mode_decomp_pipeline",
            "data": inputs,
            "runtime": {"output_dir": str(tmp_path / "runs")},
            "model": {"candidates": [candidate]},
        }
    )
    model = load_model(trained.artifacts["model"])
    np.testing.assert_allclose(model.predict(data.conditions), data.fields, atol=2e-9)


@pytest.mark.parametrize("name", ["tucker", "quadratic_manifold"])
def test_new_method_parameters_share_strict_config_resolution(name):
    with pytest.raises(ValueError, match="Unknown .* parameters"):
        resolve_representation({"decomposition": {"name": name, "params": {"unknown_option": True}}})
