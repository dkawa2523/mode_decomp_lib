"""Regression checks for self-contained field bundles and their runtime identity."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.metadata import version

import numpy as np
import pytest
from ml_platform_core.io import read_json

from ml_platform_mode_decomp.data import FieldDataset
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.model_artifact import bundle_schema, load_model, save_model
from ml_platform_mode_decomp.representation import FieldRepresentation
from ml_platform_mode_decomp.surrogate import FieldSurrogate


def _dataset(height=4, width=4):
    rng = np.random.default_rng(31)
    conditions = rng.normal(size=(8, 2))
    fields = rng.normal(size=(8, height, width, 1)) + 3
    geometry = build_domain_spec({"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}, (height, width))
    return FieldDataset(
        fields,
        geometry,
        np.asarray([f"sample_{i}" for i in range(8)]),
        conditions=conditions,
        condition_columns=("a", "b"),
    )


@pytest.mark.parametrize("surrogate", [False, True])
def test_disabled_offset_bundle_roundtrip_excludes_nan_diagnostics(tmp_path, surrogate):
    dataset = _dataset()
    representation = {"decomposition": {"name": "offset_residual", "params": {"enabled": False}}}
    if surrogate:
        model = FieldSurrogate({"representation": representation, "regression": {"name": "ridge"}}).fit(dataset)
        rep = model.representation
        expected = model.predict(dataset.conditions)
    else:
        model = FieldRepresentation(representation).fit(dataset)
        rep = model
        expected = model.reconstruct(dataset)
    # The numerical diagnostic remains available without becoming a structural contract.
    assert np.isnan(rep.raw_meta["offset_ratio_median"])
    layout = bundle_schema(model)["coefficient_layout"]
    assert "offset_ratio_median" not in layout
    json.dumps(bundle_schema(model), allow_nan=False)
    path, info_path = save_model(model, tmp_path / "model.joblib")
    info = read_json(info_path)
    assert info["input_schema"] == bundle_schema(model)
    restored = load_model(path, info_path)
    actual = restored.predict(dataset.conditions) if surrogate else restored.reconstruct(dataset)
    np.testing.assert_array_equal(actual, expected)


def test_nested_wavelet_runtime_version_is_saved_and_verified(tmp_path):
    pytest.importorskip("pywt")
    dataset = _dataset(16, 16)
    model = FieldRepresentation(
        {
            "decomposition": {
                "name": "offset_residual",
                "params": {
                    "enabled": True,
                    "inner": {
                        "name": "wavelet2d",
                        "wavelet": "db2",
                        "level": 2,
                        "mode": "symmetric",
                        "mask_policy": "error",
                    },
                },
            }
        }
    ).fit(dataset)
    expected = model.reconstruct(dataset)
    path, info_path = save_model(model, tmp_path / "nested.joblib")
    info = read_json(info_path)
    assert info["runtime"]["packages"]["PyWavelets"] == version("PyWavelets")
    np.testing.assert_array_equal(load_model(path, info_path).reconstruct(dataset), expected)


def test_mtgp_runtime_records_torch_and_gpytorch_versions(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    dataset = _dataset()
    model = FieldSurrogate(
        {
            "representation": {"decomposition": {"name": "dct2"}},
            "regression": {"name": "mtgp", "params": {"training_iter": 1}},
        }
    ).fit(dataset)
    expected = model.predict(dataset.conditions)
    path, info_path = save_model(model, tmp_path / "model.joblib")
    packages = read_json(info_path)["runtime"]["packages"]
    for package in ("torch", "gpytorch", "linear-operator"):
        assert packages[package] == version(package)
    np.testing.assert_allclose(load_model(path, info_path).predict(dataset.conditions), expected, atol=1e-6)


def test_cpu_autoencoder_bundle_has_no_sibling_weights_dependency(tmp_path):
    pytest.importorskip("torch")
    dataset = _dataset()
    model = FieldSurrogate(
        {
            "representation": {
                "decomposition": {
                    "name": "autoencoder",
                    "params": {
                        "latent_dim": 2,
                        "hidden_channels": [2],
                        "epochs": 1,
                        "batch_size": 4,
                    },
                }
            },
            "regression": {"name": "ridge"},
        }
    ).fit(dataset)
    expected = model.predict(dataset.conditions)
    path, info_path = save_model(model, tmp_path / "model.joblib")
    restored = load_model(path, info_path)
    np.testing.assert_allclose(restored.predict(dataset.conditions), expected, atol=1e-6)
    repeated = FieldSurrogate(model.candidate).fit(dataset)
    np.testing.assert_array_equal(repeated.predict(dataset.conditions), expected)


def test_field_components_and_units_survive_without_the_training_dataset(tmp_path):
    dataset = replace(_dataset(), components=("temperature",), field_units=("K",))
    model = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(dataset)
    path, info_path = save_model(model, tmp_path / "model.joblib")
    info = read_json(info_path)["input_schema"]
    assert info["components"] == ["temperature"] and info["field_units"] == ["K"]
    restored = load_model(path, info_path)
    assert restored.field_units == ("K",) and restored.components == ("temperature",)
    with pytest.raises(ValueError, match="components"):
        restored.transform(replace(dataset, components=("pressure",)))
    with pytest.raises(ValueError, match="units"):
        restored.transform(replace(dataset, field_units=("degC",)))


@pytest.mark.parametrize("solver", [None, "direct", "randomized", "incremental"])
def test_backend_compatibility_is_separate_from_package_provenance(tmp_path, solver):
    dataset = _dataset()
    decomposition = (
        {"name": "dct2"}
        if solver is None
        else {"name": "pod", "params": {"solver": solver, "n_modes": 2, "batch_size": 4}}
    )
    model = FieldRepresentation({"decomposition": decomposition}).fit(dataset)
    path, info_path = save_model(model, tmp_path / "model.joblib")
    info = read_json(info_path)
    assert "scikit-learn" not in info["required_packages"]
    info["runtime"]["packages"]["ml-platform-mode-decomp"] = "0.8.99"
    info["runtime"]["packages"]["scikit-learn"] = "0.0.provenance-only"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    np.testing.assert_array_equal(load_model(path).reconstruct(dataset), model.reconstruct(dataset))
    info["runtime"]["packages"]["numpy"] = "0.0.incompatible"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="requires package numpy"):
        load_model(path)


def test_save_refuses_overwrite_and_rolls_back_an_incomplete_pair(tmp_path, monkeypatch):
    import ml_platform_mode_decomp.model_artifact as artifacts

    model = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(_dataset())
    path, _ = save_model(model, tmp_path / "model.joblib")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_model(model, path)
    assert path.read_bytes() == original
    link = artifacts.os.link

    def fail_sidecar(source, destination):
        if str(destination).endswith("_info.json"):
            raise OSError("simulated sidecar installation failure")
        return link(source, destination)

    monkeypatch.setattr(artifacts.os, "link", fail_sidecar)
    with pytest.raises(OSError, match="sidecar"):
        save_model(model, tmp_path / "new.joblib")
    assert not (tmp_path / "new.joblib").exists()
    assert not (tmp_path / "new_info.json").exists()
