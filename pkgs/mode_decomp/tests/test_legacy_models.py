"""Actual 0.7.1 bundles, produced before the 0.8 source changes."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from ml_platform_mode_decomp import FieldDataset, load_model


@pytest.mark.parametrize("name", ["dct", "pod_ridge", "calibrated"])
def test_real_071_bundle_retains_predictions_and_calibrated_bands(name):
    folder = Path(__file__).parent / "fixtures/models_v0_7_1"
    producer = json.loads((folder / "producer.json").read_text(encoding="utf-8"))
    info = json.loads((folder / name / "model_info.json").read_text(encoding="utf-8"))
    if info["runtime"]["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        pytest.skip("Legacy joblib fixture requires the producer Python minor version")
    for package, trained in producer["runtime"].items():
        if package != "ml-platform-core" and version(package) != trained:
            pytest.skip(f"Legacy joblib fixture requires {package}=={trained}; use the locked environment")
    with np.load(folder / "inputs.npz", allow_pickle=False) as arrays:
        query = FieldDataset.from_arrays(
            arrays["fields"][48:],
            arrays["conditions"][48:],
            x=arrays["x"],
            y=arrays["y"],
            components=["temperature"],
            field_units=["K"],
            condition_columns=["a", "b"],
        )
    model = load_model(folder / name / "model.joblib")
    with np.load(folder / name / "expected.npz", allow_pickle=False) as expected:
        if name == "dct":
            np.testing.assert_allclose(model.reconstruct(query), expected["prediction"], atol=1e-12)
            np.testing.assert_allclose(model.transform(query, space="a"), expected["a"], atol=1e-12)
        else:
            np.testing.assert_allclose(model.predict(query.conditions), expected["prediction"], atol=1e-12)
            np.testing.assert_allclose(
                model.predict_coefficients(query.conditions), expected["coefficients"], atol=1e-12
            )
            if name == "calibrated":
                _, lower, upper = model.predict_interval(query.conditions)
                np.testing.assert_allclose(lower, expected["lower"], atol=1e-12)
                np.testing.assert_allclose(upper, expected["upper"], atol=1e-12)
                assert model.calibrator.field_units == ("K",)


@pytest.mark.parametrize("name", ["hodge", "polar_fft", "laplace_beltrami", "spherical_harmonics", "spherical_slepian"])
def test_real_071_geometry_backend_bundle_retains_coefficients_and_reconstruction(name):
    folder = Path(__file__).parent / "fixtures/models_v0_7_1" / name
    producer = json.loads((folder / "producer.json").read_text(encoding="utf-8"))
    info = json.loads((folder / "model_info.json").read_text(encoding="utf-8"))
    if info["runtime"]["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        pytest.skip("Legacy joblib fixture requires the producer Python minor version")
    for package, trained in producer["runtime"].items():
        if package not in {"ml-platform-core", "ml-platform-mode-decomp"} and version(package) != trained:
            pytest.skip(f"Legacy joblib fixture requires {package}=={trained}; use the locked environment")
    model = load_model(folder / "model.joblib")
    with np.load(folder / "expected.npz", allow_pickle=False) as expected:
        np.testing.assert_allclose(model.reconstruct(expected["fields"]), expected["prediction"], atol=1e-12)
        np.testing.assert_allclose(model.transform(expected["fields"], space="a"), expected["a"], atol=1e-12)


@pytest.mark.parametrize("name", ["weighted_pod", "weighted_gappy", "pod_svd", "offset_dct", "hodge", "calibrated"])
def test_real_090_schema2_bundle_retains_coordinates_fields_and_bands(name):
    folder = Path(__file__).parent / "fixtures/models_v0_9_0"
    producer = json.loads((folder / "producer.json").read_text(encoding="utf-8"))
    if producer["runtime"]["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        pytest.skip("Legacy joblib fixture requires the producer Python minor version")
    info = json.loads((folder / name / "model_info.json").read_text(encoding="utf-8"))
    for package in info["required_packages"]:
        if version(package) != producer["runtime"]["packages"][package]:
            pytest.skip(f"Legacy joblib fixture requires the producer {package}; use the locked environment")
    model = load_model(folder / name / "model.joblib")
    with np.load(folder / name / "expected.npz", allow_pickle=False) as expected:
        if name == "calibrated":
            prediction, lower, upper = model.predict_interval(expected["conditions"])
            np.testing.assert_allclose(prediction, expected["prediction"], atol=1e-11)
            np.testing.assert_allclose(lower, expected["lower"], atol=1e-11)
            np.testing.assert_allclose(upper, expected["upper"], atol=1e-11)
            np.testing.assert_allclose(
                model.predict_coefficients(expected["conditions"]), expected["coefficients"], atol=1e-11
            )
        else:
            mask = expected["mask"] if "mask" in expected else None
            a = model.transform(expected["fields"], mask, space="a")
            z = model.transform(expected["fields"], mask)
            # Old offset encoding rounded residuals to float32. Its deliberate
            # precision fix changes re-encoding, while stored coordinates must
            # still synthesize the original field within that same rounding.
            tolerance = 2e-6 if name == "offset_dct" else 1e-11
            np.testing.assert_allclose(a, expected["a"], atol=tolerance)
            np.testing.assert_allclose(z, expected["z"], atol=tolerance)
            np.testing.assert_allclose(model.inverse_transform(expected["z"]), expected["prediction"], atol=tolerance)
