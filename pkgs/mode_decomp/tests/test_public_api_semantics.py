"""Public compositions preserve coefficient domains and calibration roles."""

import numpy as np
import pytest

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate, load_model
from ml_platform_mode_decomp.model_artifact import save_model
from ml_platform_mode_decomp.uncertainty import FieldCalibrator


def _fields(values, prefix):
    values = np.asarray(values, dtype=float)
    return FieldDataset.from_arrays(
        values[:, None, None, None],
        conditions=values[:, None],
        sample_ids=[f"{prefix}_{index}" for index in range(len(values))],
    )


def test_encoding_never_returns_overflowed_latent_coordinates():
    model = FieldRepresentation(
        {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}, "coeff_post": {"name": "standardize"}}
    ).fit(_fields([0, 0.01, 0.02], "train"))
    query = np.array([[[[1e307]]]])
    assert np.isfinite(model.transform(query, space="a")).all()
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        model.transform(query)


def test_interval_flag_is_boolean_and_normalization_uses_actual_training_cases():
    training = _fields(np.linspace(0, 1, 8), "train")
    calibration = _fields(np.linspace(0, 1, 10), "calibration")
    model = FieldSurrogate({"representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}}})
    model.fit(training).calibrate(calibration, training=training)
    assert not isinstance(next(model.iter_predict(calibration, intervals=False)), tuple)
    assert len(next(model.iter_predict(calibration, intervals=True))) == 3
    with pytest.raises(ValueError, match="intervals must be boolean"):
        next(model.iter_predict(calibration, intervals="false"))
    with pytest.raises(ValueError, match="training IDs must match"):
        model.calibrate(calibration, training=_fields(np.linspace(0, 1, 8), "different"))


def test_calibration_retains_field_kind_and_upgrades_saved_bands(tmp_path):
    values = np.ones((10, 2, 3, 2))
    vector = FieldDataset.from_arrays(values, field_kind="vector")
    other = FieldDataset.from_arrays(values, field_kind="multichannel")
    calibrated = FieldCalibrator().fit(vector, np.zeros_like(values))
    with pytest.raises(ValueError, match="kind"):
        calibrated.evaluate(other, np.zeros_like(values))
    training, calibration = _fields(np.arange(8), "train"), _fields(np.arange(10), "calibration")
    model = FieldSurrogate({"representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}}})
    model.fit(training).calibrate(calibration, training=training)
    # Older bundles store target kind on the representation but not the calibrator.
    del model.calibrator.field_kind
    path, _ = save_model(model, tmp_path / "model.joblib")
    restored = load_model(path)
    assert restored.calibrator.field_kind == training.field_kind
    assert restored.calibrator.evaluate(calibration, restored.predict(calibration))["sample_count"] == len(calibration)
