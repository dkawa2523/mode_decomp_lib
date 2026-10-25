"""Bound query work while preserving physical predictions and portable output."""

import numpy as np
import pandas as pd
import pytest
from ml_platform_core.io import read_json
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import FieldDataset, FieldSurrogate, run_task
from ml_platform_mode_decomp.model_artifact import save_model


def fitted_surrogate(method="dct2", scope=None, *, dtype=np.float64, field_shape=(4, 5)):
    rng = np.random.default_rng(91)
    conditions = rng.normal(size=(60, 2)).astype(dtype)
    x, y = np.meshgrid(np.linspace(-1, 1, field_shape[1]), np.linspace(-1, 1, field_shape[0]))
    base = conditions[:, 0, None, None] * x + conditions[:, 1, None, None] * y
    base += 0.05 * conditions[:, 0, None, None] ** 2 * (x * y)
    dataset = FieldDataset.from_arrays(
        np.stack((base, 2 * base + 0.3), axis=-1),
        conditions,
        condition_columns=("a", "b"),
        components=("first", "second"),
        field_units=("m/s", "m/s"),
    )
    model = FieldSurrogate(
        {
            "representation": {"decomposition": {"name": method}},
            "regression": {"name": "ridge", "params": {"alpha": 0.1}},
        }
    ).fit(dataset.subset(np.arange(40)))
    if scope is not None:
        model.calibrate(
            dataset.subset(np.arange(40, 60)), training=dataset.subset(np.arange(40)), coverage=0.8, scope=scope
        )
    return model


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method,scope,field_shape,count",
    [
        ("fft2", "simultaneous", (4, 5), 517),
        ("dct2", "pointwise", (4, 5), 517),
        ("pod_svd", "simultaneous", (64, 64), 65),
    ],
)
def test_query_batches_match_independent_full_coefficient_reconstruction(
    method, scope, field_shape, count, dtype, monkeypatch
):
    model = fitted_surrogate(method, scope, dtype=dtype, field_shape=field_shape)
    conditions = np.random.default_rng(5).normal(size=(2 * count, 2)).astype(dtype)[::2]
    conditions.setflags(write=False)
    coefficients = model.regressor.predict(conditions)
    expected = model.representation.inverse_transform(coefficients)
    expected_parts = (expected,) if scope is None else (expected, *model.calibrator.intervals(expected))
    predict = model.regressor.predict
    calls = []

    def tracked(values):
        calls.append(len(values))
        return predict(values)

    def same_values(values, truth):
        if dtype == np.float32:
            # SGEMM can choose a different accumulation order for query tails.
            # Bound normwise error by four float32 eps at the oracle's scale;
            # float64 retains the stricter numerical equivalence check.
            tolerance = 4 * np.finfo(np.float32).eps * max(1, float(np.max(np.abs(truth))))
            np.testing.assert_allclose(values, truth, rtol=0, atol=tolerance)
        else:
            np.testing.assert_allclose(values, truth, rtol=1e-12, atol=1e-12)

    monkeypatch.setattr(model.regressor, "predict", tracked)
    same_values(model.predict_coefficients(conditions), coefficients)
    assert max(calls) <= 256 and sum(calls) == len(conditions)
    if len(conditions) > 256:
        assert len(calls) > 1
    calls.clear()
    actual = (model.predict(conditions),) if scope is None else model.predict_interval(conditions)
    assert len(calls) > 1 and max(calls) <= 256 and sum(calls) == len(conditions)
    for value, truth in zip(actual, expected_parts, strict=True):
        same_values(value, truth)
    if scope is not None:
        np.testing.assert_array_equal(model.predict(conditions), actual[0])
    batches = list(model.iter_predict(conditions, batch_size=7, intervals=scope is not None))
    for index, truth in enumerate(expected_parts):
        values = [batch if scope is None else batch[index] for batch in batches]
        assert max(map(len, values)) <= 7
        same_values(np.concatenate(values), truth)


@pytest.mark.parametrize("suffix", ["npz", "npy"])
def test_condition_only_inference_streams_saved_bands_and_preserves_output_identity(tmp_path, monkeypatch, suffix):
    model = fitted_surrogate(scope="pointwise")
    conditions = np.random.default_rng(8).normal(size=(519, 2))
    ids = [f"query_{index:04d}_観測" for index in range(len(conditions))]
    expected = model.predict_interval(conditions)
    artifact, _ = save_model(model, tmp_path / "model.joblib")
    queries = tmp_path / "conditions.csv"
    pd.DataFrame({"sample_id": ids, "b": conditions[:, 1], "a": conditions[:, 0]}).to_csv(queries, index=False)

    def unexpected_full_prediction(*args, **kwargs):
        raise AssertionError("Inference must consume query batches directly")

    monkeypatch.setattr(FieldSurrogate, "predict", unexpected_full_prediction)
    monkeypatch.setattr(FieldSurrogate, "predict_interval", unexpected_full_prediction)
    result = run_task(
        {
            "task": "mode_decomp_infer",
            "data": {"local_path": str(queries)},
            "model": {"artifact_path": str(artifact)},
            "runtime": {"output_dir": str(tmp_path / "inference")},
            "uncertainty": {"enabled": True},
            "output": {"prediction_name": f"custom_fields.{suffix}"},
        }
    )
    assert result.artifacts["predictions"].name == f"custom_fields.{suffix}"
    if suffix == "npz":
        with np.load(result.artifacts["predictions"], allow_pickle=False) as saved:
            assert saved["sample_ids"].tolist() == ids
            assert saved["components"].tolist() == ["first", "second"]
            assert saved["field_units"].tolist() == ["m/s", "m/s"]
            assert saved["interval_scope"].item() == "pointwise"
            for key, truth in zip(("predictions", "lower", "upper"), expected, strict=True):
                np.testing.assert_allclose(saved[key], truth, atol=1e-12)
    else:
        metadata = read_json(result.artifacts["prediction_metadata"])
        assert metadata["sample_count"] == len(ids)
        assert metadata["field_shape"] == [4, 5, 2]
        assert metadata["field_kind"] == "multichannel"
        assert metadata["components"] == ["first", "second"]
        assert metadata["field_units"] == ["m/s", "m/s"]
        assert metadata["field_uncertainty"]["scope"] == "pointwise"
        for name, truth in zip(("predictions", "lower", "upper"), expected, strict=True):
            values = np.load(result.run_dir / metadata["arrays"][name], mmap_mode="r", allow_pickle=False)
            assert isinstance(values, np.memmap)
            np.testing.assert_allclose(values, truth, atol=1e-12)
        assert np.load(result.artifacts["prediction_sample_ids"], allow_pickle=False).tolist() == ids
    assert not list(result.run_dir.glob(".predictions_*"))


@pytest.mark.parametrize("suffix", ["npz", "npy"])
def test_failed_prediction_batch_keeps_existing_artifacts_and_publishes_no_partial_fields(
    tmp_path, monkeypatch, suffix
):
    model = fitted_surrogate()
    artifact, _ = save_model(model, tmp_path / "model.joblib")
    queries = tmp_path / "conditions.csv"
    queries.write_text("sample_id,a,b\none,1,2\ntwo,3,4\n", encoding="utf-8")
    predict = FieldSurrogate.iter_predict

    def fail_after_first(self, conditions, **kwargs):
        yield next(predict(self, conditions[:1], **kwargs))
        raise ValueError("second query batch failed")

    monkeypatch.setattr(FieldSurrogate, "iter_predict", fail_after_first)
    with pytest.raises(RunError, match="second query batch failed") as caught:
        run_task(
            {
                "task": "mode_decomp_infer",
                "data": {"local_path": str(queries)},
                "model": {"artifact_path": str(artifact)},
                "runtime": {"output_dir": str(tmp_path / "inference")},
                "output": {"prediction_name": f"custom_fields.{suffix}"},
            }
        )
    result = caught.value.result
    assert result.artifacts["config"].is_file()
    assert "predictions" not in result.artifacts
    assert not list(result.run_dir.glob("custom_fields*"))
    assert not list(result.run_dir.glob(".predictions_*"))
