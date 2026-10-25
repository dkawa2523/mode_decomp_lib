"""Scientific counterexamples for independent coordinates and physical operations."""

from dataclasses import replace
from importlib.metadata import PackageNotFoundError

import numpy as np
import pytest
import pandas as pd
import yaml

from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.decomposition.physical import analyze_components, validate_components
from ml_platform_mode_decomp.model_artifact import load_model, save_model
from ml_platform_mode_decomp.representation import FieldRepresentation
from ml_platform_mode_decomp.surrogate import FieldSurrogate
from ml_platform_mode_decomp.catalog import build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec


@pytest.mark.parametrize("method", ["pod", "gappy_pod", "pod_svd"])
def test_independent_rank_zero_and_three_have_only_meaningful_coordinates(method, tmp_path):
    matrix = np.vstack((np.eye(4)[:3], -np.eye(4)[:3]))
    fields = np.stack((np.zeros_like(matrix), matrix), axis=-1).reshape(6, 2, 2, 2)
    data = FieldDataset.from_arrays(fields, layout="NHWC", components=["empty", "varying"])
    params = {"reg_lambda": 0} if method == "gappy_pod" else {}
    rep = FieldRepresentation({"decomposition": {"name": method, "params": params}}).fit(data)
    assert rep.packed_dim == 3
    assert rep.raw_meta["component_modes"] == [0, 3]
    assert rep.layout.component_offsets == (0, 0, 3)
    assert all(mode["component_name"] == "varying" for mode in rep.list_modes())
    np.testing.assert_allclose(rep.reconstruct(data), fields, atol=1e-12)
    if method == "gappy_pod":
        assert all(part._pod is None for part in rep.decomposer._impl._decomposers)
    artifact, _ = save_model(rep, tmp_path / "model.joblib")
    np.testing.assert_allclose(load_model(artifact).reconstruct(data), fields, atol=1e-12)


@pytest.mark.parametrize("solver", ["direct", "randomized", "incremental"])
def test_constant_profile_uses_baseline_without_a_regressor_or_dummy_coordinate(solver, tmp_path, monkeypatch):
    profile = np.array([[0.1, 1e8 + 1], [-2.0, 4.0]])
    data = FieldDataset.from_arrays(
        np.repeat(profile[None, ..., None], 6, axis=0),
        layout="NHWC",
        conditions=np.arange(6.0)[:, None],
        condition_columns=["x"],
    )

    def no_regressor(*_args, **_kwargs):
        pytest.fail("No regression backend is needed for a constant field")

    monkeypatch.setattr("ml_platform_mode_decomp.surrogate.build_regressor", no_regressor)
    model = FieldSurrogate(
        {
            "representation": {"decomposition": {"name": "pod", "params": {"solver": solver}}},
            "regression": {"name": "ridge"},
        }
    ).fit(data)
    assert model.regressor is None and model.prediction_kind == "constant_field"
    assert model.predict_coefficients(np.array([[0.5], [8.0]])).shape == (2, 0)
    np.testing.assert_allclose(model.predict(np.array([[0.5], [8.0]])), data.fields[:2], atol=1e-12)
    artifact, info = save_model(model, tmp_path / "constant.joblib")
    from ml_platform_core.io import read_json

    assert "scikit-learn" not in read_json(info)["required_packages"]
    np.testing.assert_allclose(load_model(artifact).predict(np.array([[0.5]])), data.fields[:1], atol=1e-12)
    with pytest.raises(ValueError, match="finite matrix"):
        model.predict(np.array([[np.nan]]))


def test_offset_keeps_float64_variations_and_force_skips_auto_diagnostics(monkeypatch):
    fields = np.stack([1e8 + np.array([[1, -1], [-1, 1]]) * amplitude for amplitude in (1, 2, 3)])
    data = FieldDataset.from_arrays(fields[..., None], layout="NHWC")

    def no_auto(*_args):
        pytest.fail("Explicit enabled must not sample an automatic split diagnostic")

    monkeypatch.setattr(
        "ml_platform_mode_decomp.decomposition.offset_residual.OffsetResidualDecomposer._decide_split", no_auto
    )
    rep = FieldRepresentation({"decomposition": {"name": "offset_residual", "params": {"enabled": True}}}).fit(data)
    np.testing.assert_allclose(rep.reconstruct(data), data.fields, atol=2e-8, rtol=0)
    disabled = FieldRepresentation({"decomposition": {"name": "offset_residual", "params": {"enabled": False}}}).fit(
        data
    )
    assert all(mode["additive"] for mode in disabled.list_modes())


@pytest.mark.parametrize("enabled", [False, True])
def test_nested_physical_operator_cannot_bypass_vector_meaning(enabled):
    data = FieldDataset.from_arrays(np.ones((4, 3, 3, 2)), layout="NHWC", components=["a", "b"])
    with pytest.raises(ValueError, match="fields.kind: vector"):
        FieldRepresentation(
            {"decomposition": {"name": "offset_residual", "params": {"enabled": enabled, "inner": {"name": "hodge"}}}}
        ).fit(data)


@pytest.mark.parametrize("method", ["hodge", "helmholtz"])
def test_physical_parts_do_not_depend_on_the_learning_representation(method):
    rng = np.random.default_rng(77)
    data = FieldDataset.from_arrays(
        rng.normal(size=(5, 4, 5, 2)),
        layout="NHWC",
        field_kind="vector",
        components=["u", "v"],
        field_units=["m/s", "m/s"],
    )
    rep = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(data)
    arrays, rows, summary = analyze_components(
        rep.field_spec, rep.geometry, data.fields, data.sample_ids, method=method
    )
    if method == "hodge":
        restored = sum(arrays[f"node_{part}_edge_lift"] for part in ("exact", "coexact", "harmonic"))
        restored += arrays["node_sampling_residual"]
        for row in rows:
            assert row["sum_component_edge_energy"] == pytest.approx(row["input_edge_energy"], abs=1e-9)
    else:
        restored = arrays["node_curl_free"] + arrays["node_div_free"]
        for row in rows:
            assert row["sum_component_grid_energy"] == pytest.approx(row["input_grid_energy"], abs=1e-10)
    np.testing.assert_allclose(restored, data.fields, atol=1e-10)
    assert summary["method"] == method
    with pytest.raises(ValueError, match="same physical unit"):
        analyze_components(
            replace(rep.field_spec, units=("m/s", "K")), rep.geometry, data.fields, data.sample_ids, method=method
        )


@pytest.mark.parametrize(
    "method,params",
    [
        ("zernike", {"n_max": 2}),
        ("fourier_bessel", {"m_max": 1, "n_max": 2}),
        ("fourier_jacobi", {"m_max": 1, "k_max": 2}),
    ],
)
def test_fixed_support_projection_shares_a_solve_but_varying_support_does_not(method, params, monkeypatch):
    geometry = build_domain_spec(
        {"name": "disk", "radius": 1.0, "center": [0, 0], "x_range": [-1, 1], "y_range": [-1, 1]}, (19, 19)
    )
    fields = np.random.default_rng(4).normal(size=(6, 19, 19, 2))
    expected_operator = build_decomposer({"name": method, **params})
    expected = np.stack([expected_operator.transform(field, None, geometry) for field in fields])
    actual_operator = build_decomposer({"name": method, **params})
    calls = []
    original = np.linalg.lstsq

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", counted)
    actual = actual_operator.transform_batch(fields, None, geometry)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert len(calls) == 1
    calls.clear()
    masks = np.ones(fields.shape[:-1], dtype=bool)
    masks[0, 9, 9] = False
    varying = actual_operator.transform_batch(fields, masks, geometry)
    assert len(calls) == len(fields)
    fresh = build_decomposer({"name": method, **params})
    expected = np.stack([fresh.transform(field, mask, geometry) for field, mask in zip(fields, masks, strict=True)])
    np.testing.assert_allclose(varying, expected, atol=1e-12)


def test_direct_dct_layout_cannot_be_replaced_by_an_hwc_query():
    geometry = build_domain_spec({"name": "rectangle", "x_range": [0, 1], "y_range": [0, 1]}, (3, 4))
    field = np.arange(12.0).reshape(3, 4)
    operator = build_decomposer({"name": "dct2"})
    coefficients = operator.transform(field, None, geometry)
    with pytest.raises(ValueError, match="Field layout"):
        operator.transform(field[..., None], None, geometry)
    np.testing.assert_allclose(operator.inverse_transform(coefficients, geometry), field, atol=1e-12)


@pytest.mark.parametrize("task", ["mode_decomp_pipeline", "mode_decomp_analysis"])
def test_constant_workflow_does_not_require_an_unused_optional_regression_recipe(task, tmp_path, monkeypatch):
    from ml_platform_mode_decomp import run_task
    from ml_platform_mode_decomp import model_artifact
    from ml_platform_core.result import RunResult

    root = tmp_path / "input"
    root.mkdir()
    np.save(root / "fields.npy", np.repeat(np.arange(4.0).reshape(1, 2, 2, 1), 20, axis=0))
    pd.DataFrame(
        {"sample_id": [f"s{index}" for index in range(20)], "array_index": range(20), "x": np.arange(20.0)}
    ).to_csv(root / "samples.csv", index=False)
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {"path": "fields.npy", "layout": "NHWC", "kind": "scalar", "components": ["signal"]},
        "conditions": {"format": "table", "columns": ["x"]},
        "geometry": {"kind": "rectangle", "axes": {"x": {"values": [0, 1]}, "y": {"values": [0, 1]}}},
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    actual_version = model_artifact.version

    def without_xgboost(package):
        if package == "xgboost":
            raise PackageNotFoundError(package)
        return actual_version(package)

    monkeypatch.setattr(model_artifact, "version", without_xgboost)
    candidate = {"id": "constant", "representation": {"decomposition": {"name": "pod"}}, "regression": {"name": "xgb"}}
    cfg = {
        "task": task,
        "data": {"local_path": str(root)},
        "runtime": {"output_dir": str(tmp_path / "output")},
        "metrics": {"names": ["field_rmse", "field_mae"]},
        "model": {"candidates": [candidate]},
    }
    if task == "mode_decomp_analysis":
        cfg["analysis"] = {"kind": "surrogate", "bootstrap_samples": 0}
    result = run_task(cfg)
    assert isinstance(result, RunResult)
    if task == "mode_decomp_pipeline":
        assert result.metrics["field_rmse"] == 0
    else:
        comparison = pd.read_csv(result.tables["comparison"])
        assert (comparison.loc[comparison.candidate_id == "constant", "field_rmse"] == 0).all()
    assert "xgboost" not in model_artifact.evaluation_identity(candidate)["runtime"]["packages"]


def test_plain_fit_establishes_packing_without_encoding_unused_training_coordinates(monkeypatch):
    data = FieldDataset.from_arrays(np.random.default_rng(8).normal(size=(12, 3, 4, 1)))
    original = FieldRepresentation._encode
    encoded_counts = []

    def counted(self, fields, masks, **kwargs):
        encoded_counts.append(len(fields))
        return original(self, fields, masks, **kwargs)

    monkeypatch.setattr(FieldRepresentation, "_encode", counted)
    rep = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(data)
    assert encoded_counts == [1]
    np.testing.assert_allclose(rep.reconstruct(data), data.fields, atol=1e-12)
    encoded_counts.clear()
    posted = FieldRepresentation({"decomposition": {"name": "dct2"}, "coeff_post": {"name": "standardize"}}).fit(data)
    assert encoded_counts == [len(data)]
    np.testing.assert_allclose(posted.reconstruct(data), data.fields, atol=1e-12)


def test_component_preflight_checks_inputs_without_building_a_cochain_complex(monkeypatch):
    data = FieldDataset.from_arrays(np.ones((1, 3, 4, 2)), field_kind="vector")
    rep = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(data)

    def no_complex(*_args, **_kwargs):
        pytest.fail("Preflight must not allocate cochain operators")

    monkeypatch.setattr("ml_platform_mode_decomp.decomposition.hodge.HodgeDecomposition.__init__", no_complex)
    validate_components(rep.field_spec, rep.geometry)
    with pytest.raises(ValueError, match="tolerance"):
        validate_components(rep.field_spec, rep.geometry, params={"tolerance": -1})
    periodic = replace(rep.geometry, meta={**rep.geometry.meta, "periodic_axes": ["x", "y"]})
    with pytest.raises(ValueError, match="nonperiodic"):
        validate_components(rep.field_spec, periodic)
