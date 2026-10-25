"""Input declarations preserve observation support, identities and physical measure."""

from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation
from ml_platform_mode_decomp.data import load_condition_batch, load_field_dataset
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.model_artifact import load_model, save_model
from ml_platform_mode_decomp.evaluation import prepare_objective


def test_dataset_and_query_masks_intersect_without_revealing_hidden_values():
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    amplitudes = np.random.default_rng(53).normal(size=(24, 2))
    fields = (10 + np.einsum("nk,khw->nhw", amplitudes, np.stack((x, y))))[..., None]
    model = FieldRepresentation({"decomposition": {"name": "gappy_pod", "params": {"n_modes": 2}}})
    model.fit(FieldDataset.from_arrays(fields))
    known = np.ones((3, 4), dtype=bool)
    known[0, 0] = False
    additional = np.ones((2, 3, 4), dtype=bool)
    additional[:, 0] = False
    additional[1, 1, 0] = False
    corrupted = np.where(additional[..., None], fields[:2], 1e6)
    query = FieldDataset.from_arrays(corrupted, observation_mask=known)
    actual = model.transform(query, masks=additional)
    expected = model.transform(fields[:2], masks=additional & known)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_array_equal(query.observation_mask, known)


def test_file_inputs_do_not_ignore_field_typos_or_a_requested_id_column(tmp_path):
    np.save(tmp_path / "field.npy", np.ones((2, 2, 2, 1)))
    with pytest.raises(ValueError, match="Unknown fields"):
        load_field_dataset({"local_path": str(tmp_path), "fields": {"path": "field.npy", "component": ["pressure"]}})
    (tmp_path / "conditions.csv").write_text("sample_id,temperature\na,20\nb,30\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Configured ID column"):
        load_condition_batch(
            {"local_path": str(tmp_path), "dataset_file": "conditions.csv", "id_column": "case_id"},
            condition_columns=["temperature"],
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"kind": "rectangle", "axes": {"z": {"values": [0, 1, 2]}}},
        {"kind": "rectangle", "x_range": [0, 1], "y_range": [0, 1], "raduis": 2},
        {"kind": "disk", "x_range": [-1, 1], "y_range": [-1, 1], "center": [0, 0], "radius": np.inf},
        {"kind": "sphere_grid", "lat_range": [-60, 60], "lon_range": [-150, 150], "n_lat": 3.8},
        {"kind": "rectangle", "x_range": [0, 1], "y_range": [0, 1], "axes": {"x": {"values": [0, 1, 2, 3]}}},
    ],
)
def test_geometry_cannot_silently_use_a_different_parameter_or_coordinate_system(settings):
    with pytest.raises(ValueError):
        build_domain_spec(settings, (3, 4, 1))


def test_integration_weights_reject_invalid_active_measure_and_keep_masked_locations_inactive():
    geometry = build_domain_spec({"kind": "rectangle", "x_range": [0, 1], "y_range": [0, 1]}, (2, 2, 1))
    for invalid in (-1.0, np.nan):
        weights = np.array([[invalid, 1.0], [2.0, 3.0]])
        with pytest.raises(ValueError, match="finite and non-negative"):
            replace(geometry, weights=weights).integration_weights()
        masked = replace(geometry, weights=weights, mask=np.array([[False, True], [True, True]]))
        np.testing.assert_array_equal(masked.integration_weights(), [[0, 1], [2, 3]])
    dataset = FieldDataset(np.ones((2, 2, 2, 1)), geometry, np.array(["a", "b"]))
    assert dataset.components == ("component_0",)


def test_explicit_model_provenance_expectations_are_all_checked(tmp_path):
    data = FieldDataset.from_arrays(np.ones((2, 2, 2, 1)))
    model = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(data)
    expected = {"workspace_key": "science", "input_revision": "source-1"}
    path, _ = save_model(model, tmp_path / "model.joblib", provenance=expected)
    load_model(path, expected_context=expected)
    for wrong in ({"workspace_key": "other"}, {"input_revison": "source-1"}, {"execution_release_id": "unknown"}):
        with pytest.raises(ValueError, match="Model provenance differs"):
            load_model(path, expected_context=wrong)


def test_hidden_primary_metric_is_evaluated_even_without_optional_metrics():
    data = FieldDataset.from_arrays(np.arange(16.0).reshape(2, 2, 4, 1))
    with pytest.raises(ValueError, match="explicit held-out"):
        prepare_objective(data, {"reference": "hidden"})
    hidden = np.zeros(data.fields.shape[:-1], dtype=bool)
    hidden[:, 0, 0] = True
    objective = prepare_objective(
        data, {"reference": "hidden", "names": [], "primary_metric": "hidden_r2"}, evaluation_mask=hidden
    )
    report = objective.score(data.fields + 0.2)
    assert report.metrics["hidden_r2"] == pytest.approx(1 - 0.2**2 / 16)
    assert not any(name.startswith("field_") for name in report.metrics)
    with pytest.raises(ValueError, match="cannot redefine"):
        prepare_objective(data, {"reference": "observations"}, evaluation_mask=hidden)
