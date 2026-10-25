"""Input declarations preserve observation support, identities and physical measure."""

from dataclasses import replace
import json

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


def test_explicit_manifest_argument_does_not_override_the_selected_input_format(tmp_path):
    np.save(tmp_path / "field.npy", np.ones((2, 2, 2, 1)))
    specification = {"schema": "field_dataset_v1", "fields": {"path": "field.npy"}}
    with pytest.raises(ValueError, match="accepts npy"):
        load_field_dataset({"local_path": str(tmp_path), "format": "csv"}, source_manifest=specification)
    assert len(load_field_dataset({"local_path": str(tmp_path), "format": "npy"}, source_manifest=specification)) == 2


@pytest.mark.parametrize("source", ["inline", "file"])
def test_geometry_mask_source_cannot_choose_the_other_mask_representation(tmp_path, source):
    np.save(tmp_path / "field.npy", np.ones((2, 2, 2, 1)))
    np.save(tmp_path / "mask.npy", np.ones((2, 2), dtype=bool))
    geometry = {
        "kind": "mask",
        "x_range": [0, 1],
        "y_range": [0, 1],
        "mask_source": source,
        **({"mask_path": "mask.npy"} if source == "inline" else {"mask": [[True, True], [True, True]]}),
    }
    with pytest.raises(ValueError, match="mask_source"):
        load_field_dataset({"local_path": str(tmp_path), "fields": {"path": "field.npy"}, "geometry": geometry})


def test_csv_coordinates_and_tolerance_do_not_ignore_or_coerce_explicit_settings(tmp_path):
    from copy import deepcopy
    from test_csv_import import _grid_source
    from ml_platform_mode_decomp.data import import_csv_field_dataset

    root = tmp_path / "source"
    manifest, _ = _grid_source(root)
    for change in ({"coordinates": {"x": "x", "y": "y", "z": "x"}}, {"coordinate_atol": True}):
        configured = deepcopy(manifest)
        configured["fields"].update(change)
        with pytest.raises(ValueError, match="fields.coordinates|coordinate_atol"):
            import_csv_field_dataset(root, tmp_path / "output", source_manifest=configured)
        assert not (tmp_path / "output").exists()


def test_legacy_conversion_preserves_declared_kind_and_rejects_ambiguous_metadata(tmp_path):
    from ml_platform_mode_decomp.data import import_legacy_npy_dataset

    np.save(tmp_path / "field.npy", np.ones((2, 2, 2, 2)))
    np.save(tmp_path / "cond.npy", np.ones((2, 2)))
    manifest = {
        "field_kind": "multichannel",
        "grid": {"H": 2, "W": 2, "x_range": [0, 1], "y_range": [0, 1]},
        "domain": {"type": "rectangle"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    for columns in ([], "ab"):
        with pytest.raises(ValueError, match="condition_columns"):
            import_legacy_npy_dataset(tmp_path, tmp_path / "invalid", condition_columns=columns)
    for changed in ({**manifest, "grid": {**manifest["grid"], "H": 2.8}}, {**manifest, "field_kind": "scalar"}):
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="grid.H|field_kind"):
            import_legacy_npy_dataset(tmp_path, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    converted = import_legacy_npy_dataset(tmp_path, tmp_path / "converted", condition_columns=["a", "b"])
    restored = load_field_dataset(converted)
    assert restored.field_kind == "multichannel"
    assert restored.components == ("component_0", "component_1")
    assert restored.condition_columns == ("a", "b")


def test_legacy_conversion_cannot_ignore_an_alternate_condition_file_or_duplicate_declarations(tmp_path):
    from ml_platform_mode_decomp.input_preparation import convert_field_input

    settings = {"format": "legacy-npy", "local_path": str(tmp_path)}
    with pytest.raises(ValueError, match="Unknown legacy-npy conditions"):
        convert_field_input(
            {**settings, "conditions": {"path": "other.npy", "format": "npy", "columns": ["a"]}},
            tmp_path / "output",
        )
    with pytest.raises(ValueError, match="condition columns once"):
        convert_field_input({**settings, "condition_columns": ["a"]}, tmp_path / "output", condition_columns=["b"])
    with pytest.raises(ValueError, match="geometry once"):
        convert_field_input({**settings, "geometry": {}}, tmp_path / "output", geometry={})
    assert not (tmp_path / "output").exists()


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
