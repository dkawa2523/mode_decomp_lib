"""One input description supports local inline use and reusable external datasets."""

from copy import deepcopy

import numpy as np
import pytest
import yaml
from ml_platform_core.artifacts import file_hash

from ml_platform_mode_decomp import load_model, run_task
from ml_platform_mode_decomp.data import (
    FieldDataset,
    evaluate_scattered_prediction,
    import_csv_field_dataset,
    import_scattered_field_dataset,
    load_field_dataset,
    load_scattered_observations,
)
from ml_platform_mode_decomp.data.input_spec import field_input_spec

from test_csv_import import _grid_source
from test_scattered import _write_source


def _grid_spec(root, layout="HWC"):
    return {
        "local_path": str(root),
        "format": "npy",
        "fields": {"path": "field.npy", "layout": layout, "kind": "scalar", "components": ["value"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0, "step": 1, "count": 4},
                "y": {"start": 0, "step": 1, "count": 3},
            },
        },
    }


def test_single_field_needs_only_npy_and_inline_run_description(tmp_path):
    field = np.arange(12, dtype=float).reshape(3, 4, 1)
    np.save(tmp_path / "field.npy", field)
    settings = {
        "task": "mode_decomp_representation",
        "data": _grid_spec(tmp_path),
        "representation": {"decomposition": {"name": "dct2"}},
        "runtime": {"output_dir": str(tmp_path / "runs")},
    }
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump(settings), encoding="utf-8")
    dataset = load_field_dataset(yaml.safe_load(run_yaml.read_text(encoding="utf-8")))
    assert dataset.fields.shape == (1, 3, 4, 1)
    assert dataset.sample_ids.tolist() == ["sample_000000"]
    assert {path.name for path in tmp_path.iterdir()} == {"field.npy", "run.yaml"}
    result = run_task(yaml.safe_load(run_yaml.read_text(encoding="utf-8")))
    representation = load_model(result.artifacts["model"])
    np.testing.assert_allclose(
        representation.inverse_transform(representation.transform(dataset))[0], field, atol=1e-12
    )
    assert not (tmp_path / "samples.csv").exists() and not (tmp_path / "field_dataset.yaml").exists()


def test_generated_sample_rows_preserve_npy_condition_correspondence(tmp_path):
    conditions = np.array([[3, 7], [-2, 4], [11, -5]], dtype=float)
    fields = np.arange(36, dtype=float).reshape(3, 3, 4, 1)
    np.save(tmp_path / "field.npy", fields)
    np.save(tmp_path / "cond.npy", conditions)
    data = _grid_spec(tmp_path, "NHWC")
    data["conditions"] = {"format": "npy", "path": "cond.npy", "columns": ["a", "b"]}
    dataset = load_field_dataset(data)
    assert dataset.sample_ids.tolist() == ["sample_000000", "sample_000001", "sample_000002"]
    np.testing.assert_array_equal(dataset.conditions, conditions)
    np.testing.assert_array_equal(dataset.fields, fields)
    assert not (tmp_path / "samples.csv").exists()


def test_single_mesh_vc_keeps_explicit_vertex_order_and_measure(tmp_path):
    np.save(tmp_path / "field.npy", np.array([[9.0], [3.0], [7.0]]))
    data = {
        "local_path": str(tmp_path),
        "fields": {"path": "field.npy", "layout": "VC", "kind": "scalar", "components": ["p"]},
        "geometry": {"kind": "mesh", "vertices": [[0, 0], [1, 0], [0, 1]], "faces": [[0, 1, 2]]},
    }
    dataset = load_field_dataset(data)
    assert dataset.fields.shape == (1, 3, 1)
    np.testing.assert_array_equal(dataset.fields[0, :, 0], [9, 3, 7])
    assert dataset.geometry.integration_weights().sum() == pytest.approx(0.5)
    assert [path.name for path in tmp_path.iterdir()] == ["field.npy"]


@pytest.mark.parametrize("format_name", ["csv", "scattered"])
def test_inline_import_matches_external_spec_without_temporary_manifest(tmp_path, format_name):
    source = tmp_path / "source"
    importer = import_csv_field_dataset if format_name == "csv" else import_scattered_field_dataset
    if format_name == "csv":
        manifest, _ = _grid_source(source)
    else:
        manifest, _, _ = _write_source(source)
    importer(source, tmp_path / "external", source_manifest="import.yaml")
    data = {key: deepcopy(value) for key, value in manifest.items() if key != "schema"}
    if format_name == "scattered":
        data.pop("interpolation")  # linear / mask are the useful supported defaults.
    selected, reference = field_input_spec({"local_path": str(source), "format": format_name, **data})
    assert selected == format_name
    before = {path.name: file_hash(path) for path in source.iterdir()}
    importer(source, tmp_path / "inline", source_manifest=reference)
    external, inline = (load_field_dataset(tmp_path / name) for name in ("external", "inline"))
    np.testing.assert_array_equal(inline.fields, external.fields)
    np.testing.assert_array_equal(inline.effective_mask, external.effective_mask)
    np.testing.assert_array_equal(inline.conditions, external.conditions)
    assert inline.sample_ids.tolist() == external.sample_ids.tolist()
    assert {path.name: file_hash(path) for path in source.iterdir()} == before
    if format_name == "scattered":
        # The original-point evaluator uses the same inline canonical description.
        canonical = inline.manifest
        point_report = evaluate_scattered_prediction(tmp_path / "inline", inline.fields, source_manifest=canonical)
        assert point_report["field"]["field_rmse"] == pytest.approx(0)
        assert point_report["interpolation"]["rmse"][0] > 0
        assert load_scattered_observations(tmp_path / "inline", source_manifest=canonical)["sample_ids"].tolist() == [
            "001",
            "002",
        ]


def test_inline_geometry_changes_revision_and_cannot_silently_override_external_geometry(tmp_path):
    np.save(tmp_path / "field.npy", np.ones((3, 4, 1)))
    data = _grid_spec(tmp_path)
    before = load_field_dataset(data).revision
    original_hash = file_hash(tmp_path / "field.npy")
    data["geometry"]["axes"]["x"]["step"] = 2
    assert load_field_dataset(data).revision != before
    assert file_hash(tmp_path / "field.npy") == original_hash
    data["source_manifest"] = "reusable.yaml"
    with pytest.raises(ValueError, match="inline.*source_manifest|inline.*referenced"):
        field_input_spec(data)
    with pytest.raises(ValueError, match="inline.*source_manifest|inline.*referenced"):
        load_field_dataset(data)


def test_minimal_hwc_defaults_match_array_api_without_inventing_physical_coordinates(tmp_path):
    values = np.arange(24, dtype=float).reshape(3, 4, 2)
    np.save(tmp_path / "field.npy", values)
    dataset = load_field_dataset({"local_path": str(tmp_path), "fields": {"path": "field.npy", "layout": "HWC"}})
    direct = FieldDataset.from_arrays(values, layout="HWC")
    assert dataset.manifest["fields"]["kind"] == "multichannel"
    assert dataset.components == direct.components == ("component_0", "component_1")
    np.testing.assert_array_equal(dataset.geometry.coords["x"], direct.geometry.coords["x"])
    np.testing.assert_array_equal(dataset.geometry.coords["y"], direct.geometry.coords["y"])
    np.testing.assert_array_equal(dataset.geometry.integration_weights(), direct.geometry.integration_weights())
    assert dataset.geometry.meta["weight_measure"] == "index_coordinate_measure"
    assert dataset.geometry.meta["coordinate_source"] == {"x": "index", "y": "index"}
    assert [path.name for path in tmp_path.iterdir()] == ["field.npy"]
    domain = np.ones((3, 4), dtype=bool)
    domain[1, 2] = False
    np.save(tmp_path / "domain.npy", domain)
    masked = load_field_dataset(
        {
            "local_path": str(tmp_path),
            "fields": {"path": "field.npy", "layout": "HWC"},
            "masks": {"domain": "domain.npy"},
        }
    )
    np.testing.assert_array_equal(
        masked.geometry.integration_weights(), np.where(domain, direct.geometry.integration_weights(), 0)
    )


def test_nhwc_and_condition_name_defaults_preserve_declared_physical_geometry_and_mask(tmp_path):
    np.save(tmp_path / "field.npy", np.arange(24, dtype=float).reshape(2, 3, 4, 1))
    conditions = np.asarray([[3.0, 1.0], [-7.0, 2.0]])
    np.save(tmp_path / "conditions.npy", conditions)
    domain = np.ones((3, 4), dtype=bool)
    domain[0, 0] = False
    np.save(tmp_path / "domain.npy", domain)
    dataset = load_field_dataset(
        {
            "local_path": str(tmp_path),
            "fields": {"path": "field.npy"},
            "conditions": {"format": "npy", "path": "conditions.npy"},
            "masks": {"domain": "domain.npy"},
            "geometry": {
                "kind": "rectangle",
                "coordinate_units": {"x": "m", "y": "m"},
                "axes": {
                    "x": {"start": 100, "step": 2, "count": 4},
                    "y": {"start": -2, "step": 3, "count": 3},
                },
            },
        }
    )
    assert dataset.fields.shape == (2, 3, 4, 1)
    assert dataset.manifest["fields"]["kind"] == "scalar"
    assert dataset.components == ("component_0",)
    assert dataset.condition_columns == ("condition_0", "condition_1")
    np.testing.assert_array_equal(dataset.conditions, conditions)
    np.testing.assert_array_equal(dataset.geometry.coords["x"][0], [100, 102, 104, 106])
    assert dataset.geometry.integration_weights()[0, 0] == 0
    assert dataset.geometry.integration_weights().sum() == pytest.approx(34.5)
    assert dataset.geometry.meta.get("weight_measure") != "index_coordinate_measure"
