import csv

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.artifacts import file_hash

from ml_platform_mode_decomp.data import import_csv_field_dataset, load_field_dataset


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _grid_source(root):
    root.mkdir()
    _write_csv(
        root / "samples.csv",
        ["sample_id", "array_index", "field_csv", "experiment"],
        [["002", 1, "second.csv", "b"], ["001", 0, "first.csv", "a"]],
    )
    _write_csv(root / "conditions.csv", ["id", "pressure", "temperature"], [["001", 4, 8], ["002", 5, 9]])
    x, y = [0.1, 0.4, 1.3], [-1.0, 2.0]
    physical = np.arange(24, dtype=np.float64).reshape(2, 2, 3, 2) / 4
    physical[1, 0, 0] = np.nan
    observation = np.ones((2, 2, 3), dtype=np.uint8)
    observation[1, 0, 0] = 0
    np.save(root / "observed.npy", observation)
    for index, filename in enumerate(("first.csv", "second.csv")):
        rows = [
            [y[row], physical[index, row, column, 1], x[column], physical[index, row, column, 0]]
            for row in range(2)
            for column in range(3)
        ]
        _write_csv(root / filename, ["y", "speed_x", "x", "speed_y"], [rows[i] for i in [5, 1, 3, 0, 4, 2]])
    manifest = {
        "schema": "field_csv_import_v1",
        "samples": {"path": "samples.csv", "id_column": "sample_id", "index_column": "array_index"},
        "conditions": {
            "format": "table",
            "path": "conditions.csv",
            "id_column": "id",
            "columns": ["temperature", "pressure"],
        },
        "fields": {
            "path_column": "field_csv",
            "layout": "NHWC",
            "kind": "vector",
            "components": ["vy", "vx"],
            "columns": ["speed_y", "speed_x"],
            "units": ["m/s", "m/s"],
            "coordinates": {"x": "x", "y": "y"},
        },
        "geometry": {"kind": "rectangle", "axes": {"x": {"values": x}, "y": {"values": y}}},
        "masks": {"observation": "observed.npy"},
    }
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest, physical


@pytest.mark.parametrize("condition_format", ["table", "npy"])
def test_grid_csv_import_preserves_sample_component_coordinate_and_mask_mapping(tmp_path, condition_format):
    root, target = tmp_path / "source", tmp_path / "output"
    manifest, fields = _grid_source(root)
    if condition_format == "npy":
        np.save(root / "cond.npy", np.asarray([[8, 4], [9, 5]], dtype=np.float32))
        manifest["conditions"] = {"format": "npy", "path": "cond.npy", "columns": ["temperature", "pressure"]}
        (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    before = {path.name: file_hash(path) for path in root.iterdir()}
    result = import_csv_field_dataset(root, target, source_manifest="import.yaml")
    assert result == target / "field_dataset.yaml"
    dataset = load_field_dataset(target)
    assert dataset.sample_ids.tolist() == ["002", "001"]
    assert dataset.components == ("vy", "vx")
    assert dataset.condition_columns == ("temperature", "pressure")
    np.testing.assert_array_equal(dataset.conditions, [[9, 5], [8, 4]])
    np.testing.assert_array_equal(dataset.fields, fields[[1, 0]])
    np.testing.assert_array_equal(np.load(target / "field.npy"), fields)
    assert not dataset.observation_mask[0, 0, 0]
    assert dataset.observation_mask[1].all()
    assert dataset.sample_metadata["experiment"].tolist() == ["b", "a"]
    assert dataset.manifest["fields"]["units"] == ["m/s", "m/s"]
    assert {path.name: file_hash(path) for path in root.iterdir()} == before
    assert "field_csv" not in pd.read_csv(target / "samples.csv")


@pytest.mark.parametrize("problem", ["duplicate", "missing", "outside"])
def test_csv_coordinate_errors_leave_no_partial_dataset(tmp_path, problem):
    root, target = tmp_path / "source", tmp_path / "output"
    _grid_source(root)
    frame = pd.read_csv(root / "first.csv")
    if problem == "duplicate":
        frame.loc[1, ["x", "y"]] = frame.loc[0, ["x", "y"]]
    elif problem == "missing":
        frame = frame.iloc[:-1]
    else:
        frame.loc[0, "x"] = 999
    frame.to_csv(root / "first.csv", index=False)
    with pytest.raises(
        ValueError, match={"duplicate": "Duplicate", "missing": "Missing", "outside": "outside"}[problem]
    ):
        import_csv_field_dataset(root, target, source_manifest="import.yaml")
    assert not target.exists()
    assert not list(tmp_path.glob(".field_import_*"))


def test_csv_import_requires_declared_geometry_and_does_not_overwrite(tmp_path):
    root, target = tmp_path / "source", tmp_path / "output"
    manifest, _ = _grid_source(root)
    del manifest["geometry"]
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry must be a mapping"):
        import_csv_field_dataset(root, target, source_manifest="import.yaml")
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError):
        import_csv_field_dataset(root, target, source_manifest="import.yaml")


def test_unmasked_nonfinite_csv_field_is_rejected_without_partial_destination(tmp_path):
    root, target = tmp_path / "source", tmp_path / "output"
    manifest, _ = _grid_source(root)
    manifest.pop("masks")
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Observed fields must be finite"):
        import_csv_field_dataset(root, target, source_manifest="import.yaml")
    assert not target.exists()
    assert not list(tmp_path.glob(".field_import_*"))


def test_csv_coordinate_rounding_requires_explicit_unambiguous_tolerance(tmp_path):
    root, target = tmp_path / "source", tmp_path / "output"
    manifest, fields = _grid_source(root)
    frame = pd.read_csv(root / "first.csv")
    frame.loc[0, "x"] += 1e-7
    frame.to_csv(root / "first.csv", index=False)
    with pytest.raises(ValueError, match="outside the declared grid"):
        import_csv_field_dataset(root, target, source_manifest="import.yaml")
    manifest["fields"]["coordinate_atol"] = 2e-7
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    import_csv_field_dataset(root, target, source_manifest="import.yaml")
    np.testing.assert_array_equal(np.load(target / "field.npy"), fields)
    manifest["fields"]["coordinate_atol"] = 0.2
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="half the smallest"):
        import_csv_field_dataset(root, tmp_path / "ambiguous", source_manifest="import.yaml")


def test_mesh_csv_import_uses_vertex_indices_and_declared_surface_measure(tmp_path):
    root, target = tmp_path / "source", tmp_path / "output"
    root.mkdir()
    vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    np.savez(root / "surface.npz", vertices=vertices, faces=[[0, 1, 2]])
    _write_csv(root / "samples.csv", ["sample_id", "array_index", "path"], [["007", 0, "field.csv"]])
    _write_csv(root / "field.csv", ["node", "value"], [[2, 3.5], [0, 1.5], [1, 2.5]])
    manifest = {
        "schema": "field_csv_import_v1",
        "samples": {"path": "samples.csv"},
        "fields": {
            "path_column": "path",
            "layout": "NVC",
            "kind": "scalar",
            "components": ["pressure"],
            "columns": ["value"],
            "coordinates": {"vertex_index": "node"},
        },
        "geometry": {"kind": "mesh", "mesh": {"path": "surface.npz"}},
    }
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    import_csv_field_dataset(root, target, source_manifest="import.yaml")
    dataset = load_field_dataset(target)
    assert dataset.fields.shape == (1, 3, 1)
    assert dataset.conditions is None
    assert dataset.sample_ids.tolist() == ["007"]
    np.testing.assert_array_equal(dataset.fields[0, :, 0], [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(dataset.geometry.coords["vertices"], vertices)
    assert dataset.geometry.integration_weights().sum() == pytest.approx(0.5)
    assert file_hash(root / "surface.npz") == file_hash(target / "surface.npz")


def test_single_csv_can_use_fields_path_without_a_sample_table(tmp_path):
    root, target = tmp_path / "source", tmp_path / "output"
    manifest, fields = _grid_source(root)
    manifest.pop("samples")
    manifest.pop("conditions")
    manifest.pop("masks")
    manifest["fields"].pop("path_column")
    manifest["fields"]["path"] = "first.csv"
    import_csv_field_dataset(root, target, source_manifest=manifest)
    dataset = load_field_dataset(target)
    np.testing.assert_array_equal(dataset.fields, fields[:1])
    assert dataset.conditions is None and dataset.sample_ids.tolist() == ["sample_000000"]
