from __future__ import annotations

import csv
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.artifacts import file_hash

from ml_platform_mode_decomp.data import (
    FieldSample,
    import_legacy_npy_dataset,
    load_condition_batch,
    load_field_dataset,
    split_dataset,
)
from ml_platform_mode_decomp.geometry import build_domain_spec, validate_decomposer_compatibility
from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint


def write_dataset(
    root: Path, *, count: int = 12, order=None, condition_format="table", manifest_name="field_dataset.yaml"
):
    root.mkdir(parents=True)
    fields = np.arange(count * 16, dtype=np.float32).reshape(count, 4, 4, 1)
    np.save(root / "field.npy", fields)
    np.save(root / "cond.npy", np.column_stack((np.arange(count), np.arange(count) + 0.5)).astype(np.float32))
    np.save(root / "observation.npy", np.ones((4, 4), dtype=np.uint8))
    order = list(range(count)) if order is None else order
    with (root / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "a", "b", "experiment_id", "measured_at", "is_test"])
        for index in order:
            writer.writerow(
                [
                    f"{index:03d}",
                    index,
                    index,
                    index + 0.5,
                    f"group_{index // 2}",
                    f"2026-01-{index // 2 + 1:02d}T00:00:00Z",
                    int(index >= count - 2),
                ]
            )
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "conditions": {"format": condition_format, "columns": ["a", "b"]},
        "fields": {"path": "field.npy", "layout": "NHWC", "kind": "scalar", "components": ["signal"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0.0, "step": 0.25, "count": 4, "periodic": True},
                "y": {"start": 0.0, "step": 0.25, "count": 4, "periodic": True},
            },
        },
        "masks": {"observation": "observation.npy"},
    }
    if condition_format == "npy":
        manifest["conditions"]["path"] = "cond.npy"
    target = root / manifest_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest, fields


def test_root_relative_manifest_and_array_index_preserve_pairs(tmp_path):
    order = [3, 0, 2, 1]
    _, fields = write_dataset(
        tmp_path / "input", count=4, order=order, condition_format="npy", manifest_name="nested/input.yaml"
    )
    dataset = load_field_dataset({"data": {"local_path": tmp_path / "input", "source_manifest": "nested/input.yaml"}})
    np.testing.assert_array_equal(dataset.fields, fields[order])
    np.testing.assert_array_equal(dataset.conditions[:, 0], order)
    assert dataset.sample_ids.tolist() == ["003", "000", "002", "001"]
    assert dataset.condition_columns == ("a", "b")
    assert dataset.observation_mask.shape == (4, 4)
    assert isinstance(dataset[0], FieldSample)
    assert dataset[0].sample_id == "003"
    np.testing.assert_array_equal(dataset[0].field, fields[3])


@pytest.mark.parametrize("problem", ["vertices_nan", "faces_fractional", "degenerate_face"])
def test_mesh_schema_rejects_invalid_geometry(problem):
    config = {"kind": "mesh", "vertices": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "faces": [[0, 1, 2]]}
    if problem == "vertices_nan":
        config["vertices"][0][0] = np.nan
    elif problem == "faces_fractional":
        config["faces"][0][1] = 1.5
    else:
        config["faces"] = [[0, 1, 1]]
    with pytest.raises(ValueError, match="finite|positive area"):
        build_domain_spec(config, (3, 1, 1))


@pytest.mark.parametrize("problem", ["duplicate_id", "duplicate_index", "index_missing", "fractional_index"])
def test_bad_sample_correspondence_rejected(tmp_path, problem):
    root = tmp_path / "input"
    write_dataset(root, count=4)
    rows = list(csv.reader((root / "samples.csv").open(encoding="utf-8")))
    if problem == "duplicate_id":
        rows[2][0] = rows[1][0]
    elif problem == "duplicate_index":
        rows[2][1] = rows[1][1]
    elif problem == "fractional_index":
        rows[1][1] = "0.5"
    else:
        rows.pop()
    with (root / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(ValueError):
        load_field_dataset(root)


def test_domain_and_observation_masks_do_not_change_field_values(tmp_path):
    root = tmp_path / "input"
    manifest, fields = write_dataset(root, count=4)
    mask = np.ones((4, 4), dtype=np.uint8)
    mask[0, 0] = 0
    np.save(root / "observation.npy", mask)
    fields[:, 0, 0] = np.nan
    np.save(root / "field.npy", fields)
    dataset = load_field_dataset(root)
    assert np.isnan(dataset.fields[:, 0, 0]).all()
    assert not dataset.masks[:, 0, 0].any()
    assert dataset.geometry.mask is None
    mask[0, 0] = 1
    np.save(root / "observation.npy", mask)
    with pytest.raises(ValueError, match="finite"):
        load_field_dataset(root)


def test_conditions_optional_for_representation(tmp_path):
    root = tmp_path / "input"
    manifest, _ = write_dataset(root, count=4)
    manifest.pop("conditions")
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    dataset = load_field_dataset(root)
    assert dataset.conditions is None and dataset[0].cond is None


@pytest.mark.parametrize("method", ["group", "time"])
def test_selection_cannot_weaken_outer_group_or_time_separation(tmp_path, method):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    config = {
        "method": method,
        "selection_method": "random",
        "group_column": "experiment_id",
        "time_column": "measured_at",
    }
    with pytest.raises(ValueError, match="Selection must preserve"):
        split_dataset(dataset, config, nested=True)


def test_path_escape_and_ambiguous_layout_rejected(tmp_path):
    root = tmp_path / "input"
    manifest, _ = write_dataset(root, count=4)
    manifest["fields"]["path"] = "../field.npy"
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="within"):
        load_field_dataset(root)
    manifest["fields"].update(path="field.npy", layout="NHW")
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="layout"):
        load_field_dataset(root)


def test_condition_only_csv_uses_trained_order_and_keeps_ids(tmp_path):
    path = tmp_path / "conditions.csv"
    path.write_text("sample_id,b,a,note\n001,2,1,hello\n002,4,3,world\n", encoding="utf-8")
    batch = load_condition_batch(path, ["a", "b"])
    np.testing.assert_array_equal(batch.conditions, [[1, 2], [3, 4]])
    assert batch.sample_ids.tolist() == ["001", "002"]
    assert batch.metadata["note"].tolist() == ["hello", "world"]
    path.write_text("a,a\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_condition_batch(path, ["a"])


def test_condition_only_npy_and_nonfinite_csv(tmp_path):
    path = tmp_path / "conditions.npy"
    np.save(path, [[1.0, 2.0]])
    batch = load_condition_batch(path, ["a", "b"])
    assert batch.conditions.shape == (1, 2)
    (tmp_path / "conditions.csv").write_text("a,b\n1,inf\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_condition_batch(tmp_path / "conditions.csv", ["a", "b"])


def test_direct_npy_declared_columns_must_match_model_order(tmp_path):
    path = tmp_path / "conditions.npy"
    np.save(path, [[1.0, 2.0]])
    config = {"local_path": path, "condition_columns": ["a", "b"]}
    batch = load_condition_batch(config, ["a", "b"])
    np.testing.assert_array_equal(batch.conditions, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="trained model column order"):
        load_condition_batch({**config, "condition_columns": ["b", "a"]}, ["a", "b"])


def test_condition_manifest_pairs_npy_rows_and_checks_model_columns(tmp_path):
    root = tmp_path / "input"
    write_dataset(root, count=3, order=[2, 0, 1])
    manifest = {
        "schema": "condition_batch_v1",
        "samples": {"path": "samples.csv"},
        "conditions": {"format": "npy", "path": "cond.npy", "columns": ["a", "b"]},
    }
    path = root / "conditions.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    config = {"local_path": root, "source_manifest": "conditions.yaml"}
    batch = load_condition_batch(config, ["a", "b"])
    np.testing.assert_array_equal(batch.conditions[:, 0], [2, 0, 1])
    assert batch.sample_ids.tolist() == ["002", "000", "001"]
    with pytest.raises(ValueError, match="trained model schema"):
        load_condition_batch(config, ["b", "a"])


def test_random_split_is_deterministic_and_covers_samples(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    first = split_dataset(dataset, {}, seed=11)
    assert first == split_dataset(dataset, {}, seed=11)
    fit, selection, test = (set(first[f"{name}_indices"]) for name in ("fit", "selection", "test"))
    assert not fit & selection and not fit & test and not selection & test
    assert fit | selection | test == set(range(len(dataset)))
    assert first["fingerprint"] != split_dataset(dataset, {}, seed=12)["fingerprint"]


def test_group_split_separates_outer_and_inner_groups(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    split = split_dataset(dataset, {"method": "group", "group_column": "experiment_id"}, 42)
    groups = dataset.sample_metadata["experiment_id"]
    parts = [set(groups[split[f"{name}_indices"]]) for name in ("fit", "selection", "test")]
    assert not parts[0] & parts[1] and not parts[0] & parts[2] and not parts[1] & parts[2]


def test_group_metadata_used_only_by_calibration_is_preserved(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    split = split_dataset(
        dataset,
        {"method": "random", "group_column": "experiment_id"},
        nested=False,
        calibration={
            "enabled": True,
            "calibration_method": "group",
            "calibration_size": 0.3,
            "calibration_seed": 13,
        },
    )
    groups = dataset.sample_metadata["experiment_id"]
    assert set(groups[split["train_indices"]]).isdisjoint(groups[split["calibration_indices"]])


@pytest.mark.parametrize("missing", [np.nan, pd.NaT, pd.NA])
def test_group_metadata_rejects_missing_scalars(tmp_path, missing):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    values = np.array(["one"] * len(dataset), dtype=object)
    values[2] = missing
    dataset = replace(dataset, sample_metadata={"group": values})
    with pytest.raises(ValueError, match="missing"):
        split_dataset(dataset, {"method": "group", "group_column": "group"})


def test_explicit_split_seed_is_independent_of_component_seed(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    first = split_dataset(dataset, {"seed": 19}, seed=3)
    assert first == split_dataset(dataset, {"seed": 19}, seed=99)
    assert first["seed"] == 19
    assert first["fingerprint"] != split_dataset(dataset, {"seed": 20}, seed=3)["fingerprint"]


@pytest.mark.parametrize("method", ["random", "group", "time", "fixed"])
def test_outer_only_split_accepts_two_samples_without_inner_selection(tmp_path, method):
    root = tmp_path / "input"
    write_dataset(root, count=2)
    dataset = load_field_dataset(root)
    dataset = replace(
        dataset,
        sample_metadata={
            "group": np.array(["a", "b"]),
            "time": np.array(["2026-01-01", "2026-01-02"]),
            "is_test": np.array([0, 1]),
        },
    )
    config = {"method": method}
    if method in {"group", "time"}:
        config[f"{method}_column"] = method
    if method == "fixed":
        config.update(valid_filter_column="is_test", valid_filter_value=1)
    split = split_dataset(dataset, config, nested=False)
    assert len(split["train_indices"]) == len(split["test_indices"]) == 1
    assert not set(split["train_ids"]) & set(split["test_ids"])
    assert split["nested"] is False
    assert not {"selection_method", "fit_indices", "selection_indices"} & set(split)
    with pytest.raises(ValueError, match="not used without inner holdout"):
        split_dataset(dataset, {**config, "selection_size": 0.5}, nested=False)


def test_time_split_orders_equal_timestamps_and_rejects_insufficient_times(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    config = {"method": "time", "time_column": "measured_at"}
    split = split_dataset(dataset, config)
    values = dataset.sample_metadata["measured_at"]
    assert max(values[split["fit_indices"]]) < min(values[split["selection_indices"]])
    assert max(values[split["selection_indices"]]) < min(values[split["test_indices"]])
    single_time = replace(dataset, sample_metadata={"measured_at": np.repeat("2026-01-01", len(dataset))})
    with pytest.raises(ValueError, match="distinct timestamps"):
        split_dataset(single_time, config)


def test_fixed_outer_requires_explicit_selection_method(tmp_path):
    root = tmp_path / "input"
    write_dataset(root)
    dataset = load_field_dataset(root)
    config = {"method": "fixed", "valid_filter_column": "is_test", "valid_filter_value": 1}
    with pytest.raises(ValueError, match="selection_method"):
        split_dataset(dataset, config)
    with pytest.raises(ValueError, match="valid_size is not used"):
        split_dataset(dataset, {**config, "selection_method": "random", "valid_size": 0.5})
    split = split_dataset(dataset, {**config, "selection_method": "random"})
    assert split["test_indices"] == [10, 11]
    assert split["valid_size"] is None


def test_revision_uses_contents_not_only_sample_ids(tmp_path):
    root = tmp_path / "input"
    _, fields = write_dataset(root, count=4)
    stat = (root / "field.npy").stat()
    first = load_field_dataset(root).revision
    fields[0, 0, 0] += 1
    np.save(root / "field.npy", fields)
    # Copy/synchronization tools can preserve the timestamp of same-sized files.
    os.utime(root / "field.npy", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert load_field_dataset(root).revision != first


def test_cartesian_disk_area_and_explicit_radial_fit_weights():
    config = {"name": "disk", "x_range": [-1, 1], "y_range": [-1, 1], "center": [0, 0], "radius": 1}
    geometry = build_domain_spec(config, (401, 401, 1))
    assert geometry.integration_weights().sum() == pytest.approx(np.pi, abs=0.01)
    assert geometry.weights[200, 200] > 0
    assert geometry.radial_fit_weights()[200, 200] == 0


def test_nonuniform_fft_rejected_and_geometry_semantics_in_identity():
    config = {"kind": "rectangle", "axes": {"x": {"values": [0, 1, 3, 4]}, "y": {"start": 0, "step": 1, "count": 4}}}
    geometry = build_domain_spec(config, (4, 4, 1))
    with pytest.raises(ValueError, match="equally spaced"):
        validate_decomposer_compatibility(geometry, {"name": "fft2"})
    original = geometry_fingerprint(geometry)
    assert geometry_fingerprint(replace(geometry, meta={**geometry.meta, "units": "m"})) != original
    assert (
        geometry_fingerprint(replace(geometry, meta={**geometry.meta, "mask_path": "another/location.npy"})) == original
    )


def test_sphere_area_scales_with_radius_and_duplicate_period_rejected():
    config = {"name": "sphere_grid", "lat_range": [-90, 90], "lon_range": [-180, 170], "radius": 1}
    first = build_domain_spec(config, (19, 36, 1))
    second = build_domain_spec({**config, "radius": 3}, (19, 36, 1))
    np.testing.assert_allclose(second.weights, first.weights * 9)
    config = {
        "kind": "sphere_grid",
        "axes": {
            "lat": {"start": -90, "step": 10, "count": 19},
            "lon": {"start": -180, "step": 10, "count": 37, "periodic": True},
        },
    }
    with pytest.raises(ValueError, match="repeat"):
        build_domain_spec(config, (19, 37, 1))


def test_mesh_keeps_nvc_public_shape_and_legacy_sample_adapter(tmp_path):
    root = tmp_path / "input"
    manifest, _ = write_dataset(root, count=4)
    np.save(root / "field.npy", np.ones((4, 3, 1), dtype=np.float32))
    np.savez(root / "mesh.npz", vertices=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], faces=[[0, 1, 2]])
    manifest["fields"]["layout"] = "NVC"
    manifest["geometry"] = {"kind": "mesh", "mesh": {"path": "mesh.npz"}}
    manifest.pop("masks")
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    dataset = load_field_dataset(root)
    assert dataset.fields.shape == (4, 3, 1)
    assert dataset[0].field.shape == (3, 1, 1)
    assert dataset.geometry.weights.shape == (3,)
    assert dataset.geometry.weights.sum() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "name",
    [
        "scalar_rect",
        "scalar_disk",
        "scalar_mask",
        "scalar_annulus",
        "scalar_sphere",
        "vector_rect",
        "vector_disk",
        "vector_mask",
    ],
)
def test_eight_legacy_fixtures_import_without_altering_values(tmp_path, name):
    source = Path(__file__).resolve().parents[3] / "data" / "mode_decomp_eval_dataset_v1" / name
    if not source.exists():
        pytest.skip("Repository fixture is not part of the installed wheel.")
    destination = tmp_path / name
    import_legacy_npy_dataset(source, destination)
    for filename in ("cond.npy", "field.npy"):
        assert file_hash(destination / filename) == file_hash(source / filename)
    dataset = load_field_dataset(destination)
    assert len(dataset) == 200
    assert dataset.fields.shape[-1] == (2 if name.startswith("vector") else 1)
    assert len(dataset.condition_columns) == dataset.conditions.shape[1]
    assert dataset[0].mask.shape == dataset.geometry.grid_shape
    if name.endswith("rect"):
        np.testing.assert_allclose(dataset.geometry.coords["x"][0], np.arange(64) / 64)
        assert set(dataset.geometry.meta["periodic_axes"]) == {"x", "y"}


def test_local_content_revision_is_not_replaced_by_an_arbitrary_dataset_id(tmp_path):
    root = tmp_path / "input"
    _, fields = write_dataset(root, count=4)
    config = {"local_path": root, "clearml_dataset_id": "same-user-supplied-id"}
    before = load_field_dataset(config).revision
    np.save(root / "field.npy", fields + 1)
    assert load_field_dataset(config).revision != before


def test_mesh_file_and_vertex_mask_paths_use_the_dataset_root(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    np.save(root / "field.npy", np.ones((1, 4, 1)))
    np.save(root / "active.npy", [1, 1, 1, 0])
    np.savez(root / "mesh.npz", vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], faces=[[0, 1, 2], [0, 2, 3]])
    monkeypatch.chdir(tmp_path)
    dataset = load_field_dataset(
        {
            "local_path": root,
            "fields": {"path": "field.npy", "layout": "NVC"},
            "geometry": {"kind": "mesh", "mesh": {"path": "mesh.npz"}, "vertex_mask_path": "active.npy"},
        }
    )
    np.testing.assert_array_equal(dataset.geometry.mask, [True, True, True, False])
    np.testing.assert_array_equal(dataset.geometry.coords["faces"], [[0, 1, 2]])
    assert dataset.geometry.weights.sum() == pytest.approx(0.5)
