"""One run owns raw import, shared training inputs and the saved inference boundary."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.artifacts import file_hash
from ml_platform_core.io import read_json

from ml_platform_mode_decomp import load_config, run_task
from ml_platform_mode_decomp import input_preparation
from ml_platform_mode_decomp.data import (
    import_csv_field_dataset,
    import_legacy_npy_dataset,
    import_scattered_field_dataset,
    load_field_dataset,
)
from ml_platform_mode_decomp.data import importing as legacy_importing
from ml_platform_mode_decomp.model_artifact import load_model

from test_csv_import import _grid_source, _write_csv
from test_scattered import _csv, _write_source
from test_search import payloads, write_input
from test_uncertainty_workflow import calibrated_settings


def hashes(root):
    return {path.relative_to(root): file_hash(path) for path in root.rglob("*") if path.is_file()}


def inline(format_name, root, manifest):
    return {
        "format": format_name,
        "local_path": str(root),
        **{key: value for key, value in manifest.items() if key != "schema"},
    }


def test_one_yaml_imports_raw_scattered_data_and_optional_point_scoring_preserves_predictions(tmp_path):
    source = tmp_path / "raw"
    manifest, points, _ = _write_source(source)
    points = 0.1 + 0.8 * points
    coefficients = np.linspace(-1, 1, 24)
    values = (points[:, 0] ** 2 + points[:, 1] ** 2)[None, :, None] * coefficients[:, None, None]
    values += coefficients[:, None, None] ** 2
    np.save(source / "points.npy", points)
    np.save(source / "values.npy", values)
    _csv(
        source / "samples.csv",
        ["sample_id", "array_index", "temperature"],
        [(f"case_{index:03d}", index, value) for index, value in enumerate(coefficients)],
    )
    before = hashes(source)
    cfg = {
        "task": "mode_decomp_analysis",
        "data": inline("scattered", source, manifest),
        "run": {"seed": 11},
        "runtime": {"output_dir": str(tmp_path / "on")},
        "split": {"method": "random", "valid_size": 0.25},
        "analysis": {
            "kind": "surrogate",
            "split_seeds": [7],
            "model_seeds": [13],
            "bootstrap_samples": 0,
            "scattered": {"enabled": True},
        },
        "model": {
            "candidates": [
                {"id": "pod2", "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}}}
            ]
        },
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    enabled = run_task(load_config(path))
    off = deepcopy(cfg)
    off["runtime"]["output_dir"] = str(tmp_path / "off")
    off["analysis"]["scattered"]["enabled"] = False
    disabled = run_task(off)
    on_table, off_table = (
        pd.read_csv(result.tables["comparison"]).set_index("candidate_id") for result in (enabled, disabled)
    )
    np.testing.assert_array_equal(on_table.field_rmse, off_table.field_rmse)
    np.testing.assert_array_equal(on_table.representation_rmse, off_table.representation_rmse)
    np.testing.assert_array_equal(on_table.regression_rmse, off_table.regression_rmse)
    assert 0 < on_table.loc["pod2", "observation_coverage"] < 1
    assert on_table.loc["pod2", "observation_rmse_ch0"] > 0
    assert on_table.loc["pod2", "interpolation_rmse_ch0"] > 0
    assert "observation_rmse_ch0" not in off_table
    manifest_path = enabled.artifacts["input_dataset_manifest"]
    assert manifest_path.is_file()
    stored = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert (manifest_path.parent / stored["scattered"]["path"]).is_file()
    assert hashes(source) == before


def test_raw_csv_is_converted_once_before_cv_and_calibration_and_the_saved_model_predicts(tmp_path, monkeypatch):
    source, output = tmp_path / "raw_csv", tmp_path / "runs"
    write_input(source, count=48)
    manifest = yaml.safe_load((source / "field_dataset.yaml").read_text(encoding="utf-8"))
    fields = np.load(source / "field.npy")
    samples = pd.read_csv(source / "samples.csv")
    samples["field_csv"] = [f"field_{index:03d}.csv" for index in range(len(samples))]
    samples.to_csv(source / "samples.csv", index=False)
    axis = np.linspace(-1, 1, 3)
    for index, filename in enumerate(samples.field_csv):
        _write_csv(
            source / filename,
            ["x", "y", "value"],
            [(x, y, fields[index, row, column, 0]) for row, y in enumerate(axis) for column, x in enumerate(axis)],
        )
    manifest["fields"] = {
        "path_column": "field_csv",
        "layout": "NHWC",
        "kind": "scalar",
        "components": ["signal"],
        "columns": ["value"],
        "coordinates": {"x": "x", "y": "y"},
    }
    before = hashes(source)
    cfg = calibrated_settings(source, output)
    cfg["data"] = inline("csv", source, manifest)
    conversions = []
    convert = input_preparation.convert_field_input

    def tracked_conversion(data, destination, **kwargs):
        conversions.append(Path(destination))
        return convert(data, destination, **kwargs)

    monkeypatch.setattr(input_preparation, "convert_field_input", tracked_conversion)
    result = run_task(cfg)
    assert len(conversions) == 1
    manifest_path = result.artifacts["input_dataset_manifest"]
    assert manifest_path.parent == conversions[0]
    core_manifest = read_json(result.artifacts["manifest"])
    assert Path(core_manifest["artifacts"]["input_dataset_manifest"]["path"]) == manifest_path
    dataset = load_field_dataset(manifest_path.parent)
    split = read_json(next(output.rglob("split.json")))
    proper, calibration, test = (set(split[key]) for key in ("train_ids", "calibration_ids", "test_ids"))
    assert proper.isdisjoint(calibration | test) and calibration.isdisjoint(test)
    assert len(proper | calibration | test) == len(fields)
    model = load_model(result.artifacts["model"])
    assert set(model.fit_sample_ids) == proper and set(model.calibration_sample_ids) == calibration
    for candidate in payloads(output).values():
        assert candidate["dataset_revision"] == dataset.revision
        assert set(candidate["fit_ids"]) <= proper
    for path in output.rglob("cv_folds.json"):
        for fold in read_json(path)["folds"]:
            assert set(fold["fit_ids"] + fold["validation_ids"]) == proper
    for path in output.rglob("config.yaml"):
        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert Path(saved["data"]["local_path"]) == manifest_path.parent
    assert hashes(source) == before
    conditions = tmp_path / "new.csv"
    conditions.write_text("sample_id,b,a\nnew,0.4,0.2\n", encoding="utf-8")
    expected = model.predict_interval(np.asarray([[0.2, 0.4]]))
    source.rename(tmp_path / "hidden_raw_csv")
    predicted = run_task(
        {
            "task": "mode_decomp_infer",
            "data": {"local_path": str(conditions)},
            "runtime": {"output_dir": str(tmp_path / "inference")},
            "model": {"artifact_path": str(result.artifacts["model"])},
            "uncertainty": {"enabled": True},
        }
    )
    assert len(conversions) == 1
    with np.load(predicted.artifacts["predictions"], allow_pickle=False) as saved:
        for key, values in zip(("predictions", "lower", "upper"), expected, strict=True):
            np.testing.assert_array_equal(saved[key], values)


def test_canonical_npy_executes_without_import_or_source_mutation(tmp_path, monkeypatch):
    source = tmp_path / "canonical"
    write_input(source)
    before = hashes(source)

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("Canonical NPY data must not enter the import path")

    monkeypatch.setattr(input_preparation, "convert_field_input", unexpected_conversion)
    result = run_task(
        {
            "task": "mode_decomp_representation",
            "data": {"format": "npy", "local_path": str(source)},
            "runtime": {"output_dir": str(tmp_path / "output")},
            "representation": {"decomposition": {"name": "fft2"}},
        }
    )
    assert result.artifacts["model"].is_file() and "input_dataset_manifest" not in result.artifacts
    assert hashes(source) == before


@pytest.mark.parametrize("format_name", ["csv", "scattered", "legacy-npy"])
def test_child_output_import_preserves_sources_and_referenced_axes(tmp_path, monkeypatch, format_name):
    source = tmp_path / "raw"
    if format_name == "csv":
        manifest, _ = _grid_source(source)
        importer = import_csv_field_dataset
        kwargs = {"source_manifest": "import.yaml"}
        x = manifest["geometry"]["axes"]["x"]["values"]
    elif format_name == "scattered":
        manifest, _, _ = _write_source(source)
        importer = import_scattered_field_dataset
        kwargs = {"source_manifest": "import.yaml"}
        x = np.linspace(0, 1, 5)
    else:
        source.mkdir()
        np.save(source / "cond.npy", np.asarray([[0.0], [1.0]]))
        np.save(source / "field.npy", np.arange(24, dtype=float).reshape(2, 3, 4, 1))
        manifest = {
            "field_kind": "scalar",
            "grid": {"H": 3, "W": 4, "x_range": [0, 3], "y_range": [0, 2]},
            "domain": {"type": "rectangle"},
            "geometry": {"axes": {}},
        }
        importer = import_legacy_npy_dataset
        kwargs = {}
        x = np.arange(4, dtype=float)
    np.save(source / "axis.npy", np.asarray(x))
    manifest["geometry"]["axes"]["x"] = {"path": "axis.npy"}
    if format_name == "legacy-npy":
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    before = hashes(source)
    destination = source / "outputs" / "converted"
    if format_name == "legacy-npy":
        copy_file = legacy_importing.shutil.copyfile

        def interrupted_copy(original, target):
            if Path(original).name == "field.npy":
                Path(target).write_bytes(b"interrupted field copy")
                raise OSError("simulated copy interruption")
            return copy_file(original, target)

        with monkeypatch.context() as patch:
            patch.setattr(legacy_importing.shutil, "copyfile", interrupted_copy)
            with pytest.raises(OSError, match="simulated copy interruption"):
                importer(source, destination, **kwargs)
        assert not destination.exists() and not list(destination.parent.glob(".legacy_import_*"))
    result = importer(source, destination, **kwargs)
    dataset = load_field_dataset(result)
    assert len(dataset) == 2
    np.testing.assert_array_equal(dataset.geometry.coords["x"][0], x)
    assert file_hash(source / "axis.npy") == file_hash(destination / "axis.npy")
    assert all(file_hash(source / name) == digest for name, digest in before.items())
    output_before = hashes(destination)
    with pytest.raises(FileExistsError):
        importer(source, destination, **kwargs)
    with pytest.raises(ValueError, match="separate from the source"):
        importer(source, source, **kwargs)
    assert hashes(destination) == output_before
    assert all(file_hash(source / name) == digest for name, digest in before.items())


def test_cli_raw_csv_from_current_directory_uses_default_outputs_without_modifying_inputs(tmp_path):
    source = tmp_path / "working"
    source.mkdir()
    _write_csv(source / "samples.csv", ["sample_id", "array_index", "field_csv"], [["single", 0, "field.csv"]])
    _write_csv(source / "field.csv", ["x", "y", "value"], [[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]])
    np.save(source / "axis.npy", np.asarray([0.0, 1.0]))
    cfg = {
        "task": "mode_decomp_representation",
        "data": {
            "format": "csv",
            "local_path": ".",
            "samples": {"path": "samples.csv"},
            "fields": {
                "path_column": "field_csv",
                "layout": "NHWC",
                "kind": "scalar",
                "components": ["value"],
                "columns": ["value"],
                "coordinates": {"x": "x", "y": "y"},
            },
            "geometry": {
                "kind": "rectangle",
                "axes": {"x": {"path": "axis.npy"}, "y": {"start": 0, "step": 1, "count": 2}},
            },
        },
        "representation": {"decomposition": {"name": "dct2"}},
    }
    (source / "run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    before = hashes(source)
    completed = subprocess.run(
        [sys.executable, "-m", "ml_platform_mode_decomp", "run.yaml"],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    converted = Path(result["artifacts"]["input_dataset_manifest"])
    assert converted.is_relative_to(source / "outputs")
    assert result["metrics"]["field_rmse"] < 1e-12
    assert file_hash(converted.parent / "axis.npy") == file_hash(source / "axis.npy")
    assert Path(result["artifacts"]["model"]).is_file()
    assert all(file_hash(source / name) == digest for name, digest in before.items())
