"""Durable numerical results and portable physical field output acceptance."""

import shutil
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml_platform_core.io import read_json
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import FieldDataset, FieldSpec
from ml_platform_mode_decomp.field_output import read_field_output, write_field_batches
from test_analysis import _input, _settings


def test_preview_selects_an_evaluated_sample_instead_of_an_inactive_nan_row():
    from ml_platform_mode_decomp.analysis_task import _append_report
    from ml_platform_mode_decomp.evaluation import prepare_objective

    data = FieldDataset.from_arrays(
        np.ones((2, 2, 2, 1)), sample_ids=["excluded", "scored"], sample_metadata={"importance": [0, 1]}
    )
    prediction = np.array(data.fields, copy=True)
    prediction[0] = np.nan
    prediction[1] += 2
    report = prepare_objective(data, {"sample_weight_column": "importance"}).score(prediction)
    previews = {}
    _append_report(report, {"candidate_id": "one", "split_seed": 1, "model_seed": 1}, [], [], [], previews, data)
    assert previews["one"]["sample_id"] == "scored"
    np.testing.assert_array_equal(previews["one"]["prediction"], prediction[1])


@pytest.mark.parametrize("failure", ["save", "mode"])
def test_requested_output_failure_keeps_scored_candidate(tmp_path, monkeypatch, failure):
    from ml_platform_mode_decomp import analysis_task

    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "runs", kind="representation")
    cfg["analysis"].update(allow_failures=True, save_models=failure == "save")
    cfg["report"] = {"enabled": False}
    if failure == "save":

        def unavailable(*args, **kwargs):
            raise OSError("model destination unavailable")

        monkeypatch.setattr(analysis_task, "save_model", unavailable)
    else:
        cfg["analysis"]["modes"] = {"enabled": True, "indices": [999999]}
    with pytest.raises(RunError, match="requested analysis outputs") as caught:
        run_task(cfg)
    result = caught.value.result
    comparison = pd.read_csv(result.tables["comparison"])
    assert (comparison[comparison.method != "baseline"].status == "ok").all()
    assert np.isfinite(comparison.field_rmse).all()
    summary = read_json(result.artifacts["analysis_summary"])
    assert not summary["failures"] and len(summary["output_failures"]) == 2
    assert all(item["phase"] == "requested_outputs" for item in summary["output_failures"])


def test_all_numeric_failures_keep_empty_table_headers_and_causes(tmp_path, monkeypatch):
    from ml_platform_mode_decomp import analysis_task

    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "runs", kind="representation")
    cfg["analysis"]["allow_failures"] = True
    cfg["report"] = {"enabled": False}
    monkeypatch.setattr(analysis_task, "_baseline_reports", lambda *args: [])

    def unavailable(*args, **kwargs):
        raise ValueError("no valid candidate support")

    monkeypatch.setattr(analysis_task.FieldRepresentation, "fit", unavailable)
    with pytest.raises(RunError, match="No declared candidate") as caught:
        run_task(cfg)
    result = caught.value.result
    assert pd.read_csv(result.tables["per_sample"]).empty
    assert pd.read_csv(result.tables["paired_comparisons"]).empty
    summary = read_json(result.artifacts["analysis_summary"])
    assert all("no valid candidate support" in item["error"] for item in summary["failures"])


def test_disabled_report_prepares_no_preview_and_enabled_preview_owns_small_arrays(tmp_path, monkeypatch):
    from ml_platform_mode_decomp import analysis_reporting
    from ml_platform_mode_decomp.evaluation import score_fields

    root = tmp_path / "input"
    _input(root, conditions=False)
    cfg = _settings(root, tmp_path / "runs", kind="representation")
    cfg["report"] = {"enabled": False}
    prepare = analysis_reporting.prepare_preview

    def unexpected(*args):
        raise AssertionError("disabled HTML must not prepare previews")

    monkeypatch.setattr(analysis_reporting, "prepare_preview", unexpected)
    run_task(cfg)
    fields = np.ones((3, 129, 133, 4))
    data = FieldDataset.from_arrays(fields)
    report = score_fields(data, fields.copy())
    preview = prepare(data, report, 0)
    assert preview["prediction"].shape[0] <= 33 and preview["prediction"].shape[-1] == 3
    assert not np.shares_memory(preview["prediction"], report.predictions)
    assert not np.shares_memory(preview["reference"], fields)


@pytest.mark.parametrize("suffix", ["npz", "npy"])
@pytest.mark.parametrize("domain", ["rectangle", "sphere_grid", "mesh"])
def test_physical_field_output_moves_without_model_or_input_and_preserves_existing_files(tmp_path, suffix, domain):
    from ml_platform_mode_decomp.geometry import build_domain_spec

    if domain == "rectangle":
        data = FieldDataset.from_arrays(
            np.ones((2, 4, 5, 2)),
            x=np.linspace(0, 2, 5),
            y=np.linspace(-1, 1, 4),
            components=["u", "v"],
            field_kind="vector",
            field_units=["m/s", "m/s"],
        )
    elif domain == "sphere_grid":
        geometry = build_domain_spec(
            {
                "name": domain,
                "angle_unit": "deg",
                "radius": 2,
                "axes": {"lon": {"values": [0, 90, 180, 270]}, "lat": {"values": [-45, 0, 45]}},
            },
            (3, 4, 2),
        )
        data = FieldDataset.from_arrays(
            np.ones((2, 3, 4, 2)),
            geometry=geometry,
            components=["u", "v"],
            field_kind="vector",
            field_units=["m/s", "m/s"],
        )
    else:
        geometry = build_domain_spec(
            {"name": "mesh", "vertices": [[0, 0], [1, 0], [0, 1]], "faces": [[0, 1, 2]]}, (3, 1, 2)
        )
        data = FieldDataset.from_arrays(
            np.ones((2, 3, 2)),
            layout="NVC",
            geometry=geometry,
            components=["u", "v"],
            field_kind="vector",
            field_units=["m/s", "m/s"],
        )
    observed = np.ones(data.fields.shape[:-1], dtype=bool)
    observed[0].reshape(-1)[0] = False
    support = np.ones(data.fields.shape[1:-1], dtype=bool)
    directory = tmp_path / "original"
    paths = write_field_batches(
        directory / f"fields.{suffix}",
        data.sample_ids,
        [data.fields],
        FieldSpec.from_dataset(data),
        data.geometry,
        observation_mask=observed,
        calibration_support=support,
    )
    moved = tmp_path / "moved"
    shutil.move(directory, moved)
    output = read_field_output(moved / paths["predictions"].name)
    assert output["metadata"]["schema"] == "field_output_v1"
    assert output["metadata"]["field_units"] == ["m/s", "m/s"]
    for name, coordinates in data.geometry.coords.items():
        np.testing.assert_array_equal(output["arrays"]["geometry_" + name], coordinates)
    np.testing.assert_array_equal(output["arrays"]["observation_mask"], observed)
    np.testing.assert_array_equal(output["arrays"]["calibration_support"], support)
    before = (moved / paths["predictions"].name).read_bytes()
    with pytest.raises(FileExistsError):
        write_field_batches(
            moved / paths["predictions"].name, data.sample_ids, [data.fields], FieldSpec.from_dataset(data)
        )
    assert (moved / paths["predictions"].name).read_bytes() == before


@pytest.mark.parametrize("layout", ["current", "manifest_collision", "legacy"])
def test_canonical_stage_snapshot_keeps_only_referenced_files_and_survives_raw_removal(tmp_path, layout):
    from ml_platform_mode_decomp.data import load_field_dataset
    from ml_platform_mode_decomp.input_preparation import snapshot_dataset, stage_dataset_config

    root = tmp_path / "input"
    _input(root)
    data = {"local_path": str(root)}
    if layout == "manifest_collision":
        import yaml

        manifest = root / "field_dataset.yaml"
        (root / "stage_input.json").write_text(json.dumps(yaml.safe_load(manifest.read_text())), encoding="utf-8")
        manifest.unlink()
        data["source_manifest"] = "stage_input.json"
    (root / "unrelated.txt").write_text("not an input")
    original = load_field_dataset(data)
    revision_before, ids_before = original.revision, original.sample_ids.copy()
    if layout == "legacy":
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        for name in ("field_dataset.yaml", "field.npy", "samples.csv"):
            shutil.copyfile(root / name, bundle / name)
        (bundle / "stage_input.json").write_text(
            json.dumps({"schema": "canonical_stage_input_v1", "data": {}, "dataset_revision": revision_before}),
            encoding="utf-8",
        )
    else:
        bundle = snapshot_dataset(data, tmp_path / "bundle")
    del original
    gc.collect()
    shutil.rmtree(root)
    settings, revision = stage_dataset_config(bundle)
    restored = load_field_dataset(settings)
    assert revision == restored.revision == revision_before
    assert not list(bundle.rglob("unrelated.txt"))
    np.testing.assert_array_equal(restored.sample_ids, ids_before)


def test_raw_scattered_mask_and_mode_curve_use_the_selected_reference(tmp_path):
    from test_scattered import _csv, _write_source

    source = tmp_path / "source"
    _, points, _ = _write_source(source)
    np.save(
        source / "values.npy", (points[:, 0] ** 2 + points[:, 1] ** 2)[None, :, None] + np.arange(12)[:, None, None]
    )
    _csv(source / "samples.csv", ["sample_id", "array_index", "temperature"], [[f"s{i}", i, i] for i in range(12)])
    result = run_task(
        {
            "task": "mode_decomp_representation",
            "runtime": {"output_dir": str(tmp_path / "runs")},
            "data": {"local_path": str(source), "format": "scattered", "source_manifest": "import.yaml"},
            "metrics": {"reference": "observations"},
            "representation": {"decomposition": {"name": "dct2"}},
            "analysis": {"modes": {"enabled": True, "indices": [0, 1]}},
        }
    )
    curve = pd.read_csv(result.tables["mode_reconstruction"])
    assert "field_rmse" not in curve
    assert curve.iloc[-1].observation_rmse == pytest.approx(result.metrics["observation_rmse"])
    # Raw imputation masks are resolved against the source root and copied without reordering.
    from ml_platform_mode_decomp.input_preparation import prepare_field_input
    from ml_platform_mode_decomp.policy import resolve_config
    from ml_platform_core.result import RunResult

    mask = np.ones((12, 5, 5), dtype=bool)
    mask[:, 1, 1] = False
    np.save(source / "sensors.npy", mask)
    cfg = resolve_config(
        {
            "task": "mode_decomp_analysis",
            "data": {"local_path": str(source), "format": "scattered", "source_manifest": "import.yaml"},
            "analysis": {"kind": "imputation", "observation_mask": "sensors.npy"},
        }
    )
    run_dir = tmp_path / "mask_run"
    run_dir.mkdir()
    prepared = prepare_field_input(cfg, RunResult(run_dir))
    copied = np.load(Path(prepared["data"]["local_path"]) / prepared["analysis"]["observation_mask"])
    np.testing.assert_array_equal(copied, mask)
    cfg["runtime"]["output_dir"] = str(tmp_path / "imputation")
    cfg["model"]["candidates"] = [
        {"id": "gappy", "representation": {"decomposition": {"name": "gappy_graph_fourier", "params": {"n_modes": 3}}}}
    ]
    result = run_task(cfg)
    assert np.isfinite(result.metrics["successful_evaluations"])


def test_reader_preserves_legacy_npy_fields_without_inventing_geometry(tmp_path):
    fields = np.arange(24.0).reshape(2, 3, 4, 1)
    np.save(tmp_path / "old.npy", fields)
    np.save(tmp_path / "old_sample_ids.npy", np.asarray(["one", "two"]))
    metadata = {
        "schema": "field_predictions_v1",
        "sample_count": 2,
        "field_shape": [3, 4, 1],
        "field_kind": "scalar",
        "components": ["temperature"],
        "field_units": ["K"],
        "arrays": {"predictions": "old.npy", "sample_ids": "old_sample_ids.npy"},
        "field_uncertainty": None,
    }
    (tmp_path / "old_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    restored = read_field_output(tmp_path / "old.npy")
    assert restored["metadata"]["geometry"] is None
    assert isinstance(restored["arrays"]["predictions"], np.memmap)
    np.testing.assert_array_equal(restored["arrays"]["predictions"], fields)
    np.testing.assert_array_equal(restored["arrays"]["sample_ids"], ["one", "two"])


@pytest.mark.parametrize("suffix", ["npz", "npy"])
@pytest.mark.parametrize("mismatch", ["shape", "ids", "bands"])
def test_reader_rejects_arrays_that_no_longer_match_the_saved_declarations(tmp_path, suffix, mismatch):
    data = FieldDataset.from_arrays(np.arange(24.0).reshape(2, 3, 4, 1))
    path = tmp_path / f"fields.{suffix}"
    write_field_batches(path, data.sample_ids, [data.fields], FieldSpec.from_dataset(data))
    if suffix == "npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if mismatch == "shape":
            arrays["predictions"] = arrays["predictions"][:1]
        elif mismatch == "ids":
            arrays["sample_ids"] = arrays["sample_ids"][:1]
        else:
            arrays["lower"] = arrays["predictions"]
        np.savez_compressed(path, **arrays)
    elif mismatch == "shape":
        np.save(path, data.fields[:1])
    elif mismatch == "ids":
        np.save(tmp_path / "fields_sample_ids.npy", data.sample_ids[:1])
    else:
        metadata_path = tmp_path / "fields_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["arrays"]["lower"] = "fields.npy"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="shape|sample IDs|both lower"):
        read_field_output(path)


def test_reader_rejects_an_unrelated_json_instead_of_returning_empty_fields(tmp_path):
    path = tmp_path / "model_info.json"
    path.write_text('{"model_artifact_schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="associated array files"):
        read_field_output(path)


def test_writer_rejects_support_for_a_different_field_before_consuming_batches(tmp_path):
    data = FieldDataset.from_arrays(np.ones((2, 3, 4, 1)))

    def unexpected():
        pytest.fail("Mismatched support must not consume expensive field predictions")
        yield data.fields

    path = tmp_path / "fields.npy"
    with pytest.raises(ValueError, match="sample-by-spatial"):
        write_field_batches(
            path,
            data.sample_ids,
            unexpected(),
            FieldSpec.from_dataset(data),
            observation_mask=np.ones((3, 3, 4), dtype=bool),
        )
    assert not path.exists()


def test_linear_baseline_retains_missing_value_ols_across_spatial_blocks():
    from ml_platform_mode_decomp.analysis_task import _linear_baseline

    conditions = np.random.default_rng(25).normal(size=(8, 2))
    query = np.asarray([[0.1, 0.2], [-1.0, 0.5], [2.0, -0.2]])
    x = np.linspace(-1.0, 1.0, 40000).reshape(2, 20000)
    basis = np.stack((np.sin(x), np.cos(x)))[..., None] * np.asarray([1.0, 2.0])
    fields = 3 + np.einsum("np,phwc->nhwc", conditions, basis)
    expected = 3 + np.einsum("np,phwc->nhwc", query, basis)
    observed = np.ones(fields.shape[:-1], dtype=bool)
    observed[:2, :, ::5] = False
    observed[2:4, :, ::7] = False
    observed[:, :, 0] = False
    fields[~observed] = np.nan
    training = FieldDataset.from_arrays(fields, conditions, observation_mask=observed)
    reference = FieldDataset.from_arrays(expected, query)
    actual, _ = _linear_baseline(training, reference)
    expected[:, :, 0] = np.nan
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12, equal_nan=True)
