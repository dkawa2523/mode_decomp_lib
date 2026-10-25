"""The optional acceptance command preserves held-out cases and physical units."""

import csv
from pathlib import Path
import runpy

import numpy as np
import pytest
import yaml

from ml_platform_mode_decomp import FieldDataset, load_model


ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = runpy.run_path(str(ROOT / "scripts/verify_paired_data.py"))


def test_paired_acceptance_preserves_fixed_holdout_and_reports_capacity(tmp_path):
    write = runpy.run_path(str(ROOT / "examples/create_field_mapping_dataset.py"))["_write_dataset"]
    values = np.random.default_rng(4).uniform(-1, 1, (20, 2))
    ids = [f"case_{index}" for index in range(len(values))]
    for target, name in ((False, "source"), (True, "target")):
        write(tmp_path / name, values, ids, target=target)
    with (tmp_path / "target/samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "partition"])
        writer.writerows((value, index, "test" if index >= 16 else "train") for index, value in enumerate(ids))
    cfg = {
        "task": "mode_decomp_pipeline",
        "data": {
            "local_path": str(tmp_path / "target"),
            "input": {"kind": "field", "local_path": str(tmp_path / "source")},
        },
        "split": {"method": "fixed", "valid_filter_column": "partition", "valid_filter_value": "test"},
        "model": {
            "candidates": [
                {
                    "id": "paired",
                    "input_representation": {"decomposition": {"name": "tucker", "params": {"ranks": [3, 3, 1]}}},
                    "representation": {"decomposition": {"name": "quadratic_manifold", "params": {"n_modes": 2}}},
                    "regression": {"name": "ridge", "params": {"alpha": 1e-6}},
                }
            ]
        },
    }
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    report = ACCEPTANCE["verify"](
        config, tmp_path / "acceptance", data_kind="synthetic", provenance="Unit-test generator"
    )
    assert report["success"] and report["data_kind"] == "synthetic"
    assert report["split"]["test_ids"] == ids[16:]
    assert report["split"]["train_ids"] == ids[:16]
    model = load_model(report["artifacts"]["model"])
    assert model.input_representation.fit_sample_ids == tuple(ids[:16])
    assert model.representation.fit_sample_ids == tuple(ids[:16])
    assert report["target_prediction"]["physical_grid"]["units"] == ["Pa"]
    assert report["source_representation_residual"]["units"] == ["K"]
    assert report["capacity"]["model_file_bytes"] > 0
    assert report["capacity"]["pipeline_process_peak_resident_bytes"] > 0
    assert (tmp_path / "acceptance/report.json").is_file()
    with pytest.raises(ValueError, match="never overwritten"):
        ACCEPTANCE["verify"](config, tmp_path / "acceptance", data_kind="synthetic", provenance="Unit-test generator")


def test_zero_truth_relative_error_is_explicitly_undefined():
    reference = FieldDataset.from_arrays(np.zeros((2, 3, 4, 1)), field_units=["Pa"])
    report = ACCEPTANCE["_physical_report"](reference, np.ones_like(reference.fields), training=reference)
    assert report["relative_l2_by_component"] == [None]
    assert report["metrics"]["channel_0_rmse"] == pytest.approx(1)
    assert report["training_mean_baseline"]["status"] == "zero_error_skill_undefined"
