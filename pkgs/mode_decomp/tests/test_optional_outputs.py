"""Optional YAML outputs change diagnostics, never the fitted field comparison."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.artifacts import file_hash
from ml_platform_core.config import load_run_config

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.data import import_scattered_field_dataset, load_field_dataset, load_scattered_observations

from test_scattered import _write_source


@pytest.fixture
def scattered_input(tmp_path):
    source, normalized = tmp_path / "source", tmp_path / "normalized"
    _, _, values = _write_source(source)
    controls = np.linspace(-1, 1, 12)
    np.save(source / "values.npy", values[:1] + controls[:, None, None])
    pd.DataFrame(
        {"sample_id": [f"sample_{i:02d}" for i in range(12)], "array_index": np.arange(12), "temperature": controls}
    ).to_csv(source / "samples.csv", index=False)
    import_scattered_field_dataset(source, normalized, source_manifest="import.yaml")
    return normalized


def _yaml_config(root, directory):
    task, profile = directory / "study.yaml", directory / "profile.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "task": "mode_decomp_analysis",
                "run": {"name": "optional", "seed": 23},
                "data": {"local_path": str(root)},
                "split": {"method": "random", "valid_size": 0.25},
                "analysis": {
                    "kind": "surrogate",
                    "split_seeds": [17],
                    "model_seeds": [23],
                    "bootstrap_samples": 0,
                    "modes": {"enabled": False},
                },
                "model": {
                    "candidates": [
                        {
                            "id": "pod1",
                            "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    profile.write_text(
        yaml.safe_dump({"profile": "local", "runtime": {"output_dir": str(directory / "runs")}}), encoding="utf-8"
    )
    return task, profile


def test_html_switch_preserves_tables_and_disabled_modes_need_no_indices(tmp_path, scattered_input):
    task, profile = _yaml_config(scattered_input, tmp_path)
    enabled = run_task(load_run_config(task, profile, overrides=["report.enabled=true"]))
    disabled = run_task(load_run_config(task, profile, overrides=["report.enabled=false"]))
    on, off = (pd.read_csv(result.tables["comparison"]) for result in (enabled, disabled))
    np.testing.assert_array_equal(on.field_rmse, off.field_rmse)
    assert (enabled.run_dir / "analysis.html").is_file()
    assert not (disabled.run_dir / "analysis.html").exists()
    assert disabled.tables["comparison"].is_file() and disabled.tables["per_sample"].is_file()
    assert json.loads(disabled.artifacts["analysis_summary"].read_text(encoding="utf-8"))["complete"]
    assert not (enabled.run_dir / "modes").exists() and not (disabled.run_dir / "modes").exists()


def test_original_observation_changes_revision_without_changing_canonical_field(scattered_input):
    before = load_field_dataset(scattered_input).revision
    field_hash = file_hash(scattered_input / "field.npy")
    original = load_scattered_observations(scattered_input)
    changed = np.array(original["values"], copy=True)
    changed[0, 0] += 0.25
    del original  # Release read-only mappings before replacing the NPY on Windows.
    np.save(scattered_input / "scattered" / "values.npy", changed)
    assert file_hash(scattered_input / "field.npy") == field_hash
    assert load_field_dataset(scattered_input).revision != before


def test_imputation_rejects_original_observation_scoring_before_fit(tmp_path, scattered_input):
    task, profile = _yaml_config(scattered_input, tmp_path)
    config = load_run_config(task, profile, overrides=["analysis.kind=imputation", "analysis.scattered.enabled=true"])
    with pytest.raises(ValueError, match="[Ss]cattered.*imputation"):
        run_task(config)
    assert not (tmp_path / "runs").exists()
