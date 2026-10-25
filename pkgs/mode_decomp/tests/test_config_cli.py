"""One-file execution, override precedence and the shared dictionary/CLI boundary."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from ml_platform_mode_decomp import run_task
from ml_platform_mode_decomp.config import load_config
from ml_platform_mode_decomp.policy import resolve_config
from ml_platform_mode_decomp.representation import FieldRepresentation

from test_search import write_input


def write_yaml(path, content):
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "representation, error",
    [
        ({"decompostion": {"name": "dct2"}}, "representation keys"),
        ({"decomposition": {"name": "dct2", "params": {"disk_polcy": "error"}}}, "dct2 parameters"),
    ],
)
def test_direct_task_and_array_api_reject_the_same_misspelled_representation(representation, error):
    with pytest.raises(ValueError, match=error):
        FieldRepresentation(representation)
    with pytest.raises(ValueError, match=error):
        resolve_config({"task": "mode_decomp_representation", "representation": representation})


def test_profile_is_optional_and_precedence_preserves_inline_data_and_exact_cli_replacement(tmp_path):
    np.save(tmp_path / "field.npy", np.ones((2, 3, 1)))
    data = {"local_path": str(tmp_path), "fields": {"path": "field.npy", "layout": "HWC"}}
    run = write_yaml(
        tmp_path / "run.yaml",
        {
            "task": "mode_decomp_representation",
            "run": {"seed": 13},
            "data": data,
            "representation": {"decomposition": {"name": "dct2"}},
        },
    )
    profile = write_yaml(
        tmp_path / "profile.yaml",
        {"run": {"seed": 7, "name": "from_profile"}, "runtime": {"output_dir": str(tmp_path / "runs")}},
    )
    single = load_config(run)
    assert single["data"] == data and single["run"]["seed"] == 13
    override = {"run": {"seed": 19}, "data": {"fields": {"components": ["temperature"]}}}
    before = deepcopy(override)
    merged = load_config(run, profile, overrides=override)
    assert merged["run"] == {"seed": 19, "name": "from_profile"}
    result = run_task(merged)
    assert result.metrics["field_rmse"] < 1e-12
    assert merged["data"]["fields"]["components"] == ["temperature"]
    assert override == before


def test_explicit_sensor_settings_and_saved_configuration_can_be_reused(tmp_path):
    """Replaying the saved run keeps sample order and does not invent random-mask settings."""
    import pandas as pd

    source = tmp_path / "input"
    source.mkdir()
    fields = np.arange(12.0)[:, None, None, None] * np.arange(1.0, 7.0).reshape(1, 2, 3, 1)
    np.save(source / "fields.npy", fields)
    indices = np.roll(np.arange(len(fields)), 3)
    pd.DataFrame({"sample_id": [f"s{i}" for i in indices], "array_index": indices}).to_csv(
        source / "samples.csv", index=False
    )
    observed = np.ones(fields.shape[:-1], dtype=bool)
    for index in range(len(fields)):
        observed[index].flat[index % 6] = False
    np.save(source / "sensors.npy", observed)
    manifest = write_yaml(
        source / "field_dataset.yaml",
        {
            "schema": "field_dataset_v1",
            "fields": {"path": "fields.npy", "layout": "NHWC"},
            "samples": {"path": "samples.csv"},
        },
    )
    resolved = resolve_config(
        {
            "task": "mode_decomp_analysis",
            "runtime": {"output_dir": str(tmp_path / "first")},
            "data": {"local_path": str(manifest)},
            "analysis": {
                "kind": "imputation",
                "observation_mask": "sensors.npy",
                "split_seeds": [11],
                "model_seeds": [42],
                "bootstrap_samples": 0,
            },
            "model": {
                "candidates": [
                    {"id": "pod", "representation": {"decomposition": {"name": "gappy_pod", "params": {"n_modes": 1}}}}
                ]
            },
            "report": {"enabled": False},
        }
    )
    assert resolve_config(resolved) == resolved
    assert not {"observed_fraction", "observation_pattern", "mask_seed"}.intersection(resolved["analysis"])
    first = run_task(resolved)
    replay = load_config(first.artifacts["config"], overrides={"runtime": {"output_dir": str(tmp_path / "replay")}})
    second = run_task(replay)
    before, after = (pd.read_csv(result.tables["per_sample"]) for result in (first, second))
    assert before.sample_id.tolist() == after.sample_id.tolist()
    np.testing.assert_allclose(before.mse, after.mse, atol=1e-12)
    assert first.metrics == second.metrics


def test_single_yaml_legacy_cli_and_dictionary_execute_the_same_model_with_cwd_paths(tmp_path, monkeypatch):
    write_input(tmp_path / "input")
    configs = tmp_path / "configuration"
    configs.mkdir()
    raw = {
        "task": "mode_decomp_pipeline",
        "run": {"seed": 17},
        "data": {"local_path": "input"},
        "split": {
            "method": "fixed",
            "valid_filter_column": "role",
            "valid_filter_value": "test",
        },
        "model": {
            "candidates": [
                {
                    "id": "pod",
                    "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}},
                    "regression": {"name": "ridge", "params": {"alpha": 0.0}},
                }
            ]
        },
    }
    path = write_yaml(configs / "run.yaml", raw)
    profile = write_yaml(configs / "profile.yaml", {"run": {"seed": 5}})
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "ml_platform_mode_decomp",
            str(path),
            "--profile",
            str(profile),
            "--set",
            "runtime.output_dir=cli",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    payload = json.loads(completed.stdout)
    assert Path(payload["run_dir"]).is_relative_to(tmp_path / "cli")
    monkeypatch.chdir(tmp_path)
    direct = run_task({**raw, "runtime": {"output_dir": "dictionary"}})
    assert direct.metrics == payload["metrics"] and direct.metrics["field_rmse"] < 1e-12
    with np.load(payload["artifacts"]["predictions"], allow_pickle=False) as cli:
        with np.load(direct.artifacts["predictions"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["predictions"], cli["predictions"])
            np.testing.assert_array_equal(saved["sample_ids"], cli["sample_ids"])


@pytest.mark.parametrize(
    ("arguments", "message"),
    [([], "A run YAML is required"), (["one.yaml", "--task", "two.yaml"], "Supply either CONFIG or --task, not both")],
)
def test_cli_rejects_missing_or_ambiguous_configuration_before_loading(tmp_path, arguments, message):
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "ml_platform_mode_decomp", *arguments],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert completed.returncode == 2 and message in completed.stderr
    assert not list(tmp_path.iterdir())
