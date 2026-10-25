"""One observed field can be decomposed directly, without a statistical split."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.mark.parametrize("method", ["fft2", "dct2", "hodge"])
def test_single_field_cli_writes_direct_reconstruction_and_requested_analysis(tmp_path, method):
    directory = tmp_path / "input"
    directory.mkdir()
    x, y = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7))
    field = np.stack((2 * x - y, 2 * y + x), axis=-1) if method == "hodge" else (np.cos(2 * np.pi * x) + y)[..., None]
    np.save(directory / "field.npy", field)
    components = ["u_x", "u_y"] if method == "hodge" else ["temperature"]
    units = ["m/s", "m/s"] if method == "hodge" else ["K"]
    input_spec = {
        "format": "npy",
        "local_path": str(directory),
        "fields": {
            "path": "field.npy",
            "layout": "HWC",
            "kind": "vector" if method == "hodge" else "scalar",
            "components": components,
            "units": units,
        },
        "geometry": {
            "kind": "rectangle",
            "coordinate_units": {"x": "m", "y": "m"},
            "axes": {
                "x": {"values": x[0].tolist()},
                "y": {"values": y[:, 0].tolist()},
            },
        },
    }
    additions = (
        {"components": {"enabled": True}}
        if method == "hodge"
        else {"modes": {"enabled": True, "indices": [0, 1], "cumulative_counts": [1, 2]}}
    )
    config = {
        "task": "mode_decomp_representation",
        "data": input_spec,
        "runtime": {"output_dir": str(tmp_path / "output")},
        "representation": {"decomposition": {"name": method}},
        "analysis": additions,
        "report": {"enabled": True},
    }
    task = tmp_path / "run.yaml"
    task.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "ml_platform_mode_decomp", str(task)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    artifacts = {key: Path(value) for key, value in result["artifacts"].items()}
    summary = json.loads(artifacts["analysis_summary"].read_text(encoding="utf-8"))
    assert summary["metric_scope"] == summary["reference_scope"] == "fitted_input_reconstruction"
    assert summary["fit_sample_ids"] == ["sample_000000"] and summary["sample_count"] == 1
    assert result["metrics"]["field_rmse"] < 1e-9
    assert artifacts["representation_report"].is_file() and artifacts["model"].is_file()
    with np.load(artifacts["reconstructed_fields"], allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["reconstruction"], field[None], atol=1e-9)
        assert saved["sample_ids"].tolist() == ["sample_000000"]
        assert saved["components"].tolist() == components and saved["field_units"].tolist() == units
    with np.load(artifacts["coefficients"], allow_pickle=False) as saved:
        assert saved["space"].item() == "z"
        assert saved["packed_a"].shape == (1, summary["packed_dimension"])
        assert saved["coefficients"].shape == (1, summary["target_dimension"])
    if method == "hodge":
        assert summary["component_analysis"]["status"] == "ok"
        with np.load(artifacts["component_fields"], allow_pickle=False) as saved:
            assert saved["edge_exact"].shape[0] == 1
            np.testing.assert_allclose(
                saved["edge_sampled"], saved["edge_exact"] + saved["edge_coexact"] + saved["edge_harmonic"], atol=1e-12
            )
        manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
        energy = pd.read_csv(manifest["tables"]["component_energy"]["path"])
        np.testing.assert_allclose(energy.input_edge_energy, energy.sum_component_edge_energy, atol=1e-8)
    else:
        assert summary["mode_analysis"]["status"] == "ok"
        assert artifacts["mode_report"].is_file()
