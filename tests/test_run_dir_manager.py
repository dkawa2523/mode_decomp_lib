from __future__ import annotations

from pathlib import Path

from mode_decomp_ml.pipeline.utils import RunDirManager


def test_run_dir_manager_resolves_interpolation(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    cfg = {
        "output": {"root": str(output_root), "name": "demo"},
        "task": {"name": "decomposition"},
        "run_dir": "${output.root}/${output.name}/${task.name}",
    }

    run_dir = RunDirManager(cfg).ensure()

    assert "${" not in str(cfg["run_dir"])
    assert str(run_dir) == str(output_root / "demo" / "decomposition")
