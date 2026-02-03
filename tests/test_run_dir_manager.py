from __future__ import annotations

from pathlib import Path

from mode_decomp_ml.pipeline.utils import RunDirManager


def test_run_dir_manager_resolves_interpolation(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    cfg = {
        "output_dir": str(output_root),
        "tag": "demo",
        "run_id": "${now:%Y%m%d-%H%M%S}-${hydra:job.num}",
        "run_dir": "${output_dir}/${tag}/${run_id}",
    }

    run_dir = RunDirManager(cfg).ensure()

    assert "${" not in str(cfg["run_id"])
    assert "${" not in str(cfg["run_dir"])
    assert str(run_dir).startswith(str(output_root / "demo"))
