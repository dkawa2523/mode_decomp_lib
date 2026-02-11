from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_generate_summary_benchmark_md_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "bench" / "generate_summary_benchmark_md.py"
    assert script.exists()

    dataset_root = tmp_path / "dataset"
    (dataset_root / "rectangle_scalar").mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset_meta.json").write_text(
        json.dumps(
            {
                "version": "smoke",
                "n_samples": 36,
                "fluct_ratio": 0.07,
                "noise_ratio": 0.01,
                "seed": 123,
                "cases": ["rectangle_scalar"],
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "rectangle_scalar" / "manifest.json").write_text(
        json.dumps(
            {
                "field_kind": "scalar",
                "grid": {"H": 64, "W": 64, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
                "domain": {"type": "rectangle"},
            }
        ),
        encoding="utf-8",
    )

    v1_summary_dir = tmp_path / "v1_summary"
    missing_summary_dir = tmp_path / "missing_summary"

    decomp_headers = [
        "case",
        "decompose",
        "status",
        "field_rmse",
        "field_r2",
        "field_r2_k1",
        "field_r2_k64",
        "field_r2_topk_k16",
        "field_r2_topk_k64",
        "k_req_r2_0p95",
        "n_components_required",
        "fit_time_sec",
        "decomposition_run_dir",
    ]
    _write_csv(
        v1_summary_dir / "benchmark_summary_decomposition.csv",
        decomp_headers,
        [
            {
                "case": "rectangle_scalar",
                "decompose": "fft2",
                "status": "ok",
                "field_rmse": "1e-6",
                "field_r2": "0.99",
                "field_r2_k1": "0.1",
                "field_r2_k64": "0.95",
                "field_r2_topk_k16": "0.9",
                "field_r2_topk_k64": "0.97",
                "k_req_r2_0p95": "64",
                "n_components_required": "10",
                "fit_time_sec": "0.01",
                "decomposition_run_dir": str(tmp_path / "runs" / "decomposition"),
            }
        ],
    )

    train_headers = [
        "case",
        "decompose",
        "model",
        "status",
        "val_rmse",
        "val_r2",
        "val_field_rmse",
        "val_field_r2",
        "fit_time_sec",
        "train_run_dir",
    ]
    _write_csv(
        v1_summary_dir / "benchmark_summary_train.csv",
        train_headers,
        [
            {
                "case": "rectangle_scalar",
                "decompose": "fft2",
                "model": "ridge",
                "status": "ok",
                "val_rmse": "0.1",
                "val_r2": "0.8",
                "val_field_rmse": "0.2",
                "val_field_r2": "0.85",
                "fit_time_sec": "0.02",
                "train_run_dir": str(tmp_path / "runs" / "train"),
            }
        ],
    )

    out_path = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--v1-summary-dir",
            str(v1_summary_dir),
            "--missing-summary-dir",
            str(missing_summary_dir),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    text = out_path.read_text(encoding="utf-8")
    assert "# Benchmark Summary (v1)" in text
    assert "## How to Read (Quickstart)" in text
    assert "## Metrics (Definitions)" in text
    assert "## Plot Guide (What each figure means)" in text
    assert "### rectangle_scalar" in text
    assert "**Highlights (auto)**" in text

