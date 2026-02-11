from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def _grid_xy(h: int, w: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_range[0], x_range[1], w, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy


def _mask_from_manifest(case_dir: Path, manifest: dict) -> np.ndarray:
    grid = manifest["grid"]
    h = int(grid["H"])
    w = int(grid["W"])
    domain = manifest["domain"]
    domain_type = str(domain["type"])
    if domain_type in {"rectangle", "sphere_grid"}:
        return np.ones((h, w), dtype=bool)
    if domain_type == "disk":
        xx, yy = _grid_xy(h, w, tuple(grid["x_range"]), tuple(grid["y_range"]))
        cx, cy = domain["center"]
        radius = float(domain["radius"])
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return rr2 <= radius * radius
    if domain_type == "annulus":
        xx, yy = _grid_xy(h, w, tuple(grid["x_range"]), tuple(grid["y_range"]))
        cx, cy = domain["center"]
        r_inner = float(domain["r_inner"])
        r_outer = float(domain["r_outer"])
        rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
        return (rr2 >= r_inner * r_inner) & (rr2 <= r_outer * r_outer)
    if domain_type in {"arbitrary_mask", "mask"}:
        mask_path = case_dir / "domain_mask.npy"
        assert mask_path.exists()
        loaded = np.load(mask_path, allow_pickle=False)
        return np.asarray(loaded).astype(bool)
    raise AssertionError(f"unsupported domain_type in test: {domain_type}")


def _rms(x: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(x, dtype=np.float64)[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals**2)))


def test_generate_benchmark_datasets_v1_shapes_and_ratios(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "tools" / "bench" / "generate_benchmark_datasets_v1.py"
    out_root = tmp_path / "offset_noise_36"

    # Keep it small for CI speed, but still test ratios.
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--out-root",
            str(out_root),
            "--n-samples",
            "6",
            "--fluct-ratio",
            "0.07",
            "--noise-ratio",
            "0.01",
            "--seed",
            "123",
        ],
        check=True,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    cases = [
        "rectangle_scalar",
        "rectangle_vector",
        "disk_scalar",
        "disk_vector",
        "annulus_scalar",
        "annulus_vector",
        "arbitrary_mask_scalar",
        "arbitrary_mask_vector",
        "sphere_grid_scalar",
        "sphere_grid_vector",
    ]

    for case in cases:
        case_dir = out_root / case
        assert (case_dir / "cond.npy").exists()
        assert (case_dir / "field.npy").exists()
        assert (case_dir / "manifest.json").exists()

        manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
        mask = _mask_from_manifest(case_dir, manifest)

        cond = np.load(case_dir / "cond.npy", allow_pickle=False)
        field = np.load(case_dir / "field.npy", allow_pickle=False)
        assert field.ndim == 4
        n, h, w, c = field.shape
        assert n == 6
        assert mask.shape == (h, w)
        assert cond.shape[0] == n

        if manifest["field_kind"] == "scalar":
            assert c == 1
            # offset is cond[:,0]
            for i in range(n):
                off = float(cond[i, 0])
                img = field[i, ..., 0]
                mean = float(np.mean(img[mask]))
                # mean should remain close to offset; tolerate small drift.
                assert abs(mean - off) <= 0.02 * off
                # overall fluct+noise RMS relative magnitude should be in a reasonable band.
                rel = _rms(img - off, mask) / off
                assert 0.04 <= rel <= 0.10
        else:
            assert c == 2
            # offsets are cond[:,0] (u), cond[:,1] (v)
            for i in range(n):
                ou = float(cond[i, 0])
                ov = float(cond[i, 1])
                u = field[i, ..., 0]
                v = field[i, ..., 1]
                mean_u = float(np.mean(u[mask]))
                mean_v = float(np.mean(v[mask]))
                assert abs(mean_u - ou) <= 0.02 * ou
                assert abs(mean_v - ov) <= 0.02 * ov
                rel_u = _rms(u - ou, mask) / ou
                rel_v = _rms(v - ov, mask) / ov
                assert 0.04 <= rel_u <= 0.10
                assert 0.04 <= rel_v <= 0.10

    # Root meta should exist.
    meta_path = out_root / "dataset_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("n_samples") == 6
    assert math.isclose(float(meta.get("fluct_ratio")), 0.07, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(float(meta.get("noise_ratio")), 0.01, rel_tol=0, abs_tol=1e-9)

