#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _grid_mesh(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Regular n x n grid mesh triangulated into 2*(n-1)^2 faces."""
    if n < 2:
        raise ValueError("mesh-n must be >= 2")
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    ys = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack([xv.reshape(-1), yv.reshape(-1)], axis=1)  # (V,2)

    def vid(i: int, j: int) -> int:
        return i * n + j

    faces: list[list[int]] = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j)
            d = vid(i + 1, j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    return vertices, np.asarray(faces, dtype=np.int64)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(v)
    return (v - mu) / sd


def _rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(v * v))) if v.size else 0.0


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--n-samples", type=int, default=36)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fluct-ratio", type=float, default=0.07)
    ap.add_argument("--noise-ratio", type=float, default=0.01)
    ap.add_argument("--mesh-n", type=int, default=17)  # 17x17 => 289 vertices
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_dir = out_root / "mesh_scalar"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    n_samples = int(args.n_samples)
    if n_samples <= 0:
        raise ValueError("--n-samples must be > 0")

    vertices, faces = _grid_mesh(int(args.mesh_n))
    v = int(vertices.shape[0])

    x = vertices[:, 0]
    y = vertices[:, 1]
    pi = float(np.pi)
    patterns = [
        _normalize(np.sin(2 * pi * x)),
        _normalize(np.sin(2 * pi * y)),
        _normalize(x),
        _normalize(y),
        _normalize(x * y),
        _normalize(np.cos(2 * pi * (x + y))),
    ][:3]

    cond_dim = 4
    cond = rng.uniform(-1.0, 1.0, size=(n_samples, cond_dim)).astype(np.float32)
    offset = (1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(n_samples, 1))).astype(np.float32)
    cond[:, 0:1] = offset

    field = np.zeros((n_samples, v, 1, 1), dtype=np.float32)
    for i in range(n_samples):
        off = float(offset[i, 0])
        wts = cond[i, 1:4]
        base = np.zeros((v,), dtype=np.float64)
        for ww, p in zip(wts, patterns):
            base += float(ww) * p
        base_rms = _rms(base)
        if base_rms > 0:
            base *= float((float(args.fluct_ratio) * off) / base_rms)
        noise = rng.normal(0.0, 1.0, size=(v,)).astype(np.float64)
        noise_rms = _rms(noise)
        if noise_rms > 0:
            noise *= float((float(args.noise_ratio) * off) / noise_rms)
        vals = (off + base + noise).astype(np.float32)
        field[i, :, 0, 0] = vals

    np.save(out_dir / "cond.npy", cond.astype(np.float32))
    np.save(out_dir / "field.npy", field.astype(np.float32))

    manifest = {
        "field_kind": "scalar",
        "grid": {"H": int(v), "W": 1, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        "domain": {
            "type": "mesh",
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
        },
    }
    _write_manifest(out_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

