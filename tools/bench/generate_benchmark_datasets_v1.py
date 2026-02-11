#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _grid_xy(h: int, w: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_range[0], x_range[1], w, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy


def _normalize_pattern(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    masked = np.where(mask, p, np.nan)
    mean = float(np.nanmean(masked))
    std = float(np.nanstd(masked))
    if not np.isfinite(std) or std <= 0:
        return np.zeros_like(p, dtype=np.float32)
    return ((p - mean) / std).astype(np.float32)


def _rms(x: np.ndarray, mask: np.ndarray) -> float:
    vals = np.asarray(x, dtype=np.float32)[mask]
    if vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals**2)))


def _rectangle_patterns(xx: np.ndarray, yy: np.ndarray) -> list[np.ndarray]:
    pi = float(np.pi)
    return [
        np.sin(2 * pi * xx),
        np.sin(2 * pi * yy),
        np.cos(2 * pi * (xx + yy)),
        xx,
        yy,
        xx * yy,
    ]


def _polar_patterns(xx: np.ndarray, yy: np.ndarray) -> list[np.ndarray]:
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    return [
        r,
        r**2,
        np.cos(theta),
        np.sin(theta),
        np.cos(2 * theta) * r,
        np.sin(3 * theta) * (r**2),
    ]


def _sphere_patterns(lon_deg: np.ndarray, lat_deg: np.ndarray) -> list[np.ndarray]:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    return [
        np.sin(lat),
        np.cos(lat),
        np.sin(lon),
        np.cos(lon),
        np.sin(lat) * np.cos(lon),
        np.cos(2 * lon),
    ]


def _disk_mask(xx: np.ndarray, yy: np.ndarray, *, radius: float = 1.0) -> np.ndarray:
    rr2 = (xx**2 + yy**2).astype(np.float32)
    return rr2 <= float(radius * radius)


def _annulus_mask(xx: np.ndarray, yy: np.ndarray, *, r_inner: float, r_outer: float) -> np.ndarray:
    rr2 = (xx**2 + yy**2).astype(np.float32)
    return (rr2 >= float(r_inner * r_inner)) & (rr2 <= float(r_outer * r_outer))


def _smooth3(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    m = mask.astype(np.float32)
    for _ in range(int(iters)):
        # 3x3 mean filter via slicing (no scipy).
        pad = np.pad(m, ((1, 1), (1, 1)), mode="edge")
        acc = (
            pad[0:-2, 0:-2]
            + pad[0:-2, 1:-1]
            + pad[0:-2, 2:]
            + pad[1:-1, 0:-2]
            + pad[1:-1, 1:-1]
            + pad[1:-1, 2:]
            + pad[2:, 0:-2]
            + pad[2:, 1:-1]
            + pad[2:, 2:]
        )
        m = acc / 9.0
    return m


def _arbitrary_mask(h: int, w: int, *, rng: np.random.Generator) -> np.ndarray:
    xx, yy = _grid_xy(h, w, (-1.0, 1.0), (-1.0, 1.0))
    blobs = np.zeros((h, w), dtype=np.float32)
    n_blobs = int(rng.integers(2, 4))
    for _ in range(n_blobs):
        cx = float(rng.uniform(-0.5, 0.5))
        cy = float(rng.uniform(-0.5, 0.5))
        sigma = float(rng.uniform(0.18, 0.35))
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        blobs += np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)
    blobs = _smooth3(blobs, iters=2)
    # Choose threshold so coverage is ~60-75%.
    thresh = float(np.quantile(blobs.reshape(-1), 0.35))
    mask = blobs >= thresh
    mask = _smooth3(mask.astype(np.float32), iters=1) >= 0.5
    return mask.astype(bool)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_subset(
    out_dir: Path,
    *,
    cond: np.ndarray,
    field: np.ndarray,
    manifest: dict[str, Any],
    domain_mask_path: Path | None = None,
    domain_mask: np.ndarray | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cond.npy", np.asarray(cond, dtype=np.float32))
    np.save(out_dir / "field.npy", np.asarray(field, dtype=np.float32))
    if domain_mask_path is not None and domain_mask is not None:
        np.save(out_dir / domain_mask_path.name, domain_mask.astype(np.uint8))
    _write_manifest(out_dir / "manifest.json", manifest)


def _generate_scalar_case(
    *,
    name: str,
    out_root: Path,
    n_samples: int,
    rng: np.random.Generator,
    fluct_ratio: float,
    noise_ratio: float,
) -> None:
    h, w = 64, 64
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)
    cond_dim = 4

    if name == "rectangle_scalar":
        xx, yy = _grid_xy(h, w, x_range, y_range)
        mask = np.ones((h, w), dtype=bool)
        patterns = _rectangle_patterns(xx, yy)
        domain = {"type": "rectangle"}
        grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
        domain_mask_path = None
        domain_mask = None
    elif name == "disk_scalar":
        xx, yy = _grid_xy(h, w, x_range, y_range)
        mask = _disk_mask(xx, yy, radius=1.0)
        patterns = _polar_patterns(xx, yy)
        domain = {"type": "disk", "center": [0.0, 0.0], "radius": 1.0}
        grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
        domain_mask_path = None
        domain_mask = None
    elif name == "annulus_scalar":
        xx, yy = _grid_xy(h, w, x_range, y_range)
        mask = _annulus_mask(xx, yy, r_inner=0.35, r_outer=1.0)
        patterns = _polar_patterns(xx, yy)
        domain = {"type": "annulus", "center": [0.0, 0.0], "r_inner": 0.35, "r_outer": 1.0}
        grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
        domain_mask_path = None
        domain_mask = None
    elif name == "arbitrary_mask_scalar":
        xx, yy = _grid_xy(h, w, x_range, y_range)
        domain_mask = _arbitrary_mask(h, w, rng=rng)
        mask = domain_mask
        patterns = _rectangle_patterns(xx, yy)
        domain = {"type": "arbitrary_mask", "mask_source": "file", "mask_path": "domain_mask.npy"}
        grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
        domain_mask_path = Path("domain_mask.npy")
    else:
        raise ValueError(f"unknown scalar case: {name}")

    patterns_n = [_normalize_pattern(p, mask) for p in patterns][:3]

    cond = rng.uniform(-1.0, 1.0, size=(n_samples, cond_dim)).astype(np.float32)
    offset = (1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(n_samples, 1))).astype(np.float32)
    cond[:, 0:1] = offset

    field = np.zeros((n_samples, h, w, 1), dtype=np.float32)
    for i in range(n_samples):
        off = float(offset[i, 0])
        wts = cond[i, 1:4]
        base = np.zeros((h, w), dtype=np.float32)
        for ww, p in zip(wts, patterns_n):
            base += float(ww) * p
        base_rms = _rms(base, mask)
        if base_rms > 0:
            base *= float((fluct_ratio * off) / base_rms)
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        noise_rms = _rms(noise, mask)
        if noise_rms > 0:
            noise *= float((noise_ratio * off) / noise_rms)
        img = (off + base + noise).astype(np.float32)
        img = np.where(mask, img, 0.0).astype(np.float32)
        field[i, ..., 0] = img

    manifest = {"field_kind": "scalar", "grid": grid, "domain": domain}
    _write_subset(
        out_root / name,
        cond=cond,
        field=field,
        manifest=manifest,
        domain_mask_path=domain_mask_path,
        domain_mask=domain_mask,
    )


def _generate_vector_case(
    *,
    name: str,
    out_root: Path,
    n_samples: int,
    rng: np.random.Generator,
    fluct_ratio: float,
    noise_ratio: float,
) -> None:
    if name.endswith("_vector"):
        base_name = name.replace("_vector", "_scalar")
    else:
        raise ValueError("vector case name must end with _vector")
    h, w = (18, 36) if base_name == "sphere_grid_scalar" else (64, 64)
    x_range = (-1.0, 1.0)
    y_range = (-1.0, 1.0)

    if base_name == "sphere_grid_scalar":
        lat_range = (-90.0, 90.0)
        step = 360.0 / float(w)
        lon_range = (-180.0, 180.0 - step)
        lat_vals = np.linspace(lat_range[0], lat_range[1], h, dtype=np.float32)
        lon_vals = np.linspace(lon_range[0], lon_range[1], w, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
        mask = np.ones((h, w), dtype=bool)
        patterns = _sphere_patterns(lon_grid, lat_grid)
        domain = {
            "type": "sphere_grid",
            "radius": 1.0,
            "lat_range": list(lat_range),
            "lon_range": list(lon_range),
            "angle_unit": "deg",
            "n_lat": h,
            "n_lon": w,
        }
        grid = {"H": h, "W": w, "x_range": list(lon_range), "y_range": list(lat_range)}
        domain_mask_path = None
        domain_mask = None
    else:
        xx, yy = _grid_xy(h, w, x_range, y_range)
        if base_name == "rectangle_scalar":
            mask = np.ones((h, w), dtype=bool)
            patterns = _rectangle_patterns(xx, yy)
            domain = {"type": "rectangle"}
            grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
            domain_mask_path = None
            domain_mask = None
        elif base_name == "disk_scalar":
            mask = _disk_mask(xx, yy, radius=1.0)
            patterns = _polar_patterns(xx, yy)
            domain = {"type": "disk", "center": [0.0, 0.0], "radius": 1.0}
            grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
            domain_mask_path = None
            domain_mask = None
        elif base_name == "annulus_scalar":
            mask = _annulus_mask(xx, yy, r_inner=0.35, r_outer=1.0)
            patterns = _polar_patterns(xx, yy)
            domain = {"type": "annulus", "center": [0.0, 0.0], "r_inner": 0.35, "r_outer": 1.0}
            grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
            domain_mask_path = None
            domain_mask = None
        elif base_name == "arbitrary_mask_scalar":
            domain_mask = _arbitrary_mask(h, w, rng=rng)
            mask = domain_mask
            patterns = _rectangle_patterns(xx, yy)
            domain = {"type": "arbitrary_mask", "mask_source": "file", "mask_path": "domain_mask.npy"}
            grid = {"H": h, "W": w, "x_range": list(x_range), "y_range": list(y_range)}
            domain_mask_path = Path("domain_mask.npy")
        else:
            raise ValueError(f"unknown vector case: {name}")

    patterns_n = [_normalize_pattern(p, mask) for p in patterns][:3]
    cond_dim = 8
    cond = rng.uniform(-1.0, 1.0, size=(n_samples, cond_dim)).astype(np.float32)
    off_u = (1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(n_samples, 1))).astype(np.float32)
    off_v = (1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(n_samples, 1))).astype(np.float32)
    cond[:, 0:1] = off_u
    cond[:, 1:2] = off_v

    field = np.zeros((n_samples, h, w, 2), dtype=np.float32)
    for i in range(n_samples):
        ou = float(off_u[i, 0])
        ov = float(off_v[i, 0])
        w_u = cond[i, 2:5]
        w_v = cond[i, 5:8]
        base_u = np.zeros((h, w), dtype=np.float32)
        base_v = np.zeros((h, w), dtype=np.float32)
        for ww, p in zip(w_u, patterns_n):
            base_u += float(ww) * p
        for ww, p in zip(w_v, patterns_n):
            base_v += float(ww) * p
        rms_u = _rms(base_u, mask)
        rms_v = _rms(base_v, mask)
        if rms_u > 0:
            base_u *= float((fluct_ratio * ou) / rms_u)
        if rms_v > 0:
            base_v *= float((fluct_ratio * ov) / rms_v)
        nu = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        nv = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        rms_nu = _rms(nu, mask)
        rms_nv = _rms(nv, mask)
        if rms_nu > 0:
            nu *= float((noise_ratio * ou) / rms_nu)
        if rms_nv > 0:
            nv *= float((noise_ratio * ov) / rms_nv)
        u = (ou + base_u + nu).astype(np.float32)
        v = (ov + base_v + nv).astype(np.float32)
        u = np.where(mask, u, 0.0).astype(np.float32)
        v = np.where(mask, v, 0.0).astype(np.float32)
        field[i, ..., 0] = u
        field[i, ..., 1] = v

    manifest = {"field_kind": "vector", "grid": grid, "domain": domain}
    _write_subset(
        out_root / name,
        cond=cond,
        field=field,
        manifest=manifest,
        domain_mask_path=domain_mask_path,
        domain_mask=domain_mask,
    )


def _generate_sphere_scalar_case(
    *,
    out_root: Path,
    n_samples: int,
    rng: np.random.Generator,
    fluct_ratio: float,
    noise_ratio: float,
) -> None:
    h, w = 18, 36
    lat_range = (-90.0, 90.0)
    step = 360.0 / float(w)
    lon_range = (-180.0, 180.0 - step)

    lat_vals = np.linspace(lat_range[0], lat_range[1], h, dtype=np.float32)
    lon_vals = np.linspace(lon_range[0], lon_range[1], w, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
    mask = np.ones((h, w), dtype=bool)
    patterns = _sphere_patterns(lon_grid, lat_grid)
    patterns_n = [_normalize_pattern(p, mask) for p in patterns][:3]

    cond_dim = 4
    cond = rng.uniform(-1.0, 1.0, size=(n_samples, cond_dim)).astype(np.float32)
    offset = (1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(n_samples, 1))).astype(np.float32)
    cond[:, 0:1] = offset

    field = np.zeros((n_samples, h, w, 1), dtype=np.float32)
    for i in range(n_samples):
        off = float(offset[i, 0])
        wts = cond[i, 1:4]
        base = np.zeros((h, w), dtype=np.float32)
        for ww, p in zip(wts, patterns_n):
            base += float(ww) * p
        base_rms = _rms(base, mask)
        if base_rms > 0:
            base *= float((fluct_ratio * off) / base_rms)
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        noise_rms = _rms(noise, mask)
        if noise_rms > 0:
            noise *= float((noise_ratio * off) / noise_rms)
        img = (off + base + noise).astype(np.float32)
        field[i, ..., 0] = img

    domain = {
        "type": "sphere_grid",
        "radius": 1.0,
        "lat_range": list(lat_range),
        "lon_range": list(lon_range),
        "angle_unit": "deg",
        "n_lat": h,
        "n_lon": w,
    }
    grid = {"H": h, "W": w, "x_range": list(lon_range), "y_range": list(lat_range)}
    manifest = {"field_kind": "scalar", "grid": grid, "domain": domain}
    _write_subset(out_root / "sphere_grid_scalar", cond=cond, field=field, manifest=manifest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, default="data/benchmarks/v1/offset_noise_36")
    ap.add_argument("--n-samples", type=int, default=36)
    ap.add_argument("--fluct-ratio", type=float, default=0.07)
    ap.add_argument("--noise-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    n_samples = int(args.n_samples)
    if n_samples < 2:
        raise ValueError("--n-samples must be >= 2")
    fluct_ratio = float(args.fluct_ratio)
    noise_ratio = float(args.noise_ratio)
    if fluct_ratio <= 0 or fluct_ratio >= 0.2:
        raise ValueError("--fluct-ratio must be in (0, 0.2) for benchmark v1")
    if noise_ratio <= 0 or noise_ratio >= 0.1:
        raise ValueError("--noise-ratio must be in (0, 0.1) for benchmark v1")

    rng = np.random.default_rng(int(args.seed))

    scalar_cases = ["rectangle_scalar", "disk_scalar", "annulus_scalar", "arbitrary_mask_scalar"]
    vector_cases = [c.replace("_scalar", "_vector") for c in scalar_cases]

    for case in scalar_cases:
        _generate_scalar_case(
            name=case,
            out_root=out_root,
            n_samples=n_samples,
            rng=rng,
            fluct_ratio=fluct_ratio,
            noise_ratio=noise_ratio,
        )
    _generate_sphere_scalar_case(out_root=out_root, n_samples=n_samples, rng=rng, fluct_ratio=fluct_ratio, noise_ratio=noise_ratio)

    for case in vector_cases:
        _generate_vector_case(
            name=case,
            out_root=out_root,
            n_samples=n_samples,
            rng=rng,
            fluct_ratio=fluct_ratio,
            noise_ratio=noise_ratio,
        )
    _generate_vector_case(
        name="sphere_grid_vector",
        out_root=out_root,
        n_samples=n_samples,
        rng=rng,
        fluct_ratio=fluct_ratio,
        noise_ratio=noise_ratio,
    )

    meta = {
        "version": "benchmark_v1_offset_noise_36",
        "n_samples": n_samples,
        "fluct_ratio": fluct_ratio,
        "noise_ratio": noise_ratio,
        "seed": int(args.seed),
        "cases": scalar_cases + ["sphere_grid_scalar"] + vector_cases + ["sphere_grid_vector"],
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

