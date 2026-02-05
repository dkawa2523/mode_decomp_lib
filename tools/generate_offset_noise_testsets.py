#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "mode_decomp_eval_dataset_v1" / "offset_noise_30"

N_SAMPLES = 30
COND_DIM_SCALAR = 4
COND_DIM_VECTOR = 8
SEED = 123


@dataclass
class DomainSpec:
    name: str
    height: int
    width: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    coord_kind: str
    mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None


def _grid(height: int, width: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(x_range[0], x_range[1], width, dtype=np.float32)
    y = np.linspace(y_range[0], y_range[1], height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def _normalize_pattern(pattern: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, pattern, np.nan)
    mean = np.nanmean(masked)
    std = np.nanstd(masked)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(pattern)
    return (pattern - mean) / std


def _write_csv(path: Path, xx: np.ndarray, yy: np.ndarray, field: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "f"])
        for xv, yv, fv in zip(xx.reshape(-1), yy.reshape(-1), field.reshape(-1)):
            writer.writerow([float(xv), float(yv), float(fv)])


def _write_conditions(path: Path, conds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id"] + [f"x{i+1}" for i in range(conds.shape[1])])
        for idx, row in enumerate(conds):
            writer.writerow([f"sample_{idx:04d}"] + [float(v) for v in row])


def _rectangle_patterns(xx: np.ndarray, yy: np.ndarray) -> list[np.ndarray]:
    pi = np.pi
    return [
        np.sin(2 * pi * xx),
        np.sin(2 * pi * yy),
        np.cos(2 * pi * (xx + yy)),
        xx - 0.5,
        yy - 0.5,
        (xx - 0.5) * (yy - 0.5),
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
        np.sin(3 * theta) * r**2,
    ]


def _sphere_patterns(lon: np.ndarray, lat: np.ndarray) -> list[np.ndarray]:
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    return [
        np.sin(lat_rad),
        np.cos(lat_rad),
        np.sin(lon_rad),
        np.cos(lon_rad),
        np.sin(lat_rad) * np.cos(lon_rad),
        np.cos(2 * lon_rad),
    ]


def _make_domain_specs() -> list[DomainSpec]:
    return [
        DomainSpec(
            name="scalar_rect",
            height=64,
            width=64,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            coord_kind="xy",
            mask_fn=lambda xx, yy: np.ones_like(xx, dtype=bool),
        ),
        DomainSpec(
            name="scalar_disk",
            height=64,
            width=64,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            coord_kind="xy",
            mask_fn=lambda xx, yy: (xx**2 + yy**2) <= 1.0,
        ),
        DomainSpec(
            name="scalar_annulus",
            height=64,
            width=64,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            coord_kind="xy",
            mask_fn=lambda xx, yy: (xx**2 + yy**2 >= 0.35**2) & (xx**2 + yy**2 <= 1.0),
        ),
        DomainSpec(
            name="scalar_mask",
            height=64,
            width=64,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
            coord_kind="xy",
            mask_fn=None,
        ),
        DomainSpec(
            name="scalar_sphere",
            height=18,
            width=36,
            x_range=(-180.0, 180.0),
            y_range=(-90.0, 90.0),
            coord_kind="lonlat",
            mask_fn=lambda xx, yy: np.ones_like(xx, dtype=bool),
        ),
    ]


def _load_mask(domain: DomainSpec, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    if domain.name == "scalar_mask":
        mask_path = ROOT / "data" / "mode_decomp_eval_dataset_v1" / "scalar_mask" / "mask.npy"
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask[0]
        return mask.astype(bool)
    if domain.mask_fn is None:
        return np.ones_like(xx, dtype=bool)
    return domain.mask_fn(xx, yy)


def _patterns_for_domain(domain: DomainSpec, xx: np.ndarray, yy: np.ndarray) -> list[np.ndarray]:
    if domain.coord_kind == "lonlat":
        return _sphere_patterns(xx, yy)
    if domain.name in {"scalar_disk", "scalar_annulus"}:
        return _polar_patterns(xx, yy)
    return _rectangle_patterns(xx, yy)


def _generate_scalar(domain: DomainSpec, rng: np.random.Generator) -> None:
    out_dir = OUT_ROOT / domain.name
    fields_dir = out_dir / "fields"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    xx, yy = _grid(domain.height, domain.width, domain.x_range, domain.y_range)
    mask = _load_mask(domain, xx, yy)
    patterns = [_normalize_pattern(p, mask) for p in _patterns_for_domain(domain, xx, yy)]
    patterns = patterns[:3]

    conds = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, COND_DIM_SCALAR)).astype(np.float32)
    offsets = 1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(N_SAMPLES, 1)).astype(np.float32)
    conds[:, 0:1] = offsets

    fields = np.zeros((N_SAMPLES, domain.height, domain.width, 1), dtype=np.float32)

    for idx in range(N_SAMPLES):
        offset = float(conds[idx, 0])
        weights = conds[idx, 1:4]
        fluct = np.zeros_like(xx, dtype=np.float32)
        for w, p in zip(weights, patterns):
            fluct += float(w) * p
        fluct = 0.1 * offset * fluct / max(len(patterns), 1)
        noise = 0.01 * offset * rng.normal(0.0, 1.0, size=xx.shape).astype(np.float32)
        field = offset + fluct + noise
        field = np.where(mask, field, 0.0).astype(np.float32)
        fields[idx, ..., 0] = field
        _write_csv(fields_dir / f"sample_{idx:04d}.csv", xx, yy, field)

    np.save(out_dir / "cond.npy", conds)
    np.save(out_dir / "field.npy", fields)
    np.save(out_dir / "mask.npy", mask.astype(np.uint8))
    _write_conditions(out_dir / "conditions.csv", conds)


def _generate_vector(domain: DomainSpec, rng: np.random.Generator) -> None:
    name = domain.name.replace("scalar_", "vector_")
    out_dir = OUT_ROOT / name
    fields_dir = out_dir / "fields"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    xx, yy = _grid(domain.height, domain.width, domain.x_range, domain.y_range)
    mask = _load_mask(domain, xx, yy)
    patterns = [_normalize_pattern(p, mask) for p in _patterns_for_domain(domain, xx, yy)]
    patterns = patterns[:3]

    conds = rng.uniform(-1.0, 1.0, size=(N_SAMPLES, COND_DIM_VECTOR)).astype(np.float32)
    offsets_x = 1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(N_SAMPLES, 1)).astype(np.float32)
    offsets_y = 1.0 + 0.3 * rng.uniform(0.0, 1.0, size=(N_SAMPLES, 1)).astype(np.float32)
    conds[:, 0:1] = offsets_x
    conds[:, 1:2] = offsets_y

    fields = np.zeros((N_SAMPLES, domain.height, domain.width, 2), dtype=np.float32)

    for idx in range(N_SAMPLES):
        offset_x = float(conds[idx, 0])
        offset_y = float(conds[idx, 1])
        weights_x = conds[idx, 2:5]
        weights_y = conds[idx, 5:8]
        fluct_x = np.zeros_like(xx, dtype=np.float32)
        fluct_y = np.zeros_like(xx, dtype=np.float32)
        for w, p in zip(weights_x, patterns):
            fluct_x += float(w) * p
        for w, p in zip(weights_y, patterns):
            fluct_y += float(w) * p
        fluct_x = 0.1 * offset_x * fluct_x / max(len(patterns), 1)
        fluct_y = 0.1 * offset_y * fluct_y / max(len(patterns), 1)
        noise_x = 0.01 * offset_x * rng.normal(0.0, 1.0, size=xx.shape).astype(np.float32)
        noise_y = 0.01 * offset_y * rng.normal(0.0, 1.0, size=xx.shape).astype(np.float32)
        field_x = np.where(mask, offset_x + fluct_x + noise_x, 0.0).astype(np.float32)
        field_y = np.where(mask, offset_y + fluct_y + noise_y, 0.0).astype(np.float32)
        fields[idx, ..., 0] = field_x
        fields[idx, ..., 1] = field_y
        _write_csv(fields_dir / f"sample_{idx:04d}_fx.csv", xx, yy, field_x)
        _write_csv(fields_dir / f"sample_{idx:04d}_fy.csv", xx, yy, field_y)

    np.save(out_dir / "cond.npy", conds)
    np.save(out_dir / "field.npy", fields)
    np.save(out_dir / "mask.npy", mask.astype(np.uint8))
    _write_conditions(out_dir / "conditions.csv", conds)


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for domain in _make_domain_specs():
        _generate_scalar(domain, rng)
        _generate_vector(domain, rng)

    meta = {
        "version": "offset_noise_30",
        "samples": N_SAMPLES,
        "cond_dim_scalar": COND_DIM_SCALAR,
        "cond_dim_vector": COND_DIM_VECTOR,
        "notes": {
            "offset": "constant offset per sample",
            "fluctuation": "~10% of offset magnitude",
            "noise": "~1% of offset magnitude per spatial location",
        },
        "subsets": [spec.name for spec in _make_domain_specs()]
        + [spec.name.replace("scalar_", "vector_") for spec in _make_domain_specs()],
    }
    (OUT_ROOT / "dataset_meta.json").write_text(
        __import__("json").dumps(meta, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
