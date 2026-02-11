#!/usr/bin/env python3
"""
Migrate / enrich the tracked evaluation datasets to the canonical `npy_dir` format.

This script is intentionally conservative:
- It NEVER deletes CSV artifacts by itself (use git rm --cached separately).
- It only writes `manifest.json` to dataset roots that contain `cond.npy` + `field.npy`.

Dataset roots currently live under:
  data/mode_decomp_eval_dataset_v1/
    scalar_*, vector_*
    offset_noise_30/{scalar_*, vector_*}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from mode_decomp_ml.data.manifest import validate_field_against_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read yaml: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML must be a mapping: {path}")
    return dict(data)


def _infer_field_kind(field: np.ndarray) -> str:
    arr = np.asarray(field)
    if arr.ndim != 4:
        raise ValueError(f"field.npy must be 4D (N,H,W,C), got shape={arr.shape}")
    channels = int(arr.shape[-1])
    if channels == 1:
        return "scalar"
    if channels == 2:
        return "vector"
    raise ValueError(f"Unsupported channels={channels} (expected 1 or 2)")


def _manifest_from_example_config(dataset_root: Path, example_cfg: Mapping[str, Any], field: np.ndarray) -> dict[str, Any]:
    dataset_cfg = example_cfg.get("dataset", {}) if isinstance(example_cfg.get("dataset"), Mapping) else {}
    domain_cfg = example_cfg.get("domain", {}) if isinstance(example_cfg.get("domain"), Mapping) else {}

    field_kind = _infer_field_kind(field)
    n, h, w, _ = np.asarray(field).shape
    _ = n  # unused

    grid_cfg = dataset_cfg.get("grid", {}) if isinstance(dataset_cfg.get("grid"), Mapping) else {}
    x_range = grid_cfg.get("x_range", None)
    y_range = grid_cfg.get("y_range", None)
    lon_range = grid_cfg.get("lon_range", None)
    lat_range = grid_cfg.get("lat_range", None)

    if x_range is None and lon_range is not None:
        x_range = lon_range
    if y_range is None and lat_range is not None:
        y_range = lat_range

    if x_range is None or y_range is None:
        raise ValueError(f"example_config.yaml missing grid ranges: {dataset_root}")

    domain_type = str(domain_cfg.get("name") or domain_cfg.get("type") or "").strip()
    if not domain_type:
        raise ValueError(f"example_config.yaml missing domain.name: {dataset_root}")

    domain: dict[str, Any] = {"type": domain_type}
    if domain_type == "disk":
        domain["center"] = domain_cfg.get("center", [0.0, 0.0])
        domain["radius"] = float(domain_cfg.get("radius", 1.0))
    elif domain_type == "annulus":
        domain["center"] = domain_cfg.get("center", [0.0, 0.0])
        domain["r_inner"] = float(domain_cfg.get("r_inner", 0.35))
        domain["r_outer"] = float(domain_cfg.get("r_outer", 1.0))
    elif domain_type == "sphere_grid":
        # Prefer explicit values if present; otherwise infer from grid.
        domain["radius"] = float(domain_cfg.get("radius", 1.0))
        if "angle_unit" in domain_cfg:
            domain["angle_unit"] = domain_cfg.get("angle_unit")
        domain["n_lat"] = int(domain_cfg.get("n_lat", h))
        domain["n_lon"] = int(domain_cfg.get("n_lon", w))
        if lat_range is not None:
            domain["lat_range"] = lat_range
        if lon_range is not None:
            domain["lon_range"] = lon_range
    elif domain_type in {"arbitrary_mask", "mask"}:
        # Use a fixed domain mask stored in this dataset root.
        domain["type"] = "arbitrary_mask" if domain_type == "arbitrary_mask" else "mask"
        domain["mask_source"] = "file"
        domain["mask_path"] = "mask.npy"

    return {
        "field_kind": field_kind,
        "grid": {"H": int(h), "W": int(w), "x_range": x_range, "y_range": y_range},
        "domain": domain,
    }


def _manifest_fallback(dataset_root: Path, field: np.ndarray) -> dict[str, Any]:
    field_kind = _infer_field_kind(field)
    _, h, w, _ = np.asarray(field).shape

    name = dataset_root.name.lower()
    domain: dict[str, Any]
    if "sphere" in name:
        domain = {
            "type": "sphere_grid",
            "radius": 1.0,
            "angle_unit": "deg",
            "n_lat": int(h),
            "n_lon": int(w),
            "lat_range": [-90.0, 90.0],
            "lon_range": [-180.0, 180.0],
        }
        x_range = [-180.0, 180.0]
        y_range = [-90.0, 90.0]
    elif "annulus" in name:
        domain = {"type": "annulus", "center": [0.0, 0.0], "r_inner": 0.35, "r_outer": 1.0}
        x_range = [-1.0, 1.0]
        y_range = [-1.0, 1.0]
    elif "disk" in name:
        domain = {"type": "disk", "center": [0.0, 0.0], "radius": 1.0}
        x_range = [-1.0, 1.0]
        y_range = [-1.0, 1.0]
    elif "mask" in name:
        domain = {"type": "arbitrary_mask", "mask_source": "file", "mask_path": "mask.npy"}
        x_range = [-1.0, 1.0]
        y_range = [-1.0, 1.0]
    else:
        # rectangle
        domain = {"type": "rectangle"}
        x_range = [0.0, 1.0]
        y_range = [0.0, 1.0]

    return {
        "field_kind": field_kind,
        "grid": {"H": int(h), "W": int(w), "x_range": x_range, "y_range": y_range},
        "domain": domain,
    }


def _find_dataset_roots(base: Path) -> list[Path]:
    roots: list[Path] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name.startswith("offset_noise_"):
            for sub in sorted(child.iterdir()):
                if not sub.is_dir():
                    continue
                if (sub / "cond.npy").exists() and (sub / "field.npy").exists():
                    roots.append(sub)
            continue
        if (child / "cond.npy").exists() and (child / "field.npy").exists():
            roots.append(child)
    return roots


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(manifest), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write manifest.json for eval datasets (npy_dir).")
    parser.add_argument(
        "--root",
        default="data/mode_decomp_eval_dataset_v1",
        help="Base directory to scan for dataset roots.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest.json files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files.")
    args = parser.parse_args(argv)

    base = Path(args.root)
    if not base.is_absolute():
        base = (PROJECT_ROOT / base).resolve()
    if not base.exists():
        raise FileNotFoundError(f"root not found: {base}")

    dataset_roots = _find_dataset_roots(base)
    if not dataset_roots:
        raise RuntimeError(f"No dataset roots found under: {base}")

    wrote = 0
    skipped = 0
    for root in dataset_roots:
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            skipped += 1
            continue

        field = np.load(root / "field.npy", mmap_mode="r")
        mask_path = root / "mask.npy"
        mask = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None

        example_cfg_path = root / "example_config.yaml"
        if example_cfg_path.exists():
            example_cfg = _load_yaml(example_cfg_path)
            manifest = _manifest_from_example_config(root, example_cfg, field)
        else:
            manifest = _manifest_fallback(root, field)

        # Validate against current loader contract.
        validate_field_against_manifest(field, mask, manifest)

        if args.dry_run:
            print(f"[dry-run] write {manifest_path}")
        else:
            _write_manifest(manifest_path, manifest)
        wrote += 1

    print(f"dataset roots: {len(dataset_roots)}; wrote: {wrote}; skipped(existing): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

