"""Dataset manifest loading and validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.domain.sphere_grid import sphere_grid_lat_range, sphere_grid_lon_range
_VALID_FIELD_KINDS = {"scalar": 1, "vector": 2}
_VALID_MASK_SOURCES = {"dataset", "file", "inline"}
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_root(root: str | Path) -> Path:
    path = Path(str(root))
    return path if path.is_absolute() else _PROJECT_ROOT / path


def load_manifest(root: str | Path, *, required: bool = False) -> dict[str, Any] | None:
    root_path = resolve_root(root)
    manifest_path = root_path / "manifest.json"
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError(f"manifest.json not found in dataset root: {root_path}")
        return None
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, Mapping):
        raise ValueError("manifest.json must be a JSON object")
    return _normalize_manifest(dict(data))


def manifest_domain_cfg(manifest: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    domain = _normalize_domain(manifest.get("domain"))
    domain_type = str(domain["type"])
    grid = _normalize_grid(manifest.get("grid"))
    root_path = resolve_root(root)
    cfg = {k: v for k, v in domain.items() if k not in {"type", "name"}}
    cfg["name"] = domain_type
    if domain_type != "mesh":
        cfg.setdefault("x_range", grid["x_range"])
        cfg.setdefault("y_range", grid["y_range"])
    if domain_type in {"mask", "arbitrary_mask"}:
        _apply_mask_source(cfg, domain, root_path, grid)
    return cfg


def validate_field_against_manifest(
    field: np.ndarray,
    mask: np.ndarray | None,
    manifest: Mapping[str, Any],
) -> None:
    manifest_norm = _normalize_manifest(manifest)
    grid = manifest_norm["grid"]
    field_kind = manifest_norm["field_kind"]

    field_arr = np.asarray(field)
    if field_arr.ndim == 4:
        height, width, channels = field_arr.shape[1:]
    elif field_arr.ndim == 3:
        height, width, channels = field_arr.shape
    elif field_arr.ndim == 2:
        height, width = field_arr.shape
        channels = 1
    else:
        raise ValueError(f"field must be 2D, 3D, or 4D, got shape {field_arr.shape}")

    if height != grid["H"] or width != grid["W"]:
        raise ValueError(
            f"field shape {height}x{width} does not match manifest grid {grid['H']}x{grid['W']}"
        )

    expected_channels = _VALID_FIELD_KINDS[field_kind]
    if channels != expected_channels:
        raise ValueError(
            f"field channels {channels} do not match field_kind={field_kind} (expected {expected_channels})"
        )

    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 4 and mask_arr.shape[-1] == 1:
            mask_shape = mask_arr.shape[1:3]
        elif mask_arr.ndim == 3:
            mask_shape = mask_arr.shape[1:]
        elif mask_arr.ndim == 2:
            mask_shape = mask_arr.shape
        else:
            raise ValueError(f"mask must be 2D, 3D, or (N,H,W,1), got shape {mask_arr.shape}")
        if mask_shape != (grid["H"], grid["W"]):
            raise ValueError(
                f"mask shape {mask_shape} does not match manifest grid {grid['H']}x{grid['W']}"
            )


def _normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(manifest)
    out["field_kind"] = _normalize_field_kind(manifest.get("field_kind"))
    out["grid"] = _normalize_grid(manifest.get("grid"))
    out["domain"] = _normalize_domain(manifest.get("domain"))
    return out


def _normalize_field_kind(value: Any) -> str:
    if value is None:
        raise ValueError("manifest.field_kind is required")
    field_kind = str(value).strip().lower()
    if field_kind not in _VALID_FIELD_KINDS:
        raise ValueError(f"field_kind must be one of {set(_VALID_FIELD_KINDS)}, got {field_kind}")
    return field_kind


def _normalize_grid(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest.grid is required and must be an object")
    grid = dict(value)
    height = grid.get("H", None)
    width = grid.get("W", None)
    if height is None or width is None:
        raise ValueError("manifest.grid.H and manifest.grid.W are required")
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("manifest.grid.H and manifest.grid.W must be > 0")
    x_range = grid.get("x_range", None)
    y_range = grid.get("y_range", None)
    if x_range is None or y_range is None:
        raise ValueError("manifest.grid.x_range and manifest.grid.y_range are required")
    if not hasattr(x_range, "__len__") or len(x_range) != 2:
        raise ValueError("manifest.grid.x_range must be a pair")
    if not hasattr(y_range, "__len__") or len(y_range) != 2:
        raise ValueError("manifest.grid.y_range must be a pair")
    return {
        "H": height,
        "W": width,
        "x_range": (float(x_range[0]), float(x_range[1])),
        "y_range": (float(y_range[0]), float(y_range[1])),
    }


def _normalize_domain(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest.domain is required and must be an object")
    domain = dict(value)
    domain_type = domain.get("type", None)
    if not domain_type:
        domain_type = domain.get("name", None)
    if not domain_type:
        raise ValueError("manifest.domain.type is required")
    domain_type = str(domain_type).strip()
    if not domain_type:
        raise ValueError("manifest.domain.type must be non-empty")
    domain["type"] = domain_type
    domain.pop("name", None)

    if domain_type == "disk":
        center = domain.get("center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("manifest.domain.center is required for disk")
        radius = float(domain.get("radius", 0.0))
        if radius <= 0.0:
            raise ValueError("manifest.domain.radius must be > 0 for disk")
        domain["center"] = (float(center[0]), float(center[1]))
        domain["radius"] = radius

    if domain_type == "annulus":
        center = domain.get("center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("manifest.domain.center is required for annulus")
        r_inner = float(domain.get("r_inner", 0.0))
        r_outer = float(domain.get("r_outer", 0.0))
        if r_outer <= 0.0:
            raise ValueError("manifest.domain.r_outer must be > 0 for annulus")
        if r_inner < 0.0:
            raise ValueError("manifest.domain.r_inner must be >= 0 for annulus")
        if r_inner >= r_outer:
            raise ValueError("manifest.domain.r_inner must be < r_outer for annulus")
        domain["center"] = (float(center[0]), float(center[1]))
        domain["r_inner"] = r_inner
        domain["r_outer"] = r_outer

    if domain_type == "sphere_grid":
        radius = float(domain.get("radius", 1.0))
        if radius <= 0.0:
            raise ValueError("manifest.domain.radius must be > 0 for sphere_grid")
        domain["radius"] = radius
        lat_range = domain.get("lat_range", None)
        lon_range = domain.get("lon_range", None)
        n_lat = domain.get("n_lat", None)
        n_lon = domain.get("n_lon", None)
        if lat_range is not None:
            if not hasattr(lat_range, "__len__") or len(lat_range) != 2:
                raise ValueError("manifest.domain.lat_range must be a pair")
            domain["lat_range"] = (float(lat_range[0]), float(lat_range[1]))
        if lon_range is not None:
            if not hasattr(lon_range, "__len__") or len(lon_range) != 2:
                raise ValueError("manifest.domain.lon_range must be a pair")
            domain["lon_range"] = (float(lon_range[0]), float(lon_range[1]))
        angle_unit = domain.get("angle_unit", None)
        if angle_unit is not None:
            angle_unit = str(angle_unit).strip().lower()
            if angle_unit not in {"deg", "degree", "degrees", "rad", "radian", "radians"}:
                raise ValueError(
                    "manifest.domain.angle_unit must be deg/degree/degrees or rad/radian/radians"
                )
            domain["angle_unit"] = angle_unit
        if lat_range is None and n_lat is not None:
            domain["lat_range"] = sphere_grid_lat_range(int(n_lat), angle_unit=angle_unit or "deg")
        if lon_range is None and n_lon is not None:
            domain["lon_range"] = sphere_grid_lon_range(int(n_lon), angle_unit=angle_unit or "deg")
        if n_lat is not None:
            domain["n_lat"] = int(n_lat)
        if n_lon is not None:
            domain["n_lon"] = int(n_lon)

    if domain_type in {"mask", "arbitrary_mask"}:
        mask_source = domain.get("mask_source", None)
        if mask_source is not None:
            mask_source = str(mask_source).strip()
            if mask_source not in _VALID_MASK_SOURCES:
                raise ValueError(
                    f"manifest.domain.mask_source must be one of {_VALID_MASK_SOURCES}, got {mask_source}"
                )
            domain["mask_source"] = mask_source
    return domain


def _apply_mask_source(
    cfg: dict[str, Any],
    domain: Mapping[str, Any],
    root: str | Path,
    grid: Mapping[str, Any],
) -> None:
    mask_source = domain.get("mask_source", None)
    mask_inline = domain.get("mask", None)
    mask_path = domain.get("mask_path", None)
    if mask_path is None:
        mask_path = domain.get("mask_file", None)

    if mask_source is None:
        if mask_inline is not None:
            mask_source = "inline"
        elif mask_path is not None:
            mask_source = "file"
        else:
            mask_source = "dataset"

    if mask_source == "file":
        if mask_path is None:
            raise ValueError("manifest.domain.mask_path is required for mask_source=file")
        path = Path(str(mask_path))
        if not path.is_absolute():
            path = resolve_root(root) / path
        cfg["mask_path"] = str(path)
    elif mask_source == "inline":
        if mask_inline is None:
            raise ValueError("manifest.domain.mask is required for mask_source=inline")
        _validate_inline_mask(mask_inline, grid)
        cfg["mask"] = mask_inline
    else:  # dataset
        if mask_inline is not None or mask_path is not None:
            raise ValueError("manifest.domain.mask_source=dataset cannot include mask or mask_path")

    cfg["mask_source"] = mask_source


def _validate_inline_mask(mask: Any, grid: Mapping[str, Any]) -> None:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("manifest.domain.mask must be 2D for inline masks")
    expected = (int(grid["H"]), int(grid["W"]))
    if mask_arr.shape != expected:
        raise ValueError(f"manifest.domain.mask shape {mask_arr.shape} does not match grid {expected}")
