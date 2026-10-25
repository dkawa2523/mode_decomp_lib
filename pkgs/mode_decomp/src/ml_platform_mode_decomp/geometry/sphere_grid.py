"""Utilities for configuring sphere_grid domains."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..config import integer_parameter


_DEG_UNITS = {"deg", "degree", "degrees"}
_RAD_UNITS = {"rad", "radian", "radians"}


def _normalize_unit(angle_unit: str) -> str:
    unit = str(angle_unit).strip().lower()
    if unit in _DEG_UNITS:
        return "deg"
    if unit in _RAD_UNITS:
        return "rad"
    raise ValueError("angle_unit must be deg/degree/degrees or rad/radian/radians for sphere_grid")


def sphere_grid_lon_range(n_lon: int, *, angle_unit: str = "deg") -> tuple[float, float]:
    """Return a non-overlapping longitude range for periodic grids."""
    n_lon = integer_parameter(n_lon, "n_lon")
    if n_lon < 2:
        raise ValueError("n_lon must be >= 2 for sphere_grid")
    unit = _normalize_unit(angle_unit)
    if unit == "deg":
        step = 360.0 / n_lon
        return (-180.0, 180.0 - step)
    step = (2.0 * np.pi) / n_lon
    return (-np.pi, np.pi - step)


def sphere_grid_lat_range(n_lat: int, *, angle_unit: str = "deg") -> tuple[float, float]:
    """Return a full latitude range (including poles)."""
    n_lat = integer_parameter(n_lat, "n_lat")
    if n_lat < 2:
        raise ValueError("n_lat must be >= 2 for sphere_grid")
    unit = _normalize_unit(angle_unit)
    if unit == "deg":
        return (-90.0, 90.0)
    return (-0.5 * np.pi, 0.5 * np.pi)


def sphere_grid_domain_cfg(
    n_lat: int,
    n_lon: int,
    *,
    angle_unit: str = "deg",
    radius: float = 1.0,
) -> Mapping[str, Any]:
    """Return a canonical sphere_grid domain configuration."""
    lat_range = sphere_grid_lat_range(n_lat, angle_unit=angle_unit)
    lon_range = sphere_grid_lon_range(n_lon, angle_unit=angle_unit)
    radius_value = np.asarray(radius)
    if radius_value.ndim or radius_value.dtype.kind not in "fiu" or not np.isfinite(radius_value) or radius_value <= 0:
        raise ValueError("radius must be a finite positive real number")
    return {
        "name": "sphere_grid",
        "radius": float(radius),
        "lat_range": list(lat_range),
        "lon_range": list(lon_range),
        "angle_unit": str(angle_unit),
        "n_lat": int(n_lat),
        "n_lon": int(n_lon),
        "axes": {
            "lat": {"start": lat_range[0], "step": (lat_range[1] - lat_range[0]) / (n_lat - 1), "count": n_lat},
            "lon": {
                "start": lon_range[0],
                "step": (360.0 if _normalize_unit(angle_unit) == "deg" else 2 * np.pi) / n_lon,
                "count": n_lon,
                "periodic": True,
            },
        },
    }


def fill_sphere_grid_ranges(domain_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing lat/lon ranges from n_lat/n_lon when possible."""
    if not isinstance(domain_cfg, Mapping):
        raise ValueError("domain_cfg must be a mapping")
    cfg = dict(domain_cfg)
    selectors = [cfg[key] for key in ("kind", "name", "type") if key in cfg]
    if len(selectors) > 1 and any(value != selectors[0] for value in selectors[1:]):
        raise ValueError("Sphere geometry selectors must identify the same domain")
    name = str(cfg.get("kind", cfg.get("name", cfg.get("type", "")))).strip()
    if name != "sphere_grid":
        return cfg
    cfg.pop("kind", None)
    cfg.pop("type", None)
    cfg["name"] = name
    angle_unit = cfg.get("angle_unit", "deg")
    lat_range = cfg.get("lat_range", None)
    lon_range = cfg.get("lon_range", None)
    n_lat = cfg.get("n_lat", None)
    n_lon = cfg.get("n_lon", None)
    if lat_range is None and n_lat is not None:
        cfg["lat_range"] = list(sphere_grid_lat_range(n_lat, angle_unit=angle_unit))
    if lon_range is None and n_lon is not None:
        cfg["lon_range"] = list(sphere_grid_lon_range(n_lon, angle_unit=angle_unit))
        axes = dict(cfg.get("axes", {}))
        if "lon" not in axes:
            start, _ = cfg["lon_range"]
            period = 360.0 if _normalize_unit(angle_unit) == "deg" else 2 * np.pi
            axes["lon"] = {"start": start, "step": period / n_lon, "count": n_lon, "periodic": True}
        cfg["axes"] = axes
    return cfg
