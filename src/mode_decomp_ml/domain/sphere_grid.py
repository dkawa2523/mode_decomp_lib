"""Utilities for configuring sphere_grid domains."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


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
    n_lon = int(n_lon)
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
    n_lat = int(n_lat)
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
    return {
        "name": "sphere_grid",
        "radius": float(radius),
        "lat_range": list(lat_range),
        "lon_range": list(lon_range),
        "angle_unit": str(angle_unit),
        "n_lat": int(n_lat),
        "n_lon": int(n_lon),
    }


def fill_sphere_grid_ranges(domain_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fill missing lat/lon ranges from n_lat/n_lon when possible."""
    if not isinstance(domain_cfg, Mapping):
        raise ValueError("domain_cfg must be a mapping")
    cfg = dict(domain_cfg)
    name = str(cfg.get("name", cfg.get("type", ""))).strip()
    if name != "sphere_grid":
        return cfg
    angle_unit = cfg.get("angle_unit", "deg")
    lat_range = cfg.get("lat_range", None)
    lon_range = cfg.get("lon_range", None)
    n_lat = cfg.get("n_lat", None)
    n_lon = cfg.get("n_lon", None)
    if lat_range is None and n_lat is not None:
        cfg["lat_range"] = list(sphere_grid_lat_range(int(n_lat), angle_unit=angle_unit))
    if lon_range is None and n_lon is not None:
        cfg["lon_range"] = list(sphere_grid_lon_range(int(n_lon), angle_unit=angle_unit))
    return cfg
