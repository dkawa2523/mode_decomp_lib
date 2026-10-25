"""Domain specification and coordinate generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping

import numpy as np

from ml_platform_mode_decomp.config import cfg_get

_DISK_POLICIES = {"error", "mask_zero_fill"}


@dataclass(frozen=True)
class DomainSpec:
    name: str
    coords: Dict[str, np.ndarray]
    mask: np.ndarray | None
    weights: np.ndarray | None
    meta: Dict[str, Any]
    _mass_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.mask is not None:
            _domain_mask(self)

    @property
    def grid_shape(self) -> tuple[int, int]:
        if self.name == "mesh":
            vertex_count = None
            if self.meta:
                vertex_count = self.meta.get("vertex_count")
            if vertex_count is None:
                vertices = self.coords.get("vertices")
                if vertices is not None and hasattr(vertices, "shape"):
                    vertex_count = vertices.shape[0]
            if vertex_count is None:
                raise ValueError("mesh domain is missing vertex count")
            return int(vertex_count), 1
        if not self.coords:
            raise ValueError("coords is empty")
        sample = next(iter(self.coords.values()))
        if sample.ndim < 2:
            raise ValueError(f"coords must be 2D for grid domains, got shape {sample.shape}")
        return int(sample.shape[0]), int(sample.shape[1])

    def integration_weights(self) -> np.ndarray | None:
        active = _domain_mask(self)
        if self.weights is None:
            return None
        weights = np.asarray(self.weights)
        expected = (self.grid_shape[0],) if self.name == "mesh" else self.grid_shape
        if self.name == "mesh" and weights.shape == self.grid_shape:
            weights = weights[:, 0]
        if weights.shape != expected or weights.dtype.kind not in "fiu":
            raise ValueError("Geometry weights must be real numeric values matching the spatial shape")
        if active is None:
            if not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("Active geometry weights must be finite and non-negative")
            return weights
        if not np.isfinite(weights).all(where=active) or np.any(weights < 0, where=active):
            raise ValueError("Active geometry weights must be finite and non-negative")
        return np.where(active, weights, 0.0)

    def mass_matrix(self) -> np.ndarray | None:
        return self._mass_matrix

    def radial_fit_weights(self) -> np.ndarray:
        """Explicit optional radial emphasis, distinct from physical integration area."""
        if self.name not in {"disk", "annulus"}:
            raise ValueError("Radial fit weights require disk or annulus geometry.")
        weights = self.integration_weights()
        return np.asarray(weights) * self.coords["r"]


def _domain_mask(domain: DomainSpec) -> np.ndarray | None:
    if domain.mask is None:
        return None
    mask = np.asarray(domain.mask)
    expected = (domain.grid_shape[0],) if domain.name == "mesh" else domain.grid_shape
    if domain.name == "mesh" and mask.shape == domain.grid_shape:
        mask = mask[:, 0]
    if mask.shape != expected or mask.dtype.kind not in "bui" or not np.isin(mask, (0, 1)).all():
        raise ValueError("Geometry mask must be boolean or 0/1 and match the spatial shape")
    return mask.astype(bool, copy=False)


def _normalize_field_shape(field_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(field_shape) == 2:
        return int(field_shape[0]), int(field_shape[1])
    if len(field_shape) == 3:
        return int(field_shape[0]), int(field_shape[1])
    raise ValueError(f"field_shape must be 2D or 3D, got {field_shape}")


def _parse_range(value: Any, name: str) -> tuple[float, float]:
    values = np.asarray(value)
    if values.shape != (2,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a pair of finite real numbers")
    return float(values[0]), float(values[1])


def _finite_scalar(value: Any, name: str) -> float:
    values = np.asarray(value)
    if values.ndim != 0 or values.dtype.kind not in "fiu" or not np.isfinite(values):
        raise ValueError(f"{name} must be a finite real number")
    return float(values)


def _axis_values(domain_cfg: Mapping[str, Any], name: str, count: int, range_key: str) -> np.ndarray:
    axis = (cfg_get(domain_cfg, "axes", {}) or {}).get(name)
    if axis is None:
        lower, upper = _parse_range(cfg_get(domain_cfg, range_key), range_key)
        values = np.linspace(lower, upper, count, dtype=np.float64)
    elif not isinstance(axis, Mapping):
        raise ValueError(f"geometry.axes.{name} must be a mapping.")
    elif "values" in axis:
        if any(key in axis for key in ("start", "step", "count")):
            raise ValueError("Axis values and generated coordinates cannot be combined.")
        values = np.asarray(axis["values"])
        if values.dtype.kind not in "fiu":
            raise ValueError(f"geometry.axes.{name}.values must be real numeric coordinates")
        values = values.astype(np.float64)
    else:
        if not all(key in axis for key in ("start", "step", "count")):
            raise ValueError(f"geometry.axes.{name} needs start, step, and count.")
        if (
            isinstance(axis["count"], bool)
            or not isinstance(axis["count"], (int, np.integer))
            or axis["count"] != count
        ):
            raise ValueError(f"geometry.axes.{name}.count must match the field dimension.")
        values = _finite_scalar(axis["start"], f"geometry.axes.{name}.start") + _finite_scalar(
            axis["step"], f"geometry.axes.{name}.step"
        ) * np.arange(count, dtype=np.float64)
    if values.shape != (count,) or count < 2 or not np.isfinite(values).all() or not (np.diff(values) > 0).all():
        raise ValueError(f"Axis {name} needs at least two finite, strictly increasing coordinates matching the field.")
    return values


def _quadrature_weights(values: np.ndarray, *, periodic: bool = False) -> np.ndarray:
    differences = np.diff(values)
    if periodic:
        if not np.allclose(differences, differences[0], rtol=1e-6, atol=1e-12):
            raise ValueError("Periodic generated quadrature requires equally spaced coordinates.")
        return np.full(len(values), differences[0], dtype=np.float64)
    weights = np.empty(len(values), dtype=np.float64)
    weights[0], weights[-1] = differences[0] / 2, differences[-1] / 2
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2
    return weights


def _periodic(domain_cfg: Mapping[str, Any], name: str) -> bool:
    value = ((cfg_get(domain_cfg, "axes", {}) or {}).get(name) or {}).get("periodic", False)
    if not isinstance(value, bool):
        raise ValueError(f"geometry.axes.{name}.periodic must be boolean.")
    return value


def _build_grid(domain_cfg: Mapping[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x = _axis_values(domain_cfg, "x", width, "x_range")
    y = _axis_values(domain_cfg, "y", height, "y_range")
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy


def _normalize_domain_mask(mask: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {mask_arr.shape}")
    if mask_arr.shape != shape:
        raise ValueError(f"{label} shape {mask_arr.shape} does not match {shape}")
    if mask_arr.dtype.kind not in "bui" or not np.isin(mask_arr, (0, 1)).all():
        raise ValueError(f"{label} must contain boolean or 0/1 values")
    return mask_arr.astype(bool, copy=False)


def _build_domain_spec(
    domain_cfg: Mapping[str, Any],
    field_shape: tuple[int, ...],
) -> DomainSpec:
    if domain_cfg is None:
        raise ValueError("domain_cfg is required")
    name = str(cfg_get(domain_cfg, "name", "")).strip()
    if not name:
        raise ValueError("domain.name is required")

    height, width = _normalize_field_shape(field_shape)

    if name == "mesh":
        from .mesh import mesh_measure, mesh_inputs

        vertices, faces, mask, mesh_meta = mesh_inputs(domain_cfg)
        faces, mask, weights = mesh_measure(vertices, faces, mask)
        if width != 1:
            raise ValueError("mesh field_shape must have width=1")
        if height != vertices.shape[0]:
            raise ValueError(f"mesh field_shape height {height} does not match vertices {vertices.shape[0]}")
        meta = dict(mesh_meta)
        meta.update(
            {
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
                "vertex_dim": int(vertices.shape[1]),
            }
        )
        boundary_condition = cfg_get(domain_cfg, "boundary_condition", None)
        if boundary_condition is not None:
            meta["boundary_condition"] = str(boundary_condition)
        if mask is not None:
            meta["mask_shape"] = (int(mask.shape[0]),)
        coords = {"vertices": vertices, "faces": faces}
        meta["weight_source"] = "vertex_area"
        mass_matrix_cfg = cfg_get(domain_cfg, "mass_matrix", False)
        mass_matrix = None
        if mass_matrix_cfg is not False and mass_matrix_cfg is not None:
            raise ValueError("domain.mass_matrix is retired: numerical methods use the linear-size integration weights")
        return DomainSpec(
            name=name,
            coords=coords,
            mask=mask,
            weights=weights,
            meta=meta,
            _mass_matrix=mass_matrix,
        )

    if name == "sphere_grid":
        lat_range = cfg_get(domain_cfg, "lat_range", None)
        lon_range = cfg_get(domain_cfg, "lon_range", None)
        if lat_range is None:
            lat_range = cfg_get(domain_cfg, "y_range", None)
        if lon_range is None:
            lon_range = cfg_get(domain_cfg, "x_range", None)
        axes = cfg_get(domain_cfg, "axes", {}) or {}
        if "lat" in axes:
            lat_range = (0.0, 1.0)  # Actual values are read from the explicit axis below.
        if "lon" in axes:
            lon_range = (0.0, 1.0)
        lat_min, lat_max = _parse_range(lat_range, "lat_range")
        lon_min, lon_max = _parse_range(lon_range, "lon_range")
        angle_unit = str(cfg_get(domain_cfg, "angle_unit", "deg")).strip().lower() or "deg"
        if angle_unit not in {"deg", "degree", "degrees", "rad", "radian", "radians"}:
            raise ValueError("domain.angle_unit must be deg/degree/degrees or rad/radian/radians for sphere_grid")
        n_lat = cfg_get(domain_cfg, "n_lat", None)
        n_lon = cfg_get(domain_cfg, "n_lon", None)
        if n_lat is not None and (
            isinstance(n_lat, bool) or not isinstance(n_lat, (int, np.integer)) or n_lat != height
        ):
            raise ValueError(f"domain.n_lat {n_lat} does not match field height {height}")
        if n_lon is not None and (
            isinstance(n_lon, bool) or not isinstance(n_lon, (int, np.integer)) or n_lon != width
        ):
            raise ValueError(f"domain.n_lon {n_lon} does not match field width {width}")
        if height < 2 or width < 2:
            raise ValueError("sphere_grid requires field_shape with height/width >= 2")
        radius = _finite_scalar(cfg_get(domain_cfg, "radius", 1.0), "geometry.radius")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("domain.radius must be > 0 for sphere_grid")

        explicit_config = {**dict(domain_cfg), "lat_range": (lat_min, lat_max), "lon_range": (lon_min, lon_max)}
        lat_vals = _axis_values(explicit_config, "lat", height, "lat_range")
        lon_vals = _axis_values(explicit_config, "lon", width, "lon_range")
        lat_min, lat_max = float(lat_vals[0]), float(lat_vals[-1])
        lon_min, lon_max = float(lon_vals[0]), float(lon_vals[-1])
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals, indexing="xy")
        if angle_unit in {"deg", "degree", "degrees"}:
            lat_rad = np.deg2rad(lat_grid.astype(np.float64))
            lon_rad = np.deg2rad(lon_grid.astype(np.float64))
            lat_deg = lat_grid
            lon_deg = lon_grid
            lat_vals_rad = np.deg2rad(lat_vals.astype(np.float64))
            lon_vals_rad = np.deg2rad(lon_vals.astype(np.float64))
            angle_unit_norm = "deg"
        else:
            lat_rad = lat_grid.astype(np.float64)
            lon_rad = lon_grid.astype(np.float64)
            lat_deg = np.rad2deg(lat_grid.astype(np.float64)).astype(np.float32)
            lon_deg = np.rad2deg(lon_grid.astype(np.float64)).astype(np.float32)
            lat_vals_rad = lat_vals.astype(np.float64)
            lon_vals_rad = lon_vals.astype(np.float64)
            angle_unit_norm = "rad"

        theta = lon_rad.astype(np.float64)
        phi = (0.5 * np.pi - lat_rad).astype(np.float64)
        if lat_vals_rad[0] < -np.pi / 2 - 1e-10 or lat_vals_rad[-1] > np.pi / 2 + 1e-10:
            raise ValueError("Sphere latitude must be between the poles.")
        periodic_lon = _periodic(domain_cfg, "lon")
        if periodic_lon and np.isclose(lon_vals_rad[-1] - lon_vals_rad[0], 2 * np.pi):
            raise ValueError("Periodic longitude must not repeat its endpoint.")
        if periodic_lon and not np.isclose(lon_vals_rad[-1] - lon_vals_rad[0] + np.diff(lon_vals_rad)[0], 2 * np.pi):
            raise ValueError("Periodic longitude must cover one full turn without a repeated endpoint.")
        if _periodic(domain_cfg, "lat"):
            raise ValueError("Sphere latitude cannot be periodic across the poles.")
        latitude_weights = _quadrature_weights(lat_vals_rad)
        longitude_weights = _quadrature_weights(lon_vals_rad, periodic=periodic_lon)
        weights = radius**2 * np.maximum(np.sin(phi), 0.0) * latitude_weights[:, None] * longitude_weights[None, :]

        coords = {
            "lat": lat_rad.astype(np.float64),
            "lon": lon_rad.astype(np.float64),
            "theta": theta,
            "phi": phi,
            "lat_deg": lat_deg.astype(np.float32),
            "lon_deg": lon_deg.astype(np.float32),
        }
        meta = {
            "lat_range": (float(lat_min), float(lat_max)),
            "lon_range": (float(lon_min), float(lon_max)),
            "angle_unit": angle_unit_norm,
            "radius": radius,
            "n_lat": int(height),
            "n_lon": int(width),
        }
        return DomainSpec(name=name, coords=coords, mask=None, weights=weights, meta=meta)

    xx, yy = _build_grid(domain_cfg, height, width)
    area_weights = np.outer(
        _quadrature_weights(yy[:, 0], periodic=_periodic(domain_cfg, "y")),
        _quadrature_weights(xx[0], periodic=_periodic(domain_cfg, "x")),
    )

    if name == "rectangle":
        weights = area_weights.copy()
        coords = {"x": xx, "y": yy}
        meta = {
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        return DomainSpec(name=name, coords=coords, mask=None, weights=weights, meta=meta)

    if name == "disk":
        center = cfg_get(domain_cfg, "center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("domain.center must be a pair for disk")
        cx, cy = _parse_range(center, "geometry.center")
        radius = _finite_scalar(cfg_get(domain_cfg, "radius", 0.0), "geometry.radius")
        if radius <= 0:
            raise ValueError("domain.radius must be > 0 for disk")
        radius_f = np.float32(radius)

        x_shift = xx - cx
        y_shift = yy - cy
        r = np.sqrt(x_shift**2 + y_shift**2).astype(np.float32) / radius_f
        theta = np.arctan2(y_shift, x_shift).astype(np.float32)
        mask = r <= 1.0
        # Cartesian grid cells measure dx dy; radial emphasis is an explicit fit policy.
        weights = area_weights.copy()
        weights[~mask] = 0.0
        coords = {"x": xx, "y": yy, "r": r.astype(np.float32), "theta": theta}
        meta = {
            "center": (cx, cy),
            "radius": radius,
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        return DomainSpec(name=name, coords=coords, mask=mask, weights=weights, meta=meta)

    if name == "annulus":
        center = cfg_get(domain_cfg, "center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("domain.center must be a pair for annulus")
        cx, cy = _parse_range(center, "geometry.center")
        r_inner = _finite_scalar(cfg_get(domain_cfg, "r_inner", 0.0), "geometry.r_inner")
        r_outer = _finite_scalar(cfg_get(domain_cfg, "r_outer", 0.0), "geometry.r_outer")
        if r_outer <= 0:
            raise ValueError("domain.r_outer must be > 0 for annulus")
        if r_inner < 0:
            raise ValueError("domain.r_inner must be >= 0 for annulus")
        if r_inner >= r_outer:
            raise ValueError("domain.r_inner must be < r_outer for annulus")

        x_shift = xx - cx
        y_shift = yy - cy
        r = np.sqrt(x_shift**2 + y_shift**2).astype(np.float32) / np.float32(r_outer)
        theta = np.arctan2(y_shift, x_shift).astype(np.float32)
        r_inner_norm = r_inner / r_outer
        mask = (r >= r_inner_norm) & (r <= 1.0)
        weights = area_weights.copy()
        weights[~mask] = 0.0
        coords = {"x": xx, "y": yy, "r": r.astype(np.float32), "theta": theta}
        meta = {
            "center": (cx, cy),
            "r_inner": r_inner,
            "r_outer": r_outer,
            "r_inner_norm": r_inner_norm,
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        return DomainSpec(name=name, coords=coords, mask=mask, weights=weights, meta=meta)

    if name in {"arbitrary_mask", "mask"}:
        if (
            cfg_get(domain_cfg, "mask_path", None) is not None
            or cfg_get(domain_cfg, "mask_source", "inline") != "inline"
        ):
            raise ValueError("Numerical mask geometry requires an inline mask; resolve files through data loading.")
        mask = _normalize_domain_mask(cfg_get(domain_cfg, "mask", None), (height, width), "mask")
        weights = np.where(mask, area_weights, 0.0)
        coords = {"x": xx, "y": yy}
        meta = {
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        return DomainSpec(name=name, coords=coords, mask=mask, weights=weights, meta=meta)

    raise NotImplementedError(f"Unsupported domain: {name}")


def build_domain_spec(domain_cfg: Mapping[str, Any], field_shape: tuple[int, ...]) -> DomainSpec:
    """Build numerical geometry from explicit in-memory axes or the legacy range form."""
    if not isinstance(domain_cfg, Mapping):
        raise ValueError("geometry must be a mapping.")
    config = dict(domain_cfg)
    if "kind" in config and "name" in config:
        raise ValueError("Specify geometry.kind or geometry.name, not both")
    config["name"] = config.pop("kind", config.get("name", ""))
    name = config["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("geometry.name must be a non-empty string")
    config["name"] = name = name.strip()
    metadata_keys = {
        "units",
        "coordinate_units",
        "field_units",
        "boundary_condition",
        "component_basis",
        "fit_weight_policy",
    }
    grid_keys = {"name", "axes", "x_range", "y_range"}
    domain_keys = {
        "rectangle": grid_keys,
        "disk": grid_keys | {"center", "radius"},
        "annulus": grid_keys | {"center", "r_inner", "r_outer"},
        "mask": grid_keys | {"mask", "mask_source", "mask_path"},
        "arbitrary_mask": grid_keys | {"mask", "mask_source", "mask_path"},
        "sphere_grid": grid_keys | {"lat_range", "lon_range", "n_lat", "n_lon", "radius", "angle_unit"},
        "mesh": {"name", "vertices", "faces", "mask", "vertex_mask", "mass_matrix"},
    }
    if name in domain_keys and (unknown := set(config) - domain_keys[name] - metadata_keys):
        raise ValueError(f"Unknown {name} geometry settings: {sorted(unknown, key=str)}")
    axes = config.get("axes", {})
    if not isinstance(axes, Mapping):
        raise ValueError("geometry.axes must be a mapping")
    axis_names = {"lat", "lon"} if name == "sphere_grid" else {"x", "y"}
    if set(axes) - axis_names:
        raise ValueError(f"geometry.axes for {name} must use {sorted(axis_names)}")
    if name == "sphere_grid":
        for canonical, alias in (("lat_range", "y_range"), ("lon_range", "x_range")):
            if alias in config:
                if canonical in config and not np.allclose(
                    _parse_range(config[canonical], canonical), _parse_range(config[alias], alias)
                ):
                    raise ValueError(f"Conflicting geometry {canonical} and {alias}")
                config[canonical] = config.pop(alias)
    for axis_name, axis in axes.items():
        if not isinstance(axis, Mapping) or set(axis) - {"values", "start", "step", "count", "periodic"}:
            raise ValueError(f"Unknown or invalid geometry.axes.{axis_name} settings")
        if f"{axis_name}_range" in config:
            count = field_shape[1] if axis_name in {"x", "lon"} else field_shape[0]
            values = _axis_values(config, axis_name, count, f"{axis_name}_range")
            bounds = _parse_range(config[f"{axis_name}_range"], f"{axis_name}_range")
            if not np.allclose((values[0], values[-1]), bounds, rtol=1e-12, atol=1e-12):
                raise ValueError(f"geometry.axes.{axis_name} disagrees with {axis_name}_range")
            config.pop(f"{axis_name}_range")
    result = _build_domain_spec(config, field_shape)
    metadata = {**result.meta, "weight_measure": "physical_area"}
    for key in (
        "units",
        "coordinate_units",
        "field_units",
        "boundary_condition",
        "component_basis",
        "fit_weight_policy",
    ):
        if key in config:
            metadata[key] = config[key]
    if result.name != "mesh":
        axes = config.get("axes", {}) or {}
        metadata["periodic_axes"] = sorted(name for name in axes if _periodic(config, name))
    return replace(result, meta=metadata)


def validate_decomposer_compatibility(
    domain_spec: DomainSpec,
    decomposer_cfg: Mapping[str, Any],
) -> None:
    if decomposer_cfg is None:
        raise ValueError("decomposer_cfg is required")
    method = str(cfg_get(decomposer_cfg, "name", "")).strip()
    if not method:
        raise ValueError("decompose.name is required")

    allowed_domains_by_method: dict[str, set[str] | None] = {
        # Disk / annulus special functions.
        "zernike": {"disk"},
        "pseudo_zernike": {"disk"},
        "annular_zernike": {"annulus"},
        "fourier_bessel": {"disk"},
        "fourier_jacobi": {"disk"},
        "polar_fft": {"disk", "annulus"},
        "disk_slepian": {"disk"},
        # Grid / mask-compatible.
        "rbf_expansion": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        "pod_joint": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        "pod_em": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        "pod_joint_em": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        "gappy_graph_fourier": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        # FFT/DCT family is broadly applicable, but disk needs a policy override.
        "fft2": None,
        "dct2": None,
        "fft2_lowpass": None,
        # Mesh / vector physics.
        "laplace_beltrami": {"mesh"},
        "helmholtz": {"rectangle", "arbitrary_mask", "mask"},
        "hodge": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask"},
        # Rectangle-only.
        "pswf2d_tensor": {"rectangle"},
        "tucker": {"rectangle"},
        "quadratic_manifold": {"rectangle", "disk", "annulus", "arbitrary_mask", "mask", "sphere_grid", "mesh"},
        # Wavelet can operate on masked grids (with caveats).
        "wavelet2d": {"rectangle", "disk", "arbitrary_mask", "mask"},
        # Sphere-grid.
        "spherical_harmonics": {"sphere_grid"},
        "spherical_slepian": {"sphere_grid"},
    }

    allowed = allowed_domains_by_method.get(method)
    if allowed is not None and domain_spec.name not in allowed:
        msg_by_method = {
            "zernike": "zernike requires disk domain",
            "pseudo_zernike": "pseudo_zernike requires disk domain",
            "annular_zernike": "annular_zernike requires annulus domain",
            "fourier_bessel": "fourier_bessel requires disk domain",
            "fourier_jacobi": "fourier_jacobi requires disk domain",
            "polar_fft": "polar_fft requires disk or annulus domain",
            "disk_slepian": "disk_slepian requires disk domain",
            "rbf_expansion": "rbf_expansion requires a grid domain with x/y coords",
            "pod_joint": "pod_joint requires a grid domain",
            "pod_em": "pod_em requires a grid domain",
            "pod_joint_em": "pod_joint_em requires a grid domain",
            "gappy_graph_fourier": "gappy_graph_fourier requires a grid domain",
            "laplace_beltrami": "laplace_beltrami requires mesh domain",
            "helmholtz": "helmholtz requires rectangle or mask domain",
            "hodge": "hodge requires a masked Cartesian grid domain",
            "wavelet2d": "wavelet2d requires rectangle or mask-compatible domain",
            "pswf2d_tensor": "pswf2d_tensor requires rectangle domain",
            "tucker": "tucker requires a rectangular tensor-product grid",
            "spherical_harmonics": "spherical_harmonics requires sphere_grid domain",
            "spherical_slepian": "spherical_slepian requires sphere_grid domain",
        }
        raise ValueError(msg_by_method.get(method, f"{method} is incompatible with domain {domain_spec.name}"))

    if method in {"fft2", "dct2", "fft2_lowpass"} and domain_spec.name == "disk":
        policy = str(cfg_get(decomposer_cfg, "disk_policy", "")).strip()
        if not policy:
            raise ValueError("decompose.disk_policy is required for disk domain")
        if policy not in _DISK_POLICIES:
            raise ValueError(f"decompose.disk_policy must be one of {_DISK_POLICIES}, got {policy}")
        if policy == "error":
            raise ValueError(f"{method} does not allow disk domain without mask_zero_fill")

    if method == "hodge" and ("x" not in domain_spec.coords or "y" not in domain_spec.coords):
        raise ValueError("hodge requires Cartesian coordinates x/y")

    if method in {"fft2", "dct2", "fft2_lowpass", "pswf2d_tensor", "helmholtz"}:
        if "x" not in domain_spec.coords or "y" not in domain_spec.coords:
            raise ValueError(f"{method} requires a Cartesian grid.")
        for name, values in (("x", domain_spec.coords["x"][0]), ("y", domain_spec.coords["y"][:, 0])):
            spacing = np.diff(values)
            if not spacing.size:
                if method in {"pswf2d_tensor", "helmholtz"}:
                    raise ValueError(f"{method} requires at least two {name} coordinates.")
                continue
            if not np.allclose(spacing, spacing[0], rtol=1e-6, atol=1e-12):
                raise ValueError(f"{method} requires equally spaced {name} coordinates.")
