"""Domain specification and coordinate generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

_DISK_POLICIES = {"error", "mask_zero_fill"}
_MASK_SOURCES = {"dataset", "file", "inline"}


@dataclass(frozen=True)
class DomainSpec:
    name: str
    coords: Dict[str, np.ndarray]
    mask: np.ndarray | None
    weights: np.ndarray | None
    meta: Dict[str, Any]

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


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _normalize_field_shape(field_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(field_shape) == 2:
        return int(field_shape[0]), int(field_shape[1])
    if len(field_shape) == 3:
        return int(field_shape[0]), int(field_shape[1])
    raise ValueError(f"field_shape must be 2D or 3D, got {field_shape}")


def _parse_range(value: Any, name: str) -> tuple[float, float]:
    if value is None:
        raise ValueError(f"{name} is required in domain config")
    if hasattr(value, "__len__") and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"{name} must be a pair, got {value}")


def _build_grid(domain_cfg: Mapping[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = _parse_range(_cfg_get(domain_cfg, "x_range"), "x_range")
    y_min, y_max = _parse_range(_cfg_get(domain_cfg, "y_range"), "y_range")
    x = np.linspace(x_min, x_max, width, dtype=np.float32)
    y = np.linspace(y_min, y_max, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx, yy


def _normalize_domain_mask(mask: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {mask_arr.shape}")
    if mask_arr.shape != shape:
        raise ValueError(f"{label} shape {mask_arr.shape} does not match {shape}")
    return mask_arr.astype(bool)


def _load_mask_path(mask_path: Any, shape: tuple[int, int]) -> np.ndarray:
    path = Path(str(mask_path))
    if not path.exists():
        raise FileNotFoundError(f"mask_path not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()
        raise ValueError("mask_path must point to a .npy array, not a .npz archive")
    return _normalize_domain_mask(loaded, shape, "mask")


def build_domain_spec(
    domain_cfg: Mapping[str, Any],
    field_shape: tuple[int, ...],
) -> DomainSpec:
    if domain_cfg is None:
        raise ValueError("domain_cfg is required")
    name = str(_cfg_get(domain_cfg, "name", "")).strip()
    if not name:
        raise ValueError("domain.name is required")

    height, width = _normalize_field_shape(field_shape)

    if name == "mesh":
        from .mesh import load_mesh_inputs

        vertices, faces, mask, mesh_meta = load_mesh_inputs(domain_cfg)
        if width != 1:
            raise ValueError("mesh field_shape must have width=1")
        if height != vertices.shape[0]:
            raise ValueError(
                f"mesh field_shape height {height} does not match vertices {vertices.shape[0]}"
            )
        meta = dict(mesh_meta)
        meta.update(
            {
                "vertex_count": int(vertices.shape[0]),
                "face_count": int(faces.shape[0]),
                "vertex_dim": int(vertices.shape[1]),
            }
        )
        boundary_condition = _cfg_get(domain_cfg, "boundary_condition", None)
        if boundary_condition is not None:
            meta["boundary_condition"] = str(boundary_condition)
        if mask is not None:
            meta["mask_shape"] = (int(mask.shape[0]),)
        coords = {"vertices": vertices, "faces": faces}
        return DomainSpec(name=name, coords=coords, mask=mask, weights=None, meta=meta)

    xx, yy = _build_grid(domain_cfg, height, width)
    dx = float((xx[0, -1] - xx[0, 0]) / (max(width - 1, 1)))
    dy = float((yy[-1, 0] - yy[0, 0]) / (max(height - 1, 1)))

    if name == "rectangle":
        weights = np.full((height, width), dx * dy, dtype=np.float32)
        coords = {"x": xx, "y": yy}
        meta = {
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        return DomainSpec(name=name, coords=coords, mask=None, weights=weights, meta=meta)

    if name == "disk":
        center = _cfg_get(domain_cfg, "center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("domain.center must be a pair for disk")
        cx, cy = float(center[0]), float(center[1])
        radius = float(_cfg_get(domain_cfg, "radius", 0.0))
        if radius <= 0:
            raise ValueError("domain.radius must be > 0 for disk")
        radius_f = np.float32(radius)

        x_shift = xx - cx
        y_shift = yy - cy
        r = np.sqrt(x_shift**2 + y_shift**2).astype(np.float32) / radius_f
        theta = np.arctan2(y_shift, x_shift).astype(np.float32)
        mask = r <= 1.0
        # CONTRACT: disk weights include radial factor for polar inner-products.
        weights = (r * dx * dy).astype(np.float32)
        weights[~mask] = 0.0
        coords = {"x": xx, "y": yy, "r": r.astype(np.float32), "theta": theta}
        meta = {
            "center": (cx, cy),
            "radius": radius,
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
        }
        if mask.any():
            r_max = float(r[mask].max())
            if r_max > 1.0 + 1e-6:
                raise ValueError(f"disk r must be <= 1, got max {r_max}")
        return DomainSpec(name=name, coords=coords, mask=mask, weights=weights, meta=meta)

    if name in {"arbitrary_mask", "mask"}:
        mask_source = str(_cfg_get(domain_cfg, "mask_source", "")).strip()
        mask_inline = _cfg_get(domain_cfg, "mask", None)
        mask_path = _cfg_get(domain_cfg, "mask_path", None)

        if mask_source:
            if mask_source not in _MASK_SOURCES:
                raise ValueError(f"mask_source must be one of {_MASK_SOURCES}, got {mask_source}")
            if mask_source == "dataset":
                if mask_inline is not None or mask_path is not None:
                    raise ValueError("mask_source=dataset cannot be combined with mask or mask_path")
                mask = None
            elif mask_source == "inline":
                if mask_inline is None or mask_path is not None:
                    raise ValueError("mask_source=inline requires mask and forbids mask_path")
                mask = _normalize_domain_mask(mask_inline, (height, width), "mask")
            else:  # mask_source == "file"
                if mask_path is None or mask_inline is not None:
                    raise ValueError("mask_source=file requires mask_path and forbids mask")
                mask = _load_mask_path(mask_path, (height, width))
            resolved_source = mask_source
        else:
            if mask_inline is not None and mask_path is not None:
                raise ValueError("Provide only one of mask or mask_path for arbitrary_mask")
            if mask_inline is not None:
                mask = _normalize_domain_mask(mask_inline, (height, width), "mask")
                resolved_source = "inline"
            elif mask_path is not None:
                mask = _load_mask_path(mask_path, (height, width))
                resolved_source = "file"
            else:
                raise ValueError("arbitrary_mask requires mask_source or mask/mask_path")

        weights = np.full((height, width), dx * dy, dtype=np.float32)
        if mask is not None:
            weights[~mask] = 0.0
        coords = {"x": xx, "y": yy}
        meta = {
            "x_range": (float(xx[0, 0]), float(xx[0, -1])),
            "y_range": (float(yy[0, 0]), float(yy[-1, 0])),
            "mask_source": resolved_source,
        }
        if mask_path is not None:
            meta["mask_path"] = str(Path(mask_path))
        if mask is not None:
            meta["mask_shape"] = (int(mask.shape[0]), int(mask.shape[1]))
        return DomainSpec(name=name, coords=coords, mask=mask, weights=weights, meta=meta)

    raise NotImplementedError(f"Unsupported domain: {name}")


def validate_decomposer_compatibility(
    domain_spec: DomainSpec,
    decomposer_cfg: Mapping[str, Any],
) -> None:
    if decomposer_cfg is None:
        raise ValueError("decomposer_cfg is required")
    method = str(_cfg_get(decomposer_cfg, "name", "")).strip()
    if not method:
        raise ValueError("decompose.name is required")

    if method == "zernike" and domain_spec.name != "disk":
        raise ValueError("zernike requires disk domain")

    if method == "fourier_bessel" and domain_spec.name != "disk":
        raise ValueError("fourier_bessel requires disk domain")

    if method in {"fft2", "dct2"} and domain_spec.name == "disk":
        policy = str(_cfg_get(decomposer_cfg, "disk_policy", "")).strip()
        if not policy:
            raise ValueError("decompose.disk_policy is required for disk domain")
        if policy not in _DISK_POLICIES:
            raise ValueError(f"decompose.disk_policy must be one of {_DISK_POLICIES}, got {policy}")
        if policy == "error":
            raise ValueError(f"{method} does not allow disk domain without mask_zero_fill")

    if method == "laplace_beltrami" and domain_spec.name != "mesh":
        raise ValueError("laplace_beltrami requires mesh domain")

    if method == "helmholtz" and domain_spec.name not in {"rectangle", "arbitrary_mask", "mask"}:
        raise ValueError("helmholtz requires rectangle or mask domain")
