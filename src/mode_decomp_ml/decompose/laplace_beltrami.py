"""Laplace-Beltrami decomposer for mesh domains."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

from . import BaseDecomposer, register_decomposer

_LAPLACIAN_TYPES = {"cotangent"}
_MASS_TYPES = {"lumped"}
_SOLVERS = {"auto", "dense", "eigsh"}
_MASK_POLICIES = {"forbid", "allow", "require"}
_BOUNDARY_CONDITIONS = {"neumann", "none"}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _require_cfg(cfg: Mapping[str, Any], key: str) -> Any:
    value = _cfg_get(cfg, key, None)
    if value is None:
        raise ValueError(f"decompose.{key} is required for laplace_beltrami")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"decompose.{key} must be non-empty for laplace_beltrami")
    return value


def _mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha1()
    digest.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, str]:
    field = np.asarray(field)
    if field.ndim == 1:
        return field[:, None], "N"
    if field.ndim == 2:
        return field, "NC"
    if field.ndim == 3:
        if field.shape[1] != 1:
            raise ValueError(f"mesh field must have width=1 for 3D input, got {field.shape}")
        return field[:, 0, :], "N1C"
    raise ValueError(f"mesh field must be 1D, 2D, or 3D, got shape {field.shape}")


def _normalize_mask(mask: np.ndarray | None, n_vertices: int) -> np.ndarray | None:
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2 and mask_arr.shape[1] == 1:
        mask_arr = mask_arr[:, 0]
    if mask_arr.ndim != 1:
        raise ValueError(f"mesh mask must be 1D or (N,1), got shape {mask_arr.shape}")
    if mask_arr.shape[0] != n_vertices:
        raise ValueError(f"mesh mask length {mask_arr.shape[0]} does not match vertices {n_vertices}")
    return mask_arr.astype(bool)


def _triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    if v0.shape[1] == 2:
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (
            v2[:, 0] - v0[:, 0]
        )
        return 0.5 * np.abs(cross)
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _cotangent(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = np.sum(u * v, axis=1)
    if u.shape[1] == 2:
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        denom = np.abs(cross)
    else:
        cross = np.cross(u, v)
        denom = np.linalg.norm(cross, axis=1)
    if np.any(denom <= 0):
        raise ValueError("degenerate triangle detected in mesh")
    return dot / denom


def _extract_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mask is None:
        return vertices, faces, None
    mask = mask.astype(bool)
    if mask.all():
        return vertices, faces, None
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("mesh mask removed all vertices")
    mapping = np.full(mask.shape[0], -1, dtype=np.int64)
    mapping[idx] = np.arange(idx.size, dtype=np.int64)
    face_keep = mask[faces].all(axis=1)
    faces = faces[face_keep]
    if faces.size == 0:
        raise ValueError("mesh mask removed all faces")
    faces = mapping[faces]
    vertices = vertices[idx]
    return vertices, faces, idx


def _build_cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[csr_matrix, np.ndarray, int]:
    n_vertices = vertices.shape[0]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    vi, vj, vk = vertices[i], vertices[j], vertices[k]

    cot_i = _cotangent(vj - vi, vk - vi)
    cot_j = _cotangent(vk - vj, vi - vj)
    cot_k = _cotangent(vi - vk, vj - vk)

    w_ij = 0.5 * cot_k
    w_jk = 0.5 * cot_i
    w_ki = 0.5 * cot_j

    rows = np.concatenate([i, j, j, k, k, i])
    cols = np.concatenate([j, i, k, j, i, k])
    vals = np.concatenate([w_ij, w_ij, w_jk, w_jk, w_ki, w_ki])

    adjacency = coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    adjacency.sum_duplicates()
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    laplacian = diags(degrees, dtype=np.float64) - adjacency

    area = _triangle_area(vi, vj, vk)
    if np.any(area <= 0):
        raise ValueError("mesh has degenerate faces with non-positive area")
    mass = np.zeros(n_vertices, dtype=np.float64)
    third = area / 3.0
    np.add.at(mass, i, third)
    np.add.at(mass, j, third)
    np.add.at(mass, k, third)
    if np.any(mass <= 0):
        raise ValueError("mesh mass matrix has non-positive entries")

    n_edges = int(adjacency.nnz // 2)
    return laplacian.tocsr(), mass, n_edges


@register_decomposer("laplace_beltrami")
class LaplaceBeltramiDecomposer(BaseDecomposer):
    """Laplace-Beltrami eigenbasis decomposer for mesh domains."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "laplace_beltrami"
        n_modes = _require_cfg(cfg, "n_modes")
        self._n_modes = int(n_modes)
        if self._n_modes <= 0:
            raise ValueError("decompose.n_modes must be > 0 for laplace_beltrami")
        self._laplacian_type = str(_require_cfg(cfg, "laplacian_type"))
        if self._laplacian_type not in _LAPLACIAN_TYPES:
            raise ValueError(
                f"decompose.laplacian_type must be one of {_LAPLACIAN_TYPES}, got {self._laplacian_type}"
            )
        self._mass_type = str(_require_cfg(cfg, "mass_type"))
        if self._mass_type not in _MASS_TYPES:
            raise ValueError(
                f"decompose.mass_type must be one of {_MASS_TYPES}, got {self._mass_type}"
            )
        self._boundary_condition = str(_require_cfg(cfg, "boundary_condition"))
        if self._boundary_condition not in _BOUNDARY_CONDITIONS:
            raise ValueError(
                "decompose.boundary_condition must be one of "
                f"{_BOUNDARY_CONDITIONS}, got {self._boundary_condition}"
            )
        self._mask_policy = str(_require_cfg(cfg, "mask_policy"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._solver = str(_require_cfg(cfg, "solver"))
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._dense_threshold = int(_require_cfg(cfg, "dense_threshold"))
        if self._dense_threshold <= 0:
            raise ValueError("decompose.dense_threshold must be > 0 for laplace_beltrami")
        self._eigsh_tol = float(_require_cfg(cfg, "eigsh_tol"))
        if self._eigsh_tol <= 0:
            raise ValueError("decompose.eigsh_tol must be > 0 for laplace_beltrami")
        self._eigsh_maxiter = int(_require_cfg(cfg, "eigsh_maxiter"))
        if self._eigsh_maxiter <= 0:
            raise ValueError("decompose.eigsh_maxiter must be > 0 for laplace_beltrami")

        self._basis: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._mass: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._vertex_count: int | None = None
        self._face_count: int | None = None
        self._edge_count: int | None = None
        self._mesh_hash: str | None = None
        self._channels: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_layout: str | None = None
        self._solver_effective: str | None = None

    def _resolve_mask(
        self,
        domain_spec: DomainSpec,
        masks: list[np.ndarray | None] | None,
    ) -> np.ndarray | None:
        vertices = domain_spec.coords.get("vertices")
        if vertices is None:
            raise ValueError("laplace_beltrami requires mesh vertices in domain_spec")
        n_vertices = vertices.shape[0]
        domain_mask = domain_spec.mask
        mask = domain_mask
        if masks:
            sample_mask = masks[0]
            if sample_mask is None:
                if any(mask_item is not None for mask_item in masks):
                    raise ValueError("laplace_beltrami mask must be present for all samples or none")
            else:
                sample_mask = _normalize_mask(sample_mask, n_vertices)
                for mask_item in masks[1:]:
                    if mask_item is None:
                        raise ValueError("laplace_beltrami mask must be present for all samples or none")
                    if not np.array_equal(
                        sample_mask, _normalize_mask(mask_item, n_vertices)
                    ):
                        raise ValueError("laplace_beltrami requires a fixed mask across samples")
                mask = sample_mask if mask is None else (mask & sample_mask)
        if mask is None:
            if self._mask_policy == "require":
                raise ValueError("laplace_beltrami mask_policy=require needs a mask")
            return None
        mask = _normalize_mask(mask, n_vertices)
        if self._mask_policy == "forbid" and not mask.all():
            raise ValueError("laplace_beltrami mask_policy=forbid does not allow masked vertices")
        if mask.all():
            return None
        return mask

    def _ensure_basis(self, domain_spec: DomainSpec, mask: np.ndarray | None) -> None:
        if self._basis is None:
            self._build_basis(domain_spec, mask)
            return
        if self._vertex_count is None or self._face_count is None:
            raise ValueError("laplace_beltrami basis is not initialized correctly")
        vertices = domain_spec.coords.get("vertices")
        faces = domain_spec.coords.get("faces")
        if vertices is None or faces is None:
            raise ValueError("laplace_beltrami requires mesh vertices/faces in domain_spec")
        if vertices.shape[0] != self._vertex_count or faces.shape[0] != self._face_count:
            raise ValueError("laplace_beltrami mesh size does not match cached basis")
        mesh_hash = _mesh_hash(vertices, faces)
        if self._mesh_hash is not None and mesh_hash != self._mesh_hash:
            raise ValueError("laplace_beltrami mesh geometry does not match cached basis")
        if (self._mask is None) != (mask is None):
            raise ValueError("laplace_beltrami mask presence does not match cached basis")
        if mask is not None and self._mask is not None:
            if not np.array_equal(mask, self._mask):
                raise ValueError("laplace_beltrami requires the same mask used during fit")

    def _build_basis(self, domain_spec: DomainSpec, mask: np.ndarray | None) -> None:
        vertices = domain_spec.coords.get("vertices")
        faces = domain_spec.coords.get("faces")
        if vertices is None or faces is None:
            raise ValueError("laplace_beltrami requires mesh vertices/faces in domain_spec")

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        mask = _normalize_mask(mask, vertices.shape[0]) if mask is not None else None
        vertices_use, faces_use, mask_indices = _extract_submesh(vertices, faces, mask)

        laplacian, mass, n_edges = _build_cotangent_laplacian(vertices_use, faces_use)
        n_nodes = vertices_use.shape[0]
        if self._n_modes > n_nodes:
            raise ValueError("laplace_beltrami n_modes exceeds available vertices")

        solver = self._solver
        if solver == "auto":
            solver = "dense" if n_nodes <= self._dense_threshold or self._n_modes >= n_nodes else "eigsh"
        if solver == "dense":
            dense_lap = laplacian.toarray()
            dense_mass = np.diag(mass)
            eigvals, eigvecs = eigh(dense_lap, dense_mass)
        else:
            if self._n_modes >= n_nodes:
                raise ValueError("laplace_beltrami eigsh requires n_modes < n_vertices")
            eigvals, eigvecs = eigsh(
                laplacian,
                k=self._n_modes,
                M=diags(mass, dtype=np.float64),
                which="SM",
                tol=self._eigsh_tol,
                maxiter=self._eigsh_maxiter,
            )

        eigvals = np.asarray(eigvals, dtype=np.float64)
        eigvecs = np.asarray(eigvecs, dtype=np.float64)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvecs.shape[1] > self._n_modes:
            eigvecs = eigvecs[:, : self._n_modes]
            eigvals = eigvals[: self._n_modes]

        mass_norm = np.sqrt((eigvecs ** 2 * mass[:, None]).sum(axis=0))
        if np.any(mass_norm <= 0):
            raise ValueError("laplace_beltrami eigenvectors have non-positive mass norm")
        eigvecs = eigvecs / mass_norm

        self._basis = eigvecs
        self._eigenvalues = eigvals
        self._mass = mass
        self._mask = None if mask is None else mask.astype(bool)
        self._mask_indices = mask_indices
        self._vertex_count = int(vertices.shape[0])
        self._face_count = int(faces.shape[0])
        self._edge_count = int(n_edges)
        self._mesh_hash = _mesh_hash(vertices, faces)
        self._solver_effective = solver

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "LaplaceBeltramiDecomposer":
        if domain_spec is None:
            raise ValueError("laplace_beltrami requires domain_spec for fit")
        masks: list[np.ndarray | None] | None = None
        channels: int | None = None
        if dataset is not None:
            masks = []
            vertices = domain_spec.coords.get("vertices")
            if vertices is None:
                raise ValueError("laplace_beltrami requires mesh vertices in domain_spec")
            n_vertices = vertices.shape[0]
            for idx in range(len(dataset)):
                sample = dataset[idx]
                field_2d, _ = _normalize_field(np.asarray(sample.field))
                if field_2d.shape[0] != n_vertices:
                    raise ValueError(
                        f"field length {field_2d.shape[0]} does not match vertices {n_vertices}"
                    )
                if channels is None:
                    channels = int(field_2d.shape[1])
                elif channels != field_2d.shape[1]:
                    raise ValueError("laplace_beltrami requires consistent channel count across samples")
                masks.append(None if sample.mask is None else np.asarray(sample.mask))

        mask = self._resolve_mask(domain_spec, masks)
        self._ensure_basis(domain_spec, mask)
        if channels is not None:
            self._channels = channels
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        vertices = domain_spec.coords.get("vertices")
        if vertices is None:
            raise ValueError("laplace_beltrami requires mesh vertices in domain_spec")
        n_vertices = int(vertices.shape[0])

        field_2d, layout = _normalize_field(field)
        if field_2d.shape[0] != n_vertices:
            raise ValueError(
                f"field length {field_2d.shape[0]} does not match vertices {n_vertices}"
            )
        if self._channels is not None and field_2d.shape[1] != self._channels:
            raise ValueError("laplace_beltrami field channels do not match fit")

        combined_mask = self._resolve_mask(domain_spec, None if mask is None else [mask])

        self._ensure_basis(domain_spec, combined_mask)
        if self._basis is None or self._eigenvalues is None or self._mass is None:
            raise ValueError("laplace_beltrami basis was not initialized")

        mask_indices = self._mask_indices
        coeffs = []
        for ch in range(field_2d.shape[1]):
            vec = field_2d[:, ch]
            if mask_indices is not None:
                vec = vec[mask_indices]
            if not np.isfinite(vec).all():
                raise ValueError("laplace_beltrami field has non-finite values within mask")
            coeffs.append(self._basis.T @ (self._mass * vec))

        coeff_tensor = np.stack(coeffs, axis=0)
        flat = coeff_tensor.reshape(-1, order="C")
        self._coeff_shape = coeff_tensor.shape
        self._field_layout = layout
        valid_count = int(mask_indices.size) if mask_indices is not None else n_vertices
        # REVIEW: eigen ordering and Laplace-Beltrami definition define coeff semantics.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": list(field.shape),
            "field_ndim": int(field.ndim),
            "field_layout": layout,
            "channels": int(field_2d.shape[1]),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "n_modes": int(self._n_modes),
            "vertex_count": int(self._vertex_count) if self._vertex_count is not None else n_vertices,
            "face_count": int(self._face_count) if self._face_count is not None else None,
            "edge_count": int(self._edge_count) if self._edge_count is not None else None,
            "laplacian_type": self._laplacian_type,
            "mass_type": self._mass_type,
            "boundary_condition": self._boundary_condition,
            "solver": self._solver_effective,
            "dense_threshold": int(self._dense_threshold),
            "eigsh_tol": float(self._eigsh_tol),
            "eigsh_maxiter": int(self._eigsh_maxiter),
            "mask_policy": self._mask_policy,
            "mask_valid_count": valid_count,
            "mode_order": "ascending_eigenvalue",
            "eigenvalues": [float(val) for val in self._eigenvalues],
            "projection": "laplace_beltrami_eigs",
            "mesh_hash": self._mesh_hash,
        }
        return flat

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._field_layout is None:
            raise ValueError("laplace_beltrami transform must be called before inverse_transform")
        vertices = domain_spec.coords.get("vertices")
        faces = domain_spec.coords.get("faces")
        if vertices is None or faces is None:
            raise ValueError("laplace_beltrami requires mesh vertices/faces in domain_spec")
        if self._vertex_count is None or self._face_count is None:
            raise ValueError("laplace_beltrami cached mesh is missing")
        if vertices.shape[0] != self._vertex_count or faces.shape[0] != self._face_count:
            raise ValueError("laplace_beltrami mesh does not match cached basis")
        if self._mesh_hash is not None and _mesh_hash(vertices, faces) != self._mesh_hash:
            raise ValueError("laplace_beltrami mesh geometry does not match cached basis")

        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")

        n_vertices = int(vertices.shape[0])
        mask_indices = self._mask_indices
        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = self._basis @ coeff_tensor[ch]
            if mask_indices is not None:
                full = np.zeros(n_vertices, dtype=vec.dtype)
                full[mask_indices] = vec
                field_c = full
            else:
                field_c = vec
            field_channels.append(field_c)

        field_2d = np.stack(field_channels, axis=-1)
        if self._field_layout == "N":
            return field_2d[:, 0]
        if self._field_layout == "NC":
            return field_2d
        if self._field_layout == "N1C":
            return field_2d[:, None, :]
        raise ValueError(f"Unknown field layout: {self._field_layout}")
