"""Laplace-Beltrami decomposer for mesh domains."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config, integer_parameter

from typing import Any, Mapping

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility
from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint
from ml_platform_mode_decomp.geometry.mesh import compact_mesh

from .base import EigenBasisDecomposerBase, _mask_values, require_cfg

_LAPLACIAN_TYPES = {"cotangent"}
_MASS_TYPES = {"lumped"}
_SOLVERS = {"auto", "dense", "eigsh"}
_MASK_POLICIES = {"forbid", "allow", "require"}
_BOUNDARY_CONDITIONS = {"neumann", "none"}


def _parse_n_modes(value: Any) -> int | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
        value = text
    if isinstance(value, (int, np.integer)):
        num = integer_parameter(value, "n_modes")
        if num <= 0:
            raise ValueError("decompose.n_modes must be > 0 for laplace_beltrami")
        return num
    raise ValueError("decompose.n_modes must be an int or 'auto' for laplace_beltrami")


def _auto_n_modes(n_nodes: int) -> int:
    if n_nodes <= 0:
        raise ValueError("laplace_beltrami requires at least one node")
    return max(1, min(64, n_nodes))


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
    return _mask_values(mask_arr)


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


def _build_cotangent_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[csr_matrix, int]:
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

    n_edges = int(adjacency.nnz // 2)
    return csr_matrix(laplacian), n_edges


class LaplaceBeltramiDecomposer(EigenBasisDecomposerBase):
    """Laplace-Beltrami eigenbasis decomposer for mesh domains."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_modes: Any = 64,
        laplacian_type: Any = "cotangent",
        mass_type: Any = "lumped",
        boundary_condition: Any = "neumann",
        mask_policy: Any = "allow",
        solver: Any = "auto",
        dense_threshold: Any = 2048,
        eigsh_tol: Any = 1e-06,
        eigsh_maxiter: Any = 2000,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="laplace_beltrami")
        super().__init__(cfg=cfg)
        self.name = "laplace_beltrami"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = _parse_n_modes(n_modes)
        self._n_modes_config = n_modes
        self._n_modes_effective: int | None = None
        self._laplacian_type = str(require_cfg(cfg, "laplacian_type", label="decompose"))
        if self._laplacian_type not in _LAPLACIAN_TYPES:
            raise ValueError(f"decompose.laplacian_type must be one of {_LAPLACIAN_TYPES}, got {self._laplacian_type}")
        self._mass_type = str(require_cfg(cfg, "mass_type", label="decompose"))
        if self._mass_type not in _MASS_TYPES:
            raise ValueError(f"decompose.mass_type must be one of {_MASS_TYPES}, got {self._mass_type}")
        self._boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))
        if self._boundary_condition not in _BOUNDARY_CONDITIONS:
            raise ValueError(
                f"decompose.boundary_condition must be one of {_BOUNDARY_CONDITIONS}, got {self._boundary_condition}"
            )
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._solver = str(require_cfg(cfg, "solver", label="decompose"))
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._dense_threshold = integer_parameter(
            require_cfg(cfg, "dense_threshold", label="decompose"), "dense_threshold"
        )
        if self._dense_threshold <= 0:
            raise ValueError("decompose.dense_threshold must be > 0 for laplace_beltrami")
        self._eigsh_tol = float(require_cfg(cfg, "eigsh_tol", label="decompose"))
        if self._eigsh_tol <= 0:
            raise ValueError("decompose.eigsh_tol must be > 0 for laplace_beltrami")
        self._eigsh_maxiter = integer_parameter(require_cfg(cfg, "eigsh_maxiter", label="decompose"), "eigsh_maxiter")
        if self._eigsh_maxiter <= 0:
            raise ValueError("decompose.eigsh_maxiter must be > 0 for laplace_beltrami")

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
        observation: np.ndarray | None,
    ) -> np.ndarray | None:
        vertices = domain_spec.coords.get("vertices")
        if vertices is None:
            raise ValueError("laplace_beltrami requires mesh vertices in domain_spec")
        n_vertices = vertices.shape[0]
        domain_mask = _normalize_mask(domain_spec.mask, n_vertices)
        sample_mask = _normalize_mask(observation, n_vertices)
        mask = domain_mask if sample_mask is None else sample_mask if domain_mask is None else domain_mask & sample_mask
        if mask is None:
            if self._mask_policy == "require":
                raise ValueError("laplace_beltrami mask_policy=require needs a mask")
            return None
        domain_valid = np.ones(n_vertices, dtype=bool) if domain_mask is None else domain_mask
        if self._mask_policy == "forbid" and np.any(domain_valid & ~mask):
            raise ValueError("laplace_beltrami mask_policy=forbid does not allow masked vertices")
        if mask.all():
            return None
        return mask

    def _ensure_basis(self, domain_spec: DomainSpec) -> None:
        if self._basis is None:
            self._build_basis(domain_spec)
            return
        if self._vertex_count is None or self._face_count is None:
            raise ValueError("laplace_beltrami basis is not initialized correctly")
        vertices = domain_spec.coords.get("vertices")
        faces = domain_spec.coords.get("faces")
        if vertices is None or faces is None:
            raise ValueError("laplace_beltrami requires mesh vertices/faces in domain_spec")
        if vertices.shape[0] != self._vertex_count or faces.shape[0] != self._face_count:
            raise ValueError("laplace_beltrami mesh size does not match cached basis")
        mesh_hash = geometry_fingerprint(domain_spec)
        if self._mesh_hash is not None and mesh_hash != self._mesh_hash:
            raise ValueError("laplace_beltrami mesh geometry does not match cached basis")

    def _build_basis(self, domain_spec: DomainSpec) -> None:
        vertices = domain_spec.coords.get("vertices")
        faces = domain_spec.coords.get("faces")
        if vertices is None or faces is None:
            raise ValueError("laplace_beltrami requires mesh vertices/faces in domain_spec")

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        vertices_use, faces_use, mask_indices, geometric_mass = compact_mesh(vertices, faces, domain_spec.mask)
        weights = domain_spec.integration_weights()
        if weights is None:
            raise ValueError("laplace_beltrami requires the fixed geometry's vertex mass")
        mass = np.asarray(weights, dtype=np.float64).reshape(-1)
        if mask_indices is not None:
            mass = mass[mask_indices]
        if mass.shape != geometric_mass.shape or not np.allclose(mass, geometric_mass, rtol=1e-12, atol=1e-15):
            raise ValueError("mesh geometry weights do not match its active faces")
        laplacian, n_edges = _build_cotangent_laplacian(vertices_use, faces_use)
        n_nodes = vertices_use.shape[0]
        n_modes = self._n_modes
        if n_modes is None:
            n_modes = _auto_n_modes(n_nodes)
        if n_modes > n_nodes:
            raise ValueError("laplace_beltrami n_modes exceeds available vertices")

        solver = self._solver
        if solver == "auto":
            solver = "dense" if n_nodes <= self._dense_threshold or n_modes >= n_nodes else "eigsh"
        if solver == "dense":
            dense_lap = laplacian.toarray()
            dense_mass = np.diag(mass)
            eigvals, eigvecs = eigh(dense_lap, dense_mass)
        else:
            if n_modes >= n_nodes:
                raise ValueError("laplace_beltrami eigsh requires n_modes < n_vertices")
            eigvals, eigvecs = eigsh(
                laplacian,
                k=n_modes,
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
        if eigvecs.shape[1] > n_modes:
            eigvecs = eigvecs[:, :n_modes]
            eigvals = eigvals[:n_modes]

        mass_norm = np.sqrt((eigvecs**2 * mass[:, None]).sum(axis=0))
        if np.any(mass_norm <= 0):
            raise ValueError("laplace_beltrami eigenvectors have non-positive mass norm")
        eigvecs = eigvecs / mass_norm

        self._set_basis(eigvecs, eigvals)
        self._mass = mass
        self._mask = domain_spec.mask
        self._mask_indices = mask_indices
        self._vertex_count = int(vertices.shape[0])
        self._face_count = int(faces.shape[0])
        self._edge_count = int(n_edges)
        self._mesh_hash = geometry_fingerprint(domain_spec)
        self._solver_effective = solver
        self._n_modes_effective = integer_parameter(n_modes, "n_modes")

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
                    raise ValueError(f"field length {field_2d.shape[0]} does not match vertices {n_vertices}")
                if channels is None:
                    channels = int(field_2d.shape[1])
                elif channels != field_2d.shape[1]:
                    raise ValueError("laplace_beltrami requires consistent channel count across samples")
                masks.append(None if sample.mask is None else np.asarray(sample.mask))

        for observation in masks or [None]:
            self._resolve_mask(domain_spec, observation)
        self._ensure_basis(domain_spec)
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
            raise ValueError(f"field length {field_2d.shape[0]} does not match vertices {n_vertices}")
        if self._channels is not None and field_2d.shape[1] != self._channels:
            raise ValueError("laplace_beltrami field channels do not match fit")

        combined_mask = self._resolve_mask(domain_spec, mask)

        self._ensure_basis(domain_spec)
        if self._basis is None or self._eigenvalues is None or self._mass is None:
            raise ValueError("laplace_beltrami basis was not initialized")

        mask_indices = self._mask_indices
        observed = np.ones(n_vertices, dtype=bool) if combined_mask is None else combined_mask
        if mask_indices is not None:
            observed = observed[mask_indices]
        coeffs = []
        for ch in range(field_2d.shape[1]):
            vec = field_2d[:, ch]
            if mask_indices is not None:
                vec = vec[mask_indices]
            if not np.isfinite(vec[observed]).all():
                raise ValueError("laplace_beltrami field has non-finite values within mask")
            if observed.all():
                coeffs.append(self._basis.T @ (self._mass * vec))
            else:
                sqrt_mass = np.sqrt(self._mass[observed])
                design = self._basis[observed] * sqrt_mass[:, None]
                values, _, rank, _ = np.linalg.lstsq(design, vec[observed] * sqrt_mass, rcond=None)
                if rank < self._basis.shape[1]:
                    raise ValueError("laplace_beltrami observations do not identify the requested modes")
                coeffs.append(values)

        coeff_tensor = np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_layout = layout
        valid_count = int(observed.sum())
        # REVIEW: eigen ordering and Laplace-Beltrami definition define coeff semantics.
        meta = self._coeff_meta_base(
            field_shape=list(field.shape),
            field_ndim=int(field.ndim),
            field_layout=layout,
            channels=int(field_2d.shape[1]),
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            {
                "n_modes": int(self._n_modes_effective) if self._n_modes_effective is not None else None,
                "n_modes_config": self._n_modes_config,
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
                "mesh_hash": self._mesh_hash,
            }
        )
        meta.update(self._eigen_meta(projection="laplace_beltrami_eigs"))
        self._coeff_meta = meta
        return coeff_tensor

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
        if self._mesh_hash is not None and geometry_fingerprint(domain_spec) != self._mesh_hash:
            raise ValueError("laplace_beltrami mesh geometry does not match cached basis")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)

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
