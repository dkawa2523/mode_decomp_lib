"""Graph Fourier decomposer using grid graph Laplacian eigenbasis."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

from .base import EigenBasisDecomposerBase, _combine_masks, _normalize_field, _normalize_mask, require_cfg
from mode_decomp_ml.plugins.registry import register_decomposer

_MASK_POLICIES = {"require_mask", "allow_full"}
_LAPLACIAN_TYPES = {"combinatorial", "normalized"}
_SOLVERS = {"auto", "dense", "eigsh"}
_CONNECTIVITY = {4, 8}



def _parse_n_modes(value: Any) -> int | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
        value = text
    if isinstance(value, (int, np.integer)):
        num = int(value)
        if num <= 0:
            raise ValueError("decompose.n_modes must be > 0 for graph_fourier")
        return num
    raise ValueError("decompose.n_modes must be an int or 'auto' for graph_fourier")


def _auto_n_modes(n_nodes: int) -> int:
    if n_nodes <= 0:
        raise ValueError("graph_fourier requires at least one node")
    return max(1, min(64, n_nodes))


def _edges_for_offset(node_ids: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = node_ids.shape
    if dy >= 0:
        src_rows = slice(0, height - dy)
        dst_rows = slice(dy, height)
    else:
        src_rows = slice(-dy, height)
        dst_rows = slice(0, height + dy)
    if dx >= 0:
        src_cols = slice(0, width - dx)
        dst_cols = slice(dx, width)
    else:
        src_cols = slice(-dx, width)
        dst_cols = slice(0, width + dx)
    src = node_ids[src_rows, src_cols]
    dst = node_ids[dst_rows, dst_cols]
    valid = (src >= 0) & (dst >= 0)
    return src[valid], dst[valid]


def _build_laplacian(
    mask: np.ndarray | None,
    grid_shape: tuple[int, int],
    connectivity: int,
    laplacian_type: str,
) -> tuple[csr_matrix, np.ndarray | None, int, int]:
    height, width = grid_shape
    if mask is None:
        mask_indices = None
        node_ids = np.arange(height * width, dtype=int).reshape(height, width)
        n_nodes = height * width
    else:
        mask_flat = mask.reshape(-1, order="C")
        mask_indices = np.flatnonzero(mask_flat)
        if mask_indices.size == 0:
            raise ValueError("graph_fourier mask has no valid entries")
        node_ids_flat = np.full(height * width, -1, dtype=int)
        node_ids_flat[mask_indices] = np.arange(mask_indices.size, dtype=int)
        node_ids = node_ids_flat.reshape(height, width)
        n_nodes = int(mask_indices.size)

    if n_nodes <= 0:
        raise ValueError("graph_fourier requires at least one node")

    offsets = [(1, 0), (0, 1)]
    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    for dy, dx in offsets:
        src, dst = _edges_for_offset(node_ids, dy, dx)
        if src.size == 0:
            continue
        rows_list.append(src)
        cols_list.append(dst)

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
    else:
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)

    n_edges = int(rows.size)
    if n_edges == 0:
        adjacency = csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    else:
        data = np.ones(rows.size * 2, dtype=np.float64)
        row_sym = np.concatenate([rows, cols])
        col_sym = np.concatenate([cols, rows])
        adjacency = coo_matrix((data, (row_sym, col_sym)), shape=(n_nodes, n_nodes)).tocsr()
        adjacency.sum_duplicates()

    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if laplacian_type == "combinatorial":
        laplacian = diags(degrees, dtype=np.float64) - adjacency
    elif laplacian_type == "normalized":
        inv_sqrt = np.zeros_like(degrees, dtype=np.float64)
        nonzero = degrees > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        d_inv_sqrt = diags(inv_sqrt, dtype=np.float64)
        laplacian = identity(n_nodes, dtype=np.float64) - d_inv_sqrt @ adjacency @ d_inv_sqrt
    else:
        raise ValueError(f"Unsupported laplacian_type: {laplacian_type}")
    return laplacian.tocsr(), mask_indices, n_nodes, n_edges


@register_decomposer("graph_fourier")
class GraphFourierDecomposer(EigenBasisDecomposerBase):
    """Graph Laplacian eigenbasis decomposer for arbitrary masks."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "graph_fourier"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = _parse_n_modes(n_modes)
        self._n_modes_config = n_modes
        self._n_modes_effective: int | None = None
        self._connectivity = int(require_cfg(cfg, "connectivity", label="decompose"))
        if self._connectivity not in _CONNECTIVITY:
            raise ValueError(
                f"decompose.connectivity must be one of {_CONNECTIVITY}, got {self._connectivity}"
            )
        self._laplacian_type = str(require_cfg(cfg, "laplacian_type", label="decompose"))
        if self._laplacian_type not in _LAPLACIAN_TYPES:
            raise ValueError(
                f"decompose.laplacian_type must be one of {_LAPLACIAN_TYPES}, got {self._laplacian_type}"
            )
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._solver = str(require_cfg(cfg, "solver", label="decompose"))
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._dense_threshold = int(require_cfg(cfg, "dense_threshold", label="decompose"))
        if self._dense_threshold <= 0:
            raise ValueError("decompose.dense_threshold must be > 0 for graph_fourier")
        self._eigsh_tol = float(require_cfg(cfg, "eigsh_tol", label="decompose"))
        if self._eigsh_tol <= 0:
            raise ValueError("decompose.eigsh_tol must be > 0 for graph_fourier")
        self._eigsh_maxiter = int(require_cfg(cfg, "eigsh_maxiter", label="decompose"))
        if self._eigsh_maxiter <= 0:
            raise ValueError("decompose.eigsh_maxiter must be > 0 for graph_fourier")

        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._n_nodes: int | None = None
        self._n_edges: int | None = None
        self._solver_effective: str | None = None

    def _resolve_mask(
        self,
        domain_spec: DomainSpec,
        masks: list[np.ndarray | None] | None,
    ) -> np.ndarray | None:
        domain_mask = domain_spec.mask
        mask = domain_mask
        if masks:
            sample_mask = masks[0]
            if sample_mask is None:
                if any(mask_item is not None for mask_item in masks):
                    raise ValueError("graph_fourier mask must be present for all samples or none")
            else:
                sample_mask = np.asarray(sample_mask).astype(bool)
                for mask_item in masks[1:]:
                    if mask_item is None:
                        raise ValueError("graph_fourier mask must be present for all samples or none")
                    if not np.array_equal(sample_mask, np.asarray(mask_item).astype(bool)):
                        raise ValueError("graph_fourier requires a fixed mask across samples")
                mask = sample_mask if mask is None else (mask & sample_mask)
        if mask is None:
            if self._mask_policy == "require_mask":
                raise ValueError("graph_fourier mask_policy=require_mask needs a mask")
            return None
        if mask.shape != domain_spec.grid_shape:
            raise ValueError(f"mask shape {mask.shape} does not match {domain_spec.grid_shape}")
        return mask

    def _ensure_basis(self, domain_spec: DomainSpec, mask: np.ndarray | None) -> None:
        if self._basis is None:
            self._build_basis(domain_spec, mask)
            return
        if self._grid_shape is None:
            raise ValueError("graph_fourier basis is not initialized correctly")
        if self._grid_shape != domain_spec.grid_shape:
            raise ValueError("graph_fourier domain grid does not match cached basis")
        if (self._mask is None) != (mask is None):
            raise ValueError("graph_fourier mask presence does not match cached basis")
        if mask is not None and not np.array_equal(mask, self._mask):
            raise ValueError("graph_fourier requires the same mask used during fit")

    def _build_basis(self, domain_spec: DomainSpec, mask: np.ndarray | None) -> None:
        laplacian, mask_indices, n_nodes, n_edges = _build_laplacian(
            mask, domain_spec.grid_shape, self._connectivity, self._laplacian_type
        )
        n_modes = self._n_modes
        if n_modes is None:
            n_modes = _auto_n_modes(n_nodes)
        if n_modes > n_nodes:
            raise ValueError("graph_fourier n_modes exceeds available nodes")

        solver = self._solver
        if solver == "auto":
            solver = "dense" if n_nodes <= self._dense_threshold or n_modes >= n_nodes else "eigsh"
        if solver == "dense":
            dense_lap = laplacian.toarray()
            eigvals, eigvecs = np.linalg.eigh(dense_lap)
        else:
            if n_modes >= n_nodes:
                raise ValueError("graph_fourier eigsh requires n_modes < n_nodes")
            eigvals, eigvecs = eigsh(
                laplacian,
                k=n_modes,
                which="SM",
                tol=self._eigsh_tol,
                maxiter=self._eigsh_maxiter,
            )

        eigvals = np.asarray(eigvals, dtype=np.float64)
        eigvecs = np.asarray(eigvecs, dtype=np.float64)
        order = np.argsort(eigvals)
        # REVIEW: ascending eigenvalue order defines coefficient index semantics.
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvecs.shape[1] > n_modes:
            eigvecs = eigvecs[:, : n_modes]
            eigvals = eigvals[: n_modes]

        self._set_basis(eigvecs, eigvals)
        self._mask = None if mask is None else mask.astype(bool)
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._n_nodes = int(n_nodes)
        self._n_edges = int(n_edges)
        self._solver_effective = solver
        self._n_modes_effective = int(n_modes)

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "GraphFourierDecomposer":
        if domain_spec is None:
            raise ValueError("graph_fourier requires domain_spec for fit")
        masks: list[np.ndarray | None] | None = None
        channels: int | None = None
        if dataset is not None:
            masks = []
            for idx in range(len(dataset)):
                sample = dataset[idx]
                field = np.asarray(sample.field)
                if field.ndim != 3:
                    raise ValueError(f"field must be 3D per sample, got {field.shape}")
                if field.shape[:2] != domain_spec.grid_shape:
                    raise ValueError(
                        f"field shape {field.shape[:2]} does not match domain {domain_spec.grid_shape}"
                    )
                if channels is None:
                    channels = int(field.shape[-1])
                elif channels != field.shape[-1]:
                    raise ValueError("graph_fourier requires consistent channel count across samples")
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
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        if self._channels is not None and field_3d.shape[-1] != self._channels:
            raise ValueError("graph_fourier field channels do not match fit")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)
        if combined_mask is None and self._mask_policy == "require_mask":
            raise ValueError("graph_fourier mask_policy=require_mask needs a mask")
        # CONTRACT: basis is cached per fixed mask for reproducibility.
        self._ensure_basis(domain_spec, combined_mask)
        if self._basis is None or self._eigenvalues is None:
            raise ValueError("graph_fourier basis was not initialized")

        mask_indices = self._mask_indices
        coeffs = []
        for ch in range(field_3d.shape[-1]):
            vec = field_3d[..., ch].reshape(-1, order="C")
            if mask_indices is not None:
                vec = vec[mask_indices]
            if not np.isfinite(vec).all():
                raise ValueError("graph_fourier field has non-finite values within mask")
            coeffs.append(self._basis.T @ vec)
        coeff_tensor = np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        valid_count = int(mask_indices.size) if mask_indices is not None else int(np.prod(field_3d.shape[:2]))
        # REVIEW: eigen ordering and laplacian type define coeff semantics.
        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(field_3d.shape[-1]),
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            {
                "n_modes": int(self._n_modes_effective) if self._n_modes_effective is not None else None,
                "n_modes_config": self._n_modes_config,
                "n_nodes": int(self._n_nodes) if self._n_nodes is not None else None,
                "n_edges": int(self._n_edges) if self._n_edges is not None else None,
                "laplacian_type": self._laplacian_type,
                "connectivity": int(self._connectivity),
                "solver": self._solver_effective,
                "dense_threshold": int(self._dense_threshold),
                "eigsh_tol": float(self._eigsh_tol),
                "eigsh_maxiter": int(self._eigsh_maxiter),
                "mask_policy": self._mask_policy,
                "mask_valid_count": valid_count,
                "graph_type": "grid",
            }
        )
        meta.update(self._eigen_meta(projection="graph_laplacian_eigs"))
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("graph_fourier transform must be called before inverse_transform")
        if self._grid_shape is None:
            raise ValueError("graph_fourier grid shape is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("graph_fourier domain grid does not match cached basis")
        if domain_spec.mask is not None and self._mask is None:
            raise ValueError("graph_fourier cached mask is missing but domain mask is present")
        if domain_spec.mask is not None and self._mask is not None:
            if not np.array_equal(domain_spec.mask, self._mask):
                raise ValueError("graph_fourier domain mask does not match cached basis")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        height, width = self._grid_shape
        mask_indices = self._mask_indices

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = self._basis @ coeff_tensor[ch]
            if mask_indices is not None:
                full = np.zeros(height * width, dtype=vec.dtype)
                full[mask_indices] = vec
                field_c = full.reshape(height, width, order="C")
            else:
                field_c = vec.reshape(height, width, order="C")
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat
