"""Gappy Graph Fourier decomposer: fixed Laplacian eigenbasis + per-sample LS coefficient solve."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.sparse.linalg import eigsh

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import EigenBasisDecomposerBase, _combine_masks, _normalize_field, _normalize_mask, require_cfg
from .graph_fourier import _build_laplacian, _CONNECTIVITY, _LAPLACIAN_TYPES, _SOLVERS, _auto_n_modes, _parse_n_modes

_MASK_POLICIES = {"allow_full"}


@register_decomposer("gappy_graph_fourier")
class GappyGraphFourierDecomposer(EigenBasisDecomposerBase):
    """Graph Laplacian eigenbasis with per-sample weighted ridge coefficient estimation.

    - Basis is built once from the domain mask (or full grid) and cached.
    - Sample masks may vary across samples (gappy observations); coefficients are estimated via LS.
    """

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "gappy_graph_fourier"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = _parse_n_modes(n_modes)
        self._n_modes_config = n_modes
        self._n_modes_effective: int | None = None
        self._connectivity = int(require_cfg(cfg, "connectivity", label="decompose"))
        if self._connectivity not in _CONNECTIVITY:
            raise ValueError(f"decompose.connectivity must be one of {_CONNECTIVITY}, got {self._connectivity}")
        self._laplacian_type = str(require_cfg(cfg, "laplacian_type", label="decompose"))
        if self._laplacian_type not in _LAPLACIAN_TYPES:
            raise ValueError(f"decompose.laplacian_type must be one of {_LAPLACIAN_TYPES}, got {self._laplacian_type}")
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._solver = str(require_cfg(cfg, "solver", label="decompose"))
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._dense_threshold = int(require_cfg(cfg, "dense_threshold", label="decompose"))
        if self._dense_threshold <= 0:
            raise ValueError("decompose.dense_threshold must be > 0 for gappy_graph_fourier")
        self._eigsh_tol = float(require_cfg(cfg, "eigsh_tol", label="decompose"))
        if self._eigsh_tol <= 0:
            raise ValueError("decompose.eigsh_tol must be > 0 for gappy_graph_fourier")
        self._eigsh_maxiter = int(require_cfg(cfg, "eigsh_maxiter", label="decompose"))
        if self._eigsh_maxiter <= 0:
            raise ValueError("decompose.eigsh_maxiter must be > 0 for gappy_graph_fourier")
        ridge_alpha = require_cfg(cfg, "ridge_alpha", label="decompose")
        self._ridge_alpha = float(ridge_alpha)
        if self._ridge_alpha < 0:
            raise ValueError("decompose.ridge_alpha must be >= 0 for gappy_graph_fourier")

        self._grid_shape: tuple[int, int] | None = None
        self._mask_indices: np.ndarray | None = None
        self._domain_mask: np.ndarray | None = None
        self._n_nodes: int | None = None
        self._n_edges: int | None = None
        self._solver_effective: str | None = None
        self._channels: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _ensure_basis(self, domain_spec: DomainSpec) -> None:
        if self._basis is None:
            self._build_basis(domain_spec)
            return
        if self._grid_shape != domain_spec.grid_shape:
            raise ValueError("gappy_graph_fourier domain grid does not match cached basis")
        if (self._domain_mask is None) != (domain_spec.mask is None):
            raise ValueError("gappy_graph_fourier domain mask presence does not match cached basis")
        if self._domain_mask is not None and domain_spec.mask is not None:
            if not np.array_equal(self._domain_mask, np.asarray(domain_spec.mask).astype(bool)):
                raise ValueError("gappy_graph_fourier domain mask does not match cached basis")

    def _build_basis(self, domain_spec: DomainSpec) -> None:
        # Basis is built only from the domain mask (dataset/sample masks are not used).
        mask = None if domain_spec.mask is None else np.asarray(domain_spec.mask).astype(bool)
        laplacian, mask_indices, n_nodes, n_edges = _build_laplacian(
            mask, domain_spec.grid_shape, self._connectivity, self._laplacian_type
        )
        n_modes = self._n_modes
        if n_modes is None:
            n_modes = _auto_n_modes(n_nodes)
        if n_modes > n_nodes:
            raise ValueError("gappy_graph_fourier n_modes exceeds available nodes")

        solver = self._solver
        if solver == "auto":
            solver = "dense" if n_nodes <= self._dense_threshold or n_modes >= n_nodes else "eigsh"
        if solver == "dense":
            dense_lap = laplacian.toarray()
            eigvals, eigvecs = np.linalg.eigh(dense_lap)
        else:
            if n_modes >= n_nodes:
                raise ValueError("gappy_graph_fourier eigsh requires n_modes < n_nodes")
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
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvecs.shape[1] > n_modes:
            eigvecs = eigvecs[:, :n_modes]
            eigvals = eigvals[:n_modes]

        self._set_basis(eigvecs, eigvals)
        self._grid_shape = domain_spec.grid_shape
        self._mask_indices = mask_indices
        self._domain_mask = mask
        self._n_nodes = int(n_nodes)
        self._n_edges = int(n_edges)
        self._solver_effective = solver
        self._n_modes_effective = int(n_modes)

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "GappyGraphFourierDecomposer":
        if domain_spec is None:
            raise ValueError("gappy_graph_fourier requires domain_spec for fit")
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._ensure_basis(domain_spec)
        if dataset is not None:
            channels: int | None = None
            for idx in range(len(dataset)):
                sample = dataset[idx]
                field = np.asarray(sample.field)
                if field.ndim != 3:
                    raise ValueError(f"field must be 3D per sample, got {field.shape}")
                if field.shape[:2] != domain_spec.grid_shape:
                    raise ValueError("field shape does not match domain grid")
                if channels is None:
                    channels = int(field.shape[-1])
                elif channels != field.shape[-1]:
                    raise ValueError("gappy_graph_fourier requires consistent channel count across samples")
            if channels is not None:
                self._channels = channels
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._ensure_basis(domain_spec)
        if self._basis is None or self._eigenvalues is None or self._grid_shape is None:
            raise ValueError("gappy_graph_fourier basis was not initialized")

        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
        if self._channels is not None and field_3d.shape[-1] != self._channels:
            raise ValueError("gappy_graph_fourier field channels do not match fit")

        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)

        weights = domain_spec.integration_weights()
        if weights is None:
            weights = np.ones(domain_spec.grid_shape, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != domain_spec.grid_shape:
            raise ValueError(f"weights shape {weights.shape} does not match {domain_spec.grid_shape}")
        if combined_mask is not None:
            weights = np.where(combined_mask, weights, 0.0)

        # Map to node space (domain nodes).
        mask_indices = self._mask_indices
        weights_flat = weights.reshape(-1, order="C")
        if mask_indices is not None:
            weights_nodes = weights_flat[mask_indices]
        else:
            weights_nodes = weights_flat
        valid = weights_nodes > 0
        if not np.any(valid):
            raise ValueError("gappy_graph_fourier weights are empty after masking")
        if int(np.sum(valid)) < int(self._basis.shape[1]):
            raise ValueError("gappy_graph_fourier requires at least n_modes valid points")

        sqrt_w = np.sqrt(weights_nodes[valid].astype(np.float64, copy=False))
        design = self._basis[valid]
        design_w = design * sqrt_w[:, None]
        k = design_w.shape[1]
        gram = design_w.T @ design_w + (self._ridge_alpha * np.eye(k, dtype=np.float64))

        coeffs = []
        for ch in range(field_3d.shape[-1]):
            vec_full = field_3d[..., ch].reshape(-1, order="C")
            if mask_indices is not None:
                vec_nodes = vec_full[mask_indices]
            else:
                vec_nodes = vec_full
            if not np.isfinite(vec_nodes[valid]).all():
                raise ValueError("gappy_graph_fourier field has non-finite values within valid mask")
            y_w = vec_nodes[valid].astype(np.float64, copy=False) * sqrt_w
            rhs = design_w.T @ y_w
            a = np.linalg.solve(gram, rhs)
            coeffs.append(a)
        coeff_tensor = np.stack(coeffs, axis=0)

        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
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
                "mask_valid_count": int(np.sum(valid)),
                "ridge_alpha": float(self._ridge_alpha),
                "projection": "weighted_ridge_on_fixed_basis",
                "graph_type": "grid",
            }
        )
        meta.update(self._eigen_meta(projection="graph_laplacian_eigs"))
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("gappy_graph_fourier transform must be called before inverse_transform")
        if self._grid_shape is None:
            raise ValueError("gappy_graph_fourier grid shape is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("gappy_graph_fourier domain grid does not match cached basis")
        if (domain_spec.mask is None) != (self._domain_mask is None):
            raise ValueError("gappy_graph_fourier domain mask presence mismatch")
        if domain_spec.mask is not None and self._domain_mask is not None:
            if not np.array_equal(np.asarray(domain_spec.mask).astype(bool), self._domain_mask):
                raise ValueError("gappy_graph_fourier domain mask does not match cached basis")

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
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = ["GappyGraphFourierDecomposer"]

