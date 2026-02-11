"""Disk Slepian decomposer via bandlimited operator eigenbasis (FFT-based)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import EigenBasisDecomposerBase, _combine_masks, _normalize_field, _normalize_mask, require_cfg

_SOLVERS = {"auto", "dense", "eigsh"}
_MASK_POLICIES = {"allow"}


def _freq_mask(grid_shape: tuple[int, int], freq_radius: int) -> np.ndarray:
    height, width = grid_shape
    fy = np.fft.fftfreq(height) * float(height)
    fx = np.fft.fftfreq(width) * float(width)
    fxg, fyg = np.meshgrid(fx, fy, indexing="xy")
    rr = np.sqrt(fxg * fxg + fyg * fyg)
    return rr <= float(freq_radius)


@register_decomposer("disk_slepian")
class DiskSlepianDecomposer(EigenBasisDecomposerBase):
    """Disk Slepian (bandlimited) eigenbasis decomposer for disk domain.

    Operator on disk-valid points:
      A = P_D * F^{-1} * B * F * P_D
    where B is a frequency-domain disk mask of radius `freq_radius` (index units).
    """

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "disk_slepian"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = int(n_modes)
        if self._n_modes <= 0:
            raise ValueError("decompose.n_modes must be > 0 for disk_slepian")
        freq_radius = require_cfg(cfg, "freq_radius", label="decompose")
        self._freq_radius = int(freq_radius)
        if self._freq_radius < 1:
            raise ValueError("decompose.freq_radius must be >= 1 for disk_slepian")

        self._solver = str(require_cfg(cfg, "solver", label="decompose")).strip().lower()
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._dense_threshold = int(require_cfg(cfg, "dense_threshold", label="decompose"))
        if self._dense_threshold <= 0:
            raise ValueError("decompose.dense_threshold must be > 0 for disk_slepian")
        self._eigsh_tol = float(require_cfg(cfg, "eigsh_tol", label="decompose"))
        if self._eigsh_tol <= 0:
            raise ValueError("decompose.eigsh_tol must be > 0 for disk_slepian")
        self._eigsh_maxiter = int(require_cfg(cfg, "eigsh_maxiter", label="decompose"))
        if self._eigsh_maxiter <= 0:
            raise ValueError("decompose.eigsh_maxiter must be > 0 for disk_slepian")

        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        # CONTRACT: boundary_condition is required for comparability across disk bases.
        self._boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))

        self._grid_shape: tuple[int, int] | None = None
        self._mask_indices: np.ndarray | None = None
        self._domain_mask: np.ndarray | None = None
        self._freq_mask: np.ndarray | None = None

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._channels: int | None = None
        self._solver_effective: str | None = None

    def _ensure_basis(self, domain_spec: DomainSpec) -> None:
        if self._basis is None:
            self._build_basis(domain_spec)
            return
        if self._grid_shape != domain_spec.grid_shape:
            raise ValueError("disk_slepian domain grid does not match cached basis")
        if domain_spec.mask is None:
            raise ValueError("disk_slepian requires disk domain mask")
        if self._domain_mask is None or not np.array_equal(domain_spec.mask, self._domain_mask):
            raise ValueError("disk_slepian domain mask does not match cached basis")

    def _build_basis(self, domain_spec: DomainSpec) -> None:
        if domain_spec.name != "disk":
            raise ValueError("disk_slepian requires disk domain")
        if domain_spec.mask is None:
            raise ValueError("disk_slepian requires a domain mask")
        grid_shape = domain_spec.grid_shape
        mask_flat = np.asarray(domain_spec.mask).astype(bool).reshape(-1, order="C")
        mask_indices = np.flatnonzero(mask_flat)
        if mask_indices.size == 0:
            raise ValueError("disk_slepian domain mask has no valid entries")
        n_nodes = int(mask_indices.size)
        if self._n_modes > n_nodes:
            raise ValueError("disk_slepian n_modes exceeds available disk nodes")

        bmask = _freq_mask(grid_shape, self._freq_radius)
        bmask = np.asarray(bmask, dtype=bool)

        height, width = grid_shape

        def _matvec(vec: np.ndarray) -> np.ndarray:
            v = np.asarray(vec, dtype=np.float64).reshape(-1)
            if v.size != n_nodes:
                raise ValueError("disk_slepian matvec size mismatch")
            full = np.zeros(height * width, dtype=np.float64)
            full[mask_indices] = v
            img = full.reshape(height, width, order="C")
            f = np.fft.fft2(img, norm="ortho")
            f = f * bmask
            img2 = np.fft.ifft2(f, norm="ortho").real
            out = img2.reshape(-1, order="C")[mask_indices]
            return np.asarray(out, dtype=np.float64)

        op = LinearOperator((n_nodes, n_nodes), matvec=_matvec, dtype=np.float64)

        solver = self._solver
        if solver == "auto":
            solver = "dense" if n_nodes <= self._dense_threshold else "eigsh"
        if solver == "dense":
            if n_nodes > self._dense_threshold:
                raise ValueError("disk_slepian dense solver exceeds dense_threshold")
            # Build dense operator matrix by applying matvec to basis vectors.
            eye = np.eye(n_nodes, dtype=np.float64)
            cols = [_matvec(eye[:, idx]) for idx in range(n_nodes)]
            dense = np.stack(cols, axis=1)
            eigvals, eigvecs = np.linalg.eigh(dense)
        else:
            if self._n_modes >= n_nodes:
                raise ValueError("disk_slepian eigsh requires n_modes < n_nodes")
            eigvals, eigvecs = eigsh(
                op,
                k=self._n_modes,
                which="LA",
                tol=self._eigsh_tol,
                maxiter=self._eigsh_maxiter,
            )

        eigvals = np.asarray(eigvals, dtype=np.float64)
        eigvecs = np.asarray(eigvecs, dtype=np.float64)
        order = np.argsort(-eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvecs.shape[1] > self._n_modes:
            eigvecs = eigvecs[:, : self._n_modes]
            eigvals = eigvals[: self._n_modes]

        # REVIEW: descending eigenvalue order defines coefficient index semantics.
        self._set_basis(eigvecs, eigvals)
        self._grid_shape = grid_shape
        self._mask_indices = mask_indices
        self._domain_mask = np.asarray(domain_spec.mask).astype(bool)
        self._freq_mask = bmask
        self._solver_effective = solver

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "DiskSlepianDecomposer":
        if domain_spec is None:
            raise ValueError("disk_slepian requires domain_spec for fit")
        _ = dataset
        self._ensure_basis(domain_spec)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._ensure_basis(domain_spec)
        if self._basis is None or self._eigenvalues is None or self._mask_indices is None or self._grid_shape is None:
            raise ValueError("disk_slepian basis was not initialized")

        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
        if self._channels is not None and field_3d.shape[-1] != self._channels:
            raise ValueError("disk_slepian field channels do not match fit")

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

        w = weights.reshape(-1, order="C")[self._mask_indices].astype(np.float64, copy=False)
        valid = w > 0
        if not np.any(valid):
            raise ValueError("disk_slepian weights are empty after masking")
        if int(np.sum(valid)) < self._basis.shape[1]:
            raise ValueError("disk_slepian basis has more modes than valid samples")

        sqrt_w = np.sqrt(w[valid])
        design = self._basis[valid]
        design_w = design * sqrt_w[:, None]

        coeffs = []
        for ch in range(field_3d.shape[-1]):
            vec_full = field_3d[..., ch].reshape(-1, order="C")[self._mask_indices].astype(np.float64, copy=False)
            if not np.isfinite(vec_full[valid]).all():
                raise ValueError("disk_slepian field has non-finite values within valid mask")
            field_w = vec_full[valid] * sqrt_w
            a, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
            if rank < a.shape[0]:
                raise ValueError("disk_slepian basis is rank-deficient; reduce n_modes or adjust freq_radius")
            coeffs.append(a)
        coeff_tensor = np.stack(coeffs, axis=0)

        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._channels = int(field_3d.shape[-1])
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
                "n_modes": int(self._basis.shape[1]),
                "freq_radius": int(self._freq_radius),
                "solver": self._solver_effective,
                "dense_threshold": int(self._dense_threshold),
                "eigsh_tol": float(self._eigsh_tol),
                "eigsh_maxiter": int(self._eigsh_maxiter),
                "mask_policy": self._mask_policy,
                "domain_mask_valid_count": int(self._mask_indices.size),
                "mask_valid_count": int(np.sum(valid)),
                "boundary_condition": self._boundary_condition,
            }
        )
        meta.update(self._eigen_meta(projection="disk_bandlimited_operator", mode_order="descending_eigenvalue"))
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("disk_slepian transform must be called before inverse_transform")
        if self._grid_shape is None or self._mask_indices is None:
            raise ValueError("disk_slepian grid is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("disk_slepian domain grid does not match cached basis")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        height, width = self._grid_shape

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = self._basis @ np.asarray(coeff_tensor[ch], dtype=np.float64)
            full = np.zeros(height * width, dtype=np.float64)
            full[self._mask_indices] = vec
            field_channels.append(full.reshape(height, width, order="C"))
        field_hat = np.stack(field_channels, axis=-1)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = ["DiskSlepianDecomposer"]

