"""Shared utilities for POD-family decomposers (internal helpers)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.config import integer_parameter

from .base import parse_bool


def resolve_centering(cfg: Mapping[str, Any], *, default: bool) -> bool:
    """Resolve the two nullable centering names without ignoring a contradiction."""
    center, mean_centered = cfg.get("center"), cfg.get("mean_centered")
    if center is not None and mean_centered is not None:
        if parse_bool(center, default=default) != parse_bool(mean_centered, default=default):
            raise ValueError("center and mean_centered must match when both are specified")
    return parse_bool(center if center is not None else mean_centered, default=default)


def snapshot_matrix(fields: np.ndarray, mask_indices: np.ndarray | None, *, method: str) -> np.ndarray:
    """Own one float64, LAPACK-ready work matrix without converting a whole field batch first."""
    n_samples = len(fields)
    n_features = int(np.prod(fields.shape[1:])) if mask_indices is None else len(mask_indices)
    matrix = np.empty((n_samples, n_features), dtype=np.float64, order="F")
    for index, field in enumerate(fields):
        vector = np.asarray(field).reshape(-1)
        if mask_indices is not None:
            vector = vector[mask_indices]
        if not np.isfinite(vector).all():
            raise ValueError(f"{method} requires finite snapshots")
        matrix[index] = vector
    return matrix


def snapshot_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consume an owned work matrix with the direct SVD, retaining only S and Vh."""
    from scipy.linalg import svd

    if any(not np.isfinite(row).all() for row in matrix):
        raise ValueError("POD requires finite snapshots after centering and weighting")
    _, singular, vh = svd(matrix, full_matrices=False, overwrite_a=True, check_finite=False)
    return np.asarray(singular), np.asarray(vh)


def projection_energy(
    matrix: np.ndarray, basis: np.ndarray, weights: np.ndarray | None = None, *, offset: np.ndarray | None = None
) -> np.ndarray:
    """Measure actual retained energy without allocating all sample coefficients."""
    energy = np.zeros(basis.shape[1], dtype=np.float64)
    # Bound both the input weighting and projected coordinates to a small batch.
    batch_size = max(1, min(256, 2**18 // max(matrix.shape[1], basis.shape[1], 1)))
    for start in range(0, len(matrix), batch_size):
        batch = matrix[start : start + batch_size]
        if weights is not None:
            batch = batch * weights
        projected = batch @ basis
        if offset is not None:
            projected += offset
        energy += np.einsum("ij,ij->j", projected, projected)
    return energy


def singular_value_rank(singular: np.ndarray, shape: tuple[int, int]) -> int:
    """Numerical rank in a computed subspace, relative to the leading singular value."""
    values = np.asarray(singular, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Singular values must be a finite nonnegative vector")
    if values.size == 0 or values[0] == 0:
        return 0
    tolerance = np.finfo(values.dtype).eps * max(shape) * values[0]
    return int(np.count_nonzero(values > tolerance))


def select_pod_modes(
    mode_energy: np.ndarray,
    total_energy: float,
    *,
    n_modes: int | None,
    method: str,
    energy_threshold: float | None = None,
    max_modes: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Select from nonzero candidate modes using the complete training energy.

    ``mode_energy`` can belong to an approximate/truncated basis. Its sum is
    deliberately never used as the denominator of the requested energy ratio.
    """
    values = np.asarray(mode_energy, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{method} mode energy must be finite and nonnegative")
    if total_energy == 0 and values.size == 0:
        return 0, {
            "selected_modes": 0,
            "total_training_energy": 0.0,
            "retained_training_energy": 0.0,
            "retained_energy_fraction": 1.0,
            "energy_threshold": energy_threshold,
            "energy_threshold_met": True if energy_threshold is not None else None,
        }
    if not np.isfinite(total_energy) or total_energy <= 0 or values.size == 0:
        raise ValueError(f"{method} training snapshots have zero numerical rank")
    available = len(values)
    if n_modes is not None:
        if n_modes <= 0 or n_modes > available:
            raise ValueError(f"{method} n_modes exceeds available rank")
        selected = int(n_modes)
    elif energy_threshold is None:
        selected = available
    else:
        selected = min(int(np.searchsorted(np.cumsum(values), energy_threshold * total_energy)) + 1, available)
        if max_modes is not None:
            selected = min(selected, max_modes)
    retained = float(np.sum(values[:selected]))
    fraction = float(np.clip(retained / total_energy, 0.0, 1.0))
    return selected, {
        "selected_modes": selected,
        "total_training_energy": float(total_energy),
        "retained_training_energy": retained,
        "retained_energy_fraction": fraction,
        "energy_threshold": energy_threshold,
        "energy_threshold_met": (
            bool(fraction >= energy_threshold - 16 * np.finfo(float).eps) if energy_threshold is not None else None
        ),
    }


def component_blocks_meta(meta, counts):
    """Describe independent real component blocks without padding or truncation."""
    counts = [int(count) for count in counts]
    offsets = np.cumsum([0, *counts]).tolist()
    return {
        **meta,
        "coeff_shape": [offsets[-1]],
        "coeff_layout": "D",
        "coeff_format": "component_blocks_v1",
        "component_modes": counts,
        "component_offsets": offsets,
        "complex_format": "real",
    }


def parse_n_modes(value: Any, *, method: str) -> int | None:
    """Parse n_modes where 'auto' means None."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
    if isinstance(value, (int, np.integer)):
        num = integer_parameter(value, "n_modes")
        if num <= 0:
            raise ValueError(f"decompose.n_modes must be > 0 for {method}")
        return num
    raise ValueError(f"decompose.n_modes must be an int or 'auto' for {method}")


def svd_snapshots_basis(X: np.ndarray, *, n_modes: int | None, method: str) -> np.ndarray:
    """POD basis (F,K) with scale-relative rank selection; None selects auto."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{method} snapshots must be 2D, got shape {arr.shape}")
    n_samples, n_features = arr.shape
    if n_samples <= 0 or n_features <= 0:
        raise ValueError(f"{method} snapshots must be non-empty")
    if n_modes is not None and n_modes <= 0:
        raise ValueError(f"{method} n_modes must be > 0")
    if n_modes is not None and n_modes > min(n_samples, n_features):
        raise ValueError(f"{method} n_modes exceeds available rank")
    if not np.isfinite(arr).all():
        raise ValueError(f"{method} snapshots must be finite")

    # Forming X X.T squares the condition number and can manufacture small
    # positive eigenvalues. Direct SVD preserves the numerical rank.
    _, singular, vh = np.linalg.svd(arr, full_matrices=False)
    rank = singular_value_rank(singular, arr.shape)
    if rank == 0:
        raise ValueError(f"{method} singular values are all zero")
    selected = min(64, rank) if n_modes is None else int(n_modes)
    if rank < selected:
        raise ValueError(f"{method} n_modes exceeds available rank")
    basis = vh[:selected].T
    # Truncated views keep unused singular vectors alive; full bases need no copy.
    return basis.copy(order="F") if selected < len(vh) else basis


def solve_ridge(A: np.ndarray, y: np.ndarray, ridge_alpha: float, *, method: str) -> np.ndarray:
    """Solve min ||A x - y||^2 + ridge_alpha ||x||^2 (ridge_alpha >= 0)."""
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"{method} ridge_alpha must be >= 0")
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"{method} design must be 2D")
    if y.ndim not in {1, 2}:
        raise ValueError(f"{method} targets must be a vector or multiple right-hand sides")
    if A.shape[0] != y.shape[0]:
        raise ValueError(f"{method} design/target size mismatch")
    if A.shape[1] == 0:
        return np.empty((0, *y.shape[1:]), dtype=np.float64)
    if alpha > 0:
        # Augmentation avoids squaring the condition number of partial bases.
        A = np.vstack((A, np.sqrt(alpha) * np.eye(A.shape[1])))
        y = np.concatenate((y, np.zeros((A.shape[1], *y.shape[1:]))))
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeff


def project_pod(values: np.ndarray, basis: np.ndarray, *, mean=None, projection_weights=None) -> np.ndarray:
    """Project with an explicit mean and the fitted representation of its inner product."""
    centered = np.asarray(values) if mean is None else np.asarray(values) - mean
    if projection_weights is not None:
        centered = centered * projection_weights
    return centered @ basis


def reconstruct_pod(coefficients: np.ndarray, basis: np.ndarray, *, mean=None, sqrt_weights=None) -> np.ndarray:
    """Synthesize physical values; zero-measure entries carry no recoverable residual."""
    values = coefficients @ basis.T
    if sqrt_weights is not None:
        restored = np.zeros_like(values)
        np.divide(values, sqrt_weights, out=restored, where=np.asarray(sqrt_weights) > 0)
        values = restored
    return values if mean is None else values + mean


__all__ = [
    "parse_n_modes",
    "svd_snapshots_basis",
    "solve_ridge",
    "singular_value_rank",
    "select_pod_modes",
]


def _pod_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    options = cfg.get("options")
    if options is None:
        return {}
    if not isinstance(options, Mapping) or set(options) - {"rank_select", "mode_weight"}:
        raise ValueError("POD options accepts rank_select and mode_weight")
    return options


def resolve_rank_select(cfg: Mapping[str, Any]) -> dict[str, Any]:
    options = _pod_options(cfg)
    rank_cfg = {}
    if isinstance(options, Mapping):
        rank_cfg = options.get("rank_select") or {}
    if not isinstance(rank_cfg, Mapping) or set(rank_cfg) - {"enable", "method", "energy", "max_modes"}:
        raise ValueError("options.rank_select has unsupported settings")
    enable = parse_bool(rank_cfg.get("enable", False), default=False)
    method = str(rank_cfg.get("method", "energy")).strip().lower() or "energy"
    if method != "energy":
        raise ValueError(f"options.rank_select.method must be energy, got {method}")
    energy = float(rank_cfg.get("energy", 0.99))
    if not (0.0 < energy <= 1.0):
        raise ValueError("options.rank_select.energy must be in (0, 1]")
    max_modes = rank_cfg.get("max_modes", None)
    if max_modes is not None:
        max_modes = integer_parameter(max_modes, "options.rank_select.max_modes")
        if max_modes <= 0:
            raise ValueError("options.rank_select.max_modes must be > 0")
    return {"enable": enable, "method": method, "energy": energy, "max_modes": max_modes}


def resolve_mode_weight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    options = _pod_options(cfg)
    mode_cfg = {}
    if isinstance(options, Mapping):
        mode_cfg = options.get("mode_weight") or {}
    if not isinstance(mode_cfg, Mapping) or set(mode_cfg) - {"enable", "method"}:
        raise ValueError("options.mode_weight has unsupported settings")
    enable = parse_bool(mode_cfg.get("enable", False), default=False)
    method = str(mode_cfg.get("method", "eigval_scale")).strip().lower() or "eigval_scale"
    if method != "eigval_scale":
        raise ValueError(f"options.mode_weight.method must be eigval_scale, got {method}")
    return {"enable": enable, "method": method}
