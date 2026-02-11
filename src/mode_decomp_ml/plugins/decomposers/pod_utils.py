"""Shared utilities for POD-family decomposers (internal helpers)."""
from __future__ import annotations

from typing import Any

import numpy as np


def parse_n_modes(value: Any, *, method: str) -> int | None:
    """Parse n_modes where 'auto' means None."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return None
    if isinstance(value, (int, np.integer)):
        num = int(value)
        if num <= 0:
            raise ValueError(f"decompose.n_modes must be > 0 for {method}")
        return num
    raise ValueError(f"decompose.n_modes must be an int or 'auto' for {method}")


def auto_n_modes(n_samples: int, n_features: int) -> int:
    return max(1, min(64, int(n_samples), int(n_features)))


def svd_snapshots_basis(X: np.ndarray, *, n_modes: int, method: str) -> np.ndarray:
    """Snapshot POD basis from centered data (N,F) -> basis (F,K) in the same feature space."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{method} snapshots must be 2D, got shape {arr.shape}")
    n_samples, n_features = arr.shape
    if n_samples <= 0 or n_features <= 0:
        raise ValueError(f"{method} snapshots must be non-empty")
    if n_modes <= 0:
        raise ValueError(f"{method} n_modes must be > 0")
    if n_modes > min(n_samples, n_features):
        raise ValueError(f"{method} n_modes exceeds available rank")

    gram = arr @ arr.T  # (N,N)
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    singular = np.sqrt(np.clip(eigvals, 0.0, None))
    keep = singular > 1e-12
    if not np.any(keep):
        raise ValueError(f"{method} singular values are all zero")
    singular = singular[keep]
    eigvecs = eigvecs[:, keep]
    if eigvecs.shape[1] < n_modes:
        raise ValueError(f"{method} n_modes exceeds available rank")

    singular = singular[:n_modes]
    eigvecs = eigvecs[:, :n_modes]
    basis = (arr.T @ eigvecs) / singular[None, :]
    return np.asarray(basis, dtype=np.float64)


def solve_ridge(A: np.ndarray, y: np.ndarray, ridge_alpha: float, *, method: str) -> np.ndarray:
    """Solve min ||A x - y||^2 + ridge_alpha ||x||^2 (ridge_alpha >= 0)."""
    alpha = float(ridge_alpha)
    if alpha < 0:
        raise ValueError(f"{method} ridge_alpha must be >= 0")
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"{method} design must be 2D")
    if y.ndim != 1:
        y = y.reshape(-1)
    if A.shape[0] != y.shape[0]:
        raise ValueError(f"{method} design/target size mismatch")
    if alpha > 0:
        lhs = A.T @ A + alpha * np.eye(A.shape[1], dtype=np.float64)
        rhs = A.T @ y
        return np.linalg.solve(lhs, rhs)
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeff


__all__ = ["parse_n_modes", "auto_n_modes", "svd_snapshots_basis", "solve_ridge"]

