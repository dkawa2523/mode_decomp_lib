"""Small diagnostics helpers for plots (R^2, scatter sampling, bands).

This module is intentionally small and reusable across processes to avoid
duplicating plotting/math code in entrypoints.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _broadcast_param_to(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Broadcast `value` to `shape` by inserting singleton dims if needed.

    This is intentionally permissive for common cases like:
      - mask/weights: (H,W) used with fields: (N,H,W,C)
      - weights: (V,) used with mesh fields: (V,1,1)
    """
    arr = np.asarray(value)
    if arr.shape == shape:
        return np.broadcast_to(arr, shape)
    if arr.ndim > len(shape):
        raise ValueError(f"{name} shape {arr.shape} has more dims than target {shape}")
    if arr.ndim == 0:
        return np.broadcast_to(arr, shape)

    target_ndim = len(shape)
    missing = target_ndim - arr.ndim
    if missing == 0:
        try:
            return np.broadcast_to(arr, shape)
        except Exception as e:  # pragma: no cover - rare
            raise ValueError(f"{name} shape {arr.shape} is not broadcastable to {shape}") from e

    # Try inserting singleton dims in all possible positions (ndim<=4 in practice).
    for positions in combinations(range(target_ndim), missing):
        new_shape: list[int] = []
        src = 0
        for j in range(target_ndim):
            if j in positions:
                new_shape.append(1)
            else:
                new_shape.append(int(arr.shape[src]))
                src += 1
        try:
            reshaped = arr.reshape(tuple(new_shape))
            return np.broadcast_to(reshaped, shape)
        except Exception:
            continue

    raise ValueError(f"{name} shape {arr.shape} is not broadcastable to {shape}")


def masked_weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """Compute weighted R^2 for arrays with optional mask.

    Shapes:
      - y_true/y_pred: any shape, must match
      - mask: broadcastable to y_true (bool)
      - weights: broadcastable to y_true (float)
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        m = _broadcast_param_to(m, a.shape, name="mask")
        valid &= m

    if not np.any(valid):
        return float("nan")

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = _broadcast_param_to(w, a.shape, name="weights")
        valid &= np.isfinite(w) & (w > 0)

    if not np.any(valid):
        return float("nan")

    a1 = a[valid]
    b1 = b[valid]
    if w is None:
        mu = float(np.mean(a1))
        ss_res = float(np.sum((a1 - b1) ** 2))
        ss_tot = float(np.sum((a1 - mu) ** 2))
    else:
        w1 = w[valid]
        w_sum = float(np.sum(w1))
        if w_sum <= 0:
            return float("nan")
        mu = float(np.sum(w1 * a1) / w_sum)
        ss_res = float(np.sum(w1 * (a1 - b1) ** 2))
        ss_tot = float(np.sum(w1 * (a1 - mu) ** 2))

    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def per_pixel_r2_map(
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    downsample: int = 1,
) -> np.ndarray:
    """Per-pixel R^2 across samples.

    field_true/field_pred: (N,H,W) or (N,H,W,C)
    mask: (N,H,W) or (H,W) or None
    returns: (H',W') or (H',W',C) with NaNs where undefined.
    """
    a = np.asarray(field_true, dtype=float)
    b = np.asarray(field_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("field_true and field_pred must have the same shape")
    if a.ndim not in (3, 4):
        raise ValueError(f"field arrays must be 3D or 4D, got shape {a.shape}")

    if downsample <= 0:
        raise ValueError("downsample must be >= 1")
    if downsample > 1:
        if a.ndim == 3:
            a = a[:, ::downsample, ::downsample]
            b = b[:, ::downsample, ::downsample]
        else:
            a = a[:, ::downsample, ::downsample, :]
            b = b[:, ::downsample, ::downsample, :]
        if mask is not None:
            mask_arr = np.asarray(mask).astype(bool)
            if mask_arr.ndim == 3:
                mask = mask_arr[:, ::downsample, ::downsample]
            elif mask_arr.ndim == 2:
                mask = mask_arr[::downsample, ::downsample]

    if mask is None:
        valid = np.isfinite(a) & np.isfinite(b)
    else:
        mask_arr = np.asarray(mask).astype(bool)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[None, ...]
        if mask_arr.ndim != 3:
            raise ValueError("mask must be 2D or 3D")
        if mask_arr.shape[0] == 1 and a.shape[0] > 1:
            mask_arr = np.broadcast_to(mask_arr, a.shape[:3])
        if mask_arr.shape[:3] != a.shape[:3]:
            raise ValueError("mask must match field spatial shape")
        valid = mask_arr & np.isfinite(a) & np.isfinite(b)

    # Vectorized: R^2 = 1 - ss_res/ss_tot along sample axis, per location (and channel).
    v = valid.astype(float)
    if a.ndim == 3:
        count = np.sum(v, axis=0)
        y_sum = np.sum(a * v, axis=0)
        mean = np.divide(y_sum, count, out=np.zeros_like(y_sum), where=count > 0)
        ss_res = np.sum(((a - b) ** 2) * v, axis=0)
        ss_tot = np.sum(((a - mean[None, ...]) ** 2) * v, axis=0)
        out = np.full_like(ss_tot, np.nan, dtype=float)
        good = (count >= 2) & (ss_tot > 0)
        out[good] = 1.0 - (ss_res[good] / ss_tot[good])
        return out

    # a.ndim == 4
    if v.ndim == 3:
        v = v[..., None]
    count = np.sum(v, axis=0)
    y_sum = np.sum(a * v, axis=0)
    mean = np.divide(y_sum, count, out=np.zeros_like(y_sum), where=count > 0)
    ss_res = np.sum(((a - b) ** 2) * v, axis=0)
    ss_tot = np.sum(((a - mean[None, ...]) ** 2) * v, axis=0)
    out = np.full_like(ss_tot, np.nan, dtype=float)
    good = (count >= 2) & (ss_tot > 0)
    out[good] = 1.0 - (ss_res[good] / ss_tot[good])
    return out


def sample_scatter_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_points: int = 200_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten masked values and subsample for scatter plots."""
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("y_true/y_pred must have the same size for scatter sampling")
    valid = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m = np.asarray(mask).astype(bool).reshape(-1)
        if m.shape != a.shape:
            raise ValueError("mask must match y_true shape for scatter sampling")
        valid &= m
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    n = min(int(max_points), int(idx.size))
    if n <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if n < idx.size:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=n, replace=False)
    return a[idx], b[idx]


def plot_scatter_true_pred(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    r2: float | None = None,
    max_points: int = 200_000,
) -> Path:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0:
        raise ValueError("scatter has no points")
    n = min(int(max_points), int(x.size))
    x = x[:n]
    y = y[:n]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.scatter(x, y, s=4, alpha=0.35)
    vmin = float(np.nanmin([np.min(x), np.min(y)]))
    vmax = float(np.nanmax([np.max(x), np.max(y)]))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        ax.plot([vmin, vmax], [vmin, vmax], color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("true")
    ax.set_ylabel("pred")
    if r2 is not None and np.isfinite(r2):
        title = f"{title} (R^2={float(r2):.3f})"
    ax.set_title(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_line_with_band(
    path: str | Path,
    x: Sequence[float],
    y_mean: Sequence[float],
    y_lo: Sequence[float],
    y_hi: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
) -> Path:
    x_arr = np.asarray(x, dtype=float)
    mean_arr = np.asarray(y_mean, dtype=float)
    lo_arr = np.asarray(y_lo, dtype=float)
    hi_arr = np.asarray(y_hi, dtype=float)
    if x_arr.shape != mean_arr.shape or x_arr.shape != lo_arr.shape or x_arr.shape != hi_arr.shape:
        raise ValueError("x/y_mean/y_lo/y_hi must have the same shape")
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    ax.plot(x_arr, mean_arr, marker="o", linewidth=1.6, color="#4C78A8")
    ax.fill_between(x_arr, lo_arr, hi_arr, color="#4C78A8", alpha=0.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "masked_weighted_r2",
    "per_pixel_r2_map",
    "plot_line_with_band",
    "plot_scatter_true_pred",
    "sample_scatter_points",
]
