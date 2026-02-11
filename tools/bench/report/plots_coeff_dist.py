from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.bench._util import load_json


def ensure_coeff_mode_hist_plot(
    run_dir: Path,
    *,
    out_path: Path,
    top_k: int = 8,
    cols: int = 4,
    bins: int = 40,
) -> Path | None:
    """Histogram (small-multiples) of coefficient values for top-energy dims.

    This mirrors the decomposition-side `coeff_mode_hist.png`, but runs from saved `outputs/coeffs.npz`
    so reports can standardize plots across old runs.
    """
    coeffs_npz = run_dir / "outputs" / "coeffs.npz"
    if not coeffs_npz.exists():
        return None
    try:
        coeff = np.load(coeffs_npz, allow_pickle=True).get("coeff")
    except Exception:
        return None
    if coeff is None:
        return None
    coeff = np.asarray(coeff)
    if coeff.ndim != 2 or coeff.size == 0:
        return None

    offset_dim = 0
    meta_path = run_dir / "outputs" / "states" / "coeff_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            if str(meta.get("coeff_format", "")).strip().lower() == "offset_residual_v1":
                offset_dim = int(meta.get("offset_dim", 0) or 0)
        except Exception:
            offset_dim = 0

    coeff_plot = coeff
    if 0 < offset_dim < int(coeff.shape[1]):
        coeff_plot = coeff[:, offset_dim:]
        if coeff_plot.shape[1] <= 0:
            coeff_plot = coeff
            offset_dim = 0

    if np.iscomplexobj(coeff_plot):
        coeff_plot = np.abs(coeff_plot)
    else:
        coeff_plot = np.asarray(coeff_plot, dtype=float)

    energy = np.mean(coeff_plot**2, axis=0)
    if energy.ndim != 1 or energy.size <= 0:
        return None

    top_k = max(1, int(top_k))
    cols = max(1, int(cols))
    bins = max(5, int(bins))
    order = np.argsort(-energy)[: min(top_k, int(energy.size))]
    if order.size <= 0:
        return None

    n_rows = int(np.ceil(order.size / cols))
    fig, axes = plt.subplots(
        n_rows,
        cols,
        figsize=(2.6 * cols, 2.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for idx, dim in enumerate(order.tolist()):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        values = coeff_plot[:, int(dim)].reshape(-1)
        ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
        dim0 = int(dim) + int(offset_dim)
        ax.set_title(f"dim {dim0}", fontsize=9)
        if r == n_rows - 1:
            ax.set_xlabel("coeff")
        else:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("count")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=8)
    for idx in range(order.size, n_rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path if out_path.exists() else None


def ensure_mode_energy_bar_plot(
    run_dir: Path,
    *,
    out_path: Path,
    max_bars: int = 128,
    eps: float = 1e-12,
) -> Path | None:
    """Mode-energy bar plot.

    x = mode index (or frequency-radius bins for CHW-like layouts),
    y = normalized energy (mean(coeff^2) aggregated across samples/channels).
    """
    coeffs_npz = run_dir / "outputs" / "coeffs.npz"
    if not coeffs_npz.exists():
        return None
    try:
        coeff = np.load(coeffs_npz, allow_pickle=True).get("coeff")
    except Exception:
        return None
    if coeff is None:
        return None
    A = np.asarray(coeff)
    if A.ndim != 2 or A.size == 0:
        return None

    meta_path = run_dir / "outputs" / "states" / "coeff_meta.json"
    meta: Mapping[str, Any] | None = None
    if meta_path.exists():
        try:
            loaded = load_json(meta_path)
            if isinstance(loaded, Mapping):
                meta = loaded
        except Exception:
            meta = None

    offset_dim = 0
    coeff_meta = meta or {}
    residual_meta: Mapping[str, Any] = coeff_meta
    if str(coeff_meta.get("coeff_format", "")).strip().lower() == "offset_residual_v1":
        try:
            offset_dim = int(coeff_meta.get("offset_dim", 0) or 0)
        except Exception:
            offset_dim = 0
        rm = coeff_meta.get("residual_coeff_meta")
        if isinstance(rm, Mapping):
            residual_meta = rm

    vector_dim = None
    try:
        vector_dim = residual_meta.get("vector_dim")
        vector_dim = int(vector_dim) if vector_dim is not None else None
    except Exception:
        vector_dim = None

    if offset_dim < 0:
        offset_dim = 0
    if vector_dim is None:
        vector_dim = int(A.shape[1]) - int(offset_dim)
    vector_dim = max(0, min(int(vector_dim), int(A.shape[1]) - int(offset_dim)))
    if vector_dim <= 0:
        return None
    X = A[:, offset_dim : offset_dim + vector_dim]

    coeff_layout = str(residual_meta.get("coeff_layout", "")).strip().upper()
    coeff_shape = residual_meta.get("coeff_shape")
    flatten_order = str(residual_meta.get("flatten_order", "C")).strip().upper() or "C"
    method = str(residual_meta.get("method", "")).strip().lower()

    def _safe_norm_energy(e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=float)
        e = np.where(np.isfinite(e), e, 0.0)
        s = float(np.sum(e))
        if s <= 0:
            return e
        return e / s

    x_label = "mode index"
    energy_vec: np.ndarray | None = None

    if (
        coeff_layout == "CK"
        and isinstance(coeff_shape, (list, tuple))
        and len(coeff_shape) == 2
        and int(coeff_shape[0]) > 0
        and int(coeff_shape[1]) > 0
    ):
        c = int(coeff_shape[0])
        k = int(coeff_shape[1])
        if c * k <= X.shape[1]:
            arr = np.asarray(X[:, : c * k], dtype=float).reshape((X.shape[0], c, k), order=flatten_order)
            energy_vec = np.mean(np.sum(arr**2, axis=1), axis=0)

    if energy_vec is None and coeff_layout == "K" and isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 1:
        k = int(coeff_shape[0])
        if k <= X.shape[1]:
            arr = np.asarray(X[:, :k], dtype=float)
            energy_vec = np.mean(arr**2, axis=0)

    if energy_vec is None and coeff_layout == "CHW":
        if isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 3:
            c = int(coeff_shape[0])
            h = int(coeff_shape[1])
            w = int(coeff_shape[2])
            if c * h * w <= X.shape[1] and h > 0 and w > 0:
                arr = X[:, : c * h * w].reshape((X.shape[0], c, h, w), order=flatten_order)
                e_hw = np.mean(np.sum(np.abs(arr) ** 2, axis=1), axis=0)
                KY, KX = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
                r = np.sqrt(KX**2 + KY**2)
                rbin = np.rint(r).astype(int)
                rmax = int(np.max(rbin))
                e_r = np.zeros((rmax + 1,), dtype=float)
                for b in range(rmax + 1):
                    m = rbin == b
                    if np.any(m):
                        e_r[b] = float(np.sum(e_hw[m]))
                energy_vec = e_r
                x_label = "freq radius (bin)"

    if energy_vec is None:
        e = np.mean(np.abs(X) ** 2, axis=0) if np.iscomplexobj(X) else np.mean(np.asarray(X, dtype=float) ** 2, axis=0)
        energy_vec = np.asarray(e, dtype=float)

    energy_vec = np.asarray(energy_vec, dtype=float).reshape(-1)
    if energy_vec.size <= 0 or not np.any(np.isfinite(energy_vec)):
        return None
    energy_vec = np.where(np.isfinite(energy_vec), energy_vec, 0.0)
    energy_vec = _safe_norm_energy(energy_vec)

    max_bars = max(8, int(max_bars))
    if energy_vec.size > max_bars:
        head = energy_vec[:max_bars]
        tail_sum = float(np.sum(energy_vec[max_bars:]))
        energy_show = np.concatenate([head, np.asarray([tail_sum], dtype=float)])
        x = np.arange(energy_show.size, dtype=int)
        title_suffix = f"(0..{max_bars-1} + other; total={energy_vec.size})"
    else:
        energy_show = energy_vec
        x = np.arange(energy_show.size, dtype=int)
        title_suffix = f"(total={energy_vec.size})"

    fig, ax = plt.subplots(figsize=(6.2, 2.4), constrained_layout=True)
    ax.bar(x, energy_show, color="#4C78A8", alpha=0.9, width=0.9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("energy fraction")
    if energy_show.size > 40:
        step = 16
        ticks = [0, step, 2 * step, 4 * step, 8 * step]
        ticks = [t for t in ticks if t < int(energy_show.size)]
        if ticks:
            ax.set_xticks(ticks)

    title_method = method or str(residual_meta.get("method", "")).strip() or "unknown"
    ax.set_title(f"Mode energy (dataset-level)  [{title_method}]  {title_suffix}", fontsize=10)
    ax.set_ylim(0.0, max(float(np.max(energy_show)) * 1.10, eps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path if out_path.exists() else None


def ensure_mode_value_boxplot_plot(
    run_dir: Path,
    *,
    out_path: Path,
    top_k: int = 16,
) -> Path | None:
    """Per-mode coefficient value distributions (boxplot) for top-energy dims."""
    coeffs_npz = run_dir / "outputs" / "coeffs.npz"
    if not coeffs_npz.exists():
        return None
    try:
        coeff = np.load(coeffs_npz, allow_pickle=True).get("coeff")
    except Exception:
        return None
    if coeff is None:
        return None
    A = np.asarray(coeff)
    if A.ndim != 2 or A.size == 0:
        return None

    meta_path = run_dir / "outputs" / "states" / "coeff_meta.json"
    meta: Mapping[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = load_json(meta_path)
            if isinstance(loaded, Mapping):
                meta = loaded
        except Exception:
            meta = {}

    offset_dim = 0
    residual_meta: Mapping[str, Any] = meta
    if str(meta.get("coeff_format", "")).strip().lower() == "offset_residual_v1":
        try:
            offset_dim = int(meta.get("offset_dim", 0) or 0)
        except Exception:
            offset_dim = 0
        rm = meta.get("residual_coeff_meta")
        if isinstance(rm, Mapping):
            residual_meta = rm

    vector_dim = None
    try:
        vector_dim = residual_meta.get("vector_dim")
        vector_dim = int(vector_dim) if vector_dim is not None else None
    except Exception:
        vector_dim = None

    if vector_dim is None:
        vector_dim = int(A.shape[1]) - int(offset_dim)
    vector_dim = max(0, min(int(vector_dim), int(A.shape[1]) - int(offset_dim)))
    if vector_dim <= 0:
        return None
    X = A[:, offset_dim : offset_dim + vector_dim]

    X = np.real(X) if np.iscomplexobj(X) else np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] <= 0:
        return None

    energy = np.mean(X**2, axis=0)
    if energy.ndim != 1 or energy.size <= 0:
        return None
    top_k = max(1, int(top_k))
    order = np.argsort(-energy)[: min(top_k, int(energy.size))]
    if order.size <= 0:
        return None

    data = [X[:, int(d)].reshape(-1) for d in order.tolist()]
    labels = [str(int(d) + int(offset_dim)) for d in order.tolist()]

    all_vals = np.concatenate([v[np.isfinite(v)] for v in data if v.size], axis=0) if data else np.asarray([])
    if all_vals.size:
        lo = float(np.nanpercentile(all_vals, 1))
        hi = float(np.nanpercentile(all_vals, 99))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    else:
        lo, hi = -1.0, 1.0

    method = str(residual_meta.get("method", "")).strip() or "unknown"
    fig_w = max(6.4, 0.42 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 2.7), constrained_layout=True)
    ax.boxplot(
        data,
        vert=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "#333333", "linewidth": 1.2},
        boxprops={"color": "#4C78A8"},
        whiskerprops={"color": "#4C78A8"},
        capprops={"color": "#4C78A8"},
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_xlabel("mode index (selected dims; top-energy)")
    ax.set_ylabel("coeff value")
    ax.set_title(f"Coeff value boxplot (dataset-level)  [{method}]  (top_k={len(labels)})", fontsize=10)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.05 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path if out_path.exists() else None

