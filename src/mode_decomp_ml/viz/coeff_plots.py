"""Coefficient plotting helpers (split from viz.__init__).

This module exists to keep viz.__init__ small while preserving backward-compatible imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .diagnostics import (
    masked_weighted_r2,
    per_pixel_r2_map,
    plot_line_with_band,
    plot_scatter_true_pred,
    sample_scatter_points,
)


def _resolve_raw_meta(coeff_meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if coeff_meta is None:
        return {}
    raw_meta = coeff_meta.get("raw_meta")
    if isinstance(raw_meta, Mapping):
        return raw_meta
    return coeff_meta


def coeff_energy_vector(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    coeff = np.asarray(coeff_a)
    if coeff.ndim != 2:
        raise ValueError(f"coeff_a must be 2D, got shape {coeff.shape}")
    if coeff_meta:
        coeff_shape = coeff_meta.get("coeff_shape")
        if isinstance(coeff_shape, list):
            try:
                expected = int(np.prod(coeff_shape))
            except (TypeError, ValueError):
                expected = -1
            if expected == coeff.shape[1]:
                flatten_order = str(coeff_meta.get("flatten_order", "C"))
                reshaped = coeff.reshape((coeff.shape[0], *coeff_shape), order=flatten_order)
                if np.iscomplexobj(reshaped):
                    energy = np.mean(np.abs(reshaped) ** 2, axis=0)
                else:
                    complex_format = str(coeff_meta.get("complex_format", "")).strip().lower()
                    if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and reshaped.shape[-1] == 2:
                        if complex_format == "real_imag":
                            energy = np.mean(reshaped**2, axis=0).sum(axis=-1)
                        else:
                            mag = reshaped[..., 0]
                            if complex_format == "logmag_phase":
                                mag = np.exp(mag)
                            energy = np.mean(mag**2, axis=0)
                    else:
                        energy = np.mean(reshaped**2, axis=0)
                return energy.reshape(-1, order=flatten_order)
    if np.iscomplexobj(coeff):
        return np.mean(np.abs(coeff) ** 2, axis=0)
    return np.mean(coeff**2, axis=0)


def coeff_value_magnitude(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    coeff = np.asarray(coeff_a)
    if coeff.ndim != 2:
        raise ValueError(f"coeff_a must be 2D, got shape {coeff.shape}")
    if coeff_meta:
        coeff_shape = coeff_meta.get("coeff_shape")
        if isinstance(coeff_shape, list):
            try:
                expected = int(np.prod(coeff_shape))
            except (TypeError, ValueError):
                expected = -1
            if expected == coeff.shape[1]:
                flatten_order = str(coeff_meta.get("flatten_order", "C"))
                reshaped = coeff.reshape((coeff.shape[0], *coeff_shape), order=flatten_order)
                if np.iscomplexobj(reshaped):
                    return np.abs(reshaped).reshape(-1)
                complex_format = str(coeff_meta.get("complex_format", "")).strip().lower()
                if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and reshaped.shape[-1] == 2:
                    if complex_format == "real_imag":
                        mag = np.sqrt(reshaped[..., 0] ** 2 + reshaped[..., 1] ** 2)
                    else:
                        mag = reshaped[..., 0]
                        if complex_format == "logmag_phase":
                            mag = np.exp(mag)
                    return mag.reshape(-1)
                return np.abs(reshaped).reshape(-1)
    return np.abs(coeff).reshape(-1)


def coeff_channel_norms(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None = None,
) -> np.ndarray | None:
    coeff = np.asarray(coeff_a)
    if coeff.ndim != 2 or not coeff_meta:
        return None
    channels = int(coeff_meta.get("channels", 1) or 1)
    if channels <= 1:
        return None
    coeff_shape = coeff_meta.get("coeff_shape")
    if not isinstance(coeff_shape, list):
        return None
    try:
        expected = int(np.prod(coeff_shape))
    except (TypeError, ValueError):
        return None
    if expected != coeff.shape[1]:
        return None
    flatten_order = str(coeff_meta.get("flatten_order", "C"))
    reshaped = coeff.reshape((coeff.shape[0], *coeff_shape), order=flatten_order)
    if reshaped.ndim < 3 or reshaped.shape[1] != channels:
        return None
    axes = tuple(range(2, reshaped.ndim))
    norms = np.sqrt(np.sum(np.abs(reshaped) ** 2, axis=axes))
    return norms


def plot_channel_norm_scatter(
    path: str | Path,
    norms: np.ndarray,
    *,
    title: str = "channel norms",
    max_points: int = 2000,
) -> Path:
    norms = np.asarray(norms)
    if norms.ndim != 2 or norms.shape[1] < 2:
        raise ValueError("channel norms must be 2D with at least two channels")
    n_points = min(int(max_points), norms.shape[0])
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.scatter(norms[:n_points, 0], norms[:n_points, 1], s=12, alpha=0.7)
    ax.set_xlabel("channel 0 norm")
    ax.set_ylabel("channel 1 norm")
    ax.set_title(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fft_magnitude_spectrum(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None,
) -> np.ndarray | None:
    if coeff_meta is None:
        return None
    raw_meta = _resolve_raw_meta(coeff_meta)
    method = str(raw_meta.get("method", coeff_meta.get("method", ""))).strip().lower()
    if method != "fft2":
        return None
    coeff = np.asarray(coeff_a)
    if coeff.ndim != 2:
        return None
    coeff_shape = coeff_meta.get("coeff_shape")
    if not isinstance(coeff_shape, list):
        return None
    try:
        expected = int(np.prod(coeff_shape))
    except (TypeError, ValueError):
        return None
    if expected != coeff.shape[1]:
        return None
    flatten_order = str(coeff_meta.get("flatten_order", "C"))
    reshaped = coeff.reshape((coeff.shape[0], *coeff_shape), order=flatten_order)
    complex_format = str(coeff_meta.get("complex_format", "")).strip().lower()
    if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and reshaped.shape[-1] == 2:
        if complex_format == "real_imag":
            mag = np.sqrt(reshaped[..., 0] ** 2 + reshaped[..., 1] ** 2)
        else:
            mag = reshaped[..., 0]
            if complex_format == "logmag_phase":
                mag = np.exp(mag)
    elif np.iscomplexobj(reshaped):
        mag = np.abs(reshaped)
    else:
        mag = np.abs(reshaped)
    mag_mean = np.mean(mag, axis=0)
    while mag_mean.ndim > 2:
        mag_mean = np.mean(mag_mean, axis=0)
    return mag_mean


def wavelet_band_energy(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None,
) -> tuple[list[str], np.ndarray] | None:
    if coeff_meta is None:
        return None
    raw_meta = _resolve_raw_meta(coeff_meta)
    method = str(raw_meta.get("method", coeff_meta.get("method", ""))).strip().lower()
    coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
    if method != "wavelet2d" and coeff_format != "wavedec2":
        return None
    structure = raw_meta.get("coeff_structure")
    if not isinstance(structure, Mapping):
        return None
    approx = structure.get("approx")
    details = structure.get("details")
    if not isinstance(approx, (list, tuple)) or not isinstance(details, (list, tuple)):
        return None

    def _shape_size(shape: Sequence[Any]) -> int:
        if not isinstance(shape, (list, tuple)) or not shape:
            return 0
        return int(np.prod(shape))

    approx_size = _shape_size(approx)
    detail_sizes = []
    for level in details:
        if not isinstance(level, (list, tuple)) or len(level) != 3:
            return None
        level_size = sum(_shape_size(band) for band in level)
        detail_sizes.append(level_size)
    if approx_size <= 0:
        return None

    channels = int(raw_meta.get("channels", coeff_meta.get("channels", 1)))
    per_channel = approx_size + sum(detail_sizes)
    energy_vec = coeff_energy_vector(coeff_a, coeff_meta)
    if energy_vec.size < per_channel * channels:
        return None
    energy_vec = energy_vec[: per_channel * channels].reshape(channels, per_channel)

    idx = 0
    approx_energy = energy_vec[:, idx : idx + approx_size].sum(axis=1)
    idx += approx_size
    level_energies = []
    for level_size in detail_sizes:
        level_energy = energy_vec[:, idx : idx + level_size].sum(axis=1)
        level_energies.append(level_energy)
        idx += level_size

    labels = ["A"] + [f"L{lvl}" for lvl in range(1, len(level_energies) + 1)]
    energies = [float(np.mean(approx_energy))] + [float(np.mean(energy)) for energy in level_energies]
    return labels, np.asarray(energies, dtype=float)


def spherical_l_energy(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if coeff_meta is None:
        return None
    raw_meta = _resolve_raw_meta(coeff_meta)
    method = str(raw_meta.get("method", coeff_meta.get("method", ""))).strip().lower()
    if method != "spherical_harmonics":
        return None
    lm_kind_list = raw_meta.get("lm_kind_list") or coeff_meta.get("lm_kind_list")
    if not isinstance(lm_kind_list, list) or not lm_kind_list:
        return None
    channels = int(raw_meta.get("channels", coeff_meta.get("channels", 1)))
    energy_vec = coeff_energy_vector(coeff_a, coeff_meta)
    n_modes = len(lm_kind_list)
    if energy_vec.size != channels * n_modes:
        return None
    energy_modes = energy_vec.reshape(channels, n_modes).sum(axis=0)
    l_vals = np.asarray([int(item[0]) for item in lm_kind_list], dtype=int)
    unique = np.unique(l_vals)
    energy_by_l = np.array([float(energy_modes[l_vals == level].sum()) for level in unique])
    return unique, energy_by_l


def slepian_concentration(coeff_meta: Mapping[str, Any] | None) -> np.ndarray | None:
    if coeff_meta is None:
        return None
    raw_meta = _resolve_raw_meta(coeff_meta)
    method = str(raw_meta.get("method", coeff_meta.get("method", ""))).strip().lower()
    if method != "spherical_slepian":
        return None
    values = raw_meta.get("concentration") or raw_meta.get("eigenvalues")
    if values is None:
        values = coeff_meta.get("concentration")
    if not isinstance(values, (list, tuple)):
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return arr


def plot_coeff_histogram(
    path: str | Path,
    values: np.ndarray,
    *,
    bins: int = 60,
    scale: str = "log",
) -> Path:
    data = np.asarray(values).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("coeff histogram has no valid values")
    scale = str(scale or "").strip().lower() or "log"
    if scale == "log":
        data = np.log10(data + 1e-12)
        xlabel = "log10 |coeff|"
    else:
        xlabel = "|coeff|"
    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    ax.hist(data, bins=int(bins), color="#4C78A8", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coeff_hist_by_mode(
    path: str | Path,
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None = None,
    *,
    max_modes: int = 128,
    bins: int = 60,
    sort: str = "energy",
    clip_percentile: float = 99.0,
    scale: str = "log1p",
    normalize: str = "column",
    title: str | None = None,
) -> Path:
    """2D histogram of coefficient values by mode index/rank.

    This is meant for quickly diagnosing "what coefficients look like" across a dataset.
    - x-axis: mode index (or rank when sorted by energy)
    - y-axis: coefficient value
    - color: count/density

    Notes:
    - This operates on the encoded coefficient vectors (2D: N x D). It does not require
      a particular coeff_layout, and works for complex coefficients once encoded.
    - For very large D, only the top `max_modes` are shown.
    """
    coeff = np.asarray(coeff_a, dtype=float)
    if coeff.ndim != 2:
        raise ValueError(f"coeff_a must be 2D (N,D), got shape {coeff.shape}")
    n_samples, n_dims = int(coeff.shape[0]), int(coeff.shape[1])
    if n_samples <= 0 or n_dims <= 0:
        raise ValueError("coeff_a must be non-empty")

    max_modes = max(1, int(max_modes))
    bins = max(8, int(bins))
    k_show = min(max_modes, n_dims)

    def _fft_radius_order(meta: Mapping[str, Any]) -> np.ndarray | None:
        coeff_shape = meta.get("coeff_shape")
        if not isinstance(coeff_shape, list) or not coeff_shape:
            return None
        try:
            shape = [int(x) for x in coeff_shape]
        except Exception:
            return None
        if int(np.prod(shape)) != n_dims:
            return None
        order = str(meta.get("flatten_order", "C")).strip().upper() or "C"
        channels = int(meta.get("channels", 1) or 1)
        complex_format = str(meta.get("complex_format", "")).strip().lower()
        fft_shift = bool(meta.get("fft_shift", False))

        # Expect (C,H,W,2) for real/imag-like formats.
        if len(shape) != 4 or shape[0] != channels or shape[-1] != 2:
            return None
        if complex_format not in {"real_imag", "mag_phase", "logmag_phase"}:
            return None
        h = int(shape[1])
        w = int(shape[2])
        if h <= 0 or w <= 0:
            return None

        # Create a map from multi-index -> flattened index respecting flatten_order.
        idx_map = np.arange(n_dims, dtype=int).reshape(tuple(shape), order=order)

        ky = np.fft.fftfreq(h) * float(h)
        kx = np.fft.fftfreq(w) * float(w)
        if fft_shift:
            ky = np.fft.fftshift(ky)
            kx = np.fft.fftshift(kx)
        ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
        radius = np.sqrt(kx_grid**2 + ky_grid**2)
        pos_order = np.argsort(radius.reshape(-1, order=order), kind="stable")

        # Order: channel-major, then increasing radius, then component (real/imag).
        out: list[int] = []
        ys, xs = np.divmod(pos_order, w)
        for c in range(channels):
            for y, x in zip(ys.tolist(), xs.tolist()):
                out.append(int(idx_map[c, int(y), int(x), 0]))
                out.append(int(idx_map[c, int(y), int(x), 1]))
        if len(out) != n_dims:
            return None
        return np.asarray(out, dtype=int)

    sort = str(sort or "").strip().lower() or "energy"
    energy = np.mean(coeff**2, axis=0)
    if sort in {"energy", "energy_desc"}:
        order = np.argsort(-energy)
        xlabel = "mode rank (by energy)"
    elif sort in {"index", "as_is"}:
        order = np.arange(n_dims, dtype=int)
        xlabel = "mode index"
    elif sort in {"auto", "method"}:
        order = None
        xlabel = "mode index"
        if coeff_meta is not None:
            raw = coeff_meta.get("raw_meta")
            meta_use = dict(raw) if isinstance(raw, Mapping) else dict(coeff_meta)
            method = str(meta_use.get("method", "")).strip().lower()
            if method in {"fft2", "fft2_lowpass"}:
                order = _fft_radius_order(meta_use)
                if order is not None:
                    xlabel = "mode index (freq radius order)"
        if order is None:
            order = np.arange(n_dims, dtype=int)
    else:
        raise ValueError(f"unknown sort: {sort}")

    order = order[:k_show]
    values = coeff[:, order]

    # Robust, symmetric clipping for stable visuals.
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("coeff_a has no finite values")

    clip_p = float(clip_percentile)
    if 0.0 < clip_p < 100.0:
        lo = float(np.nanpercentile(flat, 100.0 - clip_p))
        hi = float(np.nanpercentile(flat, clip_p))
        lim = max(abs(lo), abs(hi))
        if not np.isfinite(lim) or lim <= 0:
            lim = float(np.nanmax(np.abs(flat)))
    else:
        lim = float(np.nanmax(np.abs(flat)))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0

    edges = np.linspace(-lim, lim, bins + 1, dtype=float)

    hist = np.zeros((bins, k_show), dtype=float)
    for j in range(k_show):
        x = values[:, j]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        counts, _ = np.histogram(x, bins=edges)
        hist[:, j] = counts.astype(float)

    normalize = str(normalize or "").strip().lower() or "column"
    if normalize == "none":
        label = "count"
    elif normalize == "column":
        denom = hist.sum(axis=0, keepdims=True)
        hist = np.divide(hist, denom, out=np.zeros_like(hist), where=denom > 0)
        label = "density (per mode)"
    elif normalize == "global":
        denom = float(np.sum(hist))
        if denom > 0:
            hist = hist / denom
        label = "fraction"
    else:
        raise ValueError(f"unknown normalize: {normalize}")

    scale = str(scale or "").strip().lower() or "log1p"
    if scale == "linear":
        img = hist
        cbar = label
    elif scale in {"log", "log1p"}:
        img = np.log1p(hist)
        cbar = f"log1p({label})"
    else:
        raise ValueError(f"unknown scale: {scale}")

    fig_w = max(6.4, 0.06 * k_show + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6), constrained_layout=True)
    im = ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=[0.5, k_show + 0.5, float(edges[0]), float(edges[-1])],
        cmap=_colormap_with_bad("viridis"),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("coeff value")
    if title is None:
        title = f"coeff histogram by mode (top {k_show}/{n_dims})"
    ax.set_title(title)
    # Avoid unreadable ticks for large K.
    n_ticks = min(9, k_show)
    if n_ticks >= 2:
        ticks = np.linspace(1, k_show, n_ticks, dtype=int)
        ticks = np.unique(ticks)
        ax.set_xticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_topk_contrib(
    path: str | Path,
    energy: np.ndarray,
    *,
    top_k: int = 10,
) -> Path:
    energy = np.asarray(energy).reshape(-1)
    energy = energy[np.isfinite(energy)]
    if energy.size == 0:
        raise ValueError("top-k energy has no valid entries")
    total = float(np.sum(energy))
    top_k = int(top_k)
    if top_k <= 0:
        top_k = min(10, energy.size)
    top_k = min(top_k, energy.size)
    order = np.argsort(energy)[::-1][:top_k]
    top = energy[order]
    frac = top / total if total > 0 else np.zeros_like(top)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.bar(np.arange(1, top_k + 1), frac * 100.0, color="#F58518", alpha=0.85)
    ax.set_xlabel("mode rank (by energy)")
    ax.set_ylabel("energy share (%)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_energy_bars(
    path: str | Path,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ylabel: str = "mean energy",
) -> Path:
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    data = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(data)), data, color="#54A24B", alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_ylabel(ylabel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_line_series(
    path: str | Path,
    x: Sequence[float],
    y: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
) -> Path:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.plot(x_arr, y_arr, marker="o", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def coeff_energy_spectrum(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    coeff_a = np.asarray(coeff_a)
    if coeff_a.ndim != 2:
        raise ValueError(f"coeff_a must be 2D, got shape {coeff_a.shape}")
    energy = np.mean(coeff_a**2, axis=0)
    if coeff_meta is None:
        return {"kind": "index", "x": np.arange(energy.size), "y": energy}

    coeff_layout = str(coeff_meta.get("coeff_layout", "")).strip().upper()

    nm_list = coeff_meta.get("nm_list")
    channels = coeff_meta.get("channels")
    if isinstance(nm_list, list) and nm_list and isinstance(channels, int):
        n_modes = len(nm_list)
        if energy.size == channels * n_modes:
            energy_modes = energy.reshape(channels, n_modes).sum(axis=0)
            degrees = np.array([int(pair[0]) for pair in nm_list], dtype=int)
            unique = np.unique(degrees)
            energy_by_degree = np.array(
                [float(energy_modes[degrees == degree].sum()) for degree in unique],
                dtype=float,
            )
            return {"kind": "degree", "x": unique, "y": energy_by_degree}

    coeff_shape = coeff_meta.get("coeff_shape")
    # CK layouts are (channels, modes). Even if it is 2D, it should be plotted as a 1D spectrum
    # (mode index) rather than a heatmap.
    if coeff_layout == "CK" and isinstance(coeff_shape, list) and len(coeff_shape) == 2:
        try:
            c = int(channels) if isinstance(channels, int) else int(coeff_shape[0])
            k = int(coeff_shape[1])
            expected = int(c * k)
            if expected == energy.size and k > 0 and c > 0:
                flatten_order = str(coeff_meta.get("flatten_order", "C")).strip().upper() or "C"
                shaped = energy.reshape((c, k), order=flatten_order)
                energy_modes = shaped.sum(axis=0)
                return {"kind": "index", "x": np.arange(energy_modes.size), "y": energy_modes}
        except Exception:
            pass
    if isinstance(coeff_shape, list) and coeff_shape:
        try:
            expected = int(np.prod(coeff_shape))
        except (TypeError, ValueError):
            expected = -1
        if expected == energy.size:
            flatten_order = str(coeff_meta.get("flatten_order", "C"))
            shaped = energy.reshape(tuple(int(x) for x in coeff_shape), order=flatten_order)
            complex_format = str(coeff_meta.get("complex_format", "")).strip().lower()
            if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and shaped.shape[-1] == 2:
                if complex_format == "real_imag":
                    shaped = shaped.sum(axis=-1)
                else:
                    coeff_shaped = coeff_a.reshape((-1, *coeff_shape), order=flatten_order)
                    mag = coeff_shaped[..., 0]
                    if complex_format == "logmag_phase":
                        mag = np.exp(mag)
                    shaped = np.mean(mag**2, axis=0)
            while shaped.ndim > 2:
                shaped = shaped.sum(axis=0)
            if shaped.ndim == 2:
                return {"kind": "heatmap", "data": shaped}
            shaped_1d = np.asarray(shaped).reshape(-1)
            return {"kind": "index", "x": np.arange(shaped_1d.size), "y": shaped_1d}

    return {"kind": "index", "x": np.arange(energy.size), "y": energy}


def plot_coeff_spectrum(
    path: str | Path,
    spectrum: Mapping[str, Any],
    *,
    scale: str = "log",
) -> Path:
    kind = str(spectrum.get("kind", "index"))
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    if kind == "heatmap":
        data = np.asarray(spectrum.get("data"))
        # Defensive: some decomposers produce 1D coeff layouts (e.g. K). Plot as a line in that case.
        if data.ndim != 2:
            x = np.arange(data.size, dtype=float)
            y = data.reshape(-1)
            if scale == "log":
                y = np.log10(y + 1e-12)
            ax.plot(x, y, marker="o", linewidth=1.5)
            ax.set_xlabel("mode index")
            ax.set_ylabel("log10 energy" if scale == "log" else "energy")
        else:
            if scale == "log":
                data = np.log10(data + 1e-12)
            image = ax.imshow(data, origin="lower", cmap="viridis")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel("index x")
            ax.set_ylabel("index y")
    else:
        x = np.asarray(spectrum.get("x"))
        y = np.asarray(spectrum.get("y"))
        if scale == "log":
            y = np.log10(y + 1e-12)
        ax.plot(x, y, marker="o", linewidth=1.5)
        ax.set_xlabel("degree" if kind == "degree" else "mode index")
        ax.set_ylabel("log10 energy" if scale == "log" else "energy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


