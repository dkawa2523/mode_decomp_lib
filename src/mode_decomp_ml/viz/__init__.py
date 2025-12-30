"""Visualization helpers for mode decomposition outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_field_3d(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[..., None]
    if field.ndim == 3:
        return field
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _ensure_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool)


def _masked_values(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return field.reshape(-1)
    return field[mask]


def _colormap_with_bad(name: str, bad_color: str = "#dddddd") -> Any:
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color=bad_color)
    return cmap


def plot_field_grid(
    path: str | Path,
    fields: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    mask: np.ndarray | None = None,
    suptitle: str | None = None,
    cmap: str = "viridis",
) -> Path:
    if len(fields) != len(titles):
        raise ValueError("fields and titles must have the same length")
    if not fields:
        raise ValueError("fields must be non-empty")
    field_stack = [_ensure_field_3d(field) for field in fields]
    channels = field_stack[0].shape[2]
    for field in field_stack[1:]:
        if field.shape != field_stack[0].shape:
            raise ValueError("all fields must have the same shape")

    mask = _ensure_mask(mask, field_stack[0].shape[:2])
    vmin = []
    vmax = []
    for ch in range(channels):
        values = [_masked_values(field[..., ch], mask) for field in field_stack]
        merged = np.concatenate(values, axis=0)
        if merged.size == 0:
            raise ValueError("masked field has no valid entries")
        vmin.append(float(np.min(merged)))
        vmax.append(float(np.max(merged)))

    n_rows = channels
    n_cols = len(field_stack)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    cmap_obj = _colormap_with_bad(cmap)
    for col, (field, title) in enumerate(zip(field_stack, titles)):
        for row in range(channels):
            ax = axes[row][col]
            data = field[..., row]
            if mask is not None:
                data = np.where(mask, data, np.nan)
            ax.imshow(data, origin="lower", cmap=cmap_obj, vmin=vmin[row], vmax=vmax[row])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0 and channels > 1:
                ax.set_ylabel(f"ch{row}")
    if suptitle:
        fig.suptitle(suptitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_error_map(
    path: str | Path,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    cmap: str = "magma",
) -> Path:
    field_true = _ensure_field_3d(field_true)
    field_pred = _ensure_field_3d(field_pred)
    if field_true.shape != field_pred.shape:
        raise ValueError("field_true and field_pred must have the same shape")
    diff = field_pred - field_true
    if diff.shape[2] == 1:
        error = np.abs(diff[..., 0])
    else:
        error = np.linalg.norm(diff, axis=-1)
    mask = _ensure_mask(mask, error.shape)
    if mask is not None:
        error = np.where(mask, error, np.nan)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    cmap_obj = _colormap_with_bad(cmap)
    image = ax.imshow(error, origin="lower", cmap=cmap_obj)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_uncertainty_map(
    path: str | Path,
    field_std: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    cmap: str = "magma",
) -> Path:
    field_std = _ensure_field_3d(field_std)
    if field_std.shape[2] == 1:
        data = field_std[..., 0]
    else:
        data = np.linalg.norm(field_std, axis=-1)
    mask = _ensure_mask(mask, data.shape)
    if mask is not None:
        data = np.where(mask, data, np.nan)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    cmap_obj = _colormap_with_bad(cmap)
    image = ax.imshow(data, origin="lower", cmap=cmap_obj)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
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
    if isinstance(coeff_shape, list) and coeff_shape:
        try:
            expected = int(np.prod(coeff_shape))
        except (TypeError, ValueError):
            expected = -1
        if expected == energy.size:
            flatten_order = str(coeff_meta.get("flatten_order", "C"))
            shaped = energy.reshape(tuple(int(x) for x in coeff_shape), order=flatten_order)
            complex_format = str(coeff_meta.get("complex_format", ""))
            if complex_format == "real_imag" and shaped.shape[-1] == 2:
                shaped = shaped.sum(axis=-1)
            while shaped.ndim > 2:
                shaped = shaped.sum(axis=0)
            return {"kind": "heatmap", "data": shaped}

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
        if scale == "log":
            data = np.log10(data + 1e-12)
        image = ax.imshow(data, origin="lower", cmap="viridis")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("frequency x")
        ax.set_ylabel("frequency y")
    else:
        x = np.asarray(spectrum.get("x"))
        y = np.asarray(spectrum.get("y"))
        if scale == "log":
            y = np.log10(y + 1e-12)
        ax.plot(x, y, marker="o", linewidth=1.5)
        ax.set_xlabel("degree" if kind == "degree" else "index")
        ax.set_ylabel("log10 energy" if scale == "log" else "energy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "coeff_energy_spectrum",
    "plot_coeff_spectrum",
    "plot_error_map",
    "plot_field_grid",
    "plot_uncertainty_map",
]
