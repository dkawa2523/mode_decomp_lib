"""Field/domain plotting helpers (split from viz.__init__).

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
    images: list[list[Any]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for col, (field, title) in enumerate(zip(field_stack, titles)):
        for row in range(channels):
            ax = axes[row][col]
            data = field[..., row]
            if mask is not None:
                data = np.where(mask, data, np.nan)
            image = ax.imshow(data, origin="lower", cmap=cmap_obj, vmin=vmin[row], vmax=vmax[row])
            images[row][col] = image
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0 and channels > 1:
                ax.set_ylabel(f"ch{row}")
            ax.set_aspect("equal")
    for row in range(n_rows):
        image = next((img for img in images[row] if img is not None), None)
        if image is not None:
            fig.colorbar(image, ax=axes[row, :], fraction=0.046, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vector_streamplot(
    path: str | Path,
    field: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    density: float = 1.2,
    linewidth: float = 1.0,
    cmap: str = "viridis",
    title: str | None = None,
    show_background: bool = True,
    background_alpha: float = 0.35,
) -> Path:
    field = _ensure_field_3d(field)
    if field.shape[2] < 2:
        raise ValueError("vector streamplot requires at least 2 channels")
    u = np.asarray(field[..., 0], dtype=float)
    v = np.asarray(field[..., 1], dtype=float)
    mask = _ensure_mask(mask, u.shape)
    if mask is not None:
        u = np.where(mask, u, np.nan)
        v = np.where(mask, v, np.nan)
    mag = np.sqrt(u**2 + v**2)
    height, width = u.shape
    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    if show_background:
        ax.imshow(
            mag,
            origin="lower",
            cmap=_colormap_with_bad("Greys"),
            alpha=float(background_alpha),
        )
    strm = ax.streamplot(
        x,
        y,
        u,
        v,
        density=float(density),
        color=mag,
        linewidth=linewidth,
        cmap=cmap,
    )
    fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04, label="|v|")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vector_quiver(
    path: str | Path,
    field: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    stride: int | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    show_background: bool = True,
    background_alpha: float = 0.35,
) -> Path:
    field = _ensure_field_3d(field)
    if field.shape[2] < 2:
        raise ValueError("vector quiver requires at least 2 channels")
    u = np.asarray(field[..., 0], dtype=float)
    v = np.asarray(field[..., 1], dtype=float)
    mask = _ensure_mask(mask, u.shape)
    height, width = u.shape
    if stride is None:
        stride = max(1, int(max(height, width) // 32))
    stride = max(1, int(stride))
    y_idx = np.arange(0, height, stride)
    x_idx = np.arange(0, width, stride)
    X, Y = np.meshgrid(x_idx, y_idx)
    U = u[y_idx[:, None], x_idx[None, :]]
    V = v[y_idx[:, None], x_idx[None, :]]
    if mask is not None:
        M = mask[y_idx[:, None], x_idx[None, :]]
        U = np.where(M, U, np.nan)
        V = np.where(M, V, np.nan)
    mag = np.sqrt(U**2 + V**2)
    if mask is not None:
        mag_full = np.where(mask, np.sqrt(u**2 + v**2), np.nan)
    else:
        mag_full = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
    if show_background:
        ax.imshow(
            mag_full,
            origin="lower",
            cmap=_colormap_with_bad("Greys"),
            alpha=float(background_alpha),
        )
    q = ax.quiver(
        X,
        Y,
        U,
        V,
        mag,
        cmap=cmap,
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0025,
        pivot="mid",
    )
    fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="|v|")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
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
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sphere_extent(domain_spec: Any) -> tuple[float, float, float, float] | None:
    if domain_spec is None or getattr(domain_spec, "name", "") != "sphere_grid":
        return None
    coords = domain_spec.coords or {}
    lat_deg = coords.get("lat_deg")
    lon_deg = coords.get("lon_deg")
    if lat_deg is None or lon_deg is None:
        return None
    return (
        float(np.min(lon_deg)),
        float(np.max(lon_deg)),
        float(np.min(lat_deg)),
        float(np.max(lat_deg)),
    )


def _sphere_mesh(domain_spec: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if domain_spec is None or getattr(domain_spec, "name", "") != "sphere_grid":
        return None
    coords = domain_spec.coords or {}
    lat_deg = coords.get("lat_deg")
    lon_deg = coords.get("lon_deg")
    if lat_deg is None or lon_deg is None:
        return None
    return np.asarray(lon_deg), np.asarray(lat_deg)


def _plot_sphere_field(
    ax: Any,
    data: np.ndarray,
    *,
    domain_spec: Any,
    cmap_obj: Any,
    vmin: float | None,
    vmax: float | None,
    projection: str,
) -> Any:
    if projection == "plate_carre":
        extent = _sphere_extent(domain_spec)
        image = ax.imshow(data, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax, extent=extent)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        return image
    if projection == "mollweide":
        mesh = _sphere_mesh(domain_spec)
        if mesh is None:
            raise ValueError("sphere_grid requires lat/lon coords for mollweide plot")
        lon_deg, lat_deg = mesh
        lon_rad = np.deg2rad(lon_deg)
        lat_rad = np.deg2rad(lat_deg)
        image = ax.pcolormesh(lon_rad, lat_rad, data, cmap=cmap_obj, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        return image
    raise ValueError(f"Unknown sphere projection: {projection}")


def plot_domain_field_grid(
    path: str | Path,
    fields: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
    suptitle: str | None = None,
    cmap: str = "viridis",
    sphere_projection: str = "plate_carre",
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

    extent = _sphere_extent(domain_spec)
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
    images: list[list[Any]] = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    for col, (field, title) in enumerate(zip(field_stack, titles)):
        for row in range(channels):
            ax = axes[row][col]
            data = field[..., row]
            if mask is not None:
                data = np.where(mask, data, np.nan)
            if getattr(domain_spec, "name", "") == "sphere_grid":
                image = _plot_sphere_field(
                    ax,
                    data,
                    domain_spec=domain_spec,
                    cmap_obj=cmap_obj,
                    vmin=vmin[row],
                    vmax=vmax[row],
                    projection=sphere_projection,
                )
            else:
                image = ax.imshow(
                    data,
                    origin="lower",
                    cmap=cmap_obj,
                    vmin=vmin[row],
                    vmax=vmax[row],
                    extent=extent,
                )
                ax.set_aspect("equal")
            images[row][col] = image
            ax.set_xticks([])
            ax.set_yticks([])
            if extent is not None:
                ax.set_xlabel("lon")
                ax.set_ylabel("lat")
            if row == 0:
                ax.set_title(title)
            if col == 0 and channels > 1:
                ax.set_ylabel(f"ch{row}")
    for row in range(n_rows):
        image = next((img for img in images[row] if img is not None), None)
        if image is not None:
            fig.colorbar(image, ax=axes[row, :], fraction=0.046, pad=0.04)
    if suptitle:
        fig.suptitle(suptitle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_domain_error_map(
    path: str | Path,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
    cmap: str = "magma",
    sphere_projection: str = "plate_carre",
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
    extent = _sphere_extent(domain_spec)

    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    cmap_obj = _colormap_with_bad(cmap)
    if getattr(domain_spec, "name", "") == "sphere_grid":
        image = _plot_sphere_field(
            ax,
            error,
            domain_spec=domain_spec,
            cmap_obj=cmap_obj,
            vmin=None,
            vmax=None,
            projection=sphere_projection,
        )
    else:
        image = ax.imshow(error, origin="lower", cmap=cmap_obj, extent=extent)
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if extent is not None:
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lat_profile(
    path: str | Path,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
) -> Path:
    coords = getattr(domain_spec, "coords", {}) or {}
    lat_deg = coords.get("lat_deg")
    if lat_deg is None:
        raise ValueError("domain_spec must include lat_deg for lat profile")
    field_true = _ensure_field_3d(field_true)[..., 0]
    field_pred = _ensure_field_3d(field_pred)[..., 0]
    mask = _ensure_mask(mask, field_true.shape)
    data_true = field_true
    data_pred = field_pred
    if mask is not None:
        data_true = np.where(mask, data_true, np.nan)
        data_pred = np.where(mask, data_pred, np.nan)
    lat_vals = np.asarray(lat_deg)
    lat_axis = lat_vals[:, 0]
    true_mean = np.nanmean(data_true, axis=1)
    pred_mean = np.nanmean(data_pred, axis=1)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.plot(lat_axis, true_mean, label="true")
    ax.plot(lat_axis, pred_mean, label="pred")
    ax.set_xlabel("lat (deg)")
    ax.set_ylabel("mean value")
    ax.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _profile_bins(values: np.ndarray, data: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values).reshape(-1)
    data = np.asarray(data).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(data)
    if not np.any(valid):
        raise ValueError("no valid entries for profile")
    values = values[valid]
    data = data[valid]
    edges = np.linspace(values.min(), values.max(), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(centers.shape, np.nan, dtype=np.float64)
    for i in range(bins):
        mask = (values >= edges[i]) & (values < edges[i + 1])
        if np.any(mask):
            out[i] = float(np.mean(data[mask]))
    return centers, out


def plot_radial_profile(
    path: str | Path,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
    bins: int = 40,
) -> Path:
    coords = getattr(domain_spec, "coords", {}) or {}
    r = coords.get("r")
    if r is None:
        raise ValueError("domain_spec must include r for radial profile")
    field_true = _ensure_field_3d(field_true)[..., 0]
    field_pred = _ensure_field_3d(field_pred)[..., 0]
    mask = _ensure_mask(mask, field_true.shape)
    if mask is not None:
        field_true = np.where(mask, field_true, np.nan)
        field_pred = np.where(mask, field_pred, np.nan)
    centers, true_mean = _profile_bins(r, field_true, bins=bins)
    _, pred_mean = _profile_bins(r, field_pred, bins=bins)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.plot(centers, true_mean, label="true")
    ax.plot(centers, pred_mean, label="pred")
    ax.set_xlabel("r")
    ax.set_ylabel("mean value")
    ax.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_angular_profile(
    path: str | Path,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
    bins: int = 60,
) -> Path:
    coords = getattr(domain_spec, "coords", {}) or {}
    theta = coords.get("theta")
    if theta is None:
        raise ValueError("domain_spec must include theta for angular profile")
    field_true = _ensure_field_3d(field_true)[..., 0]
    field_pred = _ensure_field_3d(field_pred)[..., 0]
    mask = _ensure_mask(mask, field_true.shape)
    if mask is not None:
        field_true = np.where(mask, field_true, np.nan)
        field_pred = np.where(mask, field_pred, np.nan)
    centers, true_mean = _profile_bins(theta, field_true, bins=bins)
    _, pred_mean = _profile_bins(theta, field_pred, bins=bins)

    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.plot(centers, true_mean, label="true")
    ax.plot(centers, pred_mean, label="pred")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("mean value")
    ax.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_polar_field(
    path: str | Path,
    field: np.ndarray,
    *,
    domain_spec: Any,
    mask: np.ndarray | None = None,
    cmap: str = "viridis",
) -> Path:
    coords = getattr(domain_spec, "coords", {}) or {}
    r = coords.get("r")
    theta = coords.get("theta")
    if r is None or theta is None:
        raise ValueError("domain_spec must include r/theta for polar plot")
    field = _ensure_field_3d(field)[..., 0]
    mask = _ensure_mask(mask, field.shape)
    if mask is not None:
        field = np.where(mask, field, np.nan)
    r = np.asarray(r)
    theta = np.asarray(theta)
    import matplotlib.tri as mtri

    r_flat = r.reshape(-1)
    theta_flat = theta.reshape(-1)
    field_flat = field.reshape(-1)
    valid = np.isfinite(r_flat) & np.isfinite(theta_flat) & np.isfinite(field_flat)
    if not np.any(valid):
        raise ValueError("polar field has no valid entries")
    tri = mtri.Triangulation(theta_flat[valid], r_flat[valid])
    fig = plt.figure(figsize=(4.0, 3.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="polar")
    cmap_obj = _colormap_with_bad(cmap)
    image = ax.tripcolor(tri, field_flat[valid], cmap=cmap_obj, shading="flat")
    ax.set_thetagrids(range(0, 360, 45))
    ax.set_rlabel_position(225)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mesh_field(
    path: str | Path,
    field: np.ndarray,
    *,
    domain_spec: Any,
    cmap: str = "viridis",
) -> Path:
    coords = getattr(domain_spec, "coords", {}) or {}
    vertices = coords.get("vertices")
    faces = coords.get("faces")
    if vertices is None or faces is None:
        raise ValueError("domain_spec must include vertices/faces for mesh plot")
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    values = _ensure_field_3d(field)[..., 0].reshape(-1)
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], faces)
    fig, ax = plt.subplots(figsize=(4.0, 3.2), constrained_layout=True)
    cmap_obj = _colormap_with_bad(cmap)
    tpc = ax.tripcolor(tri, values, cmap=cmap_obj, shading="flat")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
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


