"""Evaluation metrics."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def _ensure_field_batch(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 3:
        return field[None, ...]
    if field.ndim == 4:
        return field
    raise ValueError(f"field must be 3D or 4D, got shape {field.shape}")


def _ensure_mask_batch(mask: np.ndarray, expected: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3:
        raise ValueError(f"mask must be 2D or 3D, got shape {mask.shape}")
    if mask.shape != expected:
        raise ValueError(f"mask shape {mask.shape} does not match expected {expected}")
    return mask.astype(bool)


def _ensure_coeff_batch(coeff: np.ndarray, *, name: str) -> np.ndarray:
    coeff = np.asarray(coeff)
    if coeff.ndim == 1:
        return coeff[None, :]
    if coeff.ndim == 2:
        return coeff
    raise ValueError(f"{name} must be 1D or 2D, got shape {coeff.shape}")


def _ensure_scalar_batch(field: np.ndarray, *, name: str) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[None, ...]
    if field.ndim == 3:
        return field
    raise ValueError(f"{name} must be 2D or 3D, got shape {field.shape}")


def _ensure_vector_field_batch(field: np.ndarray, *, name: str) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 3:
        field = field[None, ...]
    if field.ndim != 4:
        raise ValueError(f"{name} must be 3D or 4D, got shape {field.shape}")
    if field.shape[-1] != 2:
        raise ValueError(f"{name} must have 2 channels, got shape {field.shape}")
    return field


def _normalize_grid_spacing(grid_spacing: tuple[float, float] | None) -> tuple[float, float]:
    if grid_spacing is None:
        return 1.0, 1.0
    if not isinstance(grid_spacing, (tuple, list)) or len(grid_spacing) != 2:
        raise ValueError("grid_spacing must be a pair (dx, dy)")
    dx = float(grid_spacing[0])
    dy = float(grid_spacing[1])
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("grid_spacing must be positive")
    return dx, dy


def vector_divergence(
    field: np.ndarray,
    *,
    grid_spacing: tuple[float, float] | None = None,
) -> np.ndarray:
    field_arr = np.asarray(field)
    was_single = field_arr.ndim == 3
    field_batch = _ensure_vector_field_batch(field_arr, name="field")
    dx, dy = _normalize_grid_spacing(grid_spacing)
    u = field_batch[..., 0]
    v = field_batch[..., 1]
    du_dx = np.gradient(u, dx, axis=2)
    dv_dy = np.gradient(v, dy, axis=1)
    div = du_dx + dv_dy
    return div[0] if was_single else div


def vector_curl(
    field: np.ndarray,
    *,
    grid_spacing: tuple[float, float] | None = None,
) -> np.ndarray:
    field_arr = np.asarray(field)
    was_single = field_arr.ndim == 3
    field_batch = _ensure_vector_field_batch(field_arr, name="field")
    dx, dy = _normalize_grid_spacing(grid_spacing)
    u = field_batch[..., 0]
    v = field_batch[..., 1]
    dv_dx = np.gradient(v, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    curl = dv_dx - du_dy
    return curl[0] if was_single else curl


def field_rmse(
    field_true: np.ndarray,
    field_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    field_true = _ensure_field_batch(field_true)
    field_pred = _ensure_field_batch(field_pred)
    if field_true.shape != field_pred.shape:
        raise ValueError("field_true and field_pred must have the same shape")
    diff = field_pred - field_true
    if mask is not None:
        mask_batch = _ensure_mask_batch(mask, field_true.shape[:3])
        if not mask_batch.any():
            raise ValueError("mask has no valid entries")
        diff = diff[mask_batch]
    return float(np.sqrt(np.mean(diff**2)))


def field_r2(
    field_true: np.ndarray,
    field_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean per-sample R^2 in field space (mask-aware, unweighted)."""
    a = _ensure_field_batch(field_true)
    b = _ensure_field_batch(field_pred)
    if a.shape != b.shape:
        raise ValueError("field_true and field_pred must have the same shape")

    mask_batch = None
    if mask is not None:
        mask_batch = _ensure_mask_batch(mask, a.shape[:3])

    r2_list: list[float] = []
    for i in range(a.shape[0]):
        ai = a[i]
        bi = b[i]
        if mask_batch is None:
            valid = np.isfinite(ai) & np.isfinite(bi)
        else:
            valid = mask_batch[i][..., None] & np.isfinite(ai) & np.isfinite(bi)
        if not np.any(valid):
            continue
        y = ai[valid]
        yhat = bi[valid]
        mu = float(np.mean(y))
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - mu) ** 2))
        if ss_tot <= 0:
            continue
        r2_list.append(1.0 - (ss_res / ss_tot))

    if not r2_list:
        return float("nan")
    return float(np.mean(r2_list))


def coeff_rmse(coeff_true: np.ndarray, coeff_pred: np.ndarray) -> float:
    coeff_true = _ensure_coeff_batch(coeff_true, name="coeff_true")
    coeff_pred = _ensure_coeff_batch(coeff_pred, name="coeff_pred")
    if coeff_true.shape != coeff_pred.shape:
        raise ValueError("coeff_true and coeff_pred must have the same shape")
    return float(np.sqrt(np.mean((coeff_pred - coeff_true) ** 2)))


def _scalar_rmse(
    field_true: np.ndarray,
    field_pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    field_true = _ensure_scalar_batch(field_true, name="field_true")
    field_pred = _ensure_scalar_batch(field_pred, name="field_pred")
    if field_true.shape != field_pred.shape:
        raise ValueError("field_true and field_pred must have the same shape")
    diff = field_pred - field_true
    if mask is not None:
        mask_batch = _ensure_mask_batch(mask, field_true.shape)
        if not mask_batch.any():
            raise ValueError("mask has no valid entries")
        diff = diff[mask_batch]
    return float(np.sqrt(np.mean(diff**2)))


def _coeff_energy_vector(
    coeff: np.ndarray,
    coeff_meta: Mapping[str, object] | None = None,
) -> np.ndarray:
    coeff = _ensure_coeff_batch(coeff, name="coeff")
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
                complex_format = str(coeff_meta.get("complex_format", "")).strip().lower()
                if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and coeff_shape[-1] == 2:
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
    return np.mean(coeff**2, axis=0)


def coeff_energy_cumsum(
    coeff: np.ndarray,
    coeff_meta: Mapping[str, object] | None = None,
) -> np.ndarray:
    coeff_batch = _ensure_coeff_batch(coeff, name="coeff")
    meta_effective: Mapping[str, object] | None = coeff_meta

    # Special case: offset_residual packed vectors. Compute energy on the residual slice only.
    if coeff_meta:
        coeff_format = str(coeff_meta.get("coeff_format", "")).strip().lower()
        raw_meta = coeff_meta.get("raw_meta")
        if coeff_format != "offset_residual_v1" and isinstance(raw_meta, Mapping):
            coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
        if coeff_format == "offset_residual_v1":
            offset_dim = coeff_meta.get("offset_dim")
            if offset_dim is None and isinstance(raw_meta, Mapping):
                offset_dim = raw_meta.get("offset_dim")
            try:
                offset_dim_i = int(offset_dim)
            except Exception:
                offset_dim_i = 0
            offset_dim_i = max(0, offset_dim_i)
            if offset_dim_i >= coeff_batch.shape[1]:
                return np.zeros(0, dtype=float)
            residual_coeff_meta = coeff_meta.get("residual_coeff_meta")
            meta_effective = dict(residual_coeff_meta) if isinstance(residual_coeff_meta, Mapping) else None
            coeff_batch = coeff_batch[:, offset_dim_i:]

    energy = _coeff_energy_vector(coeff_batch, coeff_meta=meta_effective)
    if meta_effective:
        # If coeffs are channel-first, aggregate energy across channels so the resulting
        # cumsum is comparable between scalar/vector fields (same mode ordering).
        channels = meta_effective.get("channels")
        coeff_shape = meta_effective.get("coeff_shape")
        if isinstance(channels, int) and isinstance(coeff_shape, list) and coeff_shape:
            try:
                ch = int(channels)
            except Exception:
                ch = -1
            complex_format = str(meta_effective.get("complex_format", "")).strip().lower()
            # _coeff_energy_vector collapses the trailing complex axis (size=2) into magnitude energy.
            effective_shape = coeff_shape[:-1] if complex_format in {"real_imag", "mag_phase", "logmag_phase"} and int(coeff_shape[-1]) == 2 else coeff_shape
            if effective_shape and int(effective_shape[0]) == ch:
                try:
                    expected = int(np.prod([int(x) for x in effective_shape]))
                except Exception:
                    expected = -1
                if expected > 0 and energy.size == expected:
                    order = str(coeff_meta.get("flatten_order", "C")).strip().upper() or "C"
                    try:
                        energy_reshaped = energy.reshape(tuple(int(x) for x in effective_shape), order=order)
                        energy = energy_reshaped.sum(axis=0).reshape(-1, order=order)
                    except Exception:
                        pass

        # FFT-specific ordering: for unshifted FFT coefficients, low frequencies are split
        # across array edges (negative freqs at the end). Row-major cumsum becomes misleading
        # (n_req ~ H*W). Reorder by frequency radius before cumsum.
        try:
            method = str(meta_effective.get("method", "")).strip().lower()
            if method in {"fft2", "fft2_lowpass"}:
                coeff_shape = meta_effective.get("coeff_shape")
                channels = meta_effective.get("channels")
                complex_format = str(meta_effective.get("complex_format", "")).strip().lower()
                order = str(meta_effective.get("flatten_order", "C")).strip().upper() or "C"
                if isinstance(coeff_shape, list) and coeff_shape:
                    shape = [int(x) for x in coeff_shape]
                    # Drop explicit complex axis (real/imag, mag/phase, logmag/phase) if present.
                    if (
                        shape
                        and int(shape[-1]) == 2
                        and complex_format in {"real_imag", "mag_phase", "logmag_phase"}
                    ):
                        shape = shape[:-1]
                    # If energy has already been aggregated across channels, drop the channel axis.
                    if (
                        isinstance(channels, int)
                        and shape
                        and int(shape[0]) == int(channels)
                        and energy.size == int(np.prod([int(x) for x in shape[1:]]))
                    ):
                        shape = shape[1:]
                    if len(shape) == 2 and energy.size == int(shape[0] * shape[1]):
                        h = int(shape[0])
                        w = int(shape[1])
                        if h > 0 and w > 0:
                            fft_shift = bool(meta_effective.get("fft_shift", False))
                            e2 = energy.reshape((h, w), order=order)
                            ky = np.fft.fftfreq(h) * float(h)
                            kx = np.fft.fftfreq(w) * float(w)
                            if fft_shift:
                                ky = np.fft.fftshift(ky)
                                kx = np.fft.fftshift(kx)
                            ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
                            radius = np.sqrt(kx_grid**2 + ky_grid**2)
                            sort_idx = np.argsort(radius.reshape(-1, order=order), kind="stable")
                            energy = e2.reshape(-1, order=order)[sort_idx]
        except Exception:
            pass

        nm_list = meta_effective.get("nm_list")
        if isinstance(nm_list, list) and nm_list:
            n_modes = len(nm_list)
            # Support both pre- and post-channel-aggregation energy vectors.
            energy_modes: np.ndarray | None = None
            if isinstance(channels, int) and energy.size == int(channels) * n_modes:
                energy_modes = energy.reshape(int(channels), n_modes).sum(axis=0)
            elif energy.size == n_modes:
                energy_modes = energy.reshape(n_modes)
            if energy_modes is not None:
                degrees = np.asarray([int(pair[0]) for pair in nm_list], dtype=int)
                unique = np.unique(degrees)
                energy_by_degree = np.array(
                    [energy_modes[degrees == degree].sum() for degree in unique],
                    dtype=float,
                )
                total = float(energy_by_degree.sum())
                if total <= 0.0:
                    return np.zeros_like(energy_by_degree)
                return np.cumsum(energy_by_degree) / total
    total = float(np.sum(energy))
    if total <= 0.0:
        return np.zeros_like(energy)
    return np.cumsum(energy) / total


def compute_metrics(
    names: Iterable[str],
    *,
    field_true: np.ndarray,
    field_pred: np.ndarray,
    mask: np.ndarray | None = None,
    coeff_true_a: np.ndarray | None = None,
    coeff_pred_a: np.ndarray | None = None,
    coeff_true_z: np.ndarray | None = None,
    coeff_pred_z: np.ndarray | None = None,
    coeff_true: np.ndarray | None = None,
    coeff_pred: np.ndarray | None = None,
    coeff_meta: Mapping[str, object] | None = None,
    grid_spacing: tuple[float, float] | None = None,
) -> dict[str, float | np.ndarray]:
    metrics: dict[str, float | np.ndarray] = {}
    for name in names:
        if name == "field_rmse":
            metrics[name] = field_rmse(field_true, field_pred, mask=mask)
            continue
        if name == "field_r2":
            metrics[name] = field_r2(field_true, field_pred, mask=mask)
            continue
        if name == "div_rmse":
            div_true = vector_divergence(field_true, grid_spacing=grid_spacing)
            div_pred = vector_divergence(field_pred, grid_spacing=grid_spacing)
            metrics[name] = _scalar_rmse(div_true, div_pred, mask=mask)
            continue
        if name == "curl_rmse":
            curl_true = vector_curl(field_true, grid_spacing=grid_spacing)
            curl_pred = vector_curl(field_pred, grid_spacing=grid_spacing)
            metrics[name] = _scalar_rmse(curl_true, curl_pred, mask=mask)
            continue
        if name == "coeff_rmse":
            if coeff_true is None or coeff_pred is None:
                if coeff_true_a is None or coeff_pred_a is None:
                    raise ValueError("coeff_true/pred are required for coeff_rmse")
                metrics[name] = coeff_rmse(coeff_true_a, coeff_pred_a)
                continue
            metrics[name] = coeff_rmse(coeff_true, coeff_pred)
            continue
        if name == "coeff_rmse_a":
            if coeff_true_a is None or coeff_pred_a is None:
                raise ValueError("coeff_true_a/pred_a are required for coeff_rmse_a")
            metrics[name] = coeff_rmse(coeff_true_a, coeff_pred_a)
            continue
        if name == "coeff_rmse_z":
            if coeff_true_z is None or coeff_pred_z is None:
                raise ValueError("coeff_true_z/pred_z are required for coeff_rmse_z")
            metrics[name] = coeff_rmse(coeff_true_z, coeff_pred_z)
            continue
        if name == "energy_cumsum":
            if coeff_true_a is None:
                raise ValueError("coeff_true_a is required for energy_cumsum")
            # CONTRACT: energy_cumsum uses a-space coefficients for comparability.
            metrics[name] = coeff_energy_cumsum(coeff_true_a, coeff_meta=coeff_meta)
            continue
        raise ValueError(f"Unknown metric: {name}")
    return metrics


__all__ = [
    "coeff_energy_cumsum",
    "coeff_rmse",
    "compute_metrics",
    "field_r2",
    "field_rmse",
    "vector_curl",
    "vector_divergence",
]
