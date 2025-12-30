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
        diff = diff[mask_batch[..., None]]
    return float(np.sqrt(np.mean(diff**2)))


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
                energy = np.mean(reshaped**2, axis=0)
                if (
                    str(coeff_meta.get("complex_format", "")) == "real_imag"
                    and len(coeff_shape) >= 1
                    and coeff_shape[-1] == 2
                ):
                    energy = energy.sum(axis=-1)
                return energy.reshape(-1, order=flatten_order)
    return np.mean(coeff**2, axis=0)


def coeff_energy_cumsum(
    coeff: np.ndarray,
    coeff_meta: Mapping[str, object] | None = None,
) -> np.ndarray:
    energy = _coeff_energy_vector(coeff, coeff_meta=coeff_meta)
    if coeff_meta:
        nm_list = coeff_meta.get("nm_list")
        channels = coeff_meta.get("channels")
        if isinstance(nm_list, list) and nm_list and isinstance(channels, int):
            n_modes = len(nm_list)
            if energy.size == channels * n_modes:
                energy_modes = energy.reshape(channels, n_modes).sum(axis=0)
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
    "field_rmse",
    "vector_curl",
    "vector_divergence",
]
