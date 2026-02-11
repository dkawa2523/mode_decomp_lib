"""Plot helpers for the decomposition process (split from processes.decomposition)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_decomp_ml.data import FieldSample, build_dataset
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import compute_metrics, field_r2
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.preprocess import build_preprocess
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.pipeline.process_base import finalize_run, init_run
from mode_decomp_ml.viz import (
    coeff_energy_vector,
    coeff_energy_spectrum,
    coeff_value_magnitude,
    masked_weighted_r2,
    per_pixel_r2_map,
    plot_coeff_hist_by_mode,
    plot_coeff_histogram,
    plot_coeff_spectrum,
    plot_domain_error_map,
    plot_domain_field_grid,
    plot_error_map,
    plot_field_grid,
    plot_line_with_band,
    plot_line_series,
    plot_scatter_true_pred,
    sample_scatter_points,
    plot_vector_quiver,
    plot_vector_streamplot,
)


def _field_magnitude(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 2:
        return np.abs(field)
    if field.ndim == 3:
        if field.shape[2] == 1:
            return np.abs(field[..., 0])
        return np.linalg.norm(field, axis=-1)
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _resolve_cond_names(dataset: Any, conds: np.ndarray) -> list[str]:
    if hasattr(dataset, "cond_columns"):
        cols = getattr(dataset, "cond_columns")
        if isinstance(cols, list) and cols:
            return [str(col) for col in cols]
    manifest = getattr(dataset, "manifest", None)
    if isinstance(manifest, Mapping):
        cols = manifest.get("cond_columns")
        if isinstance(cols, list) and cols:
            return [str(col) for col in cols]
    n_features = conds.shape[1] if conds.ndim == 2 else 0
    return [f"x{i + 1}" for i in range(n_features)]


def _mode_energy_from_meta(coeffs: np.ndarray, coeff_meta: Mapping[str, Any]) -> np.ndarray:
    eigvals = coeff_meta.get("eigvals")
    if isinstance(eigvals, list) and eigvals:
        if isinstance(eigvals[0], list):
            arrays = [np.asarray(item, dtype=float) for item in eigvals if item]
            if arrays:
                min_len = min(arr.size for arr in arrays if arr.size > 0)
                if min_len > 0:
                    stacked = np.stack([arr[:min_len] for arr in arrays], axis=0)
                    return np.mean(stacked, axis=0)
        else:
            arr = np.asarray(eigvals, dtype=float)
            if arr.size > 0:
                return arr
    energy_vec = coeff_energy_vector(coeffs, coeff_meta)
    coeff_shape = coeff_meta.get("coeff_shape")
    channels = int(coeff_meta.get("channels", 1) or 1)
    if isinstance(coeff_shape, list) and len(coeff_shape) == 2 and int(coeff_shape[0]) == channels:
        n_modes = int(coeff_shape[1])
        if energy_vec.size >= channels * n_modes:
            reshaped = energy_vec[: channels * n_modes].reshape(channels, n_modes)
            return np.sum(reshaped, axis=0)
    return energy_vec.reshape(-1)


def _plot_spatial_stats(
    *,
    writer: ArtifactWriter,
    fields_true: np.ndarray,
    fields_pred: np.ndarray,
    masks: np.ndarray | None,
    domain_mask: np.ndarray | None,
    viz_cfg: Mapping[str, Any],
) -> None:
    stats_cfg = cfg_get(viz_cfg, "spatial_stats", {}) or {}
    if not bool(cfg_get(stats_cfg, "enabled", True)):
        return
    show_pred = bool(cfg_get(stats_cfg, "show_pred", True))
    show_error = bool(cfg_get(stats_cfg, "show_error", True))

    mask_plot = None
    if domain_mask is not None:
        mask_plot = np.asarray(domain_mask).astype(bool)
    if masks is not None:
        mask_all = np.all(np.asarray(masks).astype(bool), axis=0)
        mask_plot = mask_all if mask_plot is None else (mask_plot & mask_all)
    if mask_plot is not None and not np.any(mask_plot):
        return

    def _masked_mean(field: np.ndarray, mask_2d: np.ndarray | None) -> np.ndarray:
        if mask_2d is None:
            return np.mean(field, axis=0)
        mask_exp = mask_2d[None, ..., None] if field.ndim == 4 else mask_2d[None, ...]
        valid = mask_exp.astype(bool)
        count = valid.sum(axis=0)
        total = np.where(valid, field, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        return np.where(count > 0, mean, np.nan)

    def _masked_std(field: np.ndarray, mask_2d: np.ndarray | None) -> np.ndarray:
        if mask_2d is None:
            return np.std(field, axis=0)
        mean = _masked_mean(field, mask_2d)
        mask_exp = mask_2d[None, ..., None] if field.ndim == 4 else mask_2d[None, ...]
        valid = mask_exp.astype(bool)
        diff2 = np.where(valid, (field - mean) ** 2, 0.0)
        count = valid.sum(axis=0)
        var = np.divide(diff2.sum(axis=0), count, out=np.zeros_like(diff2.sum(axis=0)), where=count > 0)
        return np.where(count > 0, np.sqrt(var), np.nan)

    def _masked_rmse(diff: np.ndarray, mask_2d: np.ndarray | None) -> np.ndarray:
        if mask_2d is None:
            return np.sqrt(np.mean(diff**2, axis=0))
        mask_exp = mask_2d[None, ..., None] if diff.ndim == 4 else mask_2d[None, ...]
        valid = mask_exp.astype(bool)
        count = valid.sum(axis=0)
        total = np.where(valid, diff**2, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        return np.where(count > 0, np.sqrt(mean), np.nan)

    fields_true = np.asarray(fields_true)
    fields_pred = np.asarray(fields_pred)
    mean_true = _masked_mean(fields_true, mask_plot)
    std_true = _masked_std(fields_true, mask_plot)
    plot_field_grid(
        writer.plots_dir / "field_true_mean.png",
        [mean_true],
        ["mean_true"],
        mask=mask_plot,
        suptitle="field mean (true)",
    )
    plot_field_grid(
        writer.plots_dir / "field_true_std.png",
        [std_true],
        ["std_true"],
        mask=mask_plot,
        suptitle="field std (true)",
    )
    if mean_true.ndim == 3 and mean_true.shape[-1] > 1:
        try:
            plot_vector_streamplot(
                writer.plots_dir / "field_true_mean_stream.png",
                mean_true,
                mask=mask_plot,
            )
            plot_vector_quiver(
                writer.plots_dir / "field_true_mean_quiver.png",
                mean_true,
                mask=mask_plot,
            )
        except Exception:
            pass

    if show_pred:
        mean_pred = _masked_mean(fields_pred, mask_plot)
        std_pred = _masked_std(fields_pred, mask_plot)
        plot_field_grid(
            writer.plots_dir / "field_recon_mean.png",
            [mean_pred],
            ["mean_recon"],
            mask=mask_plot,
            suptitle="field mean (recon)",
        )
        plot_field_grid(
            writer.plots_dir / "field_recon_std.png",
            [std_pred],
            ["std_recon"],
            mask=mask_plot,
            suptitle="field std (recon)",
        )
        if mean_pred.ndim == 3 and mean_pred.shape[-1] > 1:
            try:
                plot_vector_streamplot(
                    writer.plots_dir / "field_recon_mean_stream.png",
                    mean_pred,
                    mask=mask_plot,
                )
                plot_vector_quiver(
                    writer.plots_dir / "field_recon_mean_quiver.png",
                    mean_pred,
                    mask=mask_plot,
                )
            except Exception:
                pass

    if show_error:
        diff = fields_pred - fields_true
        rmse = _masked_rmse(diff, mask_plot)
        plot_field_grid(
            writer.plots_dir / "field_error_rmse.png",
            [rmse],
            ["rmse"],
            mask=mask_plot,
            suptitle="field RMSE map",
        )
        if fields_true.ndim == 4 and fields_true.shape[-1] > 1:
            mag_true = np.stack([_field_magnitude(arr) for arr in fields_true], axis=0)
            mag_pred = np.stack([_field_magnitude(arr) for arr in fields_pred], axis=0)
            mag_mean = _masked_mean(mag_true, mask_plot)
            mag_std = _masked_std(mag_true, mask_plot)
            plot_field_grid(
                writer.plots_dir / "field_true_mag_mean.png",
                [mag_mean],
                ["mean_true_mag"],
                mask=mask_plot,
                suptitle="field magnitude mean (true)",
            )
            plot_field_grid(
                writer.plots_dir / "field_true_mag_std.png",
                [mag_std],
                ["std_true_mag"],
                mask=mask_plot,
                suptitle="field magnitude std (true)",
            )
            if show_pred:
                mag_mean_pred = _masked_mean(mag_pred, mask_plot)
                mag_std_pred = _masked_std(mag_pred, mask_plot)
                plot_field_grid(
                    writer.plots_dir / "field_recon_mag_mean.png",
                    [mag_mean_pred],
                    ["mean_recon_mag"],
                    mask=mask_plot,
                    suptitle="field magnitude mean (recon)",
                )
                plot_field_grid(
                    writer.plots_dir / "field_recon_mag_std.png",
                    [mag_std_pred],
                    ["std_recon_mag"],
                    mask=mask_plot,
                    suptitle="field magnitude std (recon)",
                )
            mag_rmse = _masked_rmse(mag_pred - mag_true, mask_plot)
            plot_field_grid(
                writer.plots_dir / "field_error_mag_rmse.png",
                [mag_rmse],
                ["rmse_mag"],
                mask=mask_plot,
                suptitle="field magnitude RMSE",
            )


def _basis_list_from_decomposer(decomposer: Any) -> list[np.ndarray] | None:
    impl = getattr(decomposer, "_impl", None)
    if impl is not None:
        return _basis_list_from_decomposer(impl)
    sub_list = getattr(decomposer, "_decomposers", None)
    if isinstance(sub_list, list) and sub_list:
        basis_list: list[np.ndarray] = []
        for sub in sub_list:
            sub_basis = _basis_list_from_decomposer(sub)
            if not sub_basis or len(sub_basis) != 1:
                return None
            basis_list.append(sub_basis[0])
        return basis_list
    basis = getattr(decomposer, "_basis", None)
    if isinstance(basis, list) and basis:
        return basis
    modes = getattr(decomposer, "modes", None)
    if isinstance(modes, np.ndarray):
        return [modes]
    models = getattr(decomposer, "_models", None)
    if isinstance(models, list) and models:
        basis_list = []
        for model in models:
            components = getattr(model, "components_", None)
            if components is None:
                return None
            basis_list.append(np.asarray(components, dtype=float).T)
        return basis_list
    return None


def _plot_corr_heatmap(
    path: Path,
    corr: np.ndarray,
    cond_names: Sequence[str],
    mode_labels: Sequence[str],
) -> None:
    n_rows, n_cols = corr.shape
    fig_w = min(24.0, 0.45 * n_cols + 4.0)
    fig_h = min(14.0, 0.35 * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    max_yticks = 24
    max_xticks = 24
    y_stride = max(1, int(np.ceil(n_rows / max_yticks)))
    x_stride = max(1, int(np.ceil(n_cols / max_xticks)))
    y_ticks = np.arange(0, n_rows, y_stride)
    x_ticks = np.arange(0, n_cols, x_stride)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(cond_names[idx]) for idx in y_ticks])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(mode_labels[idx]) for idx in x_ticks], rotation=45, ha="right")
    ax.set_xlabel("mode")
    ax.set_ylabel("condition")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mask_indices_from_decomposer(decomposer: Any) -> np.ndarray | None:
    impl = getattr(decomposer, "_impl", None)
    if impl is not None:
        return _mask_indices_from_decomposer(impl)
    sub_list = getattr(decomposer, "_decomposers", None)
    if isinstance(sub_list, list) and sub_list:
        return _mask_indices_from_decomposer(sub_list[0])
    mask_indices = getattr(decomposer, "_mask_indices", None)
    if mask_indices is None:
        return None
    return np.asarray(mask_indices, dtype=int)


def _grid_from_coeff_meta(coeff_meta: Mapping[str, Any]) -> tuple[int, int] | None:
    field_shape = coeff_meta.get("field_shape")
    if isinstance(field_shape, list) and len(field_shape) >= 2:
        return int(field_shape[0]), int(field_shape[1])
    return None


def _vector_to_grid(
    vec: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    mask_indices: np.ndarray | None,
) -> np.ndarray | None:
    vec = np.asarray(vec).reshape(-1)
    height, width = grid_shape
    if mask_indices is not None:
        if vec.size != mask_indices.size:
            return None
        full = np.zeros(height * width, dtype=vec.dtype)
        full[mask_indices] = vec
        return full.reshape(height, width)
    if vec.size != height * width:
        return None
    return vec.reshape(height, width)


def _plot_data_driven_diagnostics(
    *,
    writer: ArtifactWriter,
    coeffs: np.ndarray,
    coeff_meta: Mapping[str, Any],
    decomposer: Any,
    domain_spec: Any,
    viz_cfg: Mapping[str, Any],
    conds: np.ndarray | None,
    cond_names: Sequence[str] | None,
) -> None:
    data_cfg = cfg_get(viz_cfg, "data_driven", {}) or {}
    if not bool(cfg_get(data_cfg, "enabled", True)):
        return

    method = str(coeff_meta.get("method", "")).strip().lower()
    projection = str(coeff_meta.get("projection", "")).strip().lower()
    base_method = str(coeff_meta.get("base_method", "")).strip().lower()
    data_driven = {"pod", "pod_svd", "gappy_pod", "dict_learning", "autoencoder"}
    if method not in data_driven and base_method not in data_driven and projection not in {"pod", "dictionary_learning"}:
        return

    mode_energy = _mode_energy_from_meta(coeffs, coeff_meta)
    if mode_energy.size == 0:
        return
    x = np.arange(1, mode_energy.size + 1)
    plot_line_series(
        writer.plots_dir / "scree.png",
        x,
        mode_energy,
        xlabel="mode index",
        ylabel="energy",
    )
    total = float(np.sum(mode_energy))
    if total > 0:
        energy_cum = np.cumsum(mode_energy) / total
        plot_line_series(
            writer.plots_dir / "energy_cum.png",
            x,
            energy_cum,
            xlabel="mode index",
            ylabel="cumulative energy",
        )
        plot_line_series(
            writer.plots_dir / "recon_error_vs_k.png",
            x,
            1.0 - energy_cum,
            xlabel="mode index",
            ylabel="residual energy",
        )

    coeff_shape = coeff_meta.get("coeff_shape")
    channels = int(coeff_meta.get("channels", 1) or 1)
    coeff_arr = np.asarray(coeffs)
    coeff_by_mode: np.ndarray | None = None
    if isinstance(coeff_shape, list) and len(coeff_shape) == 2 and int(coeff_shape[0]) == channels:
        n_modes = int(coeff_shape[1])
        if coeff_arr.shape[1] >= channels * n_modes:
            coeff_tensor = coeff_arr[:, : channels * n_modes].reshape(coeff_arr.shape[0], channels, n_modes)
            if channels > 1:
                coeff_by_mode = np.linalg.norm(coeff_tensor, axis=1)
            else:
                coeff_by_mode = coeff_tensor[:, 0, :]
    if coeff_by_mode is None:
        coeff_by_mode = coeff_arr

    top_k = int(cfg_get(data_cfg, "coeff_top_k", 6))
    top_k = max(1, min(top_k, coeff_by_mode.shape[1]))
    order = np.argsort(-mode_energy)[:top_k]
    n_cols = int(cfg_get(data_cfg, "coeff_hist_cols", 3))
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(top_k / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for idx, mode_idx in enumerate(order):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        values = coeff_by_mode[:, int(mode_idx)].reshape(-1)
        ax.hist(values, bins=40, color="#4C78A8", alpha=0.85)
        ax.set_title(f"mode {int(mode_idx)}")
        ax.set_xlabel("coeff")
        ax.set_ylabel("count")
    for idx in range(top_k, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis("off")
    fig.savefig(writer.plots_dir / "coeff_component_hist.png", dpi=150)
    plt.close(fig)

    if conds is not None and cond_names and coeff_by_mode.size > 0:
        conds = np.asarray(conds)
        if conds.ndim == 2 and conds.shape[0] == coeff_by_mode.shape[0] and conds.shape[0] > 1:
            corr_top_k = int(cfg_get(data_cfg, "corr_top_k", 12))
            corr_top_k = max(1, min(corr_top_k, coeff_by_mode.shape[1]))
            x_vals = conds
            y_vals = coeff_by_mode
            x_mean = np.mean(x_vals, axis=0, keepdims=True)
            y_mean = np.mean(y_vals, axis=0, keepdims=True)
            x_std = np.std(x_vals, axis=0, keepdims=True)
            y_std = np.std(y_vals, axis=0, keepdims=True)
            x_std = np.where(x_std == 0, np.nan, x_std)
            y_std = np.where(y_std == 0, np.nan, y_std)
            x_norm = (x_vals - x_mean) / x_std
            y_norm = (y_vals - y_mean) / y_std
            denom = max(x_vals.shape[0] - 1, 1)
            corr_full = (x_norm.T @ y_norm) / float(denom)
            corr_full = np.nan_to_num(corr_full, nan=0.0, posinf=0.0, neginf=0.0)

            corr_order = np.argsort(-mode_energy)[:corr_top_k]
            corr_top = corr_full[:, corr_order]
            _plot_corr_heatmap(
                writer.plots_dir / "cond_coeff_corr.png",
                corr_top,
                cond_names,
                [f"m{int(idx)}" for idx in corr_order],
            )
            _plot_corr_heatmap(
                writer.plots_dir / "cond_coeff_corr_all.png",
                corr_full,
                cond_names,
                [f"m{idx}" for idx in range(corr_full.shape[1])],
            )

    basis_list = _basis_list_from_decomposer(decomposer)
    grid_shape = _grid_from_coeff_meta(coeff_meta)
    if not basis_list or grid_shape is None:
        return
    mask_indices = _mask_indices_from_decomposer(decomposer)
    n_modes = min(basis.shape[1] for basis in basis_list if basis.ndim == 2)
    max_modes = int(cfg_get(data_cfg, "max_modes", 9))
    max_modes = max(1, min(max_modes, n_modes))
    mode_indices = order[:max_modes]
    fields: list[np.ndarray] = []
    titles: list[str] = []
    for mode_idx in mode_indices:
        mode_idx = int(mode_idx)
        field = np.zeros((grid_shape[0], grid_shape[1], len(basis_list)), dtype=float)
        valid = True
        for ch, basis in enumerate(basis_list):
            if basis.ndim != 2 or mode_idx >= basis.shape[1]:
                valid = False
                break
            grid = _vector_to_grid(basis[:, mode_idx], grid_shape=grid_shape, mask_indices=mask_indices)
            if grid is None:
                valid = False
                break
            field[..., ch] = grid
        if not valid:
            continue
        fields.append(field)
        titles.append(f"mode_{mode_idx:02d}")
    if fields:
        plot_field_grid(
            writer.plots_dir / "modes_gallery.png",
            fields,
            titles,
            mask=domain_spec.mask,
            suptitle="data-driven modes",
        )
        if fields[0].shape[-1] > 1:
            mags = [_field_magnitude(field) for field in fields]
            plot_field_grid(
                writer.plots_dir / "modes_gallery_mag.png",
                mags,
                titles,
                mask=domain_spec.mask,
                suptitle="data-driven mode magnitude",
            )


def _plot_validity_diagnostics(
    *,
    writer: ArtifactWriter,
    fields_true: np.ndarray,
    fields_pred: np.ndarray,
    masks: np.ndarray | None,
    coeffs: np.ndarray,
    coeff_meta: Mapping[str, Any],
    decomposer: Any,
    domain_spec: Any,
    viz_cfg: Mapping[str, Any],
    codec: Any,
    raw_meta: Mapping[str, Any],
    preprocess: Any,
) -> None:
    validity_cfg = cfg_get(viz_cfg, "validity", {}) or {}
    if not bool(cfg_get(validity_cfg, "enabled", True)):
        return

    n_samples = int(fields_true.shape[0])
    domain_mask = getattr(domain_spec, "mask", None)
    weights = None
    try:
        weights = domain_spec.integration_weights()
    except Exception:
        weights = None

    # --- mask stats
    mask_stats_cfg = cfg_get(validity_cfg, "mask_stats", {}) or {}
    if bool(cfg_get(mask_stats_cfg, "enabled", True)):
        if masks is not None:
            frac = np.mean(np.asarray(masks).astype(bool).reshape(n_samples, -1), axis=1)
        elif domain_mask is not None:
            frac = np.full((n_samples,), float(np.mean(np.asarray(domain_mask).astype(bool))), dtype=float)
        else:
            frac = np.ones((n_samples,), dtype=float)

        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        ax.hist(frac, bins=40, color="#4C78A8", alpha=0.85)
        ax.set_xlabel("mask valid fraction")
        ax.set_ylabel("count")
        fig.savefig(writer.plots_dir / "mask_fraction_hist.png", dpi=150)
        plt.close(fig)

        if masks is not None:
            cov = np.mean(np.asarray(masks).astype(bool), axis=0).astype(float)
            plot_field_grid(
                writer.plots_dir / "mask_coverage_mean.png",
                [cov],
                ["coverage"],
                mask=domain_mask,
                suptitle="mask coverage mean",
                cmap="magma",
            )

    # --- scatter (true vs recon)
    scatter_cfg = cfg_get(validity_cfg, "scatter", {}) or {}
    if bool(cfg_get(scatter_cfg, "enabled", True)):
        max_samples = int(cfg_get(scatter_cfg, "max_samples", 8))
        max_points = int(cfg_get(scatter_cfg, "max_points", 200000))
        seed = int(cfg_get(scatter_cfg, "seed", 0))
        per_channel = bool(cfg_get(scatter_cfg, "per_channel", True))
        show_mag = bool(cfg_get(scatter_cfg, "magnitude", True))
        idxs = np.arange(min(max_samples, n_samples), dtype=int)

        mask_s = None
        if masks is not None:
            mask_s = np.asarray(masks)[idxs]
        elif domain_mask is not None:
            mask_s = np.broadcast_to(np.asarray(domain_mask).astype(bool), fields_true[idxs].shape[:3])

        ft = np.asarray(fields_true)[idxs]
        fp = np.asarray(fields_pred)[idxs]
        if ft.ndim == 4 and ft.shape[-1] > 1:
            if per_channel:
                for ch in range(ft.shape[-1]):
                    x, y = sample_scatter_points(ft[..., ch], fp[..., ch], mask=mask_s, max_points=max_points, seed=seed)
                    if x.size == 0:
                        continue
                    r2 = masked_weighted_r2(x, y)
                    plot_scatter_true_pred(
                        writer.plots_dir / f"field_scatter_true_vs_recon_ch{ch}.png",
                        x,
                        y,
                        title=f"field scatter ch{ch}",
                        r2=r2,
                        max_points=max_points,
                    )
            if show_mag:
                mag_t = np.linalg.norm(ft, axis=-1)
                mag_p = np.linalg.norm(fp, axis=-1)
                x, y = sample_scatter_points(mag_t, mag_p, mask=mask_s, max_points=max_points, seed=seed)
                if x.size:
                    r2 = masked_weighted_r2(x, y)
                    plot_scatter_true_pred(
                        writer.plots_dir / "field_scatter_true_vs_recon_mag.png",
                        x,
                        y,
                        title="field scatter magnitude",
                        r2=r2,
                        max_points=max_points,
                    )
        else:
            x, y = sample_scatter_points(ft[..., 0] if ft.ndim == 4 else ft, fp[..., 0] if fp.ndim == 4 else fp, mask=mask_s, max_points=max_points, seed=seed)
            if x.size:
                r2 = masked_weighted_r2(x, y)
                plot_scatter_true_pred(
                    writer.plots_dir / "field_scatter_true_vs_recon_ch0.png",
                    x,
                    y,
                    title="field scatter",
                    r2=r2,
                    max_points=max_points,
                )

    # --- per-pixel R^2 maps
    pixel_cfg = cfg_get(validity_cfg, "per_pixel_r2", {}) or {}
    if bool(cfg_get(pixel_cfg, "enabled", True)):
        max_samples = int(cfg_get(pixel_cfg, "max_samples", 64))
        downsample = int(cfg_get(pixel_cfg, "downsample", 1))
        hist_bins = int(cfg_get(pixel_cfg, "hist_bins", 60))
        per_channel = bool(cfg_get(pixel_cfg, "per_channel", True))
        show_mag = bool(cfg_get(pixel_cfg, "magnitude", True))
        idxs = np.arange(min(max_samples, n_samples), dtype=int)
        mask_s = None
        if masks is not None:
            mask_s = np.asarray(masks)[idxs]
        elif domain_mask is not None:
            mask_s = np.broadcast_to(np.asarray(domain_mask).astype(bool), fields_true[idxs].shape[:3])

        ft = np.asarray(fields_true)[idxs]
        fp = np.asarray(fields_pred)[idxs]
        if ft.ndim == 4 and ft.shape[-1] > 1:
            if per_channel:
                for ch in range(ft.shape[-1]):
                    r2_map = per_pixel_r2_map(ft[..., ch], fp[..., ch], mask=mask_s, downsample=downsample)
                    plot_field_grid(
                        writer.plots_dir / f"per_pixel_r2_map_ch{ch}.png",
                        [r2_map],
                        [f"r2_ch{ch}"],
                        mask=domain_mask[::downsample, ::downsample] if (domain_mask is not None and downsample > 1) else domain_mask,
                        suptitle=f"per-pixel R^2 (ch{ch})",
                        cmap="magma",
                    )
                    vals = r2_map[np.isfinite(r2_map)]
                    if vals.size:
                        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
                        ax.hist(vals, bins=hist_bins, color="#54A24B", alpha=0.85)
                        ax.set_xlabel("R^2")
                        ax.set_ylabel("count")
                        fig.savefig(writer.plots_dir / f"per_pixel_r2_hist_ch{ch}.png", dpi=150)
                        plt.close(fig)
            if show_mag:
                mag_t = np.linalg.norm(ft, axis=-1)
                mag_p = np.linalg.norm(fp, axis=-1)
                r2_map = per_pixel_r2_map(mag_t, mag_p, mask=mask_s, downsample=downsample)
                plot_field_grid(
                    writer.plots_dir / "per_pixel_r2_map_mag.png",
                    [r2_map],
                    ["r2_mag"],
                    mask=domain_mask[::downsample, ::downsample] if (domain_mask is not None and downsample > 1) else domain_mask,
                    suptitle="per-pixel R^2 (magnitude)",
                    cmap="magma",
                )
                vals = r2_map[np.isfinite(r2_map)]
                if vals.size:
                    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
                    ax.hist(vals, bins=hist_bins, color="#54A24B", alpha=0.85)
                    ax.set_xlabel("R^2")
                    ax.set_ylabel("count")
                    fig.savefig(writer.plots_dir / "per_pixel_r2_hist_mag.png", dpi=150)
                    plt.close(fig)
        else:
            ft_s = ft[..., 0] if ft.ndim == 4 else ft
            fp_s = fp[..., 0] if fp.ndim == 4 else fp
            r2_map = per_pixel_r2_map(ft_s, fp_s, mask=mask_s, downsample=downsample)
            plot_field_grid(
                writer.plots_dir / "per_pixel_r2_map_ch0.png",
                [r2_map],
                ["r2"],
                mask=domain_mask[::downsample, ::downsample] if (domain_mask is not None and downsample > 1) else domain_mask,
                suptitle="per-pixel R^2",
                cmap="magma",
            )
            vals = r2_map[np.isfinite(r2_map)]
            if vals.size:
                fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
                ax.hist(vals, bins=hist_bins, color="#54A24B", alpha=0.85)
                ax.set_xlabel("R^2")
                ax.set_ylabel("count")
                fig.savefig(writer.plots_dir / "per_pixel_r2_hist_ch0.png", dpi=150)
                plt.close(fig)

    # --- coefficient mode histograms (top energy modes)
    coeff_hist_cfg = cfg_get(validity_cfg, "coeff_mode_hist", {}) or {}
    if bool(cfg_get(coeff_hist_cfg, "enabled", True)):
        top_k = int(cfg_get(coeff_hist_cfg, "top_k", 12))
        cols = int(cfg_get(coeff_hist_cfg, "cols", 4))
        bins = int(cfg_get(coeff_hist_cfg, "bins", 40))
        top_k = max(1, top_k)
        cols = max(1, cols)
        coeff_arr = np.asarray(coeffs)
        if coeff_arr.ndim == 2 and coeff_arr.shape[0] > 0:
            energy = np.mean(coeff_arr**2, axis=0)
            order = np.argsort(-energy)[: min(top_k, energy.size)]
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
                values = coeff_arr[:, int(dim)].reshape(-1)
                ax.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
                ax.set_title(f"dim {int(dim)}", fontsize=9)
                if r == n_rows - 1:
                    ax.set_xlabel("coeff")
                else:
                    ax.set_xlabel("")
                if c == 0:
                    ax.set_ylabel("count")
                else:
                    ax.set_ylabel("")
            for idx in range(order.size, n_rows * cols):
                r = idx // cols
                c = idx % cols
                axes[r][c].axis("off")
            fig.savefig(writer.plots_dir / "coeff_mode_hist.png", dpi=150)
            plt.close(fig)

    # NOTE: A heatmap-based "histogram by mode" plot existed here but was disabled by
    # default because it is hard to read (color scale) and large for reports. We keep
    # the config hook for power users.
    coeff_by_mode_cfg = cfg_get(validity_cfg, "coeff_hist_by_mode", None)
    if isinstance(coeff_by_mode_cfg, Mapping) and bool(cfg_get(coeff_by_mode_cfg, "enabled", False)):
        max_modes = int(cfg_get(coeff_by_mode_cfg, "max_modes", 128))
        bins = int(cfg_get(coeff_by_mode_cfg, "bins", 60))
        sort = str(cfg_get(coeff_by_mode_cfg, "sort", "energy"))
        clip_percentile = float(cfg_get(coeff_by_mode_cfg, "clip_percentile", 99.0))
        scale = str(cfg_get(coeff_by_mode_cfg, "scale", "log1p"))
        normalize = str(cfg_get(coeff_by_mode_cfg, "normalize", "column"))
        coeff_arr = np.asarray(coeffs)
        if coeff_arr.ndim == 2 and coeff_arr.shape[0] > 0 and coeff_arr.shape[1] > 0:
            coeff_plot = coeff_arr
            meta_plot: Mapping[str, Any] | None = coeff_meta
            title = "coeff histogram by mode"
            try:
                if str(coeff_meta.get("coeff_format", "")).strip().lower() == "offset_residual_v1":
                    offset_dim = int(coeff_meta.get("offset_dim", 0) or 0)
                    if 0 < offset_dim < coeff_arr.shape[1]:
                        coeff_plot = coeff_arr[:, offset_dim:]
                        title = "coeff histogram by mode (residual; offset split)"
                        residual_coeff_meta = coeff_meta.get("residual_coeff_meta")
                        if isinstance(residual_coeff_meta, Mapping):
                            meta_plot = residual_coeff_meta
            except Exception:
                pass
            try:
                plot_coeff_hist_by_mode(
                    writer.plots_dir / "coeff_hist_by_mode.png",
                    coeff_plot,
                    meta_plot,
                    max_modes=max_modes,
                    bins=bins,
                    sort=sort,
                    clip_percentile=clip_percentile,
                    scale=scale,
                    normalize=normalize,
                    title=title,
                )
            except Exception:
                pass

    # --- R^2 vs K (truncated reconstruction)
    r2k_cfg = cfg_get(validity_cfg, "r2_vs_k", {}) or {}
    if bool(cfg_get(r2k_cfg, "enabled", True)):
        coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
        meta_use: Mapping[str, Any] = raw_meta
        residual_raw_meta = raw_meta.get("residual_raw_meta") if coeff_format == "offset_residual_v1" else None
        if isinstance(residual_raw_meta, Mapping):
            meta_use = residual_raw_meta
        coeff_layout = str(meta_use.get("coeff_layout", "")).strip().upper()
        method_use = str(meta_use.get("method", "")).strip().lower()
        fft_shift = bool(meta_use.get("fft_shift", False))
        if coeff_layout in {"CK", "CHW", "K"}:
            max_samples = int(cfg_get(r2k_cfg, "max_samples", 16))
            k_list = cfg_get(r2k_cfg, "k_list", cfg_get(viz_cfg, "k_list", [1, 2, 4, 8, 16]))
            if isinstance(k_list, (list, tuple)) and k_list:
                idxs = np.arange(min(max_samples, n_samples), dtype=int)
                k_vals: list[int] = []
                r2_mean: list[float] = []
                r2_p10: list[float] = []
                r2_p90: list[float] = []
                mask_s = None
                if masks is not None:
                    mask_s = np.asarray(masks)[idxs]
                elif domain_mask is not None:
                    mask_s = np.broadcast_to(np.asarray(domain_mask).astype(bool), fields_true[idxs].shape[:3])

                for k in k_list:
                    k_int = int(k)
                    if k_int <= 0:
                        continue
                    recon_proc = []
                    for idx in idxs.tolist():
                        decoded = None
                        try:
                            decoded = codec.decode(np.asarray(coeffs[idx]).reshape(-1), raw_meta)
                        except Exception:
                            decoded = None
                        if decoded is None:
                            recon_proc = []
                            break
                        if coeff_format == "offset_residual_v1":
                            if not isinstance(decoded, Mapping):
                                recon_proc = []
                                break
                            offset = decoded.get("offset")
                            residual = decoded.get("residual")
                            if offset is None or residual is None:
                                recon_proc = []
                                break
                            if not isinstance(residual_raw_meta, Mapping):
                                recon_proc = []
                                break
                            coeff_shape = residual_raw_meta.get("coeff_shape")
                            order = str(residual_raw_meta.get("flatten_order", "C")).strip().upper() or "C"
                            arr = np.asarray(residual)
                            if arr.ndim == 1:
                                if isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 2:
                                    arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1])), order=order)
                                elif isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 3:
                                    arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1]), int(coeff_shape[2])), order=order)
                                elif isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 1:
                                    arr = arr.reshape((int(coeff_shape[0]),), order=order)
                                else:
                                    recon_proc = []
                                    break
                            if coeff_layout == "CK":
                                if arr.ndim != 2 or arr.shape[1] <= 0:
                                    recon_proc = []
                                    break
                                arr_k = arr.copy()
                                if k_int < arr_k.shape[1]:
                                    arr_k[:, k_int:] = 0.0
                                recon_proc.append(
                                    decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                                )
                            elif coeff_layout == "CHW":
                                if arr.ndim != 3 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
                                    recon_proc = []
                                    break
                                s = int(np.ceil(np.sqrt(k_int)))
                                height = int(arr.shape[1])
                                width = int(arr.shape[2])
                                s_h = min(max(1, s), height)
                                s_w = min(max(1, s), width)
                                if method_use in {"fft2", "fft2_lowpass"}:
                                    shifted = arr if fft_shift else np.fft.fftshift(arr, axes=(1, 2))
                                    arr_k_shifted = np.zeros_like(shifted)
                                    cy = int(height // 2)
                                    cx = int(width // 2)
                                    y0 = int(cy - (s_h // 2))
                                    x0 = int(cx - (s_w // 2))
                                    y0 = max(0, min(y0, height - s_h))
                                    x0 = max(0, min(x0, width - s_w))
                                    arr_k_shifted[:, y0 : y0 + s_h, x0 : x0 + s_w] = shifted[:, y0 : y0 + s_h, x0 : x0 + s_w]
                                    arr_k = arr_k_shifted if fft_shift else np.fft.ifftshift(arr_k_shifted, axes=(1, 2))
                                else:
                                    # DCT-like ordering: keep top-left block.
                                    arr_k = np.zeros_like(arr)
                                    arr_k[:, :s_h, :s_w] = arr[:, :s_h, :s_w]
                                recon_proc.append(
                                    decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                                )
                            else:  # K
                                vec = np.asarray(arr).reshape(-1)
                                if vec.size <= 0:
                                    recon_proc = []
                                    break
                                vec_k = vec.copy()
                                if k_int < vec_k.size:
                                    vec_k[k_int:] = 0.0
                                recon_proc.append(
                                    decomposer.inverse_transform({"offset": offset, "residual": vec_k}, domain_spec=domain_spec)
                                )
                        else:
                            if isinstance(decoded, Mapping):
                                recon_proc = []
                                break
                            arr = np.asarray(decoded)
                            if arr.ndim == 1:
                                coeff_shape = raw_meta.get("coeff_shape")
                                if isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 2:
                                    try:
                                        arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1])), order="C")
                                    except Exception:
                                        recon_proc = []
                                        break
                                elif isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 3:
                                    try:
                                        arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1]), int(coeff_shape[2])), order="C")
                                    except Exception:
                                        recon_proc = []
                                        break
                                elif isinstance(coeff_shape, (list, tuple)) and len(coeff_shape) == 1:
                                    try:
                                        arr = arr.reshape((int(coeff_shape[0]),), order="C")
                                    except Exception:
                                        recon_proc = []
                                        break
                                else:
                                    recon_proc = []
                                    break
                            if coeff_layout == "CK":
                                if arr.ndim != 2 or arr.shape[1] <= 0:
                                    recon_proc = []
                                    break
                                arr_k = arr.copy()
                                if k_int < arr_k.shape[1]:
                                    arr_k[:, k_int:] = 0.0
                                recon_proc.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                            elif coeff_layout == "CHW":
                                if arr.ndim != 3 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
                                    recon_proc = []
                                    break
                                s = int(np.ceil(np.sqrt(k_int)))
                                height = int(arr.shape[1])
                                width = int(arr.shape[2])
                                s_h = min(max(1, s), height)
                                s_w = min(max(1, s), width)
                                if method_use in {"fft2", "fft2_lowpass"}:
                                    shifted = arr if fft_shift else np.fft.fftshift(arr, axes=(1, 2))
                                    arr_k_shifted = np.zeros_like(shifted)
                                    cy = int(height // 2)
                                    cx = int(width // 2)
                                    y0 = int(cy - (s_h // 2))
                                    x0 = int(cx - (s_w // 2))
                                    y0 = max(0, min(y0, height - s_h))
                                    x0 = max(0, min(x0, width - s_w))
                                    arr_k_shifted[:, y0 : y0 + s_h, x0 : x0 + s_w] = shifted[:, y0 : y0 + s_h, x0 : x0 + s_w]
                                    arr_k = arr_k_shifted if fft_shift else np.fft.ifftshift(arr_k_shifted, axes=(1, 2))
                                else:
                                    arr_k = np.zeros_like(arr)
                                    arr_k[:, :s_h, :s_w] = arr[:, :s_h, :s_w]
                                recon_proc.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                            else:  # K
                                vec = np.asarray(arr).reshape(-1)
                                if vec.size <= 0:
                                    recon_proc = []
                                    break
                                vec_k = vec.copy()
                                if k_int < vec_k.size:
                                    vec_k[k_int:] = 0.0
                                recon_proc.append(decomposer.inverse_transform(vec_k, domain_spec=domain_spec))
                    if not recon_proc:
                        break
                    recon_proc_b = np.stack(recon_proc, axis=0)
                    recon_b, _ = preprocess.inverse_transform(recon_proc_b, None)

                    r2_list = []
                    for local_i, idx in enumerate(idxs.tolist()):
                        m2d = None if mask_s is None else mask_s[local_i]
                        r2_list.append(masked_weighted_r2(fields_true[idx], recon_b[local_i], mask=m2d, weights=weights))
                    r2_arr = np.asarray(r2_list, dtype=float)
                    valid = np.isfinite(r2_arr)
                    if not np.any(valid):
                        continue
                    k_vals.append(k_int)
                    r2_mean.append(float(np.nanmean(r2_arr)))
                    r2_p10.append(float(np.nanpercentile(r2_arr[valid], 10)))
                    r2_p90.append(float(np.nanpercentile(r2_arr[valid], 90)))

                if k_vals:
                    plot_line_with_band(
                        writer.plots_dir / "mode_r2_vs_k.png",
                        k_vals,
                        r2_mean,
                        r2_p10,
                        r2_p90,
                        xlabel="K (modes kept)",
                        ylabel="R^2 (field recon)",
                    )


def _plot_key_decomp_dashboard(
    *,
    writer: ArtifactWriter,
    fields_true: np.ndarray,
    fields_pred: np.ndarray,
    masks: np.ndarray | None,
    coeffs: np.ndarray,
    decomposer: Any,
    domain_spec: Any,
    viz_cfg: Mapping[str, Any],
    codec: Any,
    raw_meta: Mapping[str, Any],
    preprocess: Any,
) -> None:
    dash_cfg = cfg_get(viz_cfg, "key_dashboard", {}) or {}
    if not bool(cfg_get(dash_cfg, "enabled", True)):
        return
    if fields_true.shape != fields_pred.shape:
        return

    sample_index = int(cfg_get(dash_cfg, "sample_index", cfg_get(viz_cfg, "sample_index", 0)))
    if sample_index < 0 or sample_index >= int(fields_true.shape[0]):
        sample_index = 0

    domain_mask = getattr(domain_spec, "mask", None)
    mask_sample = None
    if masks is not None:
        mask_sample = np.asarray(masks)[sample_index]
    elif domain_mask is not None:
        mask_sample = np.asarray(domain_mask).astype(bool)

    # Reduce vector fields to magnitude for the dashboard (keeps the layout compact).
    is_vector = fields_true.ndim == 4 and int(fields_true.shape[-1]) > 1
    if is_vector:
        ft_vis = np.stack([_field_magnitude(arr) for arr in np.asarray(fields_true)], axis=0)
        fp_vis = np.stack([_field_magnitude(arr) for arr in np.asarray(fields_pred)], axis=0)
        vis_label = "mag"
    else:
        ft_arr = np.asarray(fields_true)
        fp_arr = np.asarray(fields_pred)
        ft_vis = ft_arr[..., 0] if ft_arr.ndim == 4 else ft_arr
        fp_vis = fp_arr[..., 0] if fp_arr.ndim == 4 else fp_arr
        vis_label = "ch0"

    def _cmap_with_bad(name: str, bad_color: str = "#dddddd") -> Any:
        import matplotlib.pyplot as plt  # local import (Agg already configured in viz)

        cmap = plt.get_cmap(name)
        if hasattr(cmap, "copy"):
            cmap = cmap.copy()
        cmap.set_bad(color=bad_color)
        return cmap

    # --- Panel 0: R^2 vs K (top-energy modes, more robust than prefix for unordered CK layouts)
    coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
    meta_use: Mapping[str, Any] = raw_meta
    residual_raw_meta = raw_meta.get("residual_raw_meta") if coeff_format == "offset_residual_v1" else None
    if isinstance(residual_raw_meta, Mapping):
        meta_use = residual_raw_meta
    coeff_layout = str(meta_use.get("coeff_layout", "")).strip().upper()
    complex_format = str(meta_use.get("complex_format", "")).strip().lower()
    coeff_shape = meta_use.get("coeff_shape")
    flatten_order = str(meta_use.get("flatten_order", "C")).strip().upper() or "C"
    method_use = str(meta_use.get("method", "")).strip().lower()
    fft_shift = bool(meta_use.get("fft_shift", False))

    k_list = cfg_get(dash_cfg, "k_list", [1, 2, 4, 8, 16, 32, 64])
    k_list = [int(k) for k in (k_list if isinstance(k_list, (list, tuple)) else []) if int(k) > 0]
    if not k_list:
        k_list = [1, 2, 4, 8, 16, 32, 64]

    r2_k_vals: list[int] = []
    r2_mean: list[float] = []
    r2_p10: list[float] = []
    r2_p90: list[float] = []
    r2_kind = None
    r2_supported_ck = (
        coeff_layout == "CK"
        and complex_format == "real"
        and isinstance(coeff_shape, (list, tuple))
        and len(coeff_shape) == 2
    )
    r2_supported_k = (
        coeff_layout == "K"
        and complex_format == "real"
        and isinstance(coeff_shape, (list, tuple))
        and len(coeff_shape) == 1
    )
    r2_supported_chw = (
        coeff_layout == "CHW"
        and isinstance(coeff_shape, (list, tuple))
        and len(coeff_shape) == 3
        and complex_format in {"real", "complex"}
    )
    if r2_supported_ck:
        try:
            max_samples = int(cfg_get(dash_cfg, "r2_max_samples", 16))
            idxs = np.arange(min(max_samples, int(coeffs.shape[0])), dtype=int)

            decoded_list: list[tuple[np.ndarray, Any] | None] = []
            k_total = int(coeff_shape[1])
            energy = np.zeros((k_total,), dtype=np.float64)
            for idx in idxs.tolist():
                decoded = codec.decode(np.asarray(coeffs[idx]).reshape(-1), raw_meta)
                if coeff_format == "offset_residual_v1":
                    if not isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    offset = decoded.get("offset")
                    residual = decoded.get("residual")
                    if offset is None or residual is None:
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(residual)
                    if arr.ndim == 1:
                        arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1])), order=flatten_order)
                    if arr.ndim != 2 or arr.shape[1] != k_total:
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, offset))
                else:
                    if isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(decoded)
                    if arr.ndim == 1:
                        arr = arr.reshape((int(coeff_shape[0]), int(coeff_shape[1])), order=flatten_order)
                    if arr.ndim != 2 or arr.shape[1] != k_total:
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, None))

                # Channels-summed energy per mode.
                energy += np.sum(arr.astype(np.float64) ** 2, axis=0)

            if decoded_list and np.isfinite(energy).all() and np.any(energy > 0):
                order = np.argsort(-energy)  # desc
                for k_keep in k_list:
                    k_keep = min(int(k_keep), k_total)
                    if k_keep <= 0:
                        continue
                    keep = order[:k_keep]

                    recon = []
                    for pair in decoded_list:
                        if pair is None:
                            recon = []
                            break
                        arr, offset = pair
                        arr_k = np.zeros_like(arr)
                        arr_k[:, keep] = arr[:, keep]
                        if coeff_format == "offset_residual_v1":
                            recon.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        else:
                            recon.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                    if not recon:
                        break
                    recon_b = np.stack(recon, axis=0)
                    recon_b, _ = preprocess.inverse_transform(recon_b, None)

                    r2_list = []
                    for local_i, idx in enumerate(idxs.tolist()):
                        m2d = None if masks is None else np.asarray(masks)[idx]
                        recon_one = recon_b[local_i]
                        if is_vector:
                            recon_vis = _field_magnitude(recon_one)
                        else:
                            recon_vis = recon_one[..., 0] if recon_one.ndim == 3 else recon_one
                        r2_list.append(masked_weighted_r2(ft_vis[idx], recon_vis, mask=m2d, weights=None))
                    r2_arr = np.asarray(r2_list, dtype=float)
                    valid = np.isfinite(r2_arr)
                    if not np.any(valid):
                        continue
                    r2_k_vals.append(int(k_keep))
                    r2_mean.append(float(np.nanmean(r2_arr)))
                    r2_p10.append(float(np.nanpercentile(r2_arr[valid], 10)))
                    r2_p90.append(float(np.nanpercentile(r2_arr[valid], 90)))
                r2_kind = "top_energy"
        except Exception:
            r2_k_vals = []
    elif r2_supported_k:
        try:
            max_samples = int(cfg_get(dash_cfg, "r2_max_samples", 16))
            idxs = np.arange(min(max_samples, int(coeffs.shape[0])), dtype=int)

            decoded_list: list[tuple[np.ndarray, Any] | None] = []
            k_total = int(coeff_shape[0])
            energy = np.zeros((k_total,), dtype=np.float64)
            for idx in idxs.tolist():
                decoded = codec.decode(np.asarray(coeffs[idx]).reshape(-1), raw_meta)
                if coeff_format == "offset_residual_v1":
                    if not isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    offset = decoded.get("offset")
                    residual = decoded.get("residual")
                    if offset is None or residual is None:
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(residual).reshape(-1)
                    if arr.size != k_total:
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, offset))
                else:
                    if isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(decoded).reshape(-1)
                    if arr.size != k_total:
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, None))

                energy += (arr.astype(np.float64) ** 2)

            if decoded_list and np.isfinite(energy).all() and np.any(energy > 0):
                order = np.argsort(-energy)  # desc
                for k_keep in k_list:
                    k_keep = min(int(k_keep), k_total)
                    if k_keep <= 0:
                        continue
                    keep = order[:k_keep]

                    recon = []
                    for pair in decoded_list:
                        if pair is None:
                            recon = []
                            break
                        arr, offset = pair
                        arr_k = np.zeros_like(arr)
                        arr_k[keep] = arr[keep]
                        if coeff_format == "offset_residual_v1":
                            recon.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        else:
                            recon.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                    if not recon:
                        break
                    recon_b = np.stack(recon, axis=0)
                    recon_b, _ = preprocess.inverse_transform(recon_b, None)

                    r2_list = []
                    for local_i, idx in enumerate(idxs.tolist()):
                        m2d = None if masks is None else np.asarray(masks)[idx]
                        recon_one = recon_b[local_i]
                        if is_vector:
                            recon_vis = _field_magnitude(recon_one)
                        else:
                            recon_vis = recon_one[..., 0] if recon_one.ndim == 3 else recon_one
                        r2_list.append(masked_weighted_r2(ft_vis[idx], recon_vis, mask=m2d, weights=None))
                    r2_arr = np.asarray(r2_list, dtype=float)
                    valid = np.isfinite(r2_arr)
                    if not np.any(valid):
                        continue
                    r2_k_vals.append(int(k_keep))
                    r2_mean.append(float(np.nanmean(r2_arr)))
                    r2_p10.append(float(np.nanpercentile(r2_arr[valid], 10)))
                    r2_p90.append(float(np.nanpercentile(r2_arr[valid], 90)))
                r2_kind = "top_energy"
        except Exception:
            r2_k_vals = []
    elif r2_supported_chw:
        try:
            max_samples = int(cfg_get(dash_cfg, "r2_max_samples", 16))
            idxs = np.arange(min(max_samples, int(coeffs.shape[0])), dtype=int)
            decoded_list: list[tuple[np.ndarray, Any] | None] = []
            c = int(coeff_shape[0])
            h_c = int(coeff_shape[1])
            w_c = int(coeff_shape[2])
            for idx in idxs.tolist():
                decoded = codec.decode(np.asarray(coeffs[idx]).reshape(-1), raw_meta)
                if coeff_format == "offset_residual_v1":
                    if not isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    offset = decoded.get("offset")
                    residual = decoded.get("residual")
                    if offset is None or residual is None:
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(residual)
                    if arr.ndim == 1:
                        arr = arr.reshape((c, h_c, w_c), order=flatten_order)
                    if arr.ndim != 3 or arr.shape != (c, h_c, w_c):
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, offset))
                else:
                    if isinstance(decoded, Mapping):
                        decoded_list.append(None)
                        continue
                    arr = np.asarray(decoded)
                    if arr.ndim == 1:
                        arr = arr.reshape((c, h_c, w_c), order=flatten_order)
                    if arr.ndim != 3 or arr.shape != (c, h_c, w_c):
                        decoded_list.append(None)
                        continue
                    decoded_list.append((arr, None))

            if decoded_list:
                for k_keep in k_list:
                    k_keep = int(k_keep)
                    if k_keep <= 0:
                        continue
                    s = int(np.ceil(np.sqrt(float(k_keep))))
                    s_h = min(s, h_c)
                    s_w = min(s, w_c)
                    recon = []
                    for pair in decoded_list:
                        if pair is None:
                            recon = []
                            break
                        arr, offset = pair
                        if method_use in {"fft2", "fft2_lowpass"}:
                            shifted = arr if fft_shift else np.fft.fftshift(arr, axes=(1, 2))
                            arr_k_shifted = np.zeros_like(shifted)
                            cy = int(h_c // 2)
                            cx = int(w_c // 2)
                            y0 = int(cy - (s_h // 2))
                            x0 = int(cx - (s_w // 2))
                            y0 = max(0, min(y0, h_c - s_h))
                            x0 = max(0, min(x0, w_c - s_w))
                            arr_k_shifted[:, y0 : y0 + s_h, x0 : x0 + s_w] = shifted[:, y0 : y0 + s_h, x0 : x0 + s_w]
                            arr_k = arr_k_shifted if fft_shift else np.fft.ifftshift(arr_k_shifted, axes=(1, 2))
                        else:
                            arr_k = np.zeros_like(arr)
                            arr_k[:, :s_h, :s_w] = arr[:, :s_h, :s_w]
                        if coeff_format == "offset_residual_v1":
                            recon.append(
                                decomposer.inverse_transform({"offset": offset, "residual": arr_k}, domain_spec=domain_spec)
                            )
                        else:
                            recon.append(decomposer.inverse_transform(arr_k, domain_spec=domain_spec))
                    if not recon:
                        break
                    recon_b = np.stack(recon, axis=0)
                    recon_b, _ = preprocess.inverse_transform(recon_b, None)

                    r2_list = []
                    for local_i, idx in enumerate(idxs.tolist()):
                        m2d = None if masks is None else np.asarray(masks)[idx]
                        recon_one = recon_b[local_i]
                        if is_vector:
                            recon_vis = _field_magnitude(recon_one)
                        else:
                            recon_vis = recon_one[..., 0] if recon_one.ndim == 3 else recon_one
                        r2_list.append(masked_weighted_r2(ft_vis[idx], recon_vis, mask=m2d, weights=None))
                    r2_arr = np.asarray(r2_list, dtype=float)
                    valid = np.isfinite(r2_arr)
                    if not np.any(valid):
                        continue
                    r2_k_vals.append(int(k_keep))
                    r2_mean.append(float(np.nanmean(r2_arr)))
                    r2_p10.append(float(np.nanpercentile(r2_arr[valid], 10)))
                    r2_p90.append(float(np.nanpercentile(r2_arr[valid], 90)))
                r2_kind = "low_freq"
        except Exception:
            r2_k_vals = []

    # --- Panel 1: scatter true vs recon (sampled points)
    scatter_max_samples = int(cfg_get(dash_cfg, "scatter_max_samples", 8))
    scatter_max_points = int(cfg_get(dash_cfg, "scatter_max_points", 120_000))
    scatter_seed = int(cfg_get(dash_cfg, "scatter_seed", 0))
    idxs_s = np.arange(min(scatter_max_samples, int(ft_vis.shape[0])), dtype=int)
    mask_s = None
    if masks is not None:
        mask_s = np.asarray(masks)[idxs_s]
    elif domain_mask is not None:
        mask_s = np.broadcast_to(np.asarray(domain_mask).astype(bool), ft_vis[idxs_s].shape)
    x_sc, y_sc = sample_scatter_points(
        ft_vis[idxs_s],
        fp_vis[idxs_s],
        mask=mask_s,
        max_points=scatter_max_points,
        seed=scatter_seed,
    )
    r2_sc = masked_weighted_r2(x_sc, y_sc) if x_sc.size else float("nan")

    # --- Panel 2: per-pixel R^2 map (across samples)
    pixel_max_samples = int(cfg_get(dash_cfg, "pixel_max_samples", 64))
    downsample = int(cfg_get(dash_cfg, "pixel_downsample", 1))
    idxs_p = np.arange(min(pixel_max_samples, int(ft_vis.shape[0])), dtype=int)
    mask_p = None
    if masks is not None:
        mask_p = np.asarray(masks)[idxs_p]
    elif domain_mask is not None:
        mask_p = np.broadcast_to(np.asarray(domain_mask).astype(bool), ft_vis[idxs_p].shape)
    r2_map = per_pixel_r2_map(ft_vis[idxs_p], fp_vis[idxs_p], mask=mask_p, downsample=downsample)

    # --- Panel 3/4/5: sample true / error / recon with robust scaling
    true_s = ft_vis[sample_index]
    pred_s = fp_vis[sample_index]
    err_s = np.abs(pred_s - true_s)
    if mask_sample is not None:
        err_s = np.where(mask_sample, err_s, np.nan)
        true_masked = np.where(mask_sample, true_s, np.nan)
        pred_masked = np.where(mask_sample, pred_s, np.nan)
    else:
        true_masked = true_s
        pred_masked = pred_s

    vmin = float(np.nanmin([np.nanmin(true_masked), np.nanmin(pred_masked)]))
    vmax = float(np.nanmax([np.nanmax(true_masked), np.nanmax(pred_masked)]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    vmax_err = float(np.nanpercentile(err_s[np.isfinite(err_s)], float(cfg_get(dash_cfg, "error_vmax_percentile", 99.0)))) if np.any(np.isfinite(err_s)) else 1.0
    vmax_err = max(vmax_err, 1e-12)

    # Render dashboard.
    import matplotlib.pyplot as plt  # local import (Agg already configured in viz)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)

    # R^2 vs K (top-energy)
    ax = axes[0][0]
    if r2_k_vals:
        ax.plot(r2_k_vals, r2_mean, marker="o", linewidth=1.6, color="#4C78A8")
        ax.fill_between(r2_k_vals, r2_p10, r2_p90, color="#4C78A8", alpha=0.18)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel("K (modes kept)")
        ax.set_ylabel("R^2")
        if r2_kind == "low_freq":
            ax.set_title("Recon R^2 vs K (low-frequency block)")
        else:
            ax.set_title("Recon R^2 vs K (top-energy)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "R^2 vs K not available", ha="center", va="center")

    # Scatter
    ax = axes[0][1]
    if x_sc.size:
        n = min(int(x_sc.size), int(cfg_get(dash_cfg, "scatter_plot_points", 40_000)))
        ax.scatter(x_sc[:n], y_sc[:n], s=3, alpha=0.25, color="#4C78A8")
        v0 = float(np.nanmin([np.min(x_sc[:n]), np.min(y_sc[:n])]))
        v1 = float(np.nanmax([np.max(x_sc[:n]), np.max(y_sc[:n])]))
        if np.isfinite(v0) and np.isfinite(v1) and v1 > v0:
            ax.plot([v0, v1], [v0, v1], color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlabel("true")
        ax.set_ylabel("recon")
        ax.set_title(f"Scatter ({vis_label})  R^2={r2_sc:.3f}" if np.isfinite(r2_sc) else f"Scatter ({vis_label})")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Scatter not available", ha="center", va="center")

    # True sample
    ax = axes[0][2]
    im = ax.imshow(true_masked, origin="lower", cmap=_cmap_with_bad("viridis"), vmin=vmin, vmax=vmax)
    ax.set_title(f"sample_{sample_index:04d} true ({vis_label})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Per-pixel R^2 map
    ax = axes[1][0]
    im = ax.imshow(r2_map, origin="lower", cmap=_cmap_with_bad("coolwarm"), vmin=-1.0, vmax=1.0)
    ax.set_title(f"per-pixel R^2 map ({vis_label})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Error sample
    ax = axes[1][1]
    im = ax.imshow(err_s, origin="lower", cmap=_cmap_with_bad("magma"), vmin=0.0, vmax=vmax_err)
    ax.set_title("abs error (robust)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Recon sample
    ax = axes[1][2]
    im = ax.imshow(pred_masked, origin="lower", cmap=_cmap_with_bad("viridis"), vmin=vmin, vmax=vmax)
    ax.set_title(f"sample_{sample_index:04d} recon ({vis_label})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = writer.plots_dir / "key_decomp_dashboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _render_plots(
    *,
    writer: ArtifactWriter,
    fields_true: np.ndarray,
    fields_pred: np.ndarray,
    masks: np.ndarray | None,
    coeffs: np.ndarray,
    coeff_meta: Mapping[str, Any],
    decomposer: Any,
    domain_spec: Any,
    domain_cfg: Mapping[str, Any],
    viz_cfg: Mapping[str, Any],
    conds: np.ndarray | None,
    cond_names: Sequence[str] | None,
    codec: Any,
    raw_meta: Mapping[str, Any],
    preprocess: Any,
    sample_ids: Sequence[str] | None = None,
) -> None:
    n_samples = int(fields_true.shape[0])
    sample_index = int(cfg_get(viz_cfg, "sample_index", 0))
    max_samples = int(cfg_get(viz_cfg, "max_samples", 8))
    if sample_index < 0 or sample_index >= n_samples:
        raise ValueError("viz.sample_index is out of range")
    if max_samples <= 0:
        raise ValueError("viz.max_samples must be > 0")

    sample_indices = list(range(min(max_samples, n_samples)))
    if sample_index not in sample_indices:
        sample_indices = [sample_index] + sample_indices

    plot_root = writer.plots_dir
    field_true0 = fields_true[sample_index]
    for idx in sample_indices:
        field_true = fields_true[idx]
        field_pred = fields_pred[idx]
        mask_sample = masks[idx] if masks is not None else None

        sample_dir = plot_root / f"sample_{idx:04d}"
        plot_field_grid(
            sample_dir / "field_compare.png",
            [field_true, field_pred],
            ["true", "recon"],
            mask=mask_sample,
            suptitle="field_true vs field_recon",
        )
        plot_error_map(sample_dir / "error_map.png", field_true, field_pred, mask=mask_sample)

        if field_true.ndim == 3 and field_true.shape[2] > 1:
            try:
                plot_vector_streamplot(
                    sample_dir / "field_true_stream.png",
                    field_true,
                    mask=mask_sample if mask_sample is not None else domain_spec.mask,
                )
                plot_vector_streamplot(
                    sample_dir / "field_recon_stream.png",
                    field_pred,
                    mask=mask_sample if mask_sample is not None else domain_spec.mask,
                )
                plot_vector_quiver(
                    sample_dir / "field_true_quiver.png",
                    field_true,
                    mask=mask_sample if mask_sample is not None else domain_spec.mask,
                )
                plot_vector_quiver(
                    sample_dir / "field_recon_quiver.png",
                    field_pred,
                    mask=mask_sample if mask_sample is not None else domain_spec.mask,
                )
            except Exception:
                pass
            true_mag = _field_magnitude(field_true)
            pred_mag = _field_magnitude(field_pred)
            plot_field_grid(
                sample_dir / "field_compare_mag.png",
                [true_mag, pred_mag],
                ["true_mag", "recon_mag"],
                mask=mask_sample,
                suptitle="field magnitude compare",
            )
            plot_error_map(sample_dir / "error_map_mag.png", true_mag, pred_mag, mask=mask_sample)

    titles = []
    for idx in sample_indices:
        if sample_ids and idx < len(sample_ids):
            titles.append(str(sample_ids[idx]))
        else:
            titles.append(f"sample_{idx:04d}")
    plot_field_grid(
        plot_root / "field_true_samples.png",
        [fields_true[idx] for idx in sample_indices],
        titles,
        mask=domain_spec.mask,
        suptitle="field_true samples",
    )
    plot_field_grid(
        plot_root / "field_recon_samples.png",
        [fields_pred[idx] for idx in sample_indices],
        titles,
        mask=domain_spec.mask,
        suptitle="field_recon samples",
    )
    if fields_true.ndim == 4 and fields_true.shape[-1] > 1:
        true_mag_list = [_field_magnitude(fields_true[idx]) for idx in sample_indices]
        pred_mag_list = [_field_magnitude(fields_pred[idx]) for idx in sample_indices]
        plot_field_grid(
            plot_root / "field_true_mag_samples.png",
            true_mag_list,
            titles,
            mask=domain_spec.mask,
            suptitle="field_true magnitude samples",
        )
        plot_field_grid(
            plot_root / "field_recon_mag_samples.png",
            pred_mag_list,
            titles,
            mask=domain_spec.mask,
            suptitle="field_recon magnitude samples",
        )

    spectrum = coeff_energy_spectrum(coeffs, coeff_meta)
    spectrum_scale = str(cfg_get(viz_cfg, "spectrum_scale", "log")).strip().lower()
    plot_coeff_spectrum(plot_root / "coeff_spectrum.png", spectrum, scale=spectrum_scale)

    diag_cfg = cfg_get(viz_cfg, "coeff_diag", {}) or {}
    hist_bins = int(cfg_get(diag_cfg, "hist_bins", 60))
    hist_scale = str(cfg_get(diag_cfg, "hist_scale", "log")).strip().lower()
    coeff_mag = coeff_value_magnitude(coeffs, coeff_meta)
    plot_coeff_histogram(plot_root / "coeff_hist.png", coeff_mag, bins=hist_bins, scale=hist_scale)

    _plot_spatial_stats(
        writer=writer,
        fields_true=fields_true,
        fields_pred=fields_pred,
        masks=masks,
        domain_mask=domain_spec.mask,
        viz_cfg=viz_cfg,
    )

    _plot_data_driven_diagnostics(
        writer=writer,
        coeffs=coeffs,
        coeff_meta=coeff_meta,
        decomposer=decomposer,
        domain_spec=domain_spec,
        viz_cfg=viz_cfg,
        conds=conds,
        cond_names=cond_names,
    )

    # Diagnostics to understand domain/method behavior beyond aggregate metrics.
    _plot_validity_diagnostics(
        writer=writer,
        fields_true=fields_true,
        fields_pred=fields_pred,
        masks=masks,
        coeffs=coeffs,
        coeff_meta=coeff_meta,
        decomposer=decomposer,
        domain_spec=domain_spec,
        viz_cfg=viz_cfg,
        codec=codec,
        raw_meta=raw_meta,
        preprocess=preprocess,
    )

    _plot_key_decomp_dashboard(
        writer=writer,
        fields_true=fields_true,
        fields_pred=fields_pred,
        masks=masks,
        coeffs=coeffs,
        decomposer=decomposer,
        domain_spec=domain_spec,
        viz_cfg=viz_cfg,
        codec=codec,
        raw_meta=raw_meta,
        preprocess=preprocess,
    )

    domain_name = str(cfg_get(domain_cfg, "name", "")).strip().lower()
    if domain_name:
        domain_dir = plot_root / "domain"
        plot_domain_field_grid(
            domain_dir / f"field_compare_{sample_index:04d}.png",
            [field_true0, fields_pred[sample_index]],
            ["true", "recon"],
            domain_spec=domain_spec,
            mask=masks[sample_index] if masks is not None else None,
            suptitle="domain-aware field compare",
            sphere_projection=str(cfg_get(viz_cfg, "sphere_projection", "plate_carre")),
        )
        plot_domain_error_map(
            domain_dir / f"error_map_{sample_index:04d}.png",
            field_true0,
            fields_pred[sample_index],
            domain_spec=domain_spec,
            mask=masks[sample_index] if masks is not None else None,
            sphere_projection=str(cfg_get(viz_cfg, "sphere_projection", "plate_carre")),
        )

