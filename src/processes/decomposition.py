"""Process entrypoint: decomposition."""
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
from mode_decomp_ml.evaluate import compute_metrics
from mode_decomp_ml.plugins.codecs import build_coeff_codec
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.preprocess import build_preprocess
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    build_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_run_dir,
    snapshot_inputs,
    split_indices,
)
from mode_decomp_ml.viz import (
    coeff_energy_vector,
    coeff_energy_spectrum,
    coeff_value_magnitude,
    plot_coeff_histogram,
    plot_coeff_spectrum,
    plot_domain_error_map,
    plot_domain_field_grid,
    plot_error_map,
    plot_field_grid,
    plot_line_series,
    plot_vector_quiver,
    plot_vector_streamplot,
)


class _ArrayDataset:
    def __init__(
        self,
        conds: np.ndarray,
        fields: np.ndarray,
        masks: np.ndarray | None,
        metas: list[dict[str, Any]],
        *,
        name: str,
    ) -> None:
        self._conds = conds
        self._fields = fields
        self._masks = masks
        self._metas = metas
        self.name = name

    def __len__(self) -> int:
        return int(self._conds.shape[0])

    def __getitem__(self, index: int) -> FieldSample:
        idx = int(index)
        return FieldSample(
            cond=self._conds[idx],
            field=self._fields[idx],
            mask=None if self._masks is None else self._masks[idx],
            meta=self._metas[idx],
        )


def _grid_spacing_from_domain(domain_spec) -> tuple[float, float]:
    x = domain_spec.coords.get("x")
    y = domain_spec.coords.get("y")
    if x is None or y is None:
        raise ValueError("domain_spec must include x/y for div/curl metrics")
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    if dx <= 0 or dy <= 0:
        raise ValueError("domain_spec spacing must be positive for div/curl metrics")
    return dx, dy


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


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("decomposition requires config from the Hydra entrypoint")
    require_cfg_keys(
        cfg,
        ["seed", "run_dir", "dataset", "split", "domain", "decompose", "codec", "preprocess", "eval", "viz"],
    )

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    with steps.step(
        "init_run",
        outputs=[artifact_ref("configuration/run.yaml", kind="config")],
    ):
        writer.prepare_layout(clean=True)
        writer.write_run_yaml(cfg)
        snapshot_inputs(cfg, run_dir)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, domain_cfg)
    split_cfg = cfg_get(cfg, "split")

    cond_names: list[str] | None = None

    with steps.step(
        "build_dataset",
        inputs=[artifact_ref("configuration/run.yaml", kind="config")],
    ) as step:
        dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
        conds, fields, masks, metas = dataset_to_arrays(dataset)
        split_meta = split_indices(split_cfg, len(dataset), seed)
        idx = np.asarray(split_meta["train_idx"], dtype=int)
        conds = conds[idx]
        fields = fields[idx]
        masks = masks[idx] if masks is not None else None
        metas = [metas[i] for i in idx.tolist()]
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(conds))
        cond_names = _resolve_cond_names(dataset, conds)

    with steps.step(
        "preprocess_fields",
        outputs=[artifact_ref("outputs/states/preprocess/state.pkl", kind="state")],
    ):
        preprocess = build_preprocess(cfg_get(cfg, "preprocess"))
        preprocess.fit(fields, masks, split="train")
        fields_proc, masks_proc = preprocess.transform(fields, masks)
        preprocess.save_state(run_dir)

    domain_spec = build_domain_spec(domain_cfg, fields_proc.shape[1:])
    decomposer = build_decomposer(cfg_get(cfg, "decompose"))
    codec = build_coeff_codec(cfg_get(cfg, "codec"))

    fit_time_sec: float | None = None
    with steps.step(
        "fit_decomposer",
        outputs=[artifact_ref("outputs/states/decomposer/state.pkl", kind="state")],
    ) as step:
        start = time.perf_counter()
        decomposer.fit(
            dataset=_ArrayDataset(conds, fields_proc, masks_proc, metas, name=dataset.name),
            domain_spec=domain_spec,
        )
        fit_time_sec = float(time.perf_counter() - start)
        step["meta"]["fit_time_sec"] = fit_time_sec

    coeffs = []
    raw_meta: Mapping[str, Any] | None = None
    sample_ids: list[str] = []
    with steps.step(
        "encode_coeffs",
        outputs=[
            artifact_ref("outputs/states/decomposer/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/decomposer/state.pkl", kind="state"),
            artifact_ref("outputs/states/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/coeff_codec/coeff_meta.json", kind="metadata"),
            artifact_ref("outputs/states/coeff_codec/state.pkl", kind="state"),
            artifact_ref("outputs/coeffs.npz", kind="coeffs"),
        ],
    ):
        for idx in range(fields_proc.shape[0]):
            raw_coeff = decomposer.transform(
                fields_proc[idx],
                mask=None if masks_proc is None else masks_proc[idx],
                domain_spec=domain_spec,
            )
            if raw_meta is None:
                raw_meta = decomposer.coeff_meta()
            coeffs.append(codec.encode(raw_coeff, raw_meta))
        A = np.stack(coeffs, axis=0)
        if raw_meta is None:
            raise ValueError("coeff_meta missing after decomposer.transform")

        decomposer.save_coeff_meta(run_dir)
        decomposer.save_state(run_dir)
        codec.save_coeff_meta(run_dir, raw_meta, A[0])
        codec.save_state(run_dir)
        sample_ids = [meta.get("sample_id", f"sample_{i:04d}") for i, meta in enumerate(metas)]
        writer.write_coeffs(
            {
                "coeff": A,
                "cond": conds,
                "sample_ids": np.asarray(sample_ids, dtype=object),
            }
        )

    with steps.step(
        "reconstruct_field",
        outputs=[artifact_ref("outputs/preds.npz", kind="preds")],
    ):
        fields_recon = []
        for idx in range(A.shape[0]):
            raw_coeff = codec.decode(A[idx], raw_meta)
            fields_recon.append(decomposer.inverse_transform(raw_coeff, domain_spec=domain_spec))
        field_hat = np.stack(fields_recon, axis=0)
        field_hat, _ = preprocess.inverse_transform(field_hat, None)
        writer.write_preds({"field": field_hat})

    eval_cfg = cfg_get(cfg, "eval", {}) or {}
    metrics_list = list(cfg_get(eval_cfg, "metrics", []))
    if not metrics_list:
        raise ValueError("eval.metrics must be configured for decomposition")

    domain_name = str(cfg_get(domain_cfg, "name", "")).strip().lower()
    needs_mask = domain_name in {"disk", "mask", "arbitrary_mask", "mesh"}
    if needs_mask:
        masks_eval = combine_masks(
            masks,
            domain_spec.mask,
            spatial_shape=fields.shape[1:3],
            n_samples=fields.shape[0],
        )
    else:
        masks_eval = masks

    needs_divcurl = any(name in {"div_rmse", "curl_rmse"} for name in metrics_list)
    grid_spacing = _grid_spacing_from_domain(domain_spec) if needs_divcurl else None

    coeff_meta = codec.coeff_meta(raw_meta, A[0])
    with steps.step(
        "compute_metrics",
        outputs=[artifact_ref("outputs/metrics.json", kind="metrics")],
    ):
        metrics = compute_metrics(
            metrics_list,
            field_true=fields,
            field_pred=field_hat,
            mask=masks_eval,
            coeff_true_a=A,
            coeff_pred_a=A,
            coeff_meta=coeff_meta,
            grid_spacing=grid_spacing,
        )
        if fit_time_sec is not None:
            metrics["fit_time_sec"] = fit_time_sec
        writer.write_metrics(metrics)

    with steps.step(
        "render_plots",
        outputs=[artifact_ref("plots", kind="plots_dir")],
    ):
        _render_plots(
            writer=writer,
            fields_true=fields,
            fields_pred=field_hat,
            masks=masks_eval,
            coeffs=A,
            coeff_meta=coeff_meta,
            decomposer=decomposer,
            domain_spec=domain_spec,
            domain_cfg=domain_cfg,
            viz_cfg=cfg_get(cfg, "viz", {}) or {},
            conds=conds,
            cond_names=cond_names,
            sample_ids=sample_ids,
        )

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)
    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    writer.write_manifest(meta=meta, dataset_meta=dataset_meta, steps=steps.to_list())
    writer.write_steps(steps.to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
