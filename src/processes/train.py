"""Process entrypoint: train."""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_decomp_ml.plugins.models import build_regressor
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_meta,
    cfg_get,
    default_run_dir,
    load_coeff_meta,
    read_json,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    snapshot_inputs,
)
from mode_decomp_ml.pipeline.artifacts import load_coeffs_npz
from mode_decomp_ml.viz import coeff_channel_norms, plot_channel_norm_scatter


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for train")
    return task_cfg


def _resolve_preprocess_dir(cfg: Mapping[str, Any]) -> str:
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    run_dir = cfg_get(task_cfg, "preprocessing_run_dir", None)
    if run_dir is None or str(run_dir).strip() == "":
        return str(default_run_dir(cfg, "preprocessing"))
    return str(resolve_path(run_dir))


def _split_indices(n_samples: int, *, val_ratio: float, seed: int | None, shuffle: bool) -> tuple[np.ndarray, np.ndarray]:
    if n_samples <= 1 or val_ratio <= 0.0:
        idx = np.arange(n_samples, dtype=int)
        return idx, np.asarray([], dtype=int)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_samples, dtype=int)
    if shuffle:
        rng.shuffle(idx)
    split = max(1, int(n_samples * (1.0 - val_ratio)))
    split = min(split, n_samples - 1)
    return idx[:split], idx[split:]


def _kfold_indices(
    n_samples: int,
    *,
    n_splits: int,
    seed: int | None,
    shuffle: bool,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 1:
        raise ValueError("cv.folds must be >= 2")
    idx = np.arange(n_samples, dtype=int)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = idx[start:stop]
        train_idx = np.concatenate([idx[:start], idx[stop:]]) if start > 0 or stop < n_samples else idx
        yield train_idx, val_idx
        current = stop


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    mean = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - (ss_res / ss_tot)
    valid = np.isfinite(r2)
    if not np.any(valid):
        return float("nan")
    weights = ss_tot[valid]
    if np.all(weights == 0):
        return float(np.nanmean(r2[valid]))
    return float(np.sum(r2[valid] * weights) / np.sum(weights))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"rmse": _rmse(y_true, y_pred), "mae": _mae(y_true, y_pred), "r2": _r2(y_true, y_pred)}


def _plot_pred_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_dir: Path,
    max_dims: int,
    max_points: int,
    prefix: str,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    dims = y_true.shape[1]
    if dims == 0:
        return
    var = np.var(y_true, axis=0)
    order = np.argsort(-var)
    select = order[: min(int(max_dims), dims)]
    n_points = min(int(max_points), y_true.shape[0])
    def _r2_1d(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size == 0:
            return float("nan")
        mean = float(np.mean(a))
        ss_res = float(np.sum((a - b) ** 2))
        ss_tot = float(np.sum((a - mean) ** 2))
        if ss_tot <= 0.0:
            return float("nan")
        return 1.0 - (ss_res / ss_tot)
    for rank, dim in enumerate(select):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        x = y_true[:n_points, dim]
        y = y_pred[:n_points, dim]
        ax.scatter(x, y, s=12, alpha=0.7)
        vmin = float(np.nanmin([np.min(x), np.min(y)]))
        vmax = float(np.nanmax([np.max(x), np.max(y)]))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            ax.plot([vmin, vmax], [vmin, vmax], color="#333333", linestyle="--", linewidth=1.0)
        r2 = _r2_1d(x, y)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        title = f"{prefix} dim {int(dim)}"
        if np.isfinite(r2):
            title += f" (R^2={r2:.3f})"
        ax.set_title(title)
        path = out_dir / f"{prefix}_scatter_dim_{int(dim):04d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        if rank + 1 >= max_dims:
            break


def _plot_residual_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_dir: Path,
    prefix: str,
) -> None:
    residual = (y_pred - y_true).reshape(-1)
    if residual.size == 0:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.hist(residual, bins=60, color="#4C78A8", alpha=0.85)
    ax.set_xlabel("residual")
    ax.set_ylabel("count")
    path = out_dir / f"{prefix}_residual_hist.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coeff_scatter_2d(
    coeff: np.ndarray,
    *,
    out_path: Path,
    title: str,
    max_points: int,
) -> None:
    coeff = np.asarray(coeff)
    if coeff.ndim != 2 or coeff.shape[1] < 2:
        return
    var = np.var(coeff, axis=0)
    idx = np.argsort(-var)[:2]
    n_points = min(int(max_points), coeff.shape[0])
    x = coeff[:n_points, idx[0]]
    y = coeff[:n_points, idx[1]]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.scatter(x, y, s=12, alpha=0.7)
    ax.set_xlabel(f"dim {int(idx[0])}")
    ax.set_ylabel(f"dim {int(idx[1])}")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_decomposition_dir(preprocess_dir: str | Path) -> str | None:
    metrics_path = Path(preprocess_dir) / "outputs" / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = read_json(metrics_path)
    except Exception:
        return None
    run_dir = metrics.get("decomposition_run_dir")
    if run_dir is None or str(run_dir).strip() == "":
        return None
    return str(resolve_path(run_dir))


def _grid_params(param_grid: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    if not param_grid:
        return []
    keys = list(param_grid.keys())
    values = []
    for key in keys:
        value = param_grid[key]
        if isinstance(value, (list, tuple)):
            values.append(list(value))
        else:
            values.append([value])
    return (dict(zip(keys, combo)) for combo in product(*values))


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("train requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["run_dir", "model"])
    _require_task_config(cfg_get(cfg, "task", None))

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

    preprocess_dir = _resolve_preprocess_dir(cfg)
    with steps.step(
        "load_coeffs",
        inputs=[artifact_ref(f"{preprocess_dir}/outputs/coeffs.npz", kind="coeffs")],
    ):
        coeff_payload = load_coeffs_npz(preprocess_dir)
        conds = coeff_payload.get("cond")
        coeff_z = coeff_payload.get("coeff_z")
        coeff_a = coeff_payload.get("coeff_a")
        if conds is None:
            raise ValueError("preprocessing coeffs.npz missing cond array")
        if coeff_z is None and coeff_a is None:
            raise ValueError("preprocessing coeffs.npz missing coeff_z/coeff_a arrays")
        conds = np.asarray(conds)
        targets = np.asarray(coeff_z if coeff_z is not None else coeff_a)
        target_space = "z" if coeff_z is not None else "a"

    coeff_meta = None
    decomposition_dir = _resolve_decomposition_dir(preprocess_dir)
    if decomposition_dir is not None:
        try:
            coeff_meta = load_coeff_meta(decomposition_dir)
        except Exception:
            coeff_meta = None

    train_cfg = cfg_get(cfg, "train", {}) or {}
    eval_cfg = cfg_get(train_cfg, "eval", {}) or {}
    eval_enabled = bool(cfg_get(eval_cfg, "enabled", True))
    val_ratio = float(cfg_get(eval_cfg, "val_ratio", 0.2))
    eval_shuffle = bool(cfg_get(eval_cfg, "shuffle", True))
    eval_seed = cfg_get(eval_cfg, "seed", cfg_get(cfg, "seed", None))
    eval_metrics = cfg_get(eval_cfg, "metrics", ["rmse", "mae", "r2"])
    eval_metrics = [str(m).strip().lower() for m in eval_metrics if str(m).strip()]

    cv_cfg = cfg_get(train_cfg, "cv", {}) or {}
    cv_enabled = bool(cfg_get(cv_cfg, "enabled", False))
    cv_folds = int(cfg_get(cv_cfg, "folds", 5))
    cv_shuffle = bool(cfg_get(cv_cfg, "shuffle", True))
    cv_seed = cfg_get(cv_cfg, "seed", cfg_get(cfg, "seed", None))

    tuning_cfg = cfg_get(train_cfg, "tuning", {}) or {}
    tuning_enabled = bool(cfg_get(tuning_cfg, "enabled", False))
    tuning_metric = str(cfg_get(tuning_cfg, "metric", "rmse")).strip().lower()
    tuning_maximize = bool(cfg_get(tuning_cfg, "maximize", False))
    param_grid = cfg_get(tuning_cfg, "param_grid", {}) or {}

    model_cfg = dict(cfg_get(cfg, "model", {}))
    model_cfg["target_space"] = target_space
    chosen_params: dict[str, Any] = {}

    train_idx, val_idx = _split_indices(
        conds.shape[0],
        val_ratio=val_ratio,
        seed=eval_seed,
        shuffle=eval_shuffle,
    )

    if tuning_enabled:
        if not param_grid:
            raise ValueError("train.tuning.param_grid is required when tuning.enabled is true")
        if not cv_enabled and val_idx.size == 0:
            raise ValueError("tuning requires cv.enabled or a non-zero eval.val_ratio")
        best_score = -np.inf if tuning_maximize else np.inf
        best_params: dict[str, Any] = {}
        with steps.step("tune_model"):
            for params in _grid_params(param_grid):
                trial_cfg = dict(model_cfg)
                trial_cfg.update(params)
                trial_cfg["target_space"] = target_space
                if cv_enabled:
                    scores = []
                    for fold_train_idx, fold_val_idx in _kfold_indices(
                        conds.shape[0], n_splits=cv_folds, seed=cv_seed, shuffle=cv_shuffle
                    ):
                        model = build_regressor(trial_cfg)
                        model.fit(conds[fold_train_idx], targets[fold_train_idx])
                        pred = model.predict(conds[fold_val_idx])
                        metrics = _compute_metrics(targets[fold_val_idx], pred)
                        scores.append(metrics.get(tuning_metric, metrics["rmse"]))
                    score = float(np.mean(scores)) if scores else float("nan")
                else:
                    model = build_regressor(trial_cfg)
                    model.fit(conds[train_idx], targets[train_idx])
                    pred = model.predict(conds[val_idx])
                    metrics = _compute_metrics(targets[val_idx], pred)
                    score = float(metrics.get(tuning_metric, metrics["rmse"]))
                is_better = score > best_score if tuning_maximize else score < best_score
                if is_better:
                    best_score = score
                    best_params = dict(params)
        chosen_params = best_params
        model_cfg.update(best_params)

    model = build_regressor(model_cfg)

    fit_time_sec: float | None = None
    with steps.step(
        "fit_model",
        outputs=[artifact_ref("model/model.pkl", kind="model")],
    ) as step:
        start = time.perf_counter()
        model.fit(conds, targets)
        model.save_state(run_dir)
        fit_time_sec = float(time.perf_counter() - start)
        step["meta"]["fit_time_sec"] = fit_time_sec

    metrics: dict[str, Any] = {
        "target_space": target_space,
        "preprocessing_run_dir": preprocess_dir,
    }
    if chosen_params:
        metrics["tuning_params"] = chosen_params
        metrics["tuning_metric"] = tuning_metric
        metrics["tuning_maximize"] = tuning_maximize
    if fit_time_sec is not None:
        metrics["fit_time_sec"] = fit_time_sec

    if eval_enabled:
        with steps.step("evaluate_model"):
            train_pred = model.predict(conds[train_idx])
            train_scores = _compute_metrics(targets[train_idx], train_pred)
            metrics.update({f"train_{k}": float(v) for k, v in train_scores.items()})
            metrics["train_samples"] = int(train_idx.size)
            if val_idx.size > 0:
                val_pred = model.predict(conds[val_idx])
                val_scores = _compute_metrics(targets[val_idx], val_pred)
                metrics.update({f"val_{k}": float(v) for k, v in val_scores.items()})
                metrics["val_samples"] = int(val_idx.size)

    if cv_enabled:
        with steps.step("cv_evaluate"):
            scores = []
            for fold_train_idx, fold_val_idx in _kfold_indices(
                conds.shape[0], n_splits=cv_folds, seed=cv_seed, shuffle=cv_shuffle
            ):
                cv_model = build_regressor(model_cfg)
                cv_model.fit(conds[fold_train_idx], targets[fold_train_idx])
                pred = cv_model.predict(conds[fold_val_idx])
                fold_scores = _compute_metrics(targets[fold_val_idx], pred)
                scores.append(fold_scores)
            if scores:
                for key in ("rmse", "mae", "r2"):
                    vals = [float(s[key]) for s in scores if key in s]
                    if vals:
                        metrics[f"cv_{key}_mean"] = float(np.mean(vals))
                        metrics[f"cv_{key}_std"] = float(np.std(vals))
                metrics["cv_folds"] = int(cv_folds)
    with steps.step(
        "write_metrics",
        outputs=[artifact_ref("outputs/metrics.json", kind="metrics")],
    ):
        writer.write_metrics(metrics)

    viz_cfg = cfg_get(train_cfg, "viz", {}) or {}
    viz_enabled = bool(cfg_get(viz_cfg, "enabled", True))
    if viz_enabled and eval_enabled:
        with steps.step(
            "render_plots",
            outputs=[artifact_ref("plots", kind="plots_dir")],
        ):
            max_dims = int(cfg_get(viz_cfg, "max_dims", 6))
            max_points = int(cfg_get(viz_cfg, "max_points", 2000))
            if coeff_meta is not None and coeff_a is not None:
                raw_norms = coeff_channel_norms(np.asarray(coeff_a), coeff_meta)
                if raw_norms is not None:
                    plot_channel_norm_scatter(
                        writer.plots_dir / "coeff_a_channel_norm_scatter.png",
                        raw_norms,
                        title="coeff_a channel norms",
                        max_points=max_points,
                    )
            if val_idx.size > 0:
                val_pred = model.predict(conds[val_idx])
                _plot_pred_scatter(
                    targets[val_idx],
                    val_pred,
                    out_dir=writer.plots_dir,
                    max_dims=max_dims,
                    max_points=max_points,
                    prefix="val",
                )
                _plot_residual_hist(
                    targets[val_idx],
                    val_pred,
                    out_dir=writer.plots_dir,
                    prefix="val",
                )
                _plot_coeff_scatter_2d(
                    targets[val_idx],
                    out_path=writer.plots_dir / "coeff_true_scatter_val.png",
                    title="coeff true (val)",
                    max_points=max_points,
                )
                _plot_coeff_scatter_2d(
                    val_pred,
                    out_path=writer.plots_dir / "coeff_pred_scatter_val.png",
                    title="coeff pred (val)",
                    max_points=max_points,
                )
                if coeff_meta is not None and target_space == "a":
                    val_true_norms = coeff_channel_norms(targets[val_idx], coeff_meta)
                    if val_true_norms is not None:
                        plot_channel_norm_scatter(
                            writer.plots_dir / "coeff_true_channel_norm_scatter_val.png",
                            val_true_norms,
                            title="coeff true channel norms (val)",
                            max_points=max_points,
                        )
                    val_pred_norms = coeff_channel_norms(val_pred, coeff_meta)
                    if val_pred_norms is not None:
                        plot_channel_norm_scatter(
                            writer.plots_dir / "coeff_pred_channel_norm_scatter_val.png",
                            val_pred_norms,
                            title="coeff pred channel norms (val)",
                            max_points=max_points,
                        )
            else:
                train_pred = model.predict(conds[train_idx])
                _plot_pred_scatter(
                    targets[train_idx],
                    train_pred,
                    out_dir=writer.plots_dir,
                    max_dims=max_dims,
                    max_points=max_points,
                    prefix="train",
                )
                _plot_residual_hist(
                    targets[train_idx],
                    train_pred,
                    out_dir=writer.plots_dir,
                    prefix="train",
                )
                _plot_coeff_scatter_2d(
                    targets[train_idx],
                    out_path=writer.plots_dir / "coeff_true_scatter_train.png",
                    title="coeff true (train)",
                    max_points=max_points,
                )
                _plot_coeff_scatter_2d(
                    train_pred,
                    out_path=writer.plots_dir / "coeff_pred_scatter_train.png",
                    title="coeff pred (train)",
                    max_points=max_points,
                )
                if coeff_meta is not None and target_space == "a":
                    train_true_norms = coeff_channel_norms(targets[train_idx], coeff_meta)
                    if train_true_norms is not None:
                        plot_channel_norm_scatter(
                            writer.plots_dir / "coeff_true_channel_norm_scatter_train.png",
                            train_true_norms,
                            title="coeff true channel norms (train)",
                            max_points=max_points,
                        )
                    train_pred_norms = coeff_channel_norms(train_pred, coeff_meta)
                    if train_pred_norms is not None:
                        plot_channel_norm_scatter(
                            writer.plots_dir / "coeff_pred_channel_norm_scatter_train.png",
                            train_pred_norms,
                            title="coeff pred channel norms (train)",
                            max_points=max_points,
                        )

    meta = build_meta(cfg)
    writer.write_manifest(meta=meta, steps=steps.to_list(), extra={"preprocessing_run_dir": preprocess_dir})
    writer.write_steps(steps.to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
