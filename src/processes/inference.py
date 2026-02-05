"""Process entrypoint: inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_meta,
    cfg_get,
    default_run_dir,
    load_coeff_meta,
    read_dataset_meta,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    snapshot_inputs,
)
from mode_decomp_ml.pipeline.loaders import load_model_state, load_preprocess_state_from_run
from mode_decomp_ml.plugins.coeff_post import BaseCoeffPost
from mode_decomp_ml.plugins.codecs import BaseCoeffCodec
from mode_decomp_ml.plugins.decomposers import BaseDecomposer
from mode_decomp_ml.viz import (
    coeff_channel_norms,
    coeff_energy_spectrum,
    coeff_value_magnitude,
    plot_channel_norm_scatter,
    plot_coeff_histogram,
    plot_coeff_spectrum,
    plot_field_grid,
    plot_vector_quiver,
    plot_vector_streamplot,
)

try:  # optional dependency
    import optuna
except Exception:  # pragma: no cover
    optuna = None


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for inference")
    return task_cfg


def _resolve_run_dir(cfg: Mapping[str, Any], key: str, default_name: str) -> str:
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    run_dir = cfg_get(task_cfg, key, None)
    if run_dir is None or str(run_dir).strip() == "":
        return str(default_run_dir(cfg, default_name))
    return str(resolve_path(run_dir))


def _cond_columns_from_meta(dataset_meta: Mapping[str, Any]) -> list[str]:
    cond_cols = dataset_meta.get("cond_columns")
    if isinstance(cond_cols, list) and cond_cols:
        return [str(col) for col in cond_cols]
    dataset_cfg = dataset_meta.get("dataset_cfg", {})
    cond_cols = dataset_cfg.get("feature_columns") if isinstance(dataset_cfg, Mapping) else None
    if isinstance(cond_cols, list) and cond_cols:
        return [str(col) for col in cond_cols]
    raise ValueError("cond_columns are required in dataset_meta for inference")


def _cond_array_from_values(values: Mapping[str, Any], cond_columns: list[str]) -> np.ndarray:
    row = []
    for col in cond_columns:
        if col not in values:
            raise ValueError(f"inference values missing column: {col}")
        row.append(float(values[col]))
    return np.asarray(row, dtype=np.float32)[None, :]


def _cond_array_from_csv(path: str | Path, cond_columns: list[str]) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("inference csv must include header row")
        for row in reader:
            values = [float(row[col]) for col in cond_columns]
            rows.append(values)
    if not rows:
        raise ValueError("inference csv contains no rows")
    return np.asarray(rows, dtype=np.float32)


def _cond_array_from_grid(grid_cfg: Mapping[str, Any], cond_columns: list[str]) -> np.ndarray:
    axes = []
    for col in cond_columns:
        values = grid_cfg.get(col)
        if values is None:
            raise ValueError(f"grid missing values for {col}")
        axes.append(np.asarray(values, dtype=np.float32))
    meshes = np.meshgrid(*axes, indexing="xy")
    stacked = np.stack([mesh.reshape(-1) for mesh in meshes], axis=1)
    return stacked


def _cond_array_from_random(ranges_cfg: Mapping[str, Any], cond_columns: list[str], n_samples: int) -> np.ndarray:
    rng = np.random.default_rng()
    rows = []
    for _ in range(int(n_samples)):
        row = []
        for col in cond_columns:
            bounds = ranges_cfg.get(col)
            if bounds is None or len(bounds) != 2:
                raise ValueError(f"random ranges missing for {col}")
            low, high = float(bounds[0]), float(bounds[1])
            row.append(rng.uniform(low, high))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _objective_value(name: str, field: np.ndarray, coeff: np.ndarray, *, index: int | None = None) -> float:
    name = str(name or "").strip().lower()
    if name in {"field_std", "field_variance"}:
        return float(np.nanstd(field))
    if name in {"field_mean", "field_avg"}:
        return float(np.nanmean(field))
    if name == "field_abs_mean":
        return float(np.nanmean(np.abs(field)))
    if name == "coeff_value":
        if index is None:
            raise ValueError("objective index is required for coeff_value")
        return float(coeff.reshape(-1)[int(index)])
    raise ValueError(f"Unknown objective: {name}")


def _objective_values(
    name: str,
    fields: np.ndarray,
    coeffs: np.ndarray,
    *,
    index: int | None = None,
) -> np.ndarray:
    values = []
    for idx in range(fields.shape[0]):
        values.append(_objective_value(name, fields[idx], coeffs[idx], index=index))
    return np.asarray(values, dtype=np.float32)


def _field_magnitude(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 2:
        return np.abs(field)
    if field.ndim == 3:
        if field.shape[2] == 1:
            return np.abs(field[..., 0])
        return np.linalg.norm(field, axis=-1)
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _plot_objective_scatter(
    conds: np.ndarray,
    objective: np.ndarray,
    cond_columns: list[str],
    *,
    out_dir: Path,
    max_points: int,
) -> None:
    if conds.size == 0 or objective.size == 0:
        return
    n = min(int(max_points), int(conds.shape[0]))
    conds = np.asarray(conds)[:n]
    objective = np.asarray(objective)[:n]
    for idx, col in enumerate(cond_columns):
        fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        ax.scatter(conds[:, idx], objective, s=12, alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("objective")
        path = out_dir / f"objective_vs_{col}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    if len(cond_columns) >= 2:
        fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
        scatter = ax.scatter(conds[:, 0], conds[:, 1], c=objective, s=14, cmap="viridis", alpha=0.8)
        ax.set_xlabel(cond_columns[0])
        ax.set_ylabel(cond_columns[1])
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="objective")
        path = out_dir / "objective_scatter_2d.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def _plot_fields(
    field_hat: np.ndarray,
    *,
    out_dir: Path,
    max_samples: int,
    mask: np.ndarray | None,
) -> None:
    if field_hat is None:
        return
    field_hat = np.asarray(field_hat)
    n_samples = min(int(max_samples), int(field_hat.shape[0]))
    for idx in range(n_samples):
        plot_field_grid(
            out_dir / f"field_pred_{idx:04d}.png",
            [field_hat[idx]],
            [f"pred_{idx:04d}"],
            mask=mask,
        )
        if field_hat.shape[-1] > 1:
            try:
                plot_vector_streamplot(
                    out_dir / f"field_pred_stream_{idx:04d}.png",
                    field_hat[idx],
                    mask=mask,
                )
                plot_vector_quiver(
                    out_dir / f"field_pred_quiver_{idx:04d}.png",
                    field_hat[idx],
                    mask=mask,
                )
            except Exception:
                pass
            mag = _field_magnitude(field_hat[idx])
            plot_field_grid(
                out_dir / f"field_pred_mag_{idx:04d}.png",
                [mag],
                [f"pred_mag_{idx:04d}"],
                mask=mask,
            )


def _plot_field_samples_grid(
    field_hat: np.ndarray,
    *,
    out_dir: Path,
    max_samples: int,
    mask: np.ndarray | None,
) -> None:
    field_hat = np.asarray(field_hat)
    n_samples = min(int(max_samples), int(field_hat.shape[0]))
    if n_samples <= 0:
        return
    fields = [field_hat[idx] for idx in range(n_samples)]
    titles = [f"pred_{idx:04d}" for idx in range(n_samples)]
    plot_field_grid(
        out_dir / "field_pred_samples.png",
        fields,
        titles,
        mask=mask,
        suptitle="field_pred samples",
    )
    if field_hat.shape[-1] > 1:
        mags = [_field_magnitude(field_hat[idx]) for idx in range(n_samples)]
        plot_field_grid(
            out_dir / "field_pred_mag_samples.png",
            mags,
            titles,
            mask=mask,
            suptitle="field_pred magnitude samples",
        )


def _plot_field_stats(field_hat: np.ndarray, *, out_dir: Path, mask: np.ndarray | None) -> None:
    field_hat = np.asarray(field_hat)
    if field_hat.size == 0:
        return
    mean_field = np.mean(field_hat, axis=0)
    std_field = np.std(field_hat, axis=0)
    plot_field_grid(
        out_dir / "field_pred_mean.png",
        [mean_field],
        ["mean_pred"],
        mask=mask,
        suptitle="field mean (pred)",
    )
    plot_field_grid(
        out_dir / "field_pred_std.png",
        [std_field],
        ["std_pred"],
        mask=mask,
        suptitle="field std (pred)",
    )
    if field_hat.ndim == 4 and field_hat.shape[-1] > 1:
        mag = np.stack([_field_magnitude(arr) for arr in field_hat], axis=0)
        mag_mean = np.mean(mag, axis=0)
        mag_std = np.std(mag, axis=0)
        plot_field_grid(
            out_dir / "field_pred_mag_mean.png",
            [mag_mean],
            ["mean_pred_mag"],
            mask=mask,
            suptitle="field magnitude mean (pred)",
        )
        plot_field_grid(
            out_dir / "field_pred_mag_std.png",
            [mag_std],
            ["std_pred_mag"],
            mask=mask,
            suptitle="field magnitude std (pred)",
        )


def _plot_coeff_diagnostics(
    coeff_a: np.ndarray,
    coeff_meta: Mapping[str, Any] | None,
    *,
    out_dir: Path,
    hist_bins: int,
    hist_scale: str,
    spectrum_scale: str,
) -> None:
    if coeff_a is None:
        return
    spectrum = coeff_energy_spectrum(coeff_a, coeff_meta)
    plot_coeff_spectrum(out_dir / "coeff_spectrum_pred.png", spectrum, scale=spectrum_scale)
    coeff_mag = coeff_value_magnitude(coeff_a, coeff_meta)
    plot_coeff_histogram(out_dir / "coeff_hist_pred.png", coeff_mag, bins=hist_bins, scale=hist_scale)


def _save_optuna_plots(study: Any, out_dir: Path) -> None:
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
    except Exception:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ax = plot_optimization_history(study)
        ax.figure.savefig(out_dir / "optuna_history.png", dpi=150)
        plt.close(ax.figure)
    except Exception:
        pass
    try:
        ax = plot_param_importances(study)
        ax.figure.savefig(out_dir / "optuna_param_importance.png", dpi=150)
        plt.close(ax.figure)
    except Exception:
        pass
    try:
        ax = plot_slice(study)
        ax.figure.savefig(out_dir / "optuna_slice.png", dpi=150)
        plt.close(ax.figure)
    except Exception:
        pass
    try:
        ax = plot_parallel_coordinate(study)
        ax.figure.savefig(out_dir / "optuna_parallel_coordinate.png", dpi=150)
        plt.close(ax.figure)
    except Exception:
        pass


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("inference requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["run_dir", "domain"])
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

    decomposition_dir = _resolve_run_dir(cfg, "decomposition_run_dir", "decomposition")
    preprocessing_dir = _resolve_run_dir(cfg, "preprocessing_run_dir", "preprocessing")
    train_dir = _resolve_run_dir(cfg, "train_run_dir", "train")

    dataset_meta = read_dataset_meta(decomposition_dir)
    cond_columns = _cond_columns_from_meta(dataset_meta)

    coeff_meta = load_coeff_meta(decomposition_dir)
    field_shape = tuple(coeff_meta.get("field_shape", []))
    if not field_shape:
        raise ValueError("coeff_meta.field_shape is required for inference")
    domain_cfg = cfg_get(cfg, "domain")
    domain_spec = build_domain_spec(domain_cfg, field_shape)

    with steps.step(
        "load_artifacts",
        inputs=[
            artifact_ref(f"{decomposition_dir}/outputs/states/decomposer/state.pkl", kind="state"),
            artifact_ref(f"{decomposition_dir}/outputs/states/coeff_codec/state.pkl", kind="state"),
            artifact_ref(f"{preprocessing_dir}/outputs/states/coeff_post/state.pkl", kind="state"),
            artifact_ref(f"{train_dir}/model/model.pkl", kind="model"),
        ],
    ):
        decomposer = BaseDecomposer.load_state(Path(decomposition_dir) / "outputs" / "states" / "decomposer" / "state.pkl")
        codec = BaseCoeffCodec.load_state(Path(decomposition_dir) / "outputs" / "states" / "coeff_codec" / "state.pkl")
        coeff_post = BaseCoeffPost.load_state(Path(preprocessing_dir) / "outputs" / "states" / "coeff_post" / "state.pkl")
        model = load_model_state(train_dir)
        preprocess = load_preprocess_state_from_run(decomposition_dir)

    inf_cfg = cfg_get(cfg, "inference", {}) or {}
    mode = str(cfg_get(inf_cfg, "mode", "single")).strip().lower()
    sampler_name = str(cfg_get(inf_cfg, "sampler", "tpe")).strip().lower()
    objective_cfg = cfg_get(inf_cfg, "objective", {}) or {}
    objective_name = str(cfg_get(objective_cfg, "name", "")).strip()
    objective_direction = str(cfg_get(objective_cfg, "direction", "minimize")).strip().lower()
    objective_index = cfg_get(objective_cfg, "index", None)

    conds: np.ndarray | None = None
    coeff_a: np.ndarray | None = None
    coeff_std: np.ndarray | None = None
    field_hat: np.ndarray | None = None
    objective_values: np.ndarray | None = None
    study: Any | None = None

    def _predict_fields(conds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        coeff_pred, coeff_std = model.predict_with_std(conds)
        coeff_pred = np.asarray(coeff_pred)
        coeff_std = None if coeff_std is None else np.asarray(coeff_std)
        target_space = getattr(model, "target_space", None)
        if target_space not in {"a", "z"}:
            raise ValueError("model.target_space must be 'a' or 'z'")
        coeff_a = coeff_post.inverse_transform(coeff_pred) if target_space == "z" else coeff_pred
        fields = []
        raw_meta = coeff_meta.get("raw_meta") if isinstance(coeff_meta, Mapping) else None
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else coeff_meta
        for idx in range(coeff_a.shape[0]):
            raw_coeff = codec.decode(coeff_a[idx], raw_meta)
            fields.append(decomposer.inverse_transform(raw_coeff, domain_spec=domain_spec))
        field_hat = np.stack(fields, axis=0)
        field_hat, _ = preprocess.inverse_transform(field_hat, None)
        return coeff_a, field_hat, coeff_std

    results: dict[str, Any] = {"mode": mode}
    if objective_name:
        results["objective_name"] = objective_name
        results["objective_direction"] = objective_direction

    if mode == "single":
        values = cfg_get(inf_cfg, "values", None)
        if not isinstance(values, Mapping):
            raise ValueError("inference.values must be provided for single mode")
        conds = _cond_array_from_values(values, cond_columns)
        coeff_a, field_hat, coeff_std = _predict_fields(conds)
        if objective_name:
            objective_values = _objective_values(objective_name, field_hat, coeff_a, index=objective_index)
            results["objective_value"] = float(objective_values[0])
        writer.write_preds(
            {
                "coeff": coeff_a,
                "coeff_mean": coeff_a,
                "coeff_std": coeff_std,
                "field": field_hat,
                "cond": conds,
                "objective": objective_values,
            }
        )
        results["num_samples"] = int(conds.shape[0])
    elif mode == "batch":
        source = str(cfg_get(inf_cfg, "source", "grid")).strip().lower()
        if source == "csv":
            csv_path = cfg_get(inf_cfg, "csv_path", None)
            if csv_path is None:
                raise ValueError("inference.csv_path is required for batch csv source")
            conds = _cond_array_from_csv(csv_path, cond_columns)
        elif source == "random":
            ranges_cfg = cfg_get(inf_cfg, "ranges", None)
            if not isinstance(ranges_cfg, Mapping):
                raise ValueError("inference.ranges is required for batch random source")
            n_samples = int(cfg_get(inf_cfg, "n_samples", 16))
            conds = _cond_array_from_random(ranges_cfg, cond_columns, n_samples)
        else:
            grid_cfg = cfg_get(inf_cfg, "grid", None)
            if not isinstance(grid_cfg, Mapping):
                raise ValueError("inference.grid is required for batch grid source")
            conds = _cond_array_from_grid(grid_cfg, cond_columns)
        coeff_a, field_hat, coeff_std = _predict_fields(conds)
        if objective_name:
            objective_values = _objective_values(objective_name, field_hat, coeff_a, index=objective_index)
            results["objective_mean"] = float(np.mean(objective_values))
            results["objective_min"] = float(np.min(objective_values))
            results["objective_max"] = float(np.max(objective_values))
        writer.write_preds(
            {
                "coeff": coeff_a,
                "coeff_mean": coeff_a,
                "coeff_std": coeff_std,
                "field": field_hat,
                "cond": conds,
                "objective": objective_values,
            }
        )
        results["num_samples"] = int(conds.shape[0])
    elif mode == "optimize":
        if optuna is None:
            raise RuntimeError("optuna is required for optimize mode")
        ranges_cfg = cfg_get(inf_cfg, "ranges", None)
        if not isinstance(ranges_cfg, Mapping):
            raise ValueError("inference.ranges is required for optimize mode")
        if not objective_name:
            raise ValueError("inference.objective.name is required for optimize mode")
        if objective_direction not in {"minimize", "maximize"}:
            raise ValueError("objective.direction must be minimize or maximize")
        obj_index = objective_index

        if sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction=objective_direction, sampler=sampler)

        def _trial_objective(trial: optuna.Trial) -> float:
            values = {}
            for col in cond_columns:
                bounds = ranges_cfg.get(col)
                if bounds is None or len(bounds) != 2:
                    raise ValueError(f"ranges missing for {col}")
                low, high = float(bounds[0]), float(bounds[1])
                values[col] = trial.suggest_float(col, low, high)
            conds = _cond_array_from_values(values, cond_columns)
            coeff_a, field_hat, _ = _predict_fields(conds)
            score = _objective_value(objective_name, field_hat, coeff_a, index=obj_index)
            return float(score)

        n_trials = int(cfg_get(inf_cfg, "n_trials", 50))
        study.optimize(_trial_objective, n_trials=n_trials)

        best_params = study.best_params
        conds = _cond_array_from_values(best_params, cond_columns)
        coeff_a, field_hat, coeff_std = _predict_fields(conds)
        objective_values = _objective_values(objective_name, field_hat, coeff_a, index=objective_index)
        writer.write_preds(
            {
                "coeff": coeff_a,
                "coeff_mean": coeff_a,
                "coeff_std": coeff_std,
                "field": field_hat,
                "cond": conds,
                "objective": objective_values,
            }
        )
        results["best_params"] = best_params
        results["best_value"] = float(study.best_value)
        results["objective_value"] = float(objective_values[0])
        results["n_trials"] = int(n_trials)
        results["sampler"] = sampler_name
    else:
        raise ValueError(f"Unknown inference.mode: {mode}")

    if study is not None:
        trials_path = writer.outputs_dir / "optuna_trials.csv"
        best_path = writer.outputs_dir / "optuna_best.json"
        with steps.step(
            "write_optuna_artifacts",
            outputs=[
                artifact_ref("outputs/optuna_trials.csv", kind="table"),
                artifact_ref("outputs/optuna_best.json", kind="metadata"),
            ],
        ):
            param_keys = sorted({key for trial in study.trials for key in trial.params.keys()})
            with trials_path.open("w", encoding="utf-8", newline="") as fh:
                writer_csv = csv.DictWriter(
                    fh,
                    fieldnames=["number", "value", "state", *param_keys],
                )
                writer_csv.writeheader()
                for trial in study.trials:
                    row = {
                        "number": trial.number,
                        "value": trial.value,
                        "state": str(trial.state),
                    }
                    for key in param_keys:
                        row[key] = trial.params.get(key)
                    writer_csv.writerow(row)
            best_payload = {
                "best_params": study.best_params,
                "best_value": float(study.best_value),
                "direction": study.direction.name if hasattr(study.direction, "name") else str(study.direction),
            }
            best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    with steps.step(
        "write_metrics",
        outputs=[artifact_ref("outputs/metrics.json", kind="metrics")],
    ):
        writer.write_metrics(results)

    with steps.step(
        "render_plots",
        outputs=[artifact_ref("plots", kind="plots_dir")],
    ):
        viz_cfg = cfg_get(inf_cfg, "viz", {}) or {}
        max_samples = int(cfg_get(viz_cfg, "max_samples", 8))
        max_points = int(cfg_get(viz_cfg, "scatter_max_points", 2000))
        field_stats_cfg = cfg_get(viz_cfg, "field_stats", {}) or {}
        field_stats_enabled = bool(cfg_get(field_stats_cfg, "enabled", True))
        hist_bins = int(cfg_get(viz_cfg, "coeff_hist_bins", 60))
        hist_scale = str(cfg_get(viz_cfg, "coeff_hist_scale", "log")).strip().lower()
        spectrum_scale = str(cfg_get(viz_cfg, "coeff_spectrum_scale", "log")).strip().lower()

        if field_hat is not None:
            domain_mask = domain_spec.mask
            _plot_fields(field_hat, out_dir=writer.plots_dir, max_samples=max_samples, mask=domain_mask)
            _plot_field_samples_grid(field_hat, out_dir=writer.plots_dir, max_samples=max_samples, mask=domain_mask)
            if field_stats_enabled:
                _plot_field_stats(field_hat, out_dir=writer.plots_dir, mask=domain_mask)
        if coeff_a is not None:
            _plot_coeff_diagnostics(
                coeff_a,
                coeff_meta,
                out_dir=writer.plots_dir,
                hist_bins=hist_bins,
                hist_scale=hist_scale,
                spectrum_scale=spectrum_scale,
            )
            norms = coeff_channel_norms(coeff_a, coeff_meta)
            if norms is not None:
                plot_channel_norm_scatter(
                    writer.plots_dir / "coeff_channel_norm_scatter.png",
                    norms,
                    title="coeff channel norms",
                    max_points=max_points,
                )
        if objective_values is not None and conds is not None:
            _plot_objective_scatter(
                conds,
                objective_values,
                cond_columns,
                out_dir=writer.plots_dir,
                max_points=max_points,
            )
            fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
            ax.hist(objective_values, bins=40, color="#4C78A8", alpha=0.85)
            ax.set_xlabel("objective")
            ax.set_ylabel("count")
            fig.savefig(writer.plots_dir / "objective_hist.png", dpi=150)
            plt.close(fig)
        if study is not None:
            _save_optuna_plots(study, writer.plots_dir / "optuna")

    meta = build_meta(cfg)
    writer.write_manifest(
        meta=meta,
        steps=steps.to_list(),
        extra={
            "decomposition_run_dir": decomposition_dir,
            "preprocessing_run_dir": preprocessing_dir,
            "train_run_dir": train_dir,
        },
    )
    writer.write_steps(steps.to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
