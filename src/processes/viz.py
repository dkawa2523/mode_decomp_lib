"""Process entrypoint: viz.

Process layer: loads artifacts/config, then calls `mode_decomp_ml.viz` helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mode_decomp_ml.data import build_dataset
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_dataset_meta,
    build_meta,
    combine_masks,
    cfg_get,
    dataset_to_arrays,
    ensure_dir,
    load_coeff_meta,
    require_cfg_keys,
    resolve_domain_cfg,
    resolve_run_dir,
    split_indices,
)
from mode_decomp_ml.pipeline.artifacts import load_coeff_predictions, load_field_predictions, load_field_std
from mode_decomp_ml.pipeline.loaders import load_preprocess_state_from_run, load_train_artifacts
from mode_decomp_ml.tracking import maybe_log_run
from mode_decomp_ml.viz import (
    coeff_energy_spectrum,
    coeff_energy_vector,
    coeff_value_magnitude,
    fft_magnitude_spectrum,
    plot_coeff_spectrum,
    plot_coeff_histogram,
    plot_domain_error_map,
    plot_domain_field_grid,
    plot_lat_profile,
    plot_radial_profile,
    plot_angular_profile,
    plot_polar_field,
    plot_mesh_field,
    plot_energy_bars,
    plot_error_map,
    plot_field_grid,
    plot_line_series,
    plot_topk_contrib,
    plot_uncertainty_map,
    slepian_concentration,
    spherical_l_energy,
    wavelet_band_energy,
)


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for viz")
    return task_cfg


def _load_pred_coeff(predict_run_dir: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    coeff, _, meta = load_coeff_predictions(predict_run_dir)
    return coeff, meta


def _load_recon_field(reconstruct_run_dir: str) -> np.ndarray:
    return load_field_predictions(reconstruct_run_dir)


def _load_field_std(reconstruct_run_dir: str) -> tuple[np.ndarray | None, Mapping[str, Any]]:
    return load_field_std(reconstruct_run_dir)


def _normalize_k_list(
    k_list: Sequence[int],
    *,
    total_coeff: int,
    coeff_meta: Mapping[str, Any] | None,
) -> list[int]:
    if total_coeff <= 0:
        raise ValueError("total_coeff must be positive")
    complex_format = str(coeff_meta.get("complex_format", "")) if coeff_meta else ""
    coeff_shape = coeff_meta.get("coeff_shape") if coeff_meta else None
    needs_even = (
        complex_format in {"real_imag", "mag_phase", "logmag_phase"}
        and isinstance(coeff_shape, list)
        and coeff_shape[-1] == 2
    )
    max_coeff = total_coeff
    if needs_even and max_coeff % 2 != 0:
        max_coeff -= 1
    out: list[int] = []
    for value in k_list:
        k = int(value)
        if k <= 0:
            continue
        if needs_even and k % 2 != 0:
            k += 1
        k = min(k, max_coeff)
        if k not in out:
            out.append(k)
    if not out:
        out.append(max_coeff)
    return out


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("viz requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["seed", "run_dir", "dataset", "split", "viz"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    train_run_dir = cfg_get(task_cfg, "train_run_dir", None)
    predict_run_dir = cfg_get(task_cfg, "predict_run_dir", None)
    reconstruct_run_dir = cfg_get(task_cfg, "reconstruct_run_dir", None)
    if not train_run_dir or not str(train_run_dir).strip():
        raise ValueError("task.train_run_dir is required for viz")
    if not predict_run_dir or not str(predict_run_dir).strip():
        raise ValueError("task.predict_run_dir is required for viz")
    if not reconstruct_run_dir or not str(reconstruct_run_dir).strip():
        raise ValueError("task.reconstruct_run_dir is required for viz")

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    with steps.step(
        "init_run",
        outputs=[artifact_ref("run.yaml", kind="config")],
    ):
        writer.ensure_layout()
        writer.write_run_yaml(cfg)

    seed = cfg_get(cfg, "seed", None)
    dataset_cfg = cfg_get(cfg, "dataset")
    domain_cfg = cfg_get(cfg, "domain")
    domain_cfg, _ = resolve_domain_cfg(dataset_cfg, domain_cfg)
    split_cfg = cfg_get(cfg, "split")
    viz_cfg = cfg_get(cfg, "viz", {}) or {}
    sphere_projection = str(cfg_get(viz_cfg, "sphere_projection", "plate_carre")).strip().lower()
    if sphere_projection not in {"plate_carre", "mollweide"}:
        raise ValueError("viz.sphere_projection must be plate_carre or mollweide")

    with steps.step(
        "build_dataset",
        inputs=[artifact_ref("run.yaml", kind="config")],
    ) as step:
        dataset = build_dataset(dataset_cfg, domain_cfg=domain_cfg, seed=seed)
        _, fields_true_all, masks_all, _ = dataset_to_arrays(dataset)
        split_meta = split_indices(split_cfg, len(dataset), seed)
        eval_idx = np.asarray(split_meta["train_idx"], dtype=int)
        fields_true = fields_true_all[eval_idx]
        masks = masks_all[eval_idx] if masks_all is not None else None
        step["meta"]["dataset_name"] = getattr(dataset, "name", None)
        step["meta"]["num_samples"] = int(len(dataset))

    domain_name = str(cfg_get(domain_cfg, "name", "")).strip().lower()
    needs_mask = domain_name in {"disk", "mask", "arbitrary_mask", "mesh"}
    if needs_mask:
        domain_spec = build_domain_spec(domain_cfg, fields_true.shape[1:])
        masks = combine_masks(
            masks,
            domain_spec.mask,
            spatial_shape=fields_true.shape[1:3],
            n_samples=fields_true.shape[0],
        )

    with steps.step(
        "load_predictions",
        inputs=[
            artifact_ref(Path(str(reconstruct_run_dir)) / "preds.npz", kind="preds"),
            artifact_ref(Path(str(predict_run_dir)) / "preds.npz", kind="preds"),
        ],
    ):
        field_pred_full = _load_recon_field(str(reconstruct_run_dir))
        field_pred = field_pred_full[eval_idx]
        if field_pred.shape != fields_true.shape:
            raise ValueError("field_hat shape does not match dataset field shape")

    sample_index = int(cfg_get(viz_cfg, "sample_index", 0))
    if sample_index < 0 or sample_index >= fields_true.shape[0]:
        raise ValueError("viz.sample_index is out of range")

    field_true_sample = fields_true[sample_index]
    field_pred_sample = field_pred[sample_index]
    mask_sample = masks[sample_index] if masks is not None else None

    with steps.step(
        "load_train_artifacts",
        inputs=[artifact_ref(Path(str(train_run_dir)) / "states", kind="state_dir")],
    ):
        coeff_pred, preds_meta = _load_pred_coeff(str(predict_run_dir))
        coeff_pred = coeff_pred[eval_idx]
        decomposer, coeff_post, codec, model = load_train_artifacts(str(train_run_dir))
        preprocess = load_preprocess_state_from_run(str(train_run_dir))
        target_space = preds_meta.get("target_space") or getattr(model, "target_space", None)
        if target_space not in {"a", "z"}:
            raise ValueError("target_space must be 'a' or 'z' for viz")
        coeff_a = coeff_post.inverse_transform(coeff_pred) if target_space == "z" else coeff_pred

    # REVIEW: coeff_meta defines coefficient ordering and spectrum semantics.
    coeff_meta = load_coeff_meta(str(train_run_dir))
    raw_meta = coeff_meta.get("raw_meta") if isinstance(coeff_meta, Mapping) else None
    raw_meta = raw_meta if isinstance(raw_meta, Mapping) else coeff_meta
    coeff_shape = coeff_meta.get("coeff_shape")
    if isinstance(coeff_shape, list):
        expected = int(np.prod(coeff_shape))
        if coeff_a.shape[1] != expected:
            raise ValueError("coeff_a dimension does not match coeff_meta")

    field_shape = tuple(coeff_meta.get("field_shape", []))
    if not field_shape:
        raise ValueError("coeff_meta.field_shape is required for viz")
    domain_spec = build_domain_spec(domain_cfg, field_shape)

    spectrum = coeff_energy_spectrum(coeff_a, coeff_meta)
    spectrum_scale = str(cfg_get(viz_cfg, "spectrum_scale", "log")).strip().lower()

    diag_cfg = cfg_get(viz_cfg, "coeff_diag", {}) or {}
    diag_enabled = bool(cfg_get(diag_cfg, "enabled", True))
    hist_bins = int(cfg_get(diag_cfg, "hist_bins", 60))
    if hist_bins <= 0:
        hist_bins = 60
    hist_scale = str(cfg_get(diag_cfg, "hist_scale", "log")).strip().lower()
    top_k = int(cfg_get(diag_cfg, "top_k", 10))
    fft_enabled = bool(cfg_get(diag_cfg, "fft_magnitude", True))
    wavelet_enabled = bool(cfg_get(diag_cfg, "wavelet_band_energy", True))
    sh_enabled = bool(cfg_get(diag_cfg, "sh_l_energy", True))
    slepian_enabled = bool(cfg_get(diag_cfg, "slepian_concentration", True))

    energy_vec = coeff_energy_vector(coeff_a, coeff_meta) if diag_enabled else None
    coeff_mag = coeff_value_magnitude(coeff_a, coeff_meta) if diag_enabled else None
    fft_mag = fft_magnitude_spectrum(coeff_a, coeff_meta) if diag_enabled and fft_enabled else None
    wavelet_band = (
        wavelet_band_energy(coeff_a, coeff_meta) if diag_enabled and wavelet_enabled else None
    )
    sh_energy = spherical_l_energy(coeff_a, coeff_meta) if diag_enabled and sh_enabled else None
    slepian_vals = (
        slepian_concentration(coeff_meta) if diag_enabled and slepian_enabled else None
    )

    k_list = cfg_get(viz_cfg, "k_list", None)
    if k_list is None:
        k_list = [1, 2, 4, 8, 16]
    k_list = _normalize_k_list(k_list, total_coeff=coeff_a.shape[1], coeff_meta=coeff_meta)

    coeff_sample = coeff_a[sample_index]
    recon_fields = []
    for k in k_list:
        coeff_trunc = coeff_sample.copy()
        if k < coeff_trunc.size:
            coeff_trunc[k:] = 0.0
        # REVIEW: sequential recon uses flat coeff order for comparability.
        raw_coeff = codec.decode(coeff_trunc, raw_meta)
        recon_fields.append(decomposer.inverse_transform(raw_coeff, domain_spec=domain_spec))

    recon_fields = np.stack(recon_fields, axis=0)
    recon_fields, _ = preprocess.inverse_transform(recon_fields, None)

    with steps.step(
        "render_figures",
        outputs=[artifact_ref("figures", kind="figures_dir")],
    ):
        viz_root = writer.figures_dir
        sample_dir = ensure_dir(viz_root / f"sample_{sample_index:04d}")
        # CONTRACT: fixed filenames enable cross-run comparisons.
        plot_field_grid(
            sample_dir / "field_compare.png",
            [field_true_sample, field_pred_sample],
            ["true", "pred"],
            mask=mask_sample,
            suptitle="field_true vs field_hat",
        )
        plot_error_map(sample_dir / "error_map.png", field_true_sample, field_pred_sample, mask=mask_sample)
        plot_field_grid(
            sample_dir / "recon_sequence.png",
            list(recon_fields),
            [f"k={k}" for k in k_list],
            mask=mask_sample,
            suptitle="sequential reconstruction",
        )
        domain_dir = ensure_dir(viz_root / "domain")
        plot_domain_field_grid(
            domain_dir / f"field_compare_{sample_index:04d}.png",
            [field_true_sample, field_pred_sample],
            ["true", "pred"],
            domain_spec=domain_spec,
            mask=mask_sample,
            suptitle="domain-aware field compare",
            sphere_projection=sphere_projection,
        )
        plot_domain_error_map(
            domain_dir / f"error_map_{sample_index:04d}.png",
            field_true_sample,
            field_pred_sample,
            domain_spec=domain_spec,
            mask=mask_sample,
            sphere_projection=sphere_projection,
        )
        if domain_name in {"disk", "annulus"}:
            plot_polar_field(
                domain_dir / f"field_polar_true_{sample_index:04d}.png",
                field_true_sample,
                domain_spec=domain_spec,
                mask=mask_sample,
            )
            plot_polar_field(
                domain_dir / f"field_polar_pred_{sample_index:04d}.png",
                field_pred_sample,
                domain_spec=domain_spec,
                mask=mask_sample,
            )
            plot_radial_profile(
                domain_dir / f"radial_profile_{sample_index:04d}.png",
                field_true_sample,
                field_pred_sample,
                domain_spec=domain_spec,
                mask=mask_sample,
            )
            plot_angular_profile(
                domain_dir / f"angular_profile_{sample_index:04d}.png",
                field_true_sample,
                field_pred_sample,
                domain_spec=domain_spec,
                mask=mask_sample,
            )
        if domain_name == "sphere_grid":
            plot_lat_profile(
                domain_dir / f"lat_profile_{sample_index:04d}.png",
                field_true_sample,
                field_pred_sample,
                domain_spec=domain_spec,
                mask=mask_sample,
            )
        if domain_name == "mesh":
            diff = field_pred_sample - field_true_sample
            if diff.ndim == 3 and diff.shape[-1] > 1:
                error_field = np.linalg.norm(diff, axis=-1)[..., None]
            else:
                error_field = np.abs(diff)
            plot_mesh_field(
                domain_dir / f"mesh_field_{sample_index:04d}.png",
                field_true_sample,
                domain_spec=domain_spec,
            )
            plot_mesh_field(
                domain_dir / f"mesh_error_{sample_index:04d}.png",
                error_field,
                domain_spec=domain_spec,
            )
        plot_coeff_spectrum(viz_root / "coeff_spectrum.png", spectrum, scale=spectrum_scale)

        if diag_enabled:
            if coeff_mag is not None and coeff_mag.size > 0:
                plot_coeff_histogram(
                    viz_root / "coeff_hist.png",
                    coeff_mag,
                    bins=hist_bins,
                    scale=hist_scale,
                )
            if energy_vec is not None and energy_vec.size > 0:
                plot_topk_contrib(viz_root / "coeff_topk_energy.png", energy_vec, top_k=top_k)
            if fft_mag is not None:
                plot_coeff_spectrum(
                    viz_root / "fft_magnitude_spectrum.png",
                    {"kind": "heatmap", "data": fft_mag},
                    scale="log",
                )
            if wavelet_band is not None:
                labels, energies = wavelet_band
                plot_energy_bars(
                    viz_root / "wavelet_band_energy.png",
                    labels,
                    energies,
                    ylabel="mean energy",
                )
            if sh_energy is not None:
                levels, energies = sh_energy
                plot_line_series(
                    viz_root / "sh_l_energy.png",
                    levels,
                    energies,
                    xlabel="l",
                    ylabel="energy",
                )
            if slepian_vals is not None:
                plot_line_series(
                    viz_root / "slepian_concentration.png",
                    np.arange(1, len(slepian_vals) + 1),
                    slepian_vals,
                    xlabel="mode index",
                    ylabel="concentration",
                )

        uncertainty_cfg = cfg_get(cfg, "uncertainty", {}) or {}
        uncertainty_enabled = bool(cfg_get(uncertainty_cfg, "enabled", False))
        if uncertainty_enabled:
            field_std, std_meta = _load_field_std(str(reconstruct_run_dir))
            if field_std is None:
                raise ValueError("uncertainty enabled but field_std is missing from reconstruct run")
            case_indices = std_meta.get("case_indices")
            if not isinstance(case_indices, list) or not case_indices:
                raise ValueError("field_std_meta.case_indices is required for uncertainty viz")
            if field_std.shape[0] != len(case_indices):
                raise ValueError("field_std batch size does not match case_indices")
            for offset, case_idx in enumerate(case_indices):
                idx = int(case_idx)
                if idx < 0 or idx >= fields_true_all.shape[0]:
                    raise ValueError("field_std case index is out of dataset bounds")
                mask_case = masks_all[idx] if masks_all is not None else None
                # CONTRACT: uncertainty maps are persisted under figures/uncertainty_map_<id>.png.
                plot_uncertainty_map(
                    viz_root / f"uncertainty_map_{idx:04d}.png",
                    field_std[offset],
                    mask=mask_case,
                )

    dataset_meta = build_dataset_meta(dataset, dataset_cfg, split_meta, domain_cfg)

    meta = build_meta(cfg, dataset_hash=dataset_meta["dataset_hash"])
    writer.write_manifest(meta=meta, dataset_meta=dataset_meta, steps=steps.to_list())
    maybe_log_run(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
