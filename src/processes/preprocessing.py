"""Process entrypoint: preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_decomp_ml.plugins.coeff_post import build_coeff_post
from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_meta,
    cfg_get,
    default_run_dir,
    load_coeff_meta,
    require_cfg_keys,
    resolve_path,
    resolve_run_dir,
    snapshot_inputs,
)
from mode_decomp_ml.pipeline.artifacts import load_coeffs_npz
from mode_decomp_ml.viz import (
    coeff_channel_norms,
    coeff_energy_spectrum,
    coeff_value_magnitude,
    plot_channel_norm_scatter,
    plot_coeff_histogram,
    plot_coeff_spectrum,
)


_CONFIG_CANDIDATES = (
    Path("configuration/run.yaml"),
    Path("configuration/resolved.yaml"),
    Path("run.yaml"),
    Path(".hydra/config.yaml"),
    Path("hydra/config.yaml"),
)


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for preprocessing")
    return task_cfg


def _resolve_decomposition_dir(cfg: Mapping[str, Any]) -> str:
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))
    run_dir = cfg_get(task_cfg, "decomposition_run_dir", None)
    if run_dir is None or str(run_dir).strip() == "":
        return str(default_run_dir(cfg, "decomposition"))
    return str(resolve_path(run_dir))


def _load_decomposition_config(run_dir: str | Path) -> Mapping[str, Any]:
    root = Path(run_dir)
    for rel in _CONFIG_CANDIDATES:
        path = root / rel
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, Mapping) else {}
    return {}


def _resolve_decompose_name(run_dir: str | Path) -> str:
    cfg = _load_decomposition_config(run_dir)
    decompose_cfg = cfg_get(cfg, "decompose", {}) if isinstance(cfg, Mapping) else {}
    return str(cfg_get(decompose_cfg, "name", "")).strip().lower()


def _select_coeff_post(
    coeff_post_cfg: Mapping[str, Any],
    *,
    decompose_name: str,
) -> tuple[Mapping[str, Any], bool]:
    name = str(cfg_get(coeff_post_cfg, "name", "")).strip().lower()
    force = bool(cfg_get(coeff_post_cfg, "force", False))
    if force:
        return coeff_post_cfg, False
    if name != "pca":
        return coeff_post_cfg, False
    decompose_name = decompose_name.strip().lower()
    pca_blocklist = {
        "pod",
        "pod_svd",
        "pod_randomized",
        "gappy_pod",
        "dict_learning",
        "autoencoder",
    }
    if decompose_name in pca_blocklist:
        return {"name": "none"}, True
    return coeff_post_cfg, False


def _plot_coeff_scatter(path: Path, coeff: np.ndarray, *, title: str) -> None:
    coeff = np.asarray(coeff)
    if coeff.ndim != 2 or coeff.shape[1] < 2:
        return
    var = np.var(coeff, axis=0)
    idx = np.argsort(-var)[:2]
    x = coeff[:, idx[0]]
    y = coeff[:, idx[1]]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    ax.scatter(x, y, s=12, alpha=0.7)
    ax.set_xlabel(f"dim {int(idx[0])}")
    ax.set_ylabel(f"dim {int(idx[1])}")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("preprocessing requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["run_dir", "coeff_post"])
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

    decomposition_dir = _resolve_decomposition_dir(cfg)
    decompose_name = _resolve_decompose_name(decomposition_dir)
    coeff_post_cfg = cfg_get(cfg, "coeff_post", {}) or {}
    effective_coeff_post_cfg, skipped = _select_coeff_post(coeff_post_cfg, decompose_name=decompose_name)
    requested_coeff_post = str(cfg_get(coeff_post_cfg, "name", "")).strip() or "none"
    effective_coeff_post = str(cfg_get(effective_coeff_post_cfg, "name", "")).strip() or requested_coeff_post
    with steps.step(
        "load_coeffs",
        inputs=[artifact_ref(f"{decomposition_dir}/outputs/coeffs.npz", kind="coeffs")],
    ):
        coeff_payload = load_coeffs_npz(decomposition_dir)
        coeff_a = coeff_payload.get("coeff")
        conds = coeff_payload.get("cond")
        sample_ids = coeff_payload.get("sample_ids")
        if coeff_a is None or conds is None:
            raise ValueError("decomposition coeffs.npz missing coeff/cond arrays")
        coeff_a = np.asarray(coeff_a)
        conds = np.asarray(conds)
        if coeff_a.ndim != 2:
            raise ValueError("coeff array must be 2D")

    with steps.step(
        "fit_coeff_post",
        outputs=[artifact_ref("outputs/states/coeff_post/state.pkl", kind="state")],
    ):
        coeff_post = build_coeff_post(effective_coeff_post_cfg)
        coeff_post.fit(coeff_a, split="train")
        coeff_post.save_state(run_dir)
        coeff_z = coeff_post.transform(coeff_a)
        coeff_a_recon = coeff_post.inverse_transform(coeff_z)

    with steps.step(
        "write_coeffs",
        outputs=[artifact_ref("outputs/coeffs.npz", kind="coeffs")],
    ):
        writer.write_coeffs(
            {
                "coeff_a": coeff_a,
                "coeff_z": coeff_z,
                "coeff_a_recon": coeff_a_recon,
                "cond": conds,
                "sample_ids": sample_ids,
            }
        )

    coeff_rmse = float(np.sqrt(np.mean((coeff_a_recon - coeff_a) ** 2)))
    metrics = {
        "coeff_rmse": coeff_rmse,
        "latent_dim": int(coeff_z.shape[1]) if coeff_z.ndim == 2 else None,
        "method": coeff_post.name,
        "coeff_post_requested": requested_coeff_post,
        "coeff_post_effective": effective_coeff_post,
        "coeff_post_skipped": bool(skipped),
        "decompose_name": decompose_name,
        "decomposition_run_dir": decomposition_dir,
    }
    with steps.step(
        "write_metrics",
        outputs=[artifact_ref("outputs/metrics.json", kind="metrics")],
    ):
        writer.write_metrics(metrics)

    coeff_meta = None
    try:
        coeff_meta = load_coeff_meta(decomposition_dir)
    except Exception:
        coeff_meta = None

    with steps.step(
        "render_plots",
        outputs=[artifact_ref("plots", kind="plots_dir")],
    ):
        spectrum = coeff_energy_spectrum(coeff_a, coeff_meta)
        plot_coeff_spectrum(writer.plots_dir / "coeff_spectrum_raw.png", spectrum, scale="log")
        coeff_mag = coeff_value_magnitude(coeff_a, coeff_meta)
        plot_coeff_histogram(writer.plots_dir / "coeff_hist_raw.png", coeff_mag, bins=60, scale="log")
        coeff_mag_z = np.abs(coeff_z).reshape(-1)
        plot_coeff_histogram(writer.plots_dir / "coeff_hist_processed.png", coeff_mag_z, bins=60, scale="log")
        _plot_coeff_scatter(writer.plots_dir / "coeff_a_scatter.png", coeff_a, title="coeff_a scatter")
        _plot_coeff_scatter(writer.plots_dir / "coeff_z_scatter.png", coeff_z, title="coeff_z scatter")
        norms = coeff_channel_norms(coeff_a, coeff_meta)
        if norms is not None:
            plot_channel_norm_scatter(
                writer.plots_dir / "coeff_a_channel_norm_scatter.png",
                norms,
                title="coeff_a channel norms",
            )
        recon_norms = coeff_channel_norms(coeff_a_recon, coeff_meta)
        if recon_norms is not None:
            plot_channel_norm_scatter(
                writer.plots_dir / "coeff_a_recon_channel_norm_scatter.png",
                recon_norms,
                title="coeff_a_recon channel norms",
            )

    meta = build_meta(cfg)
    writer.write_manifest(meta=meta, steps=steps.to_list(), extra={"decomposition_run_dir": decomposition_dir})
    writer.write_steps(steps.to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
