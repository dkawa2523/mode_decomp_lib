#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mode_decomp_ml.data.manifest import load_manifest, manifest_domain_cfg
from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.evaluate import field_r2, field_rmse
from mode_decomp_ml.plugins.registry import build_decomposer
from mode_decomp_ml.viz.diagnostics import (
    masked_weighted_r2,
    per_pixel_r2_map,
    plot_scatter_true_pred,
    sample_scatter_points,
)


def _ensure_plugins_loaded() -> None:
    # Side-effect: registers decomposers.
    import mode_decomp_ml.plugins.decomposers  # noqa: F401


def _plot_r2_map(path: Path, r2_map: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    im = ax.imshow(r2_map, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.045)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_hist(path: Path, values: np.ndarray, title: str, bins: int = 60) -> None:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.hist(vals, bins=int(bins), color="#4C78A8", alpha=0.85)
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--obs-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--reg-lambda", type=float, default=1.0e-6)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_plots = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    field = np.load(root / "field.npy")
    cond = np.load(root / "cond.npy")
    if field.ndim != 4 or field.shape[-1] != 1:
        raise ValueError(f"expected scalar field.npy (N,H,W,1), got {field.shape}")
    n, h, w, _ = field.shape

    manifest = load_manifest(root, required=True)
    if manifest is None:
        raise ValueError("manifest.json is required")
    domain_cfg = manifest_domain_cfg(manifest, root)
    domain_spec = build_domain_spec(domain_cfg, field_shape=(h, w, 1))

    _ensure_plugins_loaded()
    decomp_cfg: dict[str, Any] = {
        "name": "gappy_pod",
        "mask_policy": "require",
        "reg_lambda": float(args.reg_lambda),
        "inner_product": "euclidean",
        "backend": "sklearn",
        "solver": "direct",
        "seed": int(args.seed),
        "options": {"mode_weight": {"enable": False}},
    }
    decomposer = build_decomposer(decomp_cfg)

    # Fit dataset: mask must be None for ALL samples.
    fit_ds = [
        SimpleNamespace(field=field[i, ..., 0], mask=None, cond=cond[i], meta={})
        for i in range(n)
    ]
    decomposer.fit(dataset=fit_ds, domain_spec=domain_spec)

    rng = np.random.default_rng(int(args.seed))
    obs_masks = np.zeros((n, h, w), dtype=bool)
    recon = np.zeros((n, h, w), dtype=np.float32)

    obs_frac = float(args.obs_frac)
    if not (0.0 < obs_frac <= 1.0):
        raise ValueError("--obs-frac must be in (0,1]")

    for i in range(n):
        m = rng.random((h, w)) < obs_frac
        obs_masks[i] = m
        c = decomposer.transform(field[i, ..., 0], mask=m, domain_spec=domain_spec)
        xhat = decomposer.inverse_transform(c, domain_spec=domain_spec)
        if xhat.ndim == 3:
            xhat = xhat[..., 0]
        recon[i] = np.asarray(xhat, dtype=np.float32)

    true = field[..., 0].astype(np.float32)

    # Metrics: full + observed-only
    full_rmse = float(field_rmse(true[..., None], recon[..., None], mask=None))
    full_r2 = float(field_r2(true[..., None], recon[..., None], mask=None))
    obs_rmse = float(field_rmse(true[..., None], recon[..., None], mask=obs_masks))
    obs_r2 = float(field_r2(true[..., None], recon[..., None], mask=obs_masks))

    metrics = {
        "n_samples": int(n),
        "grid": [int(h), int(w)],
        "obs_frac": float(obs_frac),
        "reg_lambda": float(args.reg_lambda),
        "field_rmse": full_rmse,
        "field_r2": full_r2,
        "field_rmse_obs": obs_rmse,
        "field_r2_obs": obs_r2,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plots
    frac = obs_masks.reshape(n, -1).mean(axis=1)
    _plot_hist(out_plots / "mask_fraction_hist.png", frac, "Observed fraction per sample")

    # Scatter (observed points)
    r2_scatter = masked_weighted_r2(true, recon, mask=obs_masks)
    xs, ys = sample_scatter_points(
        true, recon, mask=obs_masks, max_points=200_000, seed=int(args.seed)
    )
    if xs.size > 0:
        plot_scatter_true_pred(
            out_plots / "field_scatter_true_vs_recon_obs.png",
            xs,
            ys,
            title="gappy_pod true vs recon (observed points)",
            r2=r2_scatter,
        )

    # Per-pixel R2 across samples (full)
    r2map = per_pixel_r2_map(true, recon, mask=None, downsample=1)
    _plot_r2_map(out_plots / "per_pixel_r2_map.png", r2map, "Per-pixel R^2 (all samples)")
    _plot_hist(out_plots / "per_pixel_r2_hist.png", r2map, "Per-pixel R^2 histogram")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
