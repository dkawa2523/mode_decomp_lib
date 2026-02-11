from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.data.manifest import manifest_domain_cfg


def _cmap_with_bad(name: str, bad_color: str = "#dddddd") -> Any:
    cmap = plt.get_cmap(name)
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color=bad_color)
    return cmap


def _robust_limits(values: np.ndarray, *, q_lo: float = 1.0, q_hi: float = 99.0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, q_lo))
    hi = float(np.percentile(v, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(v))
        hi = float(np.max(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def ensure_case_problem_plots(
    *,
    dataset_root: Path,
    case: str,
    manifest: Mapping[str, Any] | None,
    out_root: Path,
) -> dict[str, Path]:
    """Generate case-level plots (domain + spatial stats + cond stats).

    This intentionally handles mesh domains separately (triangulation) to avoid unreadable Vx1 imshow plots.
    """
    case_root = dataset_root / case
    out_dir = out_root / "case_overview" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    field_path = case_root / "field.npy"
    cond_path = case_root / "cond.npy"
    if not field_path.exists() or not cond_path.exists():
        return {}

    field = np.load(field_path)
    cond = np.load(cond_path)
    if field.ndim != 4:
        return {}
    n, h, w, c = [int(x) for x in field.shape]
    field_kind = str((manifest or {}).get("field_kind", "")).strip().lower() or ("vector" if c == 2 else "scalar")

    # Build domain_spec for mask/weights plots.
    domain_cfg = None
    if manifest:
        try:
            domain_cfg = manifest_domain_cfg(manifest, case_root)
        except Exception:
            domain_cfg = None
    if not isinstance(domain_cfg, Mapping):
        domain_cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}

    domain_spec = None
    try:
        domain_spec = build_domain_spec(domain_cfg, (h, w, c))
    except Exception:
        domain_spec = None

    is_mesh = bool(getattr(domain_spec, "name", "").strip().lower() == "mesh")

    mask = None
    weights = None
    if domain_spec is not None:
        mask = getattr(domain_spec, "mask", None)
        weights = getattr(domain_spec, "integration_weights", lambda: None)()
    if mask is None:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)
        if mask.ndim == 1 and w == 1 and mask.shape[0] == h:
            mask = mask[:, None]
    if weights is None:
        weights = np.ones((h, w), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim == 1 and w == 1 and weights.shape[0] == h:
            weights = weights[:, None]

    domain_img = out_dir / "domain_overview.png"
    stats_img = out_dir / "field_stats.png"

    if is_mesh and domain_spec is not None:
        verts = np.asarray(domain_spec.coords.get("vertices"))
        faces = np.asarray(domain_spec.coords.get("faces"))
        if verts.ndim != 2 or verts.shape[1] < 2 or faces.ndim != 2 or faces.shape[1] != 3:
            is_mesh = False
        else:
            tri = mtri.Triangulation(verts[:, 0], verts[:, 1], triangles=faces.astype(int, copy=False))
            mask_v = np.asarray(mask).astype(bool).reshape(-1)
            weights_v = np.asarray(weights, dtype=float).reshape(-1)
            if mask_v.shape[0] != verts.shape[0]:
                mask_v = np.ones((verts.shape[0],), dtype=bool)
            if weights_v.shape[0] != verts.shape[0]:
                weights_v = np.ones((verts.shape[0],), dtype=float)

            tri_mask = ~np.all(mask_v[faces.astype(int, copy=False)], axis=1)
            tri_vis = mtri.Triangulation(tri.x, tri.y, triangles=tri.triangles)
            tri_vis.set_mask(tri_mask)

            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
            axes[0].triplot(tri, color="#333333", linewidth=0.25, alpha=0.35)
            m_sc = axes[0].scatter(
                verts[:, 0],
                verts[:, 1],
                c=mask_v.astype(float),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                s=10,
                linewidths=0.0,
            )
            axes[0].set_title("mesh mask (vertex)")
            axes[0].set_aspect("equal")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            fig.colorbar(m_sc, ax=axes[0], fraction=0.046, pad=0.04)

            w_show = np.where(mask_v, weights_v, np.nan)
            vmin, vmax = _robust_limits(w_show[np.isfinite(w_show)])
            w_img = axes[1].tripcolor(
                tri_vis,
                w_show,
                shading="gouraud",
                cmap=_cmap_with_bad("magma"),
                vmin=vmin,
                vmax=vmax,
            )
            axes[1].set_title("integration weights (vertex area)")
            axes[1].set_aspect("equal")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(w_img, ax=axes[1], fraction=0.046, pad=0.04)
            fig.savefig(domain_img, dpi=150)
            plt.close(fig)

            if field_kind == "vector" and c >= 2:
                values = np.linalg.norm(field[..., :2], axis=-1)
                label = "|v|"
            else:
                values = field[..., 0]
                label = "field"
            values = np.asarray(values).reshape((n, h, w))
            values = values[:, :, 0]
            mean_v = np.mean(values, axis=0)
            std_v = np.std(values, axis=0)
            mean_v = np.where(mask_v, mean_v, np.nan)
            std_v = np.where(mask_v, std_v, np.nan)
            vmin0, vmax0 = _robust_limits(mean_v[np.isfinite(mean_v)])
            vmin1, vmax1 = _robust_limits(std_v[np.isfinite(std_v)])

            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
            im0 = axes[0].tripcolor(
                tri_vis,
                mean_v,
                shading="gouraud",
                cmap=_cmap_with_bad("viridis"),
                vmin=vmin0,
                vmax=vmax0,
            )
            axes[0].set_title(f"mean({label}) over samples")
            axes[0].set_aspect("equal")
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            im1 = axes[1].tripcolor(
                tri_vis,
                std_v,
                shading="gouraud",
                cmap=_cmap_with_bad("magma"),
                vmin=vmin1,
                vmax=vmax1,
            )
            axes[1].set_title(f"std({label}) over samples")
            axes[1].set_aspect("equal")
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            fig.savefig(stats_img, dpi=150)
            plt.close(fig)

    if not is_mesh:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
        m_img = axes[0].imshow(mask.astype(float), origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"domain mask ({str(domain_cfg.get('name',''))})")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(m_img, ax=axes[0], fraction=0.046, pad=0.04)

        w_show = np.where(mask, weights, np.nan)
        vmin, vmax = _robust_limits(w_show[np.isfinite(w_show)])
        w_img = axes[1].imshow(w_show, origin="lower", cmap=_cmap_with_bad("magma"), vmin=vmin, vmax=vmax)
        axes[1].set_title("integration weights")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(w_img, ax=axes[1], fraction=0.046, pad=0.04)
        fig.savefig(domain_img, dpi=150)
        plt.close(fig)

        if field_kind == "vector" and c >= 2:
            values = np.linalg.norm(field[..., :2], axis=-1)
            label = "|v|"
        else:
            values = field[..., 0]
            label = "field"
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        mean_m = np.where(mask, mean, np.nan)
        std_m = np.where(mask, std, np.nan)
        vmin0, vmax0 = _robust_limits(mean_m[np.isfinite(mean_m)])
        vmin1, vmax1 = _robust_limits(std_m[np.isfinite(std_m)])
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), constrained_layout=True)
        im0 = axes[0].imshow(mean_m, origin="lower", cmap=_cmap_with_bad("viridis"), vmin=vmin0, vmax=vmax0)
        axes[0].set_title(f"mean({label}) over samples")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        im1 = axes[1].imshow(std_m, origin="lower", cmap=_cmap_with_bad("magma"), vmin=vmin1, vmax=vmax1)
        axes[1].set_title(f"std({label}) over samples")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.savefig(stats_img, dpi=150)
        plt.close(fig)

    # --- cond overview (offset + weight norm)
    cond_img = out_dir / "cond_overview.png"
    cond_arr = np.asarray(cond, dtype=float)
    offset = None
    wnorm = None
    if cond_arr.ndim == 2 and cond_arr.shape[0] == n and cond_arr.shape[1] >= 4:
        if field_kind == "vector" and cond_arr.shape[1] >= 8:
            offset = np.sqrt(cond_arr[:, 0] ** 2 + cond_arr[:, 1] ** 2)
            wnorm = np.linalg.norm(cond_arr[:, 2:], axis=1)
        else:
            offset = cond_arr[:, 0]
            wnorm = np.linalg.norm(cond_arr[:, 1:], axis=1)
    else:
        if cond_arr.ndim == 2 and cond_arr.shape[0] == n and cond_arr.shape[1] >= 1:
            offset = cond_arr[:, 0]
            wnorm = np.linalg.norm(cond_arr[:, 1:], axis=1) if cond_arr.shape[1] > 1 else np.zeros((n,), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    if offset is not None:
        axes[0].hist(offset[np.isfinite(offset)], bins=30, color="#4C78A8", alpha=0.85)
    axes[0].set_title("offset distribution")
    axes[0].set_xlabel("offset")
    axes[0].set_ylabel("count")
    if wnorm is not None:
        axes[1].hist(wnorm[np.isfinite(wnorm)], bins=30, color="#54A24B", alpha=0.85)
    axes[1].set_title("pattern weight-norm distribution")
    axes[1].set_xlabel("||w||")
    axes[1].set_ylabel("count")
    fig.savefig(cond_img, dpi=150)
    plt.close(fig)

    return {"domain": domain_img, "field_stats": stats_img, "cond": cond_img}

