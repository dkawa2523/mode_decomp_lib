from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DECOMP_R2_MIN = 0.95

# Allow importing `mode_decomp_ml` without installation (shim package lives at repo root).
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.bench._util import (  # noqa: E402
    fmt_compact as _fmt_compact,
    fmt_float as _fmt_float,
    fmt_sci as _fmt_sci,
    fmt_time_sec as _fmt_time_sec,
    load_json as _load_json,
    md_escape as _md_escape,
    relpath as _relpath,
    safe_slug as _safe_slug,
    to_float as _to_float,
    to_int as _to_int,
    read_csv as _read_csv,
)
from tools.bench.report.plots_case_overview import ensure_case_problem_plots as _ensure_case_problem_plots  # noqa: E402
from tools.bench.report.plots_coeff_dist import (  # noqa: E402
    ensure_coeff_mode_hist_plot as _ensure_coeff_mode_hist_plot,
    ensure_mode_energy_bar_plot as _ensure_mode_energy_bar_plot,
    ensure_mode_value_boxplot_plot as _ensure_mode_value_boxplot_plot,
)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def _md_bullets(items: list[str]) -> str:
    if not items:
        return ""
    return "\n".join([f"- {it}" for it in items]) + "\n"


def _fmt_cfg(cfg_name: Any) -> str:
    cfg_name = str(cfg_name or "").strip()
    return f"`{cfg_name}`" if cfg_name else ""


def _fmt_range(x: np.ndarray) -> str:
    a = np.asarray(x, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "(n/a)"
    return f"[{float(np.min(a)):.3g}, {float(np.max(a)):.3g}] (median {float(np.median(a)):.3g})"


# NOTE: plotting helpers live in tools/bench/report/* (imported above).

def _pick_best_ok_max(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    ok = [r for r in rows if str(r.get("status", "")) == "ok"]
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in ok:
        v = _to_float(r.get(key))
        if v is None:
            continue
        scored.append((v, r))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _median(values: list[float]) -> float | None:
    vals = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        vals.append(fv)
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def _how_to_read_lines() -> list[str]:
    lines: list[str] = []
    lines.append("## How to Read (Quickstart)")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "まず見る: `Global Best (per case)`（ケースごとの最良を俯瞰）。",
                "次に見る: 各ケースの `Highlights (auto)` と `key_decomp_dashboard.png`（処理の全体像を最短で把握）。",
                "分解: `field_rmse`, `field_r2` は **全係数での再構成**の良さ（可逆変換はほぼ1.0になりやすい）。",
                "分解: `k_req_r2_0.95`, `field_r2_topk_k64` は **圧縮としての良さ**（本資料の主指標）。",
                "分解: `field_r2_k*` / `mode_r2_vs_k.png` は **prefix-trunc（係数の並び順依存）**なので、wavelet/RBF等では解釈しづらい場合がある。",
                "学習: `val_rmse/val_r2` は **係数空間**（decomposer/codecでスケールが違うため手法間比較に注意）。",
                "学習: `val_field_rmse/val_field_r2` は **field空間**（比較しやすい、本資料の主指標）。",
                "異常検知: `status=failed` と `error`（CSV）を確認。図は `key_decomp_dashboard.png` が最短。",
            ]
        ).rstrip()
    )
    lines.append("")
    return lines


def _metrics_def_lines() -> list[str]:
    rows = [
        [
            "`field_rmse`",
            "decomposition",
            "eval mask内のRMSE（真値field vs 再構成field）。",
            "mask = domain mask ∩ dataset mask（存在する場合）、weightsはdomain weightsがあれば使用。",
            "小さいほど良い。",
        ],
        [
            "`field_r2`",
            "decomposition",
            "eval mask内のR^2（**全係数**で再構成）。",
            "同上",
            "可逆変換（DCT/FFT等）はほぼ1.0になりやすい。",
        ],
        [
            "`field_r2_k{1,4,16,64}`",
            "decomposition",
            "係数を **prefix-trunc**（先頭Kだけ残して残り0）して再構成したR^2。",
            "同上",
            "係数順に意味が薄い手法（wavelet/RBF/dict等）では解釈が難しい場合がある。未対応layoutは空欄。",
        ],
        [
            "`field_r2_topk_k{1,4,16,64}`",
            "decomposition",
            "係数エネルギー `mean(coeff^2)` の大きい順に **top-K** を残して再構成したR^2。",
            "同上",
            "順序依存が小さく、手法間比較の補助に有用。",
        ],
        [
            "`n_components_required` (`n_req`)",
            "decomposition",
            "`energy_cumsum>=0.9` に到達する最小K（係数エネルギーの累積）。",
            "coeffエネルギー（layout依存、channelsは合算）。",
            "offset優勢データでは小さく出やすい。Kの“必要数”の目安。`fft2` は周波数半径順で累積するため、負周波数が配列末尾にあっても過大評価されにくい。",
        ],
        [
            "`k_req_r2_0.95`",
            "decomposition",
            "`field_r2_topk` が 0.95 に到達する最小K（gridから求める）。",
            "同上",
            "圧縮としての主指標（小さいほど良い）。未計算の場合は空欄。",
        ],
        [
            "`val_rmse/val_r2`",
            "train",
            "cond→coeff（target_space）予測の指標（validation）。",
            "係数空間（codec/coeff_postに依存）。",
            "手法間比較は注意。係数のスケールが違うと同じ意味にならない。",
        ],
        [
            "`val_field_rmse/val_field_r2`",
            "train",
            "予測係数→decode→inverse_transformで復元したfieldの指標（validation）。",
            "mask = domain mask ∩ dataset mask（存在する場合）。",
            "手法間比較しやすい主指標。",
        ],
        [
            "`decomp_r2`",
            "train table",
            "そのdecompose(cfg)の `field_r2`（分解が十分再構成できているかの参照）。",
            "-",
            "train性能の比較前に、分解自体が破綻していないか確認する。",
        ],
    ]
    lines: list[str] = []
    lines.append("## Metrics (Definitions)")
    lines.append("")
    lines.append(
        _md_table(
            ["metric", "stage", "definition", "mask & weights", "notes"],
            [[_md_escape(c) for c in r] for r in rows],
        ).rstrip()
    )
    lines.append("")
    lines.append("_(空欄は「未計算（係数レイアウト非対応 / field_eval未対応 / 例外でスキップ）」を意味します。)_")
    lines.append("")
    return lines


def _plot_guide_lines() -> list[str]:
    rows = [
        [
            "`plots/key_decomp_dashboard.png`",
            "decomposition",
            "ダッシュボード（R^2 vs K / scatter / true / recon / abs error / per-pixel R^2）。",
            "まずこれを見る。",
        ],
        [
            "`plots/mode_r2_vs_k.png`",
            "decomposition",
            "R^2 vs K（`field_r2_k*`と同系、prefix-trunc）。",
            "Kでの劣化の仕方を見る（順序依存に注意）。",
        ],
        [
            "`plots/field_scatter_true_vs_recon_*.png`",
            "decomposition",
            "真値 vs 再構成の散布図（R^2付き）。",
            "バイアス（傾き/切片）や外れ値を確認。",
        ],
        [
            "`plots/per_pixel_r2_map_*.png` / `*_hist_*.png`",
            "decomposition",
            "位置ごとのR^2（サンプル方向の系列で算出）。",
            "境界/特定領域だけ弱い等の空間バイアスを検知。",
        ],
        [
            "`runs/benchmarks/v1/summary/mode_energy_bar/<case>/<decompose(cfg)>.png`",
            "report",
            "各手法の「モード番号→モード強度」を棒グラフ化（データセット全体、残差ベース）。",
            "どのモードにエネルギーが集中しているか（圧縮のしやすさ）を把握。",
        ],
        [
            "`runs/benchmarks/v1/summary/mode_value_boxplot/<case>/<decompose(cfg)>.png`",
            "report",
            "各手法のモード係数の分布をboxplotで可視化（データセット全体、上位モード）。",
            "符号（正負）・ばらつき・外れ値の影響（robust範囲）を確認。",
        ],
        [
            "`runs/benchmarks/v1/summary/mode_value_hist/<case>/<decompose(cfg)>.png`",
            "report",
            "各手法の上位モード係数のヒスト（small-multiples）。",
            "heavy-tail/バイアス（平均のズレ）/スケール差を確認。",
        ],
        [
            "`plots/coeff_spectrum.png`",
            "decomposition",
            "係数エネルギースペクトル（layoutに応じて index/degree/2D を表示）。",
            "どのモードが支配的か、top-Kの妥当性確認。",
        ],
        [
            "`train/plots/val_residual_hist.png`",
            "train",
            "係数予測残差の分布（val）。",
            "外れ値や系統誤差の有無。",
        ],
        [
            "`train/plots/field_eval/field_scatter_true_vs_pred_*.png`",
            "train",
            "field空間での真値 vs 予測散布図（val）。",
            "fieldとして妥当か（係数空間より解釈しやすい）。",
        ],
        [
            "`train/plots/field_eval/per_pixel_r2_map_*.png`",
            "train",
            "field空間での位置ごとのR^2（val）。",
            "予測が空間的にどこで崩れているか。",
        ],
    ]
    lines: list[str] = []
    lines.append("## Plot Guide (What each figure means)")
    lines.append("")
    lines.append(
        _md_table(
            ["file", "stage", "what it is", "what to look for"],
            [[_md_escape(c) for c in r] for r in rows],
        ).rstrip()
    )
    lines.append("")
    lines.append("_(図が存在しない場合は、(a)係数レイアウト非対応、(b)例外でスキップ、(c)そのrunで無効化、のいずれかです。)_")
    lines.append("")
    return lines


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _deep_update(dst: dict[str, Any], src: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            dst[key] = _deep_update(dict(dst[key]), value)
        else:
            dst[key] = value
    return dst


def _resolve_defaults_item(item: Any, *, config_dir: Path) -> Path | None:
    if isinstance(item, str):
        token = item.strip()
        if not token or token == "_self_":
            return None
        if token.startswith("/"):
            return config_dir / f"{token.lstrip('/')}.yaml"
        return config_dir / f"{token}.yaml"
    if isinstance(item, Mapping):
        group, name = next(iter(item.items()))
        return config_dir / str(group) / f"{str(name)}.yaml"
    return None


def _load_yaml_with_defaults(path: Path, *, config_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    cfg = _load_yaml(path) or {}
    if not isinstance(cfg, Mapping):
        return {}
    defaults = cfg.get("defaults", []) or []
    merged: dict[str, Any] = {}
    if isinstance(defaults, list):
        for item in defaults:
            sub_path = _resolve_defaults_item(item, config_dir=config_dir)
            if sub_path is None:
                continue
            merged = _deep_update(merged, _load_yaml_with_defaults(sub_path, config_dir=config_dir))
    # Merge the current file last.
    cfg_no_defaults = dict(cfg)
    cfg_no_defaults.pop("defaults", None)
    merged = _deep_update(merged, cfg_no_defaults)
    return merged


def _resolve_decomposer_from_cfg_yaml(cfg_name: str, *, cfg_root: Path, config_dir: Path) -> str | None:
    p = cfg_root / f"{cfg_name}.yaml"
    cfg = _load_yaml_with_defaults(p, config_dir=config_dir)
    if not cfg:
        return None
    name = cfg.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    decompose = cfg.get("decompose")
    if isinstance(decompose, Mapping):
        dname = decompose.get("name")
        if isinstance(dname, str) and dname.strip():
            return dname.strip()
    return None


METHOD_DESC: dict[str, str] = {
    "fft2": "2D FFT（周期境界）。高速だがマスクは0埋めが必要。",
    "dct2": "2D DCT（Dirichlet境界）。高速で安定。",
    "fft2_lowpass": "FFTの中心低周波ブロックのみ保持（周波数離散化で係数次元を削減）。",
    "wavelet2d": "2D Wavelet（多重解像度）。局所構造に強い。",
    "pswf2d_tensor": "PSWF（近似的な帯域制限基底、tensor版）。",
    "graph_fourier": "グラフラプラシアン固有基底。mask/不規則領域に適用可能（固定mask前提になりやすい）。",
    "gappy_graph_fourier": "固定基底 + 観測maskでridge最小二乗（可変mask向け）。",
    "pod": "POD（SVD/PCA）。データ駆動の低ランク分解。",
    "pod_svd": "POD（SVD実装）。",
    "pod_em": "欠損対応POD（EM/ALSで欠損を推定しつつ基底学習）。可変maskに強い。",
    "pod_joint": "ベクトル場を結合してPOD（u,v相関を活用）。",
    "pod_joint_em": "欠損対応のjoint POD（可変mask + u,v相関）。",
    "dict_learning": "Dictionary Learning（スパース符号化）。係数が疎になりやすい。",
    "autoencoder": "Autoencoder（非線形圧縮）。Torch依存（環境により無効）。",
    "rbf_expansion": "RBF基底 + ridge最小二乗。任意mask/可変maskにも適用しやすい。",
    "zernike": "Zernike（disk直交基底）。",
    "pseudo_zernike": "Pseudo-Zernike（Zernike一般化、disk）。",
    "fourier_bessel": "Fourier-Bessel（disk分離基底）。",
    "fourier_jacobi": "Fourier×Jacobi（disk分離基底、Zernike一般化）。",
    "polar_fft": "極座標リサンプル + FFT/DCT（近似、disk/annulus）。",
    "disk_slepian": "Disk Slepian（帯域制限 + 空間集中）。",
    "annular_zernike": "Annulus向けZernike系基底。",
    "helmholtz": "Helmholtz分解（周期境界、FFT）。ベクトル場のcurl-free/div-free分離。",
    "helmholtz_poisson": "Poissonソルバ系Helmholtz（periodic/dirichlet/neumann）。",
    "spherical_harmonics": "球面調和関数（sphere_grid）。",
    "spherical_slepian": "球面Slepian（sphere_grid）。",
    "laplace_beltrami": "Laplace-Beltrami固有基底（mesh）。",
    "gappy_pod": "Gappy POD（観測mask下で係数推定）。",
}


def _case_desc(manifest: Mapping[str, Any]) -> dict[str, str]:
    field_kind = str(manifest.get("field_kind", "unknown"))
    grid = manifest.get("grid", {}) if isinstance(manifest.get("grid"), Mapping) else {}
    domain = manifest.get("domain", {}) if isinstance(manifest.get("domain"), Mapping) else {}
    h = grid.get("H")
    w = grid.get("W")
    x_range = grid.get("x_range")
    y_range = grid.get("y_range")
    domain_type = str(domain.get("type", "unknown"))
    notes = ""
    if domain_type == "disk":
        notes = f"center={domain.get('center')}, radius={domain.get('radius')}"
    elif domain_type == "annulus":
        notes = f"center={domain.get('center')}, r_inner={domain.get('r_inner')}, r_outer={domain.get('r_outer')}"
    elif domain_type == "arbitrary_mask":
        notes = f"mask={domain.get('mask_path')}"
    elif domain_type == "sphere_grid":
        notes = f"n_lat={domain.get('n_lat')}, n_lon={domain.get('n_lon')}, lon_range={domain.get('lon_range')}"
    elif domain_type == "mesh":
        notes = "planar triangulated grid mesh (289 verts)"
    return {
        "field_kind": field_kind,
        "grid": f"{h}x{w}",
        "range": f"x={x_range}, y={y_range}",
        "domain": domain_type,
        "notes": notes,
    }


def _pick_best_ok(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any] | None:
    ok = [r for r in rows if str(r.get("status", "")) == "ok"]
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in ok:
        v = _to_float(r.get(key))
        if v is None:
            continue
        scored.append((v, r))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def _embed_images(lines: list[str], *, title: str, paths: list[Path], base_dir: Path, max_images: int = 4) -> None:
    existing = [p for p in paths if p.exists()][:max_images]
    if not existing:
        return
    items = [(p.name, p) for p in existing]
    _embed_image_grid(lines, title=title, items=items, base_dir=base_dir, n_cols=2, img_width=360)


def _html_escape(text: str) -> str:
    s = str(text or "")
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def _embed_image_grid(
    lines: list[str],
    *,
    title: str,
    items: list[tuple[str, Path]],
    base_dir: Path,
    n_cols: int = 2,
    img_width: int = 360,
) -> None:
    existing: list[tuple[str, Path]] = []
    for caption, p in items:
        if p is None:
            continue
        if not p.exists():
            continue
        existing.append((str(caption or ""), p))
    if not existing:
        return
    n_cols = max(1, int(n_cols))
    img_width = max(80, int(img_width))

    lines.append(f"**{title}**")
    lines.append("")
    lines.append("<table>")
    for i, (caption, p) in enumerate(existing):
        if i % n_cols == 0:
            lines.append("  <tr>")
        rel = _relpath(p, base_dir=base_dir)
        cap = _html_escape(caption)
        lines.append(
            "    <td style=\"text-align:center; vertical-align:top; padding:6px;\">"
            f"<div style=\"font-size:12px; margin-bottom:4px;\">{cap}</div>"
            f"<img src=\"{rel}\" width=\"{img_width}\" />"
            "</td>"
        )
        if i % n_cols == n_cols - 1:
            lines.append("  </tr>")
    rem = len(existing) % n_cols
    if rem:
        for _ in range(n_cols - rem):
            lines.append("    <td></td>")
        lines.append("  </tr>")
    lines.append("</table>")
    lines.append("")


def _embed_image_bundle_grid(
    lines: list[str],
    *,
    title: str,
    bundles: list[tuple[str, list[tuple[str, Path]]]],
    base_dir: Path,
    n_cols: int = 2,
    img_width: int = 320,
) -> None:
    """Embed a grid where each cell can contain multiple images (stacked)."""
    existing: list[tuple[str, list[tuple[str, Path]]]] = []
    for caption, paths in bundles:
        keep: list[tuple[str, Path]] = []
        for subcap, p in paths:
            if p is None or not p.exists():
                continue
            keep.append((str(subcap or ""), p))
        if keep:
            existing.append((str(caption or ""), keep))
    if not existing:
        return
    n_cols = max(1, int(n_cols))
    img_width = max(80, int(img_width))

    lines.append(f"**{title}**")
    lines.append("")
    lines.append("<table>")
    for i, (caption, plots) in enumerate(existing):
        if i % n_cols == 0:
            lines.append("  <tr>")
        cap = _html_escape(caption)
        lines.append("    <td style=\"text-align:center; vertical-align:top; padding:6px;\">")
        lines.append(f"      <div style=\"font-size:12px; margin-bottom:4px;\">{cap}</div>")
        for subcap, p in plots:
            rel = _relpath(p, base_dir=base_dir)
            if subcap:
                lines.append(f"      <div style=\"font-size:11px; color:#444; margin:2px 0;\">{_html_escape(subcap)}</div>")
            lines.append(f"      <img src=\"{rel}\" width=\"{img_width}\" />")
        lines.append("    </td>")
        if i % n_cols == n_cols - 1:
            lines.append("  </tr>")
    rem = len(existing) % n_cols
    if rem:
        for _ in range(n_cols - rem):
            lines.append("    <td></td>")
        lines.append("  </tr>")
    lines.append("</table>")
    lines.append("")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default="data/benchmarks/v1/offset_noise_36")
    ap.add_argument("--v1-summary-dir", type=str, default="runs/benchmarks/v1/summary")
    ap.add_argument("--missing-summary-dir", type=str, default="runs/benchmarks/v1_missing_methods/summary")
    ap.add_argument(
        "--gappy-metrics",
        type=str,
        default="runs/benchmarks/v1_missing_methods/gappy_pod_rectangle_scalar/metrics.json",
    )
    ap.add_argument("--out", type=str, default="summary_benchmark.md")
    args = ap.parse_args()

    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    v1_summary_dir = (PROJECT_ROOT / args.v1_summary_dir).resolve()
    missing_summary_dir = (PROJECT_ROOT / args.missing_summary_dir).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    cfg_root = PROJECT_ROOT / "configs" / "decompose"
    config_dir = PROJECT_ROOT / "configs"
    train_basic_cfg = yaml.safe_load((config_dir / "train" / "basic.yaml").read_text(encoding="utf-8")) or {}
    train_eval_cfg = train_basic_cfg.get("eval") if isinstance(train_basic_cfg, Mapping) else {}
    val_ratio = float(train_eval_cfg.get("val_ratio", 0.2)) if isinstance(train_eval_cfg, Mapping) else 0.2
    val_shuffle = bool(train_eval_cfg.get("shuffle", True)) if isinstance(train_eval_cfg, Mapping) else True

    method_cache: dict[str, str] = {}

    def _method_for_cfg(cfg_name: str) -> str:
        key = str(cfg_name or "").strip()
        if not key:
            return ""
        cached = method_cache.get(key)
        if cached is not None:
            return cached
        dec = _resolve_decomposer_from_cfg_yaml(key, cfg_root=cfg_root, config_dir=config_dir) or ""
        method_cache[key] = dec
        return dec

    dataset_meta_path = dataset_root / "dataset_meta.json"
    dataset_meta = _load_json(dataset_meta_path) if dataset_meta_path.exists() else {}
    cases = list(dataset_meta.get("cases", [])) if isinstance(dataset_meta.get("cases"), list) else []
    # Add mesh_scalar if present (generated for missing methods coverage).
    if (dataset_root / "mesh_scalar" / "manifest.json").exists() and "mesh_scalar" not in cases:
        cases.append("mesh_scalar")

    # Load manifests for case descriptions.
    case_manifests: dict[str, dict[str, Any]] = {}
    for case in cases:
        mp = dataset_root / case / "manifest.json"
        if mp.exists():
            case_manifests[case] = _load_json(mp)

    # Load benchmark summaries (v1 + missing).
    decomp_rows = _read_csv(v1_summary_dir / "benchmark_summary_decomposition.csv") + _read_csv(
        missing_summary_dir / "benchmark_summary_decomposition.csv"
    )
    train_rows = _read_csv(v1_summary_dir / "benchmark_summary_train.csv") + _read_csv(
        missing_summary_dir / "benchmark_summary_train.csv"
    )

    # Index rows by case.
    decomp_by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in decomp_rows:
        c = str(r.get("case", "")).strip()
        if c in decomp_by_case:
            decomp_by_case[c].append(r)
    decomp_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for c, rows in decomp_by_case.items():
        for r in rows:
            k = (c, str(r.get("decompose", "")).strip())
            if k[1]:
                # Prefer the first occurrence (v1 rows appear before missing rows).
                decomp_lookup.setdefault(k, r)
    train_by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in train_rows:
        c = str(r.get("case", "")).strip()
        if c in train_by_case:
            train_by_case[c].append(r)

    # Build the set of executed decomposers (resolved from configs).
    executed_decomposers: set[str] = set()
    for r in decomp_rows:
        cfg_name = str(r.get("decompose", "")).strip()
        if not cfg_name:
            continue
        dec = _method_for_cfg(cfg_name)
        if dec:
            executed_decomposers.add(dec)
    gappy_metrics_path = (PROJECT_ROOT / args.gappy_metrics).resolve()
    if gappy_metrics_path.exists():
        executed_decomposers.add("gappy_pod")

    # Start document.
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Benchmark Summary (v1)")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Dataset root: `{_relpath(dataset_root, base_dir=PROJECT_ROOT)}`")
    if dataset_meta:
        lines.append(f"- n_samples: {dataset_meta.get('n_samples')}")
        lines.append(f"- fluct_ratio: {dataset_meta.get('fluct_ratio')}")
        lines.append(f"- noise_ratio: {dataset_meta.get('noise_ratio')}")
    lines.append(f"- seed: {dataset_meta.get('seed')}")
    lines.append("")

    lines.append("## Background / Goals")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "目的: 各ドメイン上の場（scalar/vector）を **モード分解→係数化**し、条件 `cond` から係数（ひいてはfield）を予測するパイプラインを手法横断で比較する。",
                "本資料は (1) **分解の妥当性**（再構成と圧縮）と、(2) **学習の妥当性**（cond→coeff→field）を同じケースで見比べられるように整理する。",
                "重要: `field_r2` は **全係数での再構成**（可逆変換はほぼ1.0になりやすい）。圧縮としては `k_req_r2_0.95` / `field_r2_topk_k*` を主に見る。",
                "異常系: `status=failed` / `error` は手法の前提不一致や数値不安定の可能性が高いので、ケース別に原因を切り分ける。",
            ]
        ).rstrip()
    )
    lines.append("")

    # --- reading guide (for third-party readers)
    lines.extend(_how_to_read_lines())
    lines.extend(_metrics_def_lines())

    lines.append("## Data Generation / Problem Setting (共通)")
    lines.append("")
    lines.append("- 各サンプルは `field = offset + fluct + noise` を満たす（offset優勢）。")
    lines.append("- `fluct` は offset の **5-10% 程度**（v1は `fluct_ratio=0.07`）。")
    lines.append("- `noise` は offset の **約1%**（v1は `noise_ratio=0.01`）。")
    lines.append("")

    lines.append("**cond の定義**")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "scalar: `cond.shape=(N,4)`、`cond[:,0]=offset`、`cond[:,1:4]=pattern weights`",
                "vector: `cond.shape=(N,8)`、`cond[:,0]=offset_u`、`cond[:,1]=offset_v`、`cond[:,2:5]=weights_u`、`cond[:,5:8]=weights_v`",
            ]
        ).rstrip()
    )
    lines.append("")

    lines.append("**fluct/noise のスケーリング（概念式）**")
    lines.append("")
    lines.append("```text")
    lines.append("base = sum_j w_j * pattern_j            # patternはmask内でmean=0/std=1に正規化")
    lines.append("base *= (fluct_ratio * offset) / RMS(base, mask)")
    lines.append("noise ~ N(0,1)")
    lines.append("noise *= (noise_ratio * offset) / RMS(noise, mask)")
    lines.append("field = offset + base + noise")
    lines.append("```")
    lines.append("")

    lines.append("**patterns（先頭3つのみ使用）**")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "rectangle/arbitrary_mask: `sin(2πx)`, `sin(2πy)`, `cos(2π(x+y))`",
                "disk/annulus: `r`, `r^2`, `cos(theta)`",
                "sphere_grid: `sin(lat)`, `cos(lat)`, `sin(lon)`",
            ]
        ).rstrip()
    )
    lines.append("")

    lines.append("**この問題で期待される挙動（判断のコツ）**")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "offset優勢のため、DC（定数）を1モードで持つ手法は `K=1` から `R^2` が上がりやすい。",
                "fluctは低次パターン中心のため、低次数基底（DCT/低次Zernike/低次Graph Fourier等）が有利になりやすい。",
                "mask境界付近だけ誤差が大きい場合は、境界条件・補間・maskの扱いの不整合が疑わしい（`per_pixel_r2_map` で確認）。",
            ]
        ).rstrip()
    )
    lines.append("")

    lines.append("**mask の扱い（ドメイン別）**")
    lines.append("")
    lines.append(
        _md_bullets(
            [
                "rectangle/sphere_grid: 全点有効（maskなし）。",
                "disk/annulus: 幾何マスク（領域外は0埋め、評価は領域内のみ）。",
                "arbitrary_mask: 固定の不規則マスク（`domain_mask.npy`）。領域外は0埋め、評価はmask内のみ。",
            ]
        ).rstrip()
    )
    lines.append("")
    lines.append("## Cases (ドメイン別テストケース)")
    lines.append("")
    case_rows: list[list[str]] = []
    for case in cases:
        manifest = case_manifests.get(case, {})
        desc = _case_desc(manifest) if manifest else {"field_kind": "", "grid": "", "range": "", "domain": "", "notes": ""}
        case_rows.append(
            [
                f"`{case}`",
                _md_escape(desc["domain"]),
                _md_escape(desc["field_kind"]),
                _md_escape(desc["grid"]),
                _md_escape(desc["range"]),
                _md_escape(desc["notes"]),
            ]
        )
    lines.append(_md_table(["case", "domain", "field", "grid", "range", "notes"], case_rows))

    lines.append("## Methods (実行した分解手法の特徴)")
    lines.append("")
    # Render a method table in a stable order.
    method_list = sorted(executed_decomposers) if executed_decomposers else sorted(METHOD_DESC.keys())
    method_rows: list[list[str]] = []
    for m in method_list:
        method_rows.append([f"`{m}`", _md_escape(METHOD_DESC.get(m, ""))])
    lines.append(_md_table(["method", "description"], method_rows))

    lines.extend(_plot_guide_lines())

    lines.append("## Results")
    lines.append("")
    lines.append("### Global Best (per case)")
    lines.append("")
    best_rows: list[list[str]] = []
    for case in cases:
        drows = decomp_by_case.get(case, [])
        trows = train_by_case.get(case, [])
        best_d = _pick_best_ok(drows, key="field_rmse")
        best_t = _pick_best_ok(trows, key="val_rmse")
        # Safer pick: only consider train runs whose decomposer reconstructs well in this benchmark.
        good_t = []
        for r in trows:
            if str(r.get("status", "")) != "ok":
                continue
            cfg_name = str(r.get("decompose", "")).strip()
            drow = decomp_lookup.get((case, cfg_name))
            if drow is None or str(drow.get("status", "")) != "ok":
                continue
            r2 = _to_float(drow.get("field_r2"))
            if r2 is None or r2 < TRAIN_DECOMP_R2_MIN:
                continue
            good_t.append(r)
        best_t_good = _pick_best_ok(good_t, key="val_rmse")
        best_rows.append(
            [
                f"`{case}`",
                _md_escape(best_d.get("decompose") if best_d else ""),
                _fmt_sci(best_d.get("field_rmse") if best_d else ""),
                _fmt_float(best_d.get("field_r2") if best_d else "", digits=6),
                _md_escape(best_t.get("decompose") if best_t else ""),
                _fmt_sci(best_t.get("val_rmse") if best_t else ""),
                _fmt_float(best_t.get("val_r2") if best_t else "", digits=6),
                _fmt_sci(best_t.get("val_field_rmse") if best_t else ""),
                _fmt_float(best_t.get("val_field_r2") if best_t else "", digits=6),
                _md_escape(best_t_good.get("decompose") if best_t_good else ""),
                _fmt_sci(best_t_good.get("val_rmse") if best_t_good else ""),
            ]
        )
    lines.append(
        _md_table(
            [
                "case",
                "best_decomp(cfg)",
                "rmse",
                "r2",
                "best_train(cfg)",
                "val_rmse",
                "val_r2",
                "val_field_rmse",
                "val_field_r2",
                f"best_train(cfg) (decomp_r2>={TRAIN_DECOMP_R2_MIN})",
                "val_rmse",
            ],
            best_rows,
        )
    )

    # --- cross-case summary (method-centric)
    lines.append("### Method Summary (Across cases)")
    lines.append("")
    lines.append(
        "_各methodについて、各case内でそのmethodの成功runが複数ある場合は「代表として最良（主に r2_topk_k64 最大）」を1つ選び、"
        "その代表の指標の中央値を示します（空欄はその指標が未計算/未実行）。_"
    )
    lines.append("")

    cases_scalar = [c for c in cases if str(c).endswith("_scalar")]
    cases_vector = [c for c in cases if str(c).endswith("_vector")]

    def _method_summary_rows(case_list: list[str]) -> list[list[str]]:
        summary: list[dict[str, Any]] = []
        for method in method_list:
            n_cases = 0
            decomp_r2_topk64_vals: list[float] = []
            decomp_kreq_vals: list[float] = []
            train_val_field_r2_vals: list[float] = []

            for case in case_list:
                cand = []
                for r in decomp_by_case.get(case, []):
                    if str(r.get("status", "")) != "ok":
                        continue
                    cfg_name = str(r.get("decompose", "")).strip()
                    if not cfg_name:
                        continue
                    if _method_for_cfg(cfg_name) != method:
                        continue
                    cand.append(r)
                if not cand:
                    continue
                n_cases += 1
                best_by_topk = _pick_best_ok_max(cand, key="field_r2_topk_k64")
                best_by_r2 = _pick_best_ok_max(cand, key="field_r2")
                best_row = best_by_topk or best_by_r2 or cand[0]
                v = _to_float(best_row.get("field_r2_topk_k64"))
                if v is not None:
                    decomp_r2_topk64_vals.append(v)
                kreq = _to_float(best_row.get("k_req_r2_0p95"))
                if kreq is not None:
                    decomp_kreq_vals.append(kreq)

            for case in case_list:
                scored = []
                for r in train_by_case.get(case, []):
                    if str(r.get("status", "")) != "ok":
                        continue
                    cfg_name = str(r.get("decompose", "")).strip()
                    if not cfg_name:
                        continue
                    if _method_for_cfg(cfg_name) != method:
                        continue
                    v_rmse = _to_float(r.get("val_field_rmse"))
                    v_r2 = _to_float(r.get("val_field_r2"))
                    if v_rmse is None or v_r2 is None:
                        continue
                    scored.append((v_rmse, v_r2))
                if not scored:
                    continue
                scored.sort(key=lambda t: t[0])  # best rmse
                train_val_field_r2_vals.append(float(scored[0][1]))

            med_topk64 = _median(decomp_r2_topk64_vals)
            med_kreq = _median(decomp_kreq_vals)
            med_val_field_r2 = _median(train_val_field_r2_vals)
            if n_cases == 0 and med_val_field_r2 is None and med_topk64 is None:
                continue
            summary.append(
                {
                    "method": method,
                    "n_cases": n_cases,
                    "med_topk64": med_topk64,
                    "med_kreq": med_kreq,
                    "med_val_field_r2": med_val_field_r2,
                }
            )

        def _sort_key(item: dict[str, Any]) -> tuple[float, float, float, str]:
            vf = item.get("med_val_field_r2")
            d64 = item.get("med_topk64")
            kq = item.get("med_kreq")
            # sort: higher val_field_r2, higher topk64, lower k_req
            vf_key = float(vf) if vf is not None else -1.0
            d64_key = float(d64) if d64 is not None else -1.0
            kq_key = float(kq) if kq is not None else 1e9
            return (-vf_key, -d64_key, kq_key, str(item.get("method", "")))

        summary.sort(key=_sort_key)
        rows_out: list[list[str]] = []
        for item in summary:
            rows_out.append(
                [
                    f"`{item['method']}`",
                    str(int(item["n_cases"])),
                    _fmt_float(item.get("med_topk64"), digits=6),
                    _fmt_compact(item.get("med_kreq"), digits=2),
                    _fmt_float(item.get("med_val_field_r2"), digits=6),
                ]
            )
        return rows_out

    lines.append("#### Scalar cases")
    lines.append("")
    lines.append(
        _md_table(
            ["method", "n_cases", "median_r2_topk_k64", "median_k_req_0.95", "median_val_field_r2"],
            _method_summary_rows(cases_scalar),
        )
    )
    lines.append("#### Vector cases")
    lines.append("")
    lines.append(
        _md_table(
            ["method", "n_cases", "median_r2_topk_k64", "median_k_req_0.95", "median_val_field_r2"],
            _method_summary_rows(cases_vector),
        )
    )

    # Per-case sections.
    for case in cases:
        lines.append(f"### {case}")
        lines.append("")
        manifest = case_manifests.get(case, {})
        if manifest:
            desc = _case_desc(manifest)
            lines.append("**Problem setting**")
            lines.append("")
            lines.append(f"- domain: `{desc['domain']}` ({desc['notes']})")
            lines.append(f"- field: `{desc['field_kind']}`")
            lines.append(f"- grid: `{desc['grid']}`, {desc['range']}")
            if desc["field_kind"] == "scalar":
                lines.append("- cond: `(N,4)`")
            elif desc["field_kind"] == "vector":
                lines.append("- cond: `(N,8)`")
            if desc["domain"] in {"rectangle", "sphere_grid"}:
                lines.append("- mask: all-valid (no mask)")
            elif desc["domain"] in {"disk", "annulus"}:
                lines.append("- mask: geometric domain mask (outside is 0-filled; evaluation uses inside only)")
            elif desc["domain"] == "arbitrary_mask":
                lines.append("- mask: fixed irregular mask (`domain_mask.npy`; evaluation uses mask==true only)")
            elif desc["domain"] == "mesh":
                lines.append("- mask: n/a (values are on vertices)")

            # Dataset-level condition ranges (helps third-party readers understand the problem).
            case_root = dataset_root / case
            cond_path = case_root / "cond.npy"
            if cond_path.exists():
                try:
                    cond_arr = np.load(cond_path)
                    if cond_arr.ndim == 2:
                        lines.append(f"- n_samples: {int(cond_arr.shape[0])}")
                        lines.append(f"- cond_dim: {int(cond_arr.shape[1])}")
                        if desc["field_kind"] == "scalar" and cond_arr.shape[1] >= 4:
                            offset = cond_arr[:, 0]
                            w = cond_arr[:, 1:4]
                            lines.append(f"- offset_range: {_fmt_range(offset)}")
                            lines.append(f"- weight_norm_range: {_fmt_range(np.linalg.norm(w, axis=1))}")
                            lines.append(f"- weight_component_range: {_fmt_range(w)}")
                        elif desc["field_kind"] == "vector" and cond_arr.shape[1] >= 8:
                            ou = cond_arr[:, 0]
                            ov = cond_arr[:, 1]
                            w = cond_arr[:, 2:]
                            lines.append(f"- offset_u_range: {_fmt_range(ou)}")
                            lines.append(f"- offset_v_range: {_fmt_range(ov)}")
                            lines.append(f"- offset_mag_range: {_fmt_range(np.sqrt(ou**2 + ov**2))}")
                            lines.append(f"- weight_norm_range: {_fmt_range(np.linalg.norm(w, axis=1))}")
                except Exception:
                    pass

            # Train/val/test split notes for reproducibility.
            # Decomposition: split=all (uses all samples). Train: internal random hold-out on coeffs.
            try:
                n_samples = int(dataset_meta.get("n_samples", 0) or 0)
                if n_samples <= 0 and cond_path.exists():
                    n_samples = int(np.load(cond_path).shape[0])
            except Exception:
                n_samples = 0
            if n_samples > 0:
                n_train = n_samples
                n_val = 0
                if val_ratio > 0.0 and n_samples > 1:
                    n_train = max(1, int(n_samples * (1.0 - val_ratio)))
                    n_train = min(n_train, n_samples - 1)
                    n_val = n_samples - n_train
                seed = dataset_meta.get("seed", None)
                lines.append(f"- decomposition split: all ({n_samples} samples)")
                lines.append(
                    f"- train/val split (train.basic): val_ratio={val_ratio:g}, shuffle={val_shuffle}, seed={seed} -> train={n_train}, val={n_val}"
                )
                lines.append("- test split: none (v1 benchmark has no dedicated test set)")
            lines.append("")

            # Case overview plots (domain mask/weights, field mean/std, cond hist).
            plots = _ensure_case_problem_plots(
                dataset_root=dataset_root,
                case=case,
                manifest=manifest,
                out_root=v1_summary_dir,
            )
            if plots:
                items: list[tuple[str, Path]] = []
                if plots.get("domain") is not None:
                    items.append(("domain / mask / weights", Path(str(plots["domain"]))))
                if plots.get("field_stats") is not None:
                    items.append(("field stats (mean/std)", Path(str(plots["field_stats"]))))
                if plots.get("cond") is not None:
                    items.append(("cond stats (offset / ||w||)", Path(str(plots["cond"]))))
                _embed_image_grid(
                    lines,
                    title="Case overview plots",
                    items=items,
                    base_dir=PROJECT_ROOT,
                    n_cols=3,
                    img_width=260,
                )

        # --- decomposition table
        rows = list(decomp_by_case.get(case, []))
        # Sort ok by rmse, failed last.
        def _sort_key(r: dict[str, Any]) -> tuple[int, float]:
            status = str(r.get("status", ""))
            ok_rank = 0 if status == "ok" else 1
            rmse = _to_float(r.get("field_rmse"))
            return ok_rank, rmse if rmse is not None else float("inf")

        rows.sort(key=_sort_key)
        best = _pick_best_ok(rows, key="field_rmse")

        # --- auto highlights (decomposition + train)
        best_comp = _pick_best_ok(rows, key="k_req_r2_0p95")
        best_topk = _pick_best_ok_max(rows, key="field_r2_topk_k64")
        rows_t_all = list(train_by_case.get(case, []))
        best_t_coeff = _pick_best_ok(rows_t_all, key="val_rmse")
        best_t_field = _pick_best_ok(rows_t_all, key="val_field_rmse")
        if best_t_field is None:
            best_t_field = _pick_best_ok_max(rows_t_all, key="val_field_r2")
        mismatch = False
        if best_t_coeff is not None and best_t_field is not None:
            mismatch = str(best_t_coeff.get("decompose", "")).strip() != str(best_t_field.get("decompose", "")).strip()

        bullets: list[str] = []
        if best is not None:
            cfg_name = str(best.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            bullets.append(
                "decomposition: best full recon = "
                + f"{_fmt_cfg(cfg_name)}"
                + (f" (`{m}`)" if m else "")
                + f" (field_rmse={_fmt_sci(best.get('field_rmse'))}, field_r2={_fmt_float(best.get('field_r2'), digits=6)})"
            )
        if best_comp is not None:
            cfg_name = str(best_comp.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            bullets.append(
                "decomposition: best compression proxy = "
                + f"{_fmt_cfg(cfg_name)}"
                + (f" (`{m}`)" if m else "")
                + f" (k_req_r2_0.95={_fmt_compact(_to_float(best_comp.get('k_req_r2_0p95')))}, "
                + f"r2_topk_k64={_fmt_float(best_comp.get('field_r2_topk_k64'), digits=6)})"
            )
        if best_topk is not None:
            cfg_name = str(best_topk.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            bullets.append(
                "decomposition: best top-energy@64 = "
                + f"{_fmt_cfg(cfg_name)}"
                + (f" (`{m}`)" if m else "")
                + f" (r2_topk_k64={_fmt_float(best_topk.get('field_r2_topk_k64'), digits=6)}, "
                + f"k_req_r2_0.95={_fmt_compact(_to_float(best_topk.get('k_req_r2_0p95')))} )"
            )
        if best_t_coeff is not None:
            cfg_name = str(best_t_coeff.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            model = str(best_t_coeff.get("model", "")).strip()
            bullets.append(
                "train: best coeff-space = "
                + f"{_fmt_cfg(cfg_name)}"
                + (f" (`{m}`)" if m else "")
                + (f" (`{model}`)" if model else "")
                + f" (val_rmse={_fmt_sci(best_t_coeff.get('val_rmse'))}, val_r2={_fmt_float(best_t_coeff.get('val_r2'), digits=6)})"
            )
        if best_t_field is not None:
            cfg_name = str(best_t_field.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            model = str(best_t_field.get("model", "")).strip()
            bullets.append(
                "train: best field-space = "
                + f"{_fmt_cfg(cfg_name)}"
                + (f" (`{m}`)" if m else "")
                + (f" (`{model}`)" if model else "")
                + f" (val_field_rmse={_fmt_sci(best_t_field.get('val_field_rmse'))}, "
                + f"val_field_r2={_fmt_float(best_t_field.get('val_field_r2'), digits=6)})"
            )
        if mismatch:
            bullets.append("train: mismatch detected (best coeff-space != best field-space)")

        if bullets:
            lines.append("**Highlights (auto)**")
            lines.append("")
            lines.append(_md_bullets(bullets).rstrip())
            lines.append("")

        headers = [
            "decompose(cfg)",
            "method",
            "status",
            "rmse",
            "r2",
            "r2_k1",
            "r2_k64",
            "fit",
            "n_req",
            "k_req_r2_0.95",
            "run",
        ]
        table_rows: list[list[str]] = []
        for r in rows:
            cfg_name = str(r.get("decompose", "")).strip()
            method = _method_for_cfg(cfg_name)
            status = str(r.get("status", "")).strip()
            rmse = _fmt_sci(r.get("field_rmse"))
            r2 = _fmt_float(r.get("field_r2"), digits=6)
            r2_k1 = _fmt_float(r.get("field_r2_k1"), digits=6)
            r2_k64 = _fmt_float(r.get("field_r2_k64"), digits=6)
            fit = _fmt_time_sec(r.get("fit_time_sec"))
            nreq = _to_int(r.get("n_components_required"))
            nreq_s = "" if nreq is None else str(nreq)
            kreq = _to_int(r.get("k_req_r2_0p95"))
            kreq_s = "" if kreq is None else str(kreq)
            run_dir = r.get("decomposition_run_dir") or ""
            run_link = ""
            if run_dir:
                run_rel = _relpath(run_dir, base_dir=PROJECT_ROOT)
                run_link = f"[run]({run_rel})"
            label = f"`{cfg_name}`"
            if best is not None and cfg_name == str(best.get("decompose", "")).strip():
                label = f"**{label}**"
            table_rows.append(
                [label, f"`{method}`" if method else "", status, rmse, r2, r2_k1, r2_k64, fit, nreq_s, kreq_s, run_link]
            )
        lines.append("**Decomposition (field reconstruction)**")
        lines.append("")
        lines.append(_md_table(headers, table_rows))

        # Per-method mode energy (dataset-level) bar plots.
        # x = mode index (or freq-radius bins for CHW-like layouts), y = energy fraction.
        items: list[tuple[str, Path]] = []
        out_dir = v1_summary_dir / "mode_energy_bar" / case
        for r in rows:
            if str(r.get("status", "")) != "ok":
                continue
            cfg_name = str(r.get("decompose", "")).strip()
            if not cfg_name:
                continue
            run_dir_str = str(r.get("decomposition_run_dir") or "").strip()
            if not run_dir_str:
                continue
            run_dir = Path(run_dir_str)
            method = _method_for_cfg(cfg_name)
            plot_path = out_dir / f"{_safe_slug(cfg_name)}.png"
            plot_path = _ensure_mode_energy_bar_plot(run_dir, out_path=plot_path, max_bars=128)
            if plot_path is None or not plot_path.exists():
                continue
            caption = f"{cfg_name} ({method})"
            items.append((caption, plot_path))
        if items:
            _embed_image_grid(
                lines,
                title="Mode energy by index (dataset-level; per decomposer)",
                items=items,
                base_dir=PROJECT_ROOT,
                n_cols=2,
                img_width=320,
            )
        else:
            lines.append("**Mode energy by index (dataset-level; per decomposer)**")
            lines.append("")
            lines.append("_(no mode-energy plots found)_")
            lines.append("")

        # Mode coefficient value distributions (top methods only, to keep the report compact).
        dist_scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            if str(r.get("status", "")) != "ok":
                continue
            v = _to_float(r.get("field_r2_topk_k64"))
            if v is None:
                v = _to_float(r.get("field_r2"))
            if v is None:
                continue
            dist_scored.append((v, r))
        dist_scored.sort(key=lambda t: t[0], reverse=True)
        dist_top_n = 6
        bundles: list[tuple[str, list[tuple[str, Path]]]] = []
        out_box_dir = v1_summary_dir / "mode_value_boxplot" / case
        out_hist_dir = v1_summary_dir / "mode_value_hist" / case
        for _, r in dist_scored[:dist_top_n]:
            cfg_name = str(r.get("decompose", "")).strip()
            if not cfg_name:
                continue
            run_dir_str = str(r.get("decomposition_run_dir") or "").strip()
            if not run_dir_str:
                continue
            run_dir = Path(run_dir_str)
            method = _method_for_cfg(cfg_name)
            box_path = out_box_dir / f"{_safe_slug(cfg_name)}.png"
            box_path = _ensure_mode_value_boxplot_plot(run_dir, out_path=box_path, top_k=16) or box_path
            hist_path = out_hist_dir / f"{_safe_slug(cfg_name)}.png"
            hist_path = _ensure_coeff_mode_hist_plot(run_dir, out_path=hist_path, top_k=8, cols=4, bins=40) or hist_path
            bundles.append(
                (
                    f"{cfg_name} ({method})",
                    [
                        ("boxplot (top modes)", box_path),
                        ("hist (top modes)", hist_path),
                    ],
                )
            )
        if bundles:
            _embed_image_bundle_grid(
                lines,
                title=f"Mode coefficient value distributions (top {dist_top_n}; boxplot + hist)",
                bundles=bundles,
                base_dir=PROJECT_ROOT,
                n_cols=2,
                img_width=320,
            )

        # --- compression leaderboard (top-energy @K)
        comp_scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            if str(r.get("status", "")) != "ok":
                continue
            v = _to_float(r.get("field_r2_topk_k64"))
            if v is None:
                continue
            comp_scored.append((v, r))
        comp_scored.sort(key=lambda t: t[0], reverse=True)
        comp_rows: list[list[str]] = []
        for _, r in comp_scored[:5]:
            cfg_name = str(r.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            nreq = _to_int(r.get("n_components_required"))
            comp_rows.append(
                [
                    _fmt_cfg(cfg_name),
                    f"`{m}`" if m else "",
                    _fmt_float(r.get("field_r2_topk_k64"), digits=6),
                    _fmt_float(r.get("field_r2_topk_k16"), digits=6),
                    _fmt_compact(_to_float(r.get("k_req_r2_0p95"))),
                    "" if nreq is None else str(nreq),
                ]
            )
        lines.append("**Compression leaderboard (Top-energy @K)**")
        lines.append("")
        lines.append(
            _md_table(
                ["decompose(cfg)", "method", "r2_topk_k64", "r2_topk_k16", "k_req_r2_0.95", "n_req"],
                comp_rows,
            )
        )

        # Embed a few key plots for the best method.
        if best is not None:
            run_dir = Path(str(best.get("decomposition_run_dir")))
            plot_root = run_dir / "plots"
            cfg_name = str(best.get("decompose", "")).strip()
            method = _method_for_cfg(cfg_name)
            # Prefer the standardized report plot for coeff_mode_hist (consistent size/top_k).
            mode_energy_std = v1_summary_dir / "mode_energy_bar" / case / f"{_safe_slug(cfg_name)}.png"
            items = [
                ("dashboard", plot_root / "key_decomp_dashboard.png"),
                ("R^2 vs K", plot_root / "mode_r2_vs_k.png"),
                ("mode energy (bar)", mode_energy_std if mode_energy_std.exists() else mode_energy_std),
                ("sample true/recon", plot_root / "domain" / "field_compare_0000.png"),
            ]
            _embed_image_grid(
                lines,
                title=f"Key decomposition plots (best_rmse={cfg_name} / {method})",
                items=[(cap, p) for cap, p in items if p.exists()],
                base_dir=PROJECT_ROOT,
                n_cols=2,
                img_width=360,
            )

        # --- train table
        rows_t = list(train_by_case.get(case, []))

        def _sort_key_t(r: dict[str, Any]) -> tuple[int, float]:
            status = str(r.get("status", ""))
            ok_rank = 0 if status == "ok" else 1
            v = _to_float(r.get("val_rmse"))
            return ok_rank, v if v is not None else float("inf")

        rows_t.sort(key=_sort_key_t)
        best_t = _pick_best_ok(rows_t, key="val_rmse")
        best_t_field2 = _pick_best_ok(rows_t, key="val_field_rmse")
        if best_t_field2 is None:
            best_t_field2 = _pick_best_ok_max(rows_t, key="val_field_r2")
        headers_t = [
            "decompose(cfg)",
            "method",
            "decomp_r2",
            "model",
            "status",
            "val_rmse",
            "val_r2",
            "val_field_rmse",
            "val_field_r2",
            "fit",
            "run",
        ]
        table_rows_t: list[list[str]] = []
        for r in rows_t:
            cfg_name = str(r.get("decompose", "")).strip()
            method = _method_for_cfg(cfg_name)
            drow = decomp_lookup.get((case, cfg_name))
            decomp_r2 = _fmt_float(drow.get("field_r2") if drow else None, digits=6)
            status = str(r.get("status", "")).strip()
            val_rmse = _fmt_sci(r.get("val_rmse"))
            val_r2 = _fmt_float(r.get("val_r2"), digits=6)
            val_field_rmse = _fmt_sci(r.get("val_field_rmse"))
            val_field_r2 = _fmt_float(r.get("val_field_r2"), digits=6)
            fit = _fmt_time_sec(r.get("fit_time_sec"))
            model = str(r.get("model", "")).strip()
            run_dir = r.get("train_run_dir") or ""
            run_link = ""
            if run_dir:
                run_rel = _relpath(run_dir, base_dir=PROJECT_ROOT)
                run_link = f"[run]({run_rel})"
            label = f"`{cfg_name}`"
            if best_t is not None and cfg_name == str(best_t.get("decompose", "")).strip():
                label = f"**{label}**"
            table_rows_t.append(
                [
                    label,
                    f"`{method}`" if method else "",
                    decomp_r2,
                    f"`{model}`",
                    status,
                    val_rmse,
                    val_r2,
                    val_field_rmse,
                    val_field_r2,
                    fit,
                    run_link,
                ]
            )

        lines.append("**Train (cond -> coeff prediction)**")
        lines.append("")
        lines.append(_md_table(headers_t, table_rows_t))

        # --- train leaderboard (field-space)
        scored_field: list[tuple[float, dict[str, Any]]] = []
        for r in rows_t:
            if str(r.get("status", "")) != "ok":
                continue
            v = _to_float(r.get("val_field_rmse"))
            if v is None:
                continue
            scored_field.append((v, r))
        scored_field.sort(key=lambda t: t[0])
        field_rows: list[list[str]] = []
        for _, r in scored_field[:5]:
            cfg_name = str(r.get("decompose", "")).strip()
            m = _method_for_cfg(cfg_name)
            model = str(r.get("model", "")).strip()
            run_dir = r.get("train_run_dir") or ""
            run_link = f"[run]({_relpath(run_dir, base_dir=PROJECT_ROOT)})" if run_dir else ""
            field_rows.append(
                [
                    _fmt_cfg(cfg_name),
                    f"`{m}`" if m else "",
                    f"`{model}`" if model else "",
                    _fmt_sci(r.get("val_field_rmse")),
                    _fmt_float(r.get("val_field_r2"), digits=6),
                    run_link,
                ]
            )
        lines.append("**Train leaderboard (field-space)**")
        lines.append("")
        lines.append(
            _md_table(
                ["decompose(cfg)", "method", "model", "val_field_rmse", "val_field_r2", "run"],
                field_rows,
            )
        )

        best_t_plot = best_t_field2 or best_t
        if best_t_plot is not None:
            run_dir = Path(str(best_t_plot.get("train_run_dir")))
            plot_root = run_dir / "plots"
            items = [
                ("val residual hist", plot_root / "val_residual_hist.png"),
                ("val scatter (dim0)", plot_root / "val_scatter_dim_0000.png"),
                ("field scatter (val)", plot_root / "field_eval" / "field_scatter_true_vs_pred_ch0.png"),
                ("per-pixel R^2 map (val)", plot_root / "field_eval" / "per_pixel_r2_map_ch0.png"),
            ]
            _embed_image_grid(
                lines,
                title=f"Key train plots (best_field_eval={best_t_plot.get('decompose')})",
                items=[(cap, p) for cap, p in items if p.exists()],
                base_dir=PROJECT_ROOT,
                n_cols=2,
                img_width=360,
            )

        lines.append("")

    # Special gappy_pod evaluation.
    if gappy_metrics_path.exists():
        metrics = _load_json(gappy_metrics_path)
        out_dir = gappy_metrics_path.parent
        lines.append("## Special Evaluation: gappy_pod (rectangle_scalar, observed mask)")
        lines.append("")
        lines.append(f"- metrics: `{_relpath(gappy_metrics_path, base_dir=PROJECT_ROOT)}`")
        lines.append("")
        lines.append(
            _md_table(
                ["metric", "value"],
                [[f"`{k}`", _md_escape(v)] for k, v in metrics.items()],
            )
        )
        img_candidates = [
            out_dir / "plots" / "field_scatter_true_vs_recon_obs.png",
            out_dir / "plots" / "per_pixel_r2_map.png",
            out_dir / "plots" / "per_pixel_r2_hist.png",
            out_dir / "plots" / "mask_fraction_hist.png",
        ]
        _embed_images(
            lines,
            title="Key gappy_pod plots",
            paths=img_candidates,
            base_dir=PROJECT_ROOT,
            max_images=4,
        )
        lines.append("")

    lines.append("## PDF conversion")
    lines.append("")
    lines.append("- 画像は相対パスで埋め込み済み（Markdown `![](...)` または HTML `<img>`）。")
    lines.append("- 例: `pandoc summary_benchmark.md -o summary_benchmark.pdf`")
    lines.append("- もし画像が出ない場合: `pandoc --from markdown+raw_html summary_benchmark.md -o summary_benchmark.pdf`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
