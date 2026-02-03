"""Generate synthetic datasets and evaluate special-function decomposers."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.domain.sphere_grid import sphere_grid_domain_cfg
from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.viz import (
    coeff_energy_spectrum,
    coeff_value_magnitude,
    plot_coeff_histogram,
    plot_coeff_spectrum,
    plot_error_map,
    plot_field_grid,
)


DATA_ROOT = PROJECT_ROOT / "data" / "synthetic_special_suite"
OUT_ROOT = PROJECT_ROOT / "outputs" / "special_suite"


@dataclass
class DomainCase:
    name: str
    grid_shape: tuple[int, int]
    mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    domain_cfg: Mapping[str, Any]


def _grid_xy(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, width)
    ys = np.linspace(0.0, 1.0, height)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


def _mask_disk(xx: np.ndarray, yy: np.ndarray, *, center=(0.5, 0.5), radius=0.45) -> np.ndarray:
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    return (dx * dx + dy * dy) <= float(radius) ** 2


def _mask_annulus(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    center=(0.5, 0.5),
    r_inner=0.2,
    r_outer=0.45,
) -> np.ndarray:
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    rr = dx * dx + dy * dy
    return (rr >= float(r_inner) ** 2) & (rr <= float(r_outer) ** 2)


def _sphere_grid(lat_n: int, lon_n: int) -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(-np.pi / 2, np.pi / 2, lat_n)
    lons = np.linspace(-np.pi, np.pi, lon_n, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def _offset_dominant_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    offset: float,
    weights: np.ndarray,
) -> np.ndarray:
    patterns = [
        np.sin(2 * np.pi * xx),
        np.cos(2 * np.pi * yy),
        np.sin(2 * np.pi * (xx + yy)),
        np.cos(4 * np.pi * xx) * np.cos(2 * np.pi * yy),
    ]
    field = offset + sum(w * p for w, p in zip(weights, patterns))
    return field


def _sphere_field(lat: np.ndarray, lon: np.ndarray, *, offset: float, weights: np.ndarray) -> np.ndarray:
    patterns = [
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
        np.cos(2 * lat) * np.cos(2 * lon),
    ]
    field = offset + sum(w * p for w, p in zip(weights, patterns))
    return field


def _mesh_domain() -> tuple[Mapping[str, Any], np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    cfg = {"name": "mesh", "vertices": vertices, "faces": faces}
    return cfg, vertices, faces


def _generate_dataset(
    domain: DomainCase,
    *,
    n_groups: int,
    reps: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int]]:
    height, width = domain.grid_shape
    xx, yy = _grid_xy(height, width)
    mask = domain.mask_fn(xx, yy) if domain.mask_fn else None
    fields = []
    conds = []
    group_ids = []
    rng = np.random.default_rng(0)

    for gid in range(n_groups):
        offset = rng.uniform(0.5, 1.5)
        weights = rng.uniform(-0.1, 0.1, size=4) * offset
        for _ in range(reps):
            noise = rng.normal(scale=noise_scale * offset, size=(height, width))
            base = _offset_dominant_field(xx, yy, offset=offset, weights=weights)
            field = base + noise
            if mask is not None:
                field = np.where(mask, field, 0.0)
            fields.append(field[..., None])
            conds.append(np.array([offset, *weights], dtype=np.float64))
            group_ids.append(gid)
    field_arr = np.stack(fields, axis=0)
    cond_arr = np.stack(conds, axis=0)
    mask_arr = mask.astype(bool) if mask is not None else None
    return cond_arr, field_arr, mask_arr, group_ids


def _generate_sphere_dataset(
    lat_n: int,
    lon_n: int,
    *,
    n_groups: int,
    reps: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[int], np.ndarray, np.ndarray]:
    lat, lon = _sphere_grid(lat_n, lon_n)
    fields = []
    conds = []
    group_ids = []
    rng = np.random.default_rng(1)
    for gid in range(n_groups):
        offset = rng.uniform(0.5, 1.5)
        weights = rng.uniform(-0.1, 0.1, size=4) * offset
        for _ in range(reps):
            noise = rng.normal(scale=noise_scale * offset, size=(lat_n, lon_n))
            base = _sphere_field(lat, lon, offset=offset, weights=weights)
            fields.append((base + noise)[..., None])
            conds.append(np.array([offset, *weights], dtype=np.float64))
            group_ids.append(gid)
    mask = np.ones((lat_n, lon_n), dtype=bool)
    return np.stack(conds), np.stack(fields), mask, group_ids, lat, lon


def _save_dataset(root: Path, cond: np.ndarray, field: np.ndarray, mask: np.ndarray | None, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "cond.npy", cond)
    np.save(root / "field.npy", field)
    if mask is not None:
        np.save(root / "mask.npy", mask.astype(bool))
    with (root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x_train.T @ x_train
    reg = ridge * np.eye(xtx.shape[0])
    coef = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    return x_test @ coef


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _evaluate_decomposer(
    name: str,
    cfg: Mapping[str, Any],
    domain_spec: Any,
    fields: np.ndarray,
    mask: np.ndarray | None,
    cond: np.ndarray,
    group_ids: list[int],
    out_dir: Path,
) -> dict[str, Any]:
    def _skip_code(stage: str, exc: Exception) -> str:
        if isinstance(exc, ImportError):
            return "missing_dependency"
        message = str(exc).lower()
        if "rank-deficient" in message:
            return "rank_deficient"
        if "requires domain mask" in message:
            return "missing_domain_mask"
        if "requires dataset mask" in message:
            return "missing_dataset_mask"
        if "weights are empty" in message:
            return "empty_weights"
        if "more modes than valid samples" in message:
            return "insufficient_samples"
        return f"{stage}_error"

    def _skip(stage: str, exc: Exception) -> dict[str, Any]:
        return {
            "method": name,
            "status": "skip",
            "skip_code": _skip_code(stage, exc),
            "skip_reason": f"{stage}: {type(exc).__name__}: {exc}",
        }

    try:
        decomposer = build_decomposer(cfg)
    except Exception as exc:
        return _skip("build", exc)

    if hasattr(decomposer, "fit"):
        try:
            class _Dataset:
                def __len__(self) -> int:
                    return fields.shape[0]

                def __getitem__(self, idx: int):
                    return type(
                        "Sample",
                        (),
                        {"field": fields[idx], "mask": mask},
                    )

            decomposer.fit(dataset=_Dataset(), domain_spec=domain_spec)
        except Exception as exc:
            return _skip("fit", exc)

    try:
        coeff = decomposer.transform(fields[0], mask=mask, domain_spec=domain_spec)
        field_hat = decomposer.inverse_transform(coeff, domain_spec=domain_spec)
    except Exception as exc:
        return _skip("transform", exc)
    rmse = float(np.sqrt(np.mean((field_hat - fields[0]) ** 2)))

    coeffs = []
    try:
        for idx in range(fields.shape[0]):
            coeffs.append(decomposer.transform(fields[idx], mask=mask, domain_spec=domain_spec).reshape(-1))
        coeffs = np.stack(coeffs, axis=0)
    except Exception as exc:
        return _skip("transform_batch", exc)
    coeffs_mag = np.abs(coeffs) if np.iscomplexobj(coeffs) else coeffs

    # stability: within-group std normalized by mean magnitude
    stability_vals = []
    for gid in sorted(set(group_ids)):
        idx = [i for i, g in enumerate(group_ids) if g == gid]
        group_coeff = coeffs_mag[idx]
        std = np.mean(np.std(group_coeff, axis=0))
        mean_mag = np.mean(np.abs(group_coeff))
        stability_vals.append(float(std / (mean_mag + 1e-12)))
    stability = float(np.mean(stability_vals)) if stability_vals else 0.0
    stability_med = float(np.median(stability_vals)) if stability_vals else 0.0

    # model usefulness: ridge from cond -> coeff
    n = cond.shape[0]
    split = int(n * 0.8)
    pred = _ridge_fit_predict(cond[:split], coeffs_mag[:split], cond[split:], ridge=1e-6)
    usefulness = _r2_score(coeffs_mag[split:], pred)

    coeff_meta = decomposer.coeff_meta()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_field_grid(
        out_dir / "field_recon.png",
        [fields[0], field_hat],
        ["true", "recon"],
        mask=mask,
        suptitle=f"{name} reconstruction",
    )
    plot_error_map(out_dir / "error_map.png", fields[0], field_hat, mask=mask)
    spectrum = coeff_energy_spectrum(coeffs_mag, coeff_meta)
    plot_coeff_spectrum(out_dir / "coeff_spectrum.png", spectrum)
    plot_coeff_histogram(out_dir / "coeff_hist.png", coeff_value_magnitude(coeffs_mag, coeff_meta))

    return {
        "method": name,
        "status": "ok",
        "skip_code": "",
        "skip_reason": "",
        "rmse": rmse,
        "coeff_stability": stability,
        "coeff_stability_med": stability_med,
        "model_r2": usefulness,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Special-function evaluation suite.")
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.02)
    args = parser.parse_args(argv)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    domains = [
        DomainCase(
            name="rectangle",
            grid_shape=(32, 32),
            mask_fn=None,
            domain_cfg={"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        ),
        DomainCase(
            name="disk",
            grid_shape=(32, 32),
            mask_fn=lambda xx, yy: _mask_disk(xx, yy),
            domain_cfg={
                "name": "disk",
                "center": [0.5, 0.5],
                "radius": 0.45,
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
            },
        ),
        DomainCase(
            name="annulus",
            grid_shape=(32, 32),
            mask_fn=lambda xx, yy: _mask_annulus(xx, yy),
            domain_cfg={
                "name": "annulus",
                "center": [0.5, 0.5],
                "r_inner": 0.2,
                "r_outer": 0.45,
                "x_range": [0.0, 1.0],
                "y_range": [0.0, 1.0],
            },
        ),
    ]

    summary_rows = []

    for domain in domains:
        cond, field, mask, groups = _generate_dataset(
            domain,
            n_groups=args.n_groups,
            reps=args.reps,
            noise_scale=args.noise_scale,
        )
        manifest = {
            "field_kind": "scalar",
            "grid": {"H": domain.grid_shape[0], "W": domain.grid_shape[1], "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
            "domain": {"type": domain.name, **domain.domain_cfg},
        }
        root = DATA_ROOT / domain.name
        _save_dataset(root, cond, field, mask, manifest)

        domain_spec = build_domain_spec(domain.domain_cfg, field.shape[1:3])

        methods = []
        if domain.name == "rectangle":
            methods = [
                ("fft2", {"name": "fft2", "disk_policy": "error"}),
                ("dct2", {"name": "dct2", "disk_policy": "error"}),
                ("wavelet2d", {"name": "wavelet2d", "wavelet": "db1", "level": 2, "mask_policy": "error", "mode": "symmetric"}),
                ("pswf2d_tensor", {"name": "pswf2d_tensor", "c_x": 2.0, "c_y": 2.0, "n_x": 32, "n_y": 32, "mask_policy": "error"}),
                ("pod_svd", {"name": "pod_svd", "mask_policy": "ignore_masked_points", "n_modes": 8, "inner_product": "euclidean"}),
                ("pod", {"name": "pod", "mask_policy": "ignore_masked_points", "n_modes": 8, "inner_product": "euclidean"}),
                ("dict_learning", {"name": "dict_learning", "mask_policy": "ignore_masked_points", "n_components": 8}),
                ("graph_fourier", {"name": "graph_fourier", "n_modes": 8, "connectivity": 4, "laplacian_type": "combinatorial", "mask_policy": "allow_full", "solver": "dense", "dense_threshold": 2048, "eigsh_tol": 1.0e-6, "eigsh_maxiter": 1000}),
            ]
        if domain.name == "disk":
            methods = [
                ("zernike", {"name": "zernike", "n_max": 8, "ordering": "n_then_m", "normalization": "orthonormal", "boundary_condition": "dirichlet", "mask_policy": "ignore_masked_points"}),
                ("fourier_bessel", {"name": "fourier_bessel", "n_max": 6, "m_max": 6, "ordering": "m_then_n", "normalization": "orthonormal", "boundary_condition": "dirichlet", "mask_policy": "ignore_masked_points"}),
            ]
        if domain.name == "annulus":
            methods = [
                ("annular_zernike", {"name": "annular_zernike", "n_max": 8, "ordering": "n_then_m", "normalization": "orthonormal", "boundary_condition": "dirichlet", "mask_policy": "ignore_masked_points"}),
            ]

        for method_name, cfg in methods:
            out_dir = OUT_ROOT / domain.name / method_name
            result = _evaluate_decomposer(
                method_name,
                cfg,
                domain_spec,
                field,
                mask,
                cond,
                groups,
                out_dir,
            )
            summary_rows.append({"domain": domain.name, **result})

    # sphere grid
    lat_n, lon_n = 36, 72
    cond, field, mask, groups, lat, lon = _generate_sphere_dataset(
        lat_n,
        lon_n,
        n_groups=args.n_groups,
        reps=args.reps,
        noise_scale=args.noise_scale,
    )
    sphere_domain = dict(sphere_grid_domain_cfg(lat_n, lon_n, angle_unit="deg", radius=1.0))
    lon_range = sphere_domain["lon_range"]
    manifest = {
        "field_kind": "scalar",
        "grid": {"H": lat_n, "W": lon_n, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        "domain": {"type": "sphere_grid", **sphere_domain},
    }
    root = DATA_ROOT / "sphere_grid"
    _save_dataset(root, cond, field, mask, manifest)
    sphere_cfg = dict(manifest["domain"])
    sphere_cfg["name"] = "sphere_grid"
    sphere_cfg["x_range"] = [-1.0, 1.0]
    sphere_cfg["y_range"] = [-1.0, 1.0]
    domain_spec = build_domain_spec(sphere_cfg, field.shape[1:3])

    sphere_methods = [
        ("spherical_harmonics", {"name": "spherical_harmonics", "l_max": 1, "mask_policy": "allow", "backend": "scipy", "real_form": True, "norm": "ortho"}),
        ("spherical_slepian", {"name": "spherical_slepian", "l_max": 8, "k": 10, "mask_policy": "allow", "backend": "scipy", "region_mask": "dataset"}),
    ]
    for method_name, cfg in sphere_methods:
        out_dir = OUT_ROOT / "sphere_grid" / method_name
        result = _evaluate_decomposer(
            method_name,
            cfg,
            domain_spec,
            field,
            mask,
            cond,
            groups,
            out_dir,
        )
        summary_rows.append({"domain": "sphere_grid", **result})

    # mesh (laplace_beltrami)
    mesh_cfg, vertices, faces = _mesh_domain()
    n_vertices = vertices.shape[0]
    mesh_field = np.zeros((args.n_groups * args.reps, n_vertices, 1, 1), dtype=np.float64)
    mesh_cond = np.zeros((mesh_field.shape[0], 2), dtype=np.float64)
    rng = np.random.default_rng(3)
    groups = []
    for gid in range(args.n_groups):
        offset = rng.uniform(0.5, 1.5)
        for rep in range(args.reps):
            idx = gid * args.reps + rep
            noise = rng.normal(scale=args.noise_scale * offset, size=(n_vertices, 1, 1))
            mesh_field[idx] = offset + noise
            mesh_cond[idx] = np.array([offset, 1.0])
            groups.append(gid)
    manifest = {
        "field_kind": "scalar",
        "grid": {"H": n_vertices, "W": 1, "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "domain": {"type": "mesh", "vertices": vertices.tolist(), "faces": faces.tolist()},
    }
    root = DATA_ROOT / "mesh"
    _save_dataset(root, mesh_cond, mesh_field, None, manifest)
    domain_spec = build_domain_spec(mesh_cfg, mesh_field.shape[1:])
    lap_cfg = {
        "name": "laplace_beltrami",
        "n_modes": n_vertices,
        "laplacian_type": "cotangent",
        "mass_type": "lumped",
        "boundary_condition": "neumann",
        "mask_policy": "allow",
        "solver": "dense",
        "dense_threshold": 128,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 10000,
    }
    out_dir = OUT_ROOT / "mesh" / "laplace_beltrami"
    result = _evaluate_decomposer(
        "laplace_beltrami",
        lap_cfg,
        domain_spec,
        mesh_field,
        None,
        mesh_cond,
        groups,
        out_dir,
    )
    summary_rows.append({"domain": "mesh", **result})

    summary_path = OUT_ROOT / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "domain",
                "method",
                "status",
                "skip_code",
                "skip_reason",
                "rmse",
                "coeff_stability",
                "coeff_stability_med",
                "model_r2",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
