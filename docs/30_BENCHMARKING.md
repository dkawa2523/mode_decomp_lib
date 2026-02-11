# Benchmarking

This repo supports end-to-end benchmarking across domains and decomposers using a fixed dataset generator and a
standard artifact layout.

## Stages

- `decomposition`: field -> raw_coeff -> coeff (via codec) and reconstruction diagnostics
- `preprocessing`: optional coefficient post-processing (e.g., PCA) on top of decomposition outputs
- `train`: cond -> predicted coeff, plus optional field-space eval by decoding and inverse-transforming
- `inference`: cond -> coeff/field prediction (optionally with uncertainty if the model supports it)

Each stage writes a run directory with standardized subfolders (see `docs/04_ARTIFACTS_AND_VERSIONING.md`).

## Dataset (Benchmark v1)

- Root: `data/benchmarks/v1/offset_noise_36/`
- Cases: `rectangle_*`, `disk_*`, `annulus_*`, `arbitrary_mask_*`, `sphere_grid_*` (plus optional extensions)
- The dataset is generated (reproducibly) by the benchmark generator tools in `tools/bench/`.

## Running

Minimal run (example):

```bash
python3 tools/bench/run_benchmark_v1.py --cases rectangle_scalar --stages decomposition
python3 tools/bench/generate_summary_benchmark_md.py --out summary_benchmark.md
```

## Outputs

Run outputs (per case/method) typically include:

- `outputs/metrics.json`: numeric metrics (RMSE, R^2, energy-based diagnostics)
- `outputs/coeffs.npz`: coefficient arrays and metadata snapshots
- `plots/`: key diagnostics (e.g., `key_decomp_dashboard.png`, `mode_r2_vs_k.png`, per-pixel R^2 maps)

## Plot Filename Contract (v1)

Common decomposition plots (when enabled / available):

- `plots/key_decomp_dashboard.png`
- `plots/mode_r2_vs_k.png`
- `plots/field_scatter_true_vs_recon_ch0.png` (and `*_mag.png` for vectors)
- `plots/per_pixel_r2_map_ch0.png` and `plots/per_pixel_r2_hist_ch0.png`
- `plots/coeff_spectrum.png`
- `plots/coeff_mode_hist.png` (top-energy mode value hist)

Summary/report outputs:

- `summary_benchmark.md`: benchmark report with embedded figures (can be converted to PDF)
- CSV/Markdown leaderboards under the benchmark run's `summary/` directory (if enabled)

## Reading Results

- Full reconstruction metrics (`field_r2`) can be misleading for perfectly invertible transforms (FFT/DCT etc.).
- Prefer compression-oriented diagnostics:
  - `field_r2_topk_k*` and `k_req_r2_0.95` when present
  - `mode_r2_vs_k.png` and `key_decomp_dashboard.png` for visual diagnosis
