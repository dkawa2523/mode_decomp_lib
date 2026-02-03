# User Quickstart (run.yaml)

This project provides a single-entry `run.yaml` for non-Hydra users. Use it to run doctor/train/eval/viz without touching multiple configs.

## 1) Minimal run.yaml
```yaml
dataset: data/<dataset_name>   # dataset root with manifest.json
task: train                    # train/predict/reconstruct/eval/viz/bench/doctor
pipeline:
  decomposer: zernike
  codec: zernike_pack_v1
  coeff_post: none
  model: ridge
output:
  root: runs
  tag: demo
```

## 2) Run (or dry-run)
```bash
python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml --dry-run
python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml
```

## 2.1) Other example run.yaml
- `examples/run_scalar_rect_fft2_ridge.yaml`
- `examples/run_scalar_rect_pod_pca_ridge.yaml`
- `examples/run_scalar_mask_pod_ridge.yaml`
- `examples/run_scalar_disk_zernike.yaml`
- `examples/run_scalar_disk_zernike_pca_mtgp.yaml` (requires `gpytorch` + `torch`)
- `examples/run_scalar_annulus_annular_zernike_ridge.yaml`
- `examples/run_scalar_rect_wavelet_ridge.yaml` (requires `pywt`)
- `examples/run_sphere_sh_ridge.yaml` (requires `pyshtools`)

## 3) Optional tweaks
Use `params:` for per-module overrides.
```yaml
params:
  decompose:
    n_max: 8
  model:
    alpha: 0.5
```

## Notes
- Dataset `manifest.json` is the source of truth for domain/grid/field_kind.
- Legacy datasets without manifest require `params.domain` in run.yaml.
- `output.root`/`output.tag` control where runs are written.
- `pipeline.codec` is required (use `none` for real-valued coeffs, or the method-specific codecs).
- `coeff_post: power_yeojohnson` targets skewness while keeping an exact inverse; `coeff_post: quantile` is better for heavy-tailed/outlier-heavy data but its inverse is approximate at extremes.

## 4) Benchmark quick/full (matrix)
Benchmark runs are driven by `scripts/bench/matrix.yaml` and can be run via:
```bash
bash scripts/bench/run_p0p1_p2ready.sh   # quick profile (minimal set)
bash scripts/bench/run_full.sh           # full profile (optional deps are skipped)
```
Notes:
- `scripts/bench/run_matrix.py --profile quick|full` is the underlying runner.
- Optional methods are skipped when their config or dependency is missing.
