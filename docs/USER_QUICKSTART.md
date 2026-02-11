# User Quickstart (run.yaml)

This project provides a single-entry `run.yaml` for non-Hydra users. Use it to run decomposition/preprocessing/train/inference without touching multiple configs.

## 1) Minimal run.yaml
```yaml
dataset:
  root: data/<dataset_root>        # recommended: `npy_dir` dataset (cond.npy/field.npy/mask.npy + manifest.json)
  mask_policy: allow_none          # require|allow_none|forbid
task: decomposition            # decomposition/preprocessing/train/inference/pipeline/doctor
pipeline:
  decomposer: zernike
  codec: zernike_pack_v1
  coeff_post: none
  model: ridge
output:
  root: runs
  name: demo
```

## 2) Run (or dry-run)
```bash
python3 -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml --dry-run
python3 -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml
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
  train:
    eval:
      val_ratio: 0.1
    cv:
      enabled: true
      folds: 5
  inference:
    viz:
      max_samples: 4
```

## Notes

## Further reading

- Benchmarking and reports: `docs/30_BENCHMARKING.md`
- Code map / architecture: `docs/31_CODE_TOUR.md`
- `npy_dir` が推奨です（`cond.npy/field.npy/mask.npy` + `manifest.json`）。`csv_fields` は外部データ互換のために残しています。
- `output.root`/`output.name` で出力先を指定（`runs/<name>/<process>/...`）。
- `pipeline.codec` is required (use `none` for real-valued coeffs, or the method-specific codecs).
- `coeff_post: power_yeojohnson` targets skewness while keeping an exact inverse; `coeff_post: quantile` is better for heavy-tailed/outlier-heavy data but its inverse is approximate at extremes.
- POD系で `coeff_post: pca` を使いたい場合は `coeff_post.force: true` を明示してください（自動で無効化されるため）。

## 4) Pipeline sweep
複数分解法・複数学習法は `task: pipeline` で実行し、leaderboard を出力します。
