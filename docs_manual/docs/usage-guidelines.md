# 利用方法ガイドライン

このページは “まず動かす” と “困った時に切り分ける” を最短でできるようにするためのガイドです。

## 最小の成功（まずここ）

1. eval dataset（小）を使う
2. `task=pipeline` で分解→学習→推論まで回す

```bash
# 例: rectangle/scalar を dct2 + ridge で回す
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run \
  task=pipeline \
  dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect \
  dataset.mask_policy=allow_none \
  decompose=analytic/dct2 \
  model=ridge
```

## domain 別の指針（実務で迷う点）

- rectangle:
  - baseline: `dct2` / `fft2`
  - 局所構造: `wavelet2d`
- disk/annulus:
  - まずは解析系（`zernike`/`pseudo_zernike`/`annular_zernike`）
  - “近似でも良い” 場合は `polar_fft`
- arbitrary_mask:
  - 可変maskなら `gappy_*` や `pod_em` 系を優先
- sphere_grid:
  - 全域なら `spherical_harmonics`
  - ROI があるなら `spherical_slepian`（問題設定を合わせる）

## 係数次元を下げたい（PCA）

モード分解の後段に `coeff_post=pca` を入れて次元削減できます。

ポイント:

- 学習指標は coeff 空間より **field 空間**（復元後）で比較する
- `n_components` を小さくし過ぎると `field_r2` が落ちるので `mode_r2_vs_k.png` を併用して決める

## うまくいかない時の切り分け

1. decomposition の `plots/key_decomp_dashboard.png` を見る
   - R^2 vs K が早く飽和: 手法の前提（境界条件/ROI）が合っていない可能性
   - per-pixel R^2 が境界で落ちる: mask/境界条件/補間の影響
2. `mask_fraction_hist.png` を見る（mask domain の場合）
   - 観測密度が低いと “係数が決まらない” ことがある（valid < K）
3. `coeff_spectrum.png` を見る
   - 低次に集中: offset 優勢。`offset_residual` を検討

### 1) モード分解の比較（decomposition）
```bash
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect dataset.mask_policy=allow_none \
  domain=rectangle decompose=fft2,dct2,pod_svd model=ridge coeff_post=none
```

### 2) 学習モデル比較（train）
```bash
# 同じモード分解（dct2）のまま model を変えて比較（val_field_* を見る）
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run -m task=pipeline \
  dataset=npy_dir dataset.root=data/mode_decomp_eval_dataset_v1/scalar_rect dataset.mask_policy=allow_none \
  domain=rectangle decompose=dct2 coeff_post=none model=ridge,gpr
```

## 精度を上げる tips（テンプレ）

- `field` が offset 優勢なら offset/residual を分けて考える（offset_split）。
- `field_r2` は “全係数での再構成” なので、圧縮比較は `field_r2_topk_k64` / `k_req_r2_0.95` を見る。
