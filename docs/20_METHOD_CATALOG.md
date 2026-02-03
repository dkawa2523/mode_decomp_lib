# Method Catalog（一次分解 + 係数後処理）

この文書は “候補一覧” です。実装は `work/tasks/` で段階的に行います。

## 一次分解（Decomposer）カテゴリ
### 円（disk）
- Zernike（P0）
- Fourier–Bessel（P1, Implemented）

### 矩形（rect）
- FFT2（P0）
- DCT2（P0）
- DST2（P1）
- PSWF2D tensor（研究枠, P2, Implemented）

### ベクトル場（vector）
- Helmholtz（div/curl, P2, Implemented）

### データ駆動（grid）
- POD / SVD（P1, Implemented）
- Dictionary Learning（P2, Implemented）
  - 有効: 局所/繰り返しパターンを少数辞書で表現できる場合、maskが固定できる場合
  - 無効: サンプルごとにmaskが変わる場合、線形結合で再構成できない場合

### 非線形 / Deep（grid）
- Conv Autoencoder（P2, Implemented）
- VAE（P2, TODO）

### 任意マスク/不規則点
- RBF expansion（P0–P1）
- Graph Fourier（P1, Implemented）

### 多解像
- Wavelet（P1）

### 曲面/メッシュ
- Laplace-Beltrami eigs（P2, Implemented）

### 球面（sphere_grid）
- Spherical Harmonics（P2, Implemented）
- Slepian（P2, Implemented）
  - region は dataset mask または spherical cap 指定（config）

## 係数後処理（CoeffPost）
- normalize（P0）
- complex -> mag/phase（FFT採用時にP0）
- PowerTransform (Yeo-Johnson)（P1, Implemented）
- PCA / TruncatedSVD（P0–P1）
- Dictionary Learning（P2, Implemented）
  - 有効: 係数に疎な辞書表現が期待できる場合
  - 無効: 係数比較や直交基底の解釈を重視する場合
- ICA / NMF（P2）
- energy threshold / low-order selection（P1）

## 回帰モデル（Regressor）
- Ridge（P0）
- GPR（`gpr`, P1, Implemented）
- ElasticNet（`elasticnet`, P1, Implemented）
- GBDT（`xgb` / `lgbm` / `catboost`, P2, Implemented, optional）
  - 非線形・相互作用が強い条件に強い（依存: xgboost / lightgbm / catboost）
- MultiTask ElasticNet（`multitask_elasticnet`, P1）
- MultiTask Lasso（`multitask_lasso`, P2, Implemented）
  - Ridgeとの差分: 共有疎性で特徴選択が効く（少数の共通因子が支配的なとき有効）
  - 不向き: 出力ごとに必要特徴がバラバラ/密な場合

## Uncertainty（coeff -> field, GPR）
- Implemented（P1）: coeff_std -> field_std を MC で近似（`uncertainty: gpr_mc`）
- coeff予測は独立正規（diag std）を仮定
- MCで係数サンプル -> inverse_transform -> field std を近似
- coeff間の相関や分布形状の違いは扱わない（近似）

## Short Examples（run.yaml）
- FFT2 + Ridge（rect）: `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml`
- PSWF2D tensor + Ridge（rect）: `python -m mode_decomp_ml.run --config examples/run_scalar_rect_pswf_ridge.yaml`
- Zernike + Ridge（disk）: `python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike.yaml`
- POD + Ridge（mask）: `python -m mode_decomp_ml.run --config examples/run_scalar_mask_pod_ridge.yaml`
- 成果物は `runs/<tag>/<run_id>/` に保存される
- 他の例は `examples/` を参照
