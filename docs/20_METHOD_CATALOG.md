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

### ベクトル場（vector）
- Helmholtz（div/curl, P2, Implemented）

### データ駆動（grid）
- POD / SVD（P1, Implemented）
- Dictionary Learning（P2, Implemented）
  - 有効: 局所/繰り返しパターンを少数辞書で表現できる場合、maskが固定できる場合
  - 無効: サンプルごとにmaskが変わる場合、線形結合で再構成できない場合

### 非線形 / Deep（grid）
- Conv Autoencoder / VAE（P2, Implemented）

### 任意マスク/不規則点
- RBF expansion（P0–P1）
- Graph Fourier（P1, Implemented）

### 多解像
- Wavelet（P1）

### 曲面/メッシュ
- Laplace-Beltrami eigs（P2, Implemented）

## 係数後処理（CoeffPost）
- normalize（P0）
- complex -> mag/phase（FFT採用時にP0）
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
- MultiTask ElasticNet（`multitask_elasticnet`, P1）

## Uncertainty（coeff -> field, GPR）
- Implemented（P1）: coeff_std -> field_std を MC で近似（`uncertainty: gpr_mc`）
- coeff予測は独立正規（diag std）を仮定
- MCで係数サンプル -> inverse_transform -> field std を近似
- coeff間の相関や分布形状の違いは扱わない（近似）

## Short Examples（configs/examples）
- POD + Ridge（fast）: `python -m mode_decomp_ml.cli.run --config-name examples/pod_ridge`
- POD + GPR + uncertainty（small data）: `python -m mode_decomp_ml.cli.run --config-name examples/pod_gpr_uncertainty`
- どちらも seed/split を固定済みで、成果物は `outputs/benchmark/...` に保存される
