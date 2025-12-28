了解しました。ご指摘どおり、

* **分解前の「派生量（勾配やラプラシアン等）」を“特徴量”として増やす**よりも、
* **（特殊関数などの）モード分解で得た係数に対して、PCA / ICA などで再表現（圧縮・直交化・独立化）**する方が、
  *学習精度向上*・*主要成分抽出*・*比較評価*に直結しやすい

――という整理に改めます。

そのため、前回出していた **「分解前の派生量生成（grad/laplacian等）」は今回の“コア候補”から削除**し、代わりに **「モード分解後係数の特徴量化（係数後処理）」**として PCA/ICA/… を中心に **全体を再提示**します。
（将来必要になれば `aux_features/` として別枠で追加できる設計にしますが、いまは候補リストから外します。）

---

# 全体パイプライン（推奨の整理）

**Stage 0**: 入力（2Dスカラー/ベクトル場 + マスク + メタ）
**Stage 1**: 前処理（ノイズ・欠損・リサンプリング・座標）
**Stage 2**: （ベクトル場なら）表現変換（成分別 / div-curl 等）
**Stage 3**: **一次モード分解（特殊関数/固定基底/多解像/任意境界）** → 係数 `a`
**Stage 4**: **係数後処理（= “特徴量化”）**：PCA/ICA/… → 学習用特徴 `z`
**Stage 5**: 学習（条件→`z` もしくは条件→`a`）
**Stage 6**: 逆変換（`z`→`a`→場）＋評価・可視化

> 重要：今回の「特徴量生成」は **Stage 4（係数後処理）** を指す前提に統一します。

---

# 1) 前処理（Preprocess）候補一覧

> 目的：分解の安定化・再現性・境界形状の一般化（円以外も含む）
> 出力フォーマット統一：`field(H,W,C)`, `mask(H,W)`, `meta(dict)`

| Hydraグループ案          | 手法ID例                                                 | 目的                  | 対象  | 境界   | 優先    |
| ------------------- | ----------------------------------------------------- | ------------------- | --- | ---- | ----- |
| preprocess/scale    | `standardize` `minmax` `robust_scale` `log1p`         | 条件・値のスケール統一         | S/V | 任意   | P0    |
| preprocess/missing  | `keep_mask` `nearest_fill` `griddata_fill`            | 欠損処理（マスク維持）         | S/V | 任意   | P0    |
| preprocess/outlier  | `mad_clip` `hampel` `iqr_clip`                        | 外れ値耐性               | S/V | 任意   | P0    |
| preprocess/denoise  | `gaussian` `median` `bilateral` `nl_means`            | 空間ノイズ除去             | S/V | 任意   | P0–P1 |
| preprocess/detrend  | `remove_mean` `remove_plane` `remove_low_order_basis` | 大域トレンド除去（低次モード除去含む） | S/V | 任意   | P0    |
| preprocess/resample | `to_cart_grid` `to_polar_grid` `adaptive_grid`        | 格子化/座標統一            | S/V | 円/任意 | P0    |
| preprocess/interp   | `spline` `rbf_interpolator` `gp_interpolator`         | 補間の精度向上（疎点対応）       | S/V | 任意   | P0–P1 |
| preprocess/mask     | `domain_mask` `inpaint`                               | 任意形状マスク・欠損領域        | S/V | 任意形状 | P0    |
| preprocess/weight   | `area_weight` `jacobian_weight`                       | 非一様サンプリング補正         | S/V | 任意   | P1    |
| preprocess/align    | `center_align` `rotation_align`                       | 位置/回転の正規化           | S/V | 任意   | P1    |
| preprocess/augment  | `rot` `flip` `symmetry`                               | データ拡張（対称性が成立する場合のみ） | S/V | 任意   | P1    |

---

# 2) ベクトル場の表現変換（Vector transform）候補一覧

> 「ベクトルをどう扱うか」は、分解手法とは分離してスイッチ可能にしておくのが重要

| Hydraグループ案       | 手法ID例             | 何をするか               | 出力チャネル      | 優先 |
| ---------------- | ----------------- | ------------------- | ----------- | -- |
| vector/transform | `componentwise`   | vx,vy を別々に以降の分解へ    | C=2         | P0 |
| vector/transform | `div_curl`        | div, curl の2スカラーへ変換 | C=2（スカラー2枚） | P1 |
| vector/transform | `helmholtz_hodge` | Hodge分解（発散0/回転0など）  | 複数          | P2 |

---

# 3) 一次モード分解（Decomposition）候補一覧

（特殊関数・固定基底・FFT/Bessel/Zernike を含む）

## 3-A) 円形/円環に強い（Zernike / Bessel / 極座標系）

| Hydraグループ案        | 手法ID例             | 代表                        | 対象       | 領域/境界      | 優先     |
| ----------------- | ----------------- | ------------------------- | -------- | ---------- | ------ |
| decompose/disk    | `zernike`         | **Zernike**               | S（Vは成分別） | 円          | **P0** |
| decompose/disk    | `pseudo_zernike`  | Pseudo-Zernike            | S        | 円          | P1     |
| decompose/annulus | `annular_zernike` | Annular Zernike           | S        | 円環         | P2     |
| decompose/disk    | `fourier_bessel`  | **Fourier–Bessel / Dini** | S        | 円（境界条件が効く） | P1     |
| decompose/disk    | `hankel`          | Hankel(Bessel)            | S        | 放射対称寄り     | P2     |
| decompose/polar   | `polar_fft`       | θ方向FFT                    | S/V      | 極座標（θ周期）   | P1     |

> **注**：Bessel系は “直交性” が境界条件（Dirichlet/Neumann等）で変わるので、`boundary_condition` を config に含める前提で扱うのが設計上安全です。

## 3-B) 矩形・周期（FFT/DCT/DST/直交多項式）

| Hydraグループ案     | 手法ID例             | 代表               | 対象  | 領域/境界      | 優先    |
| -------------- | ----------------- | ---------------- | --- | ---------- | ----- |
| decompose/rect | `fft2`            | **2D FFT**       | S/V | 周期（矩形）     | P0    |
| decompose/rect | `dct2`            | **DCT**          | S/V | 矩形（端不連続緩和） | P0    |
| decompose/rect | `dst2`            | DST（Dirichlet相当） | S/V | 矩形         | P1    |
| decompose/rect | `legendre_2d`     | Legendre         | S/V | 矩形         | P0–P1 |
| decompose/rect | `chebyshev_2d`    | Chebyshev        | S/V | 矩形         | P1    |
| decompose/rect | `bspline_surface` | B-spline基底       | S/V | 矩形/パッチ     | P1    |

## 3-C) 任意境界・不規則点（汎用）

| Hydraグループ案       | 手法ID例                 | 代表             | 対象  | 領域/境界      | 優先    |
| ---------------- | --------------------- | -------------- | --- | ---------- | ----- |
| decompose/domain | `rbf_expansion`       | **RBF基底展開**    | S/V | 任意形状       | P0–P1 |
| decompose/domain | `graph_laplacian`     | Graph Fourier  | S/V | 点群/不規則点    | P1    |
| decompose/domain | `laplacian_eigen_fem` | Laplace固有（FEM） | S   | 任意境界（条件指定） | P2    |

## 3-D) 多解像（Wavelet系）

| Hydraグループ案           | 手法ID例            | 代表         | 対象  | 領域         | 優先    |
| -------------------- | ---------------- | ---------- | --- | ---------- | ----- |
| decompose/multiscale | `dwt2`           | 2D DWT     | S/V | 矩形（マスク設計要） | P0–P1 |
| decompose/multiscale | `wavelet_packet` | Packet     | S/V | 矩形         | P1    |
| decompose/multiscale | `swt2`           | SWT（シフト不変） | S/V | 矩形         | P2    |

## 3-E) 曲面（将来枠）

| Hydraグループ案        | 手法ID例                   | 代表                 | 対象 | 領域     | 優先 |
| ----------------- | ----------------------- | ------------------ | -- | ------ | -- |
| decompose/surface | `laplace_beltrami_eigs` | Laplace–Beltrami   | S  | 曲面メッシュ | P2 |
| decompose/surface | `manifold_harmonics`    | Manifold Harmonics | S  | 曲面メッシュ | P2 |
| decompose/surface | `spherical_harmonics`   | 球面調和               | S  | 球面     | P2 |

---

# 4) 係数後処理（＝“特徴量化”）候補一覧

**（特殊関数/FFT等の一次分解の「後」）**

ここが、今回のご要望の中心です。
一次分解の係数 `a` から、学習に使う特徴 `z` を作るステージ。

## 4-A) 係数の整形・正規化（学習安定化）

| Hydraグループ案           | 手法ID例                      | 何をするか            | 目的           | 優先            |        |    |
| -------------------- | -------------------------- | ---------------- | ------------ | ------------- | ------ | -- |
| coeff_post/normalize | `per_mode_standardize`     | 各係数次元を標準化        | 学習安定         | P0            |        |    |
| coeff_post/normalize | `energy_normalize`         | 係数エネルギーで正規化      | サンプル間スケール差除去 | P1            |        |    |
| coeff_post/transform | `log_abs`                  |                  | a            | のlog（符号別管理も可） | 長い裾を圧縮 | P1 |
| coeff_post/transform | `complex_to_mag_phase`     | FFT等の複素→振幅/位相    | モデル入力を実数化    | P0（FFT採用時）    |        |    |
| coeff_post/invariant | `rotation_invariant_pairs` | Zernike等のmペア→大きさ | 回転不変化（必要な場合） | P1            |        |    |

## 4-B) 次元圧縮・直交化・独立化（PCA/ICA等）

| Hydraグループ案        | 手法ID例             | 代表           | 目的            | 備考      | 優先        |
| ----------------- | ----------------- | ------------ | ------------- | ------- | --------- |
| coeff_post/reduce | `pca`             | **PCA**      | 圧縮・ノイズ低減      | 逆変換でa復元 | **P0–P1** |
| coeff_post/reduce | `truncated_svd`   | TruncatedSVD | 大規模係数に強い      | 疎/巨大向け  | P1        |
| coeff_post/whiten | `pca_whiten`      | Whitening    | 線形/距離ベース学習が安定 | PCAの派生  | P1        |
| coeff_post/reduce | `ica`             | **ICA**      | 独立成分抽出        | 逆変換で復元  | P2        |
| coeff_post/reduce | `factor_analysis` | FA           | 潜在因子モデル       | ノイズ分離   | P2        |
| coeff_post/reduce | `nmf`             | NMF          | 非負成分抽出        | 非負制約必要  | P2        |

## 4-C) 係数選択（「必要な成分だけ」へ）

| Hydraグループ案        | 手法ID例                | 何をするか       | 目的        | 優先 |
| ----------------- | -------------------- | ----------- | --------- | -- |
| coeff_post/select | `low_order_only`     | 低次モードだけ残す   | 過学習抑制・解釈性 | P0 |
| coeff_post/select | `energy_threshold`   | エネルギー比で打ち切り | 自動次元決定    | P1 |
| coeff_post/select | `group_by_frequency` | 周波数帯でまとめる   | 比較軸の統一    | P1 |

> **ここまでを“特徴量化の本体”として採用**し、
> 以前出していた「分解前の派生量」は今回の設計対象から外します（ご要望どおり）。

---

# 5) 学習モデル（Regression）候補一覧（係数 or 圧縮特徴を予測）

| Hydraグループ案   | 手法ID例                                | 目的         | 対象      | 優先    |
| ------------ | ------------------------------------ | ---------- | ------- | ----- |
| model/linear | `ridge` `lasso` `elasticnet`         | まず堅実に比較    | S/V（係数） | P0    |
| model/gp     | `gpr` `multioutput_gpr`              | 少量データ＋不確実性 | 係数/圧縮特徴 | P0–P1 |
| model/tree   | `random_forest` `xgboost` `lightgbm` | 非線形・頑健     | 係数/圧縮特徴 | P1    |
| model/nn     | `mlp`                                | 改良しやすいNN   | 係数/圧縮特徴 | P1    |
| model/nn     | `cnn_direct`                         | 条件→画像を直接予測 | 分布直予測   | P2    |

---

# 6) 評価・比較（共通化して必ず出すもの）

| Hydraグループ案   | 手法ID例                               | 内容           | 優先 |
| ------------ | ----------------------------------- | ------------ | -- |
| eval/metrics | `rmse_field` `mae_field` `r2_coeff` | 場誤差＋係数誤差     | P0 |
| eval/recon   | `reconstruct_and_diffmap`           | 再構成ヒートマップ＋差分 | P0 |
| eval/cv      | `kfold` `group_kfold`               | 手法比較の信頼性     | P1 |
| eval/report  | `summary_table`                     | 手法×指標の一覧CSV  | P0 |

---

# まとめ：今回の“再提示”で変えた点

* ✅ **「特徴量（派生量）生成」を分解前から撤去**
* ✅ **“特徴量化”＝「モード分解後係数の後処理」に統一**

  * PCA / SVD / Whitening / ICA / NMF / 係数選択 … を `coeff_post/*` として独立カテゴリ化
* ✅ Zernike / Fourier–Bessel / FFT(2D) / DCT を含めて、一次分解を網羅的に整理

---

もし次のステップに進めるなら、上の棚卸しをそのまま **Hydra config のグループ構造**に落とし込むために、

* `configs/decompose/*`
* `configs/coeff_post/*`
* `configs/model/*`
* `configs/tasks/*`（比較実験セット）

を具体的なファイル名・デフォルト構成（P0中心MVP）として提示できます。

「まずは **P0だけ**で基盤の最小実装スコープを確定したい」前提で、次に **P0の“推奨デフォルト組合せ（3〜5パターン）”**も出します。
