# 23. Methods Add-on Spec（追加手法の仕様）

本ドキュメントは、追加する以下の手法を “既存コードと衝突せず” “拡張しやすく” 導入するための仕様メモです。

対象:
- Decomposer（解析基底）: Annular Zernike / Spherical Harmonics / Slepian / Wavelet2D / PSWF
- CoeffPost: Quantile / Power(Yeo-Johnson)
- Codec: FFT複素表現統一
- Model: XGB/LGBM/CatBoost / MultiTaskLasso / Multi-task GP

---

## 1) Decomposer + Codec

### 1.1 Annular Zernike
- domain: `annulus`（center, r_inner, r_outer）
- params: `n_max`, `m_max`, `normalization`, `boundary_condition`
- state: なし（解析基底）
- codec: `zernike_pack_v1`（(n,m,cos/sin)順序をmetaで固定）
- test: roundtrip（係数→再構成）+ 既存zernikeとの互換チェック（r_inner=0で一致する範囲）

依存:
- numpy/scipy（多項式・積分）

---

### 1.2 Spherical Harmonics（球面調和）
- domain: `sphere_grid`（lat_range / lon_range / angle_unit / radius）
- params: `l_max`, `real_form`, `norm`, `backend`
- state: なし（解析基底）
- codec: `sh_pack_v1`（(l,m)順序固定）
- test: roundtrip + 既知関数（低次Y_lm）再現

依存（backendで切替）:
- `backend=pyshtools` の場合: `pyshtools`（optional dependency）
- `backend=scipy` の場合: 追加依存なし（scipyは必須）

---

### 1.3 Slepian（領域集中基底）
- domain: `sphere_grid`
- params: `l_max`, `region_mask`（dataset mask or cap）, `k`
- state: basis（eigenvalues + eigenvectors）
- codec: `slepian_pack_v1`（k次元）
- test: basis集中度（eigenvaluesの単調性など）+ roundtrip

依存:
- `pyshtools`（Slepian実装）
- optional dependency として扱い、未導入なら registry から除外 or 明確エラー

---

### 1.4 Wavelet2D
- domain: `rectangle`（disk/mask は “0埋め+mask評価” で利用可能）
- params: `wavelet`（db2等）, `level`, `mode`, `mask_policy`（error / zero_fill）
- state: なし（解析基底、ただしwavelet選択はmetaに保存）
- codec: `wavelet_pack_v1`（階層係数を flatten、metaにshape）
- test: roundtrip（lossless）+ 係数数 L の一貫性

依存:
- `PyWavelets`（pywt）

---

### 1.5 PSWF（Prolate Spheroidal）—研究枠
現実解: まず `pswf2d_tensor`（rectangle向け、1D PSWFの外積）で導入。
- domain: rectangle
- params: `c_x`, `c_y`, `n_x`, `n_y`, `mask_policy`
- state: なし（解析基底）
- codec: `tensor_pack_v1`
- test: 近似的なroundtrip（許容誤差を明示）
- note:
  - 現行は **DPSS（離散PSWF）** の外積で近似（`c_x/c_y` は DPSS の time-bandwidth として扱う）
  - mask domain は原則非対応（`mask_policy=zero_fill` は研究用途のみ）
  - 係数順は `(y, x)` の row-major（meta の `mode_axes/mode_shape` を参照）
  - 将来: 厳密PSWF（eigenbasis系）への拡張余地を残す

依存:
- SciPy（`scipy.signal.windows.dpss`）
- optional dependency 推奨（精度/安定性検証が必須）

---

## 2) CoeffPost（係数後処理）

### 2.1 QuantileTransform
- params: `output_distribution`, `n_quantiles`, `subsample`
- fit: train-only
- inverse: 可能（端部は近似になることがある）
- test: fit/transform/inverse の挙動 + 分布変換の妥当性

依存:
- scikit-learn

### 2.2 PowerTransform（Yeo-Johnson）
- params: `standardize`
- fit: train-only
- inverse: 可能
- test: fit/transform/inverse の数値安定性

依存:
- scikit-learn

---

## 3) Codec（FFT複素表現統一）
- codec: `fft_complex_codec_v1`
- modes:
  - `real_imag`（lossless）
  - `mag_phase`
  - `logmag_phase`
- test: lossless mode は roundtrip を保証

依存:
- numpy

---

## 4) Model

### 4.1 GBDT（XGB/LGBM/CatBoost）
- 方針: `IndependentMultiOutputWrapper` で多出力を統一
- optional dependency:
  - xgboost / lightgbm / catboost
- test: 小データで train/predict が通る + shape契約

### 4.2 MultiTaskLasso
- native multi-output
- scikit-learn
- test: train/predict + sparsityが働く簡単例

### 4.3 Multi-task GP（相関）
- 目的: 出力間相関を学習（P2相当）
- 推奨: GPyTorch で実装（optional）
- test: 小さな toy で train/predict 通過（重いので CI ではスキップも可）

---

## 共通の受け入れ基準（追加手法すべて）
- registry に登録され、`--list-plugins` 等で列挙可能
- run.yaml から選択できる
- artifact（states/）に必要な meta/state を保存する
- benchmark sweep に最低1ケース追加し、比較可能性を壊さない
