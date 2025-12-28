# Task 060 (P0): 係数後処理（CoeffPost）の共通インターフェースとパイプライン

## 目的
“特徴量化”を **分解後係数に対する後処理** として固定し、
PCA等のfitが必要な処理を train/serve skew なしで扱える枠を作る。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md
- depends_on: work/tasks/020_hydra_mvp.md

## スコープ
### In
- `CoeffPost` base class（fit/transform/inverse_transform）を実装
- registry（method key -> class）を実装しHydraから選択できる
- `coeff_post.fit` と `coeff_post.transform` Process を実装
- P0として `normalize(per_mode)` と `complex_to_mag_phase` を実装

### Out
- PCA/ICA/NMFの実装（次タスク）

## 実装方針（Codex向け）
### 1) 入出力
- 入力：coeff matrix `A` shape (N, K)（サンプル×係数次元）
- 出力：`Z` shape (N, K2)
- `inverse_transform` で `A_hat` を復元できること（学習→再構成で必要）

### 2) skew禁止のための設計
- `fit` は train split だけで実行し state を保存
- predict時は state をロードして `transform` のみ実行
- これを `processes/` に明示的に分ける（fit/transformを混ぜない）

### 3) P0実装
- per_mode_standardize:
  - 各係数次元ごとに mean/std を計算して標準化
  - state: mean,std
- complex_to_mag_phase（FFT系で必要）:
  - 複素係数 A_complex を `[mag, phase]` に変換して実数化
  - inverse: mag/phase -> complex

### 4) 保存
- state は `artifacts/coeff_post/state.pkl`
- 変換後のZも必要なら保存（train用）

## ライブラリ候補
- numpy
- scikit-learn（PCA等は次タスク）
- joblib/pickle（state保存）

## Acceptance Criteria（完了条件）
- [ ] coeff_post を config で切替できる（少なくとも normalize が動く）
- [ ] fit状態が保存され、推論時にロードしてtransformできる
- [ ] complex係数を実数特徴へ変換できる（FFTの前提）

## Verification（検証手順）
- [ ] 合成Aで fit->transform->inverse_transform の誤差が小さい
- [ ] trainとpredictで同じstateが使われることをログで確認
