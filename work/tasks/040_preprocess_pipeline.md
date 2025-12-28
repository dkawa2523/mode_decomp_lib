# Task 040 (P0): 前処理パイプライン（mask安全・順序固定・Hydra切替）

## 目的
分解精度と学習精度を安定させるため、前処理をプラグイン化し、
maskを壊さずに順序通り適用できるパイプラインを実装する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/030_data_domain_io.md

## スコープ
### In
- `PreprocessOp` registry を実装し、configでリスト指定できるようにする
- P0として最低限のopsを実装（scale/missing/outlier/denoise/detrend/resample）
- 各opは FieldSample を入力し FieldSample を返す（mask維持）
- 前処理のパラメータと順序は artifact として保存

### Out
- 学習時だけのaugmentation（P1で追加）
- GP補間など重い補間（P1で追加）

## 実装方針（Codex向け）
### 1) パイプラインの設計
- `PreprocessPipeline(ops: List[PreprocessOp])`
- `pipeline.apply(sample) -> sample2`
- 各opは `fit(dataset)` を持っても良いが、P0では基本 `fit` 不要で実装

### 2) mask安全のルール
- 欠損補間は「mask=False領域」を埋めてもよいが、mask自体は保存
  - 例：inpaintで値を埋めるが、maskは別で保持して評価時にmask内だけ評価
- ただし、`mask_policy` を config で選べるようにする
  - `keep`: maskを保持（デフォルト）
  - `expand`: 補間した領域を有効化（明示した時のみ）

### 3) P0 ops（例）
- scale: standardize / robust_scale
- missing: nearest_fill（簡易）
- outlier: mad_clip
- denoise: gaussian / median
- detrend: remove_mean / remove_plane
- resample: to_polar_grid（disk用）、to_cart_grid（points->grid等）

### 4) artifact
- run dir に preprocess config を保存（Hydraで自動）
- 追加で `preprocess_report.json`（各opの統計）を出す

## ライブラリ候補
- numpy
- scipy（signal / ndimage / interpolate）
- scikit-image（任意：inpaint等）

## Acceptance Criteria（完了条件）
- [ ] preprocess pipeline が config で順序指定できる
- [ ] maskが壊れない（keep policyで維持される）
- [ ] tiny dataset で preprocess -> decompose が通る

## Verification（検証手順）
- [ ] `python -m processes.preprocess preprocess=basic` で出力artifactが生成される
- [ ] 前処理前後で shape と mask が期待通りである（ログ/テスト）
