# Task 070 (P0): 学習モデル（Regressor）: scikit-learn回帰ベースラインの実装

## 目的
まず比較の基準となる頑健な回帰ベースライン（Ridge等）を実装し、
条件→係数（または潜在z）を学習・推論できるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/020_hydra_mvp.md
- depends_on: work/tasks/061_coeff_post_pca.md

## スコープ
### In
- `Regressor` base class と registry を実装
- P0: Ridge / ElasticNet を実装（multi-output対応）
- 目的変数として `target_space: {coeff, latent}` を config 化
- model artifact の save/load を実装（joblib）
- train/predict Process から呼び出せるようにする

### Out
- 深層学習モデルの本格導入（次タスク）

## 実装方針（Codex向け）
### 1) multi-output の扱い
- 係数次元 K が大きいので multi-output が前提
- sklearnでは：
  - `Ridge` は multi-target でそのまま扱える
  - `Lasso` 等は `MultiOutputRegressor` が必要な場合あり
- configで `multioutput_wrapper: none|multioutput` を選べるようにする

### 2) 入力 X（condition）
- `X = condition_table[sample_id]` を統一
- 可能なら `ConditionEncoder` を入れて前処理（標準化など）
- conditionの列順序を `condition_meta.json` として保存（比較可能性）

### 3) 出力 Y
- `target_space=coeff`: Y = A（係数）
- `target_space=latent`: Y = Z（PCA等後）
- どちらも `predict` の戻りは同じ型（np.ndarray）に統一

### 4) 保存
- `artifacts/model/model.pkl`
- `artifacts/model/model_meta.json`（target_space, input columns, K等）

## ライブラリ候補
- scikit-learn（Ridge, ElasticNet, MultiOutputRegressor）
- joblib
- numpy/pandas

## Acceptance Criteria（完了条件）
- [ ] ridge baseline が registry に登録される
- [ ] train -> predict が動作し、artifactに保存される
- [ ] target_space を切り替えても同じフローで走る

## Verification（検証手順）
- [ ] tinyデータで train/predict が完走し、preds が保存される
- [ ] 同じseedで同じ結果になる（少なくともRidge）
