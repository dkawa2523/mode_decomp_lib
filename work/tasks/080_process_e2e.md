# Task 080 (P0): Process実装：train/predict/reconstruct/eval の一連を通す

## 目的
前処理→分解→係数後処理→学習→推論→逆変換→評価 の “比較可能な一本の線” を作る。
以降の手法追加はこの線にプラグインを挿すだけで済むようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/040_preprocess_pipeline.md
- depends_on: work/tasks/051_decompose_zernike.md
- depends_on: work/tasks/061_coeff_post_pca.md
- depends_on: work/tasks/070_models_sklearn_baseline.md

## スコープ
### In
- Process: preprocess, decompose_transform, coeff_post_fit, train, predict, reconstruct, eval を実装
- artifact契約（docs/04）に従って保存
- evalで field-space metric と coeff-space metric を出力
- 最小の end-to-end run を tiny dataset で成立させる

### Out
- leaderboard/viz の充実（次タスク）

## 実装方針（Codex向け）
### 1) 実行順序（例）
1. preprocess: raw -> cleaned
2. decompose_transform: cleaned -> coeff A
3. coeff_post_fit: A_train -> coeff_post state
4. coeff_post_transform: A -> Z（train/val/test）
5. train: condition_train -> Z_train
6. predict: condition_test -> Z_hat
7. reconstruct: Z_hat -> A_hat -> field_hat
8. eval: field_hat vs field（mask内）

### 2) データの整合
- sample_id の順序を常に保存（indexファイル）
- A/Z の行が sample_id に対応していることを保証

### 3) artifact
- 各Processが自分のrun dirに出す
- 次工程が前工程のrun dirを参照する方式にするか、
  “pipeline runner” を作って1つのrun dirにまとめるかを決める
  - P0は “pipeline runner” を推奨（比較が楽）

### 4) eval指標
- RMSE/relL2（mask内）
- 係数RMSE（同じcoeff_metaのときのみ）
- explained variance（PCAなら）

## ライブラリ候補
- numpy
- pandas
- scikit-learn（metrics）

## Acceptance Criteria（完了条件）
- [ ] tinyデータで end-to-end が1コマンドで完走する
- [ ] run dir に meta/config/model/metrics/preds が揃う
- [ ] 同じseedで結果が再現する（少なくとも数値が近い）

## Verification（検証手順）
- [ ] `python -m processes.pipeline_run ...`（または同等）で完走
- [ ] 出力の `metrics.json` が読みやすい形で保存される
