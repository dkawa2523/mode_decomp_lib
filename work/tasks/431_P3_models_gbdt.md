# Task: 431 Add: Model GBDT（XGB/LGBM/CatBoost, optional, multioutput wrapper）

- Priority: P2
- Status: todo
- Depends on: 430, 410
- Unblocks: 440, 490

## Intent
条件テーブル(cond)→係数(latent)の非線形ベースラインとして、
XGBoost / LightGBM / CatBoost をモデルプラグインとして追加する（optional dependency）。

## Context / Constraints
- 多出力は `IndependentMultiOutputWrapper` で統一（タスク430）
- optional dependency を明確化し、未導入環境では registry から除外 or 明確エラー
- まずは “動く比較ベースライン” を最優先（最適化は後）

## Plan
- [ ] `model=xgb`, `model=lgbm`, `model=catboost` を追加
- [ ] それぞれ最小パラメータ（seed、depth、n_estimators等）を safe default 化
- [ ] 依存が無い場合のエラー文を統一（インストール方法を提示）
- [ ] tests: toyデータで train/predict が通る（依存あり環境）
- [ ] docs: いつ有効か（非線形/相互作用/スケール頑健）を短く追記

## Acceptance Criteria
- [ ] 3モデルが registry に追加され、run.yaml から選べる
- [ ] optional dependency 不在時に分かりやすいエラー
- [ ] 多出力（L>1）で学習・推論が通る（小規模でOK）

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike_pca_xgb.yaml --dry-run`
- Expected:
  - 依存ありなら実行できる、無ければ導入指示が出る
