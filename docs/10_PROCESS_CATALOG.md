# Process Catalog（Process一覧とI/O）

本プロジェクトでは、すべての作業を Process として切り出し、単独実行可能にする。

---

## Entrypoints（現行repo）
- run.yaml 1枚運用: `python -m mode_decomp_ml.run --config run.yaml`（`src/mode_decomp_ml/run.py`）
- Hydra直呼び: `python -m mode_decomp_ml.cli.run ...`（`src/mode_decomp_ml/cli/run.py`）

---

## Data Schema (P0)
- cond: shape [D]
- field: shape [H, W, C] (scalar: C=1, vector: C=2)
- mask: shape [H, W] (optional; mask外は評価対象外)
- meta: dict（domain種、スケールなど）

## P0で必須のProcess
### 1) decompose.transform
- In: field + mask + domain
- Out: coeff `a` + `coeff_meta.json`

### 2) coeff_post.fit / transform
- In: train coeff A_train
- Out: z_train（state保存）, z_val, z_test

### 3) train
- In: condition + z_train（または a_train）
- Out: model artifact

### 4) predict
- In: condition + model
- Out: z_hat（または a_hat）

### 5) reconstruct
- In: z_hat + coeff_post state + decomposer state
- Out: field_hat

### 6) eval
- In: field_hat + field
- Out: metrics.json

### 7) viz
- In: run dir（preds.npz/metrics.json）
- Out: viz images + summary tables
- Note: process層がI/Oとrun_dirを扱い、描画ロジックは `mode_decomp_ml/viz` に集約
  - domain-aware 追加図: `figures/domain/`（disk/annulus の polar、sphere_grid の投影、mesh の頂点表示）
  - sphere_grid 投影は `viz.sphere_projection: plate_carre | mollweide` で切替

### 8) leaderboard
- In: 複数run dir
- Out: 集計CSV/Markdown

### 9) doctor
- In: config
- Out: 環境/データ/最小smoke結果

---

## 補足: preprocess は内部ステップ
- preprocess は train 時に fit/transform され、reconstruct/viz で inverse が適用される内部ステップ
- state は `states/preprocess/state.pkl` に保存

## P1以降のProcess
- multirun benchmark（method sweep）
  - 例: `python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none,pca`
  - 各runは `runs/<tag>/<run_id>/` に保存
- domain変換（points/mesh）
