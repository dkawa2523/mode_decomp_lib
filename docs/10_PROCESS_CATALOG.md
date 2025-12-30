# Process Catalog（Process一覧とI/O）

本プロジェクトでは、すべての作業を Process として切り出し、単独実行可能にする。

---

## Data Schema (P0)
- cond: shape [D]
- field: shape [H, W, C] (scalar: C=1, vector: C=2)
- mask: shape [H, W] (optional; mask外は評価対象外)
- meta: dict（domain種、スケールなど）

## P0で必須のProcess
### 1) preprocess
- In: raw field + mask + domain
- Out: cleaned field + mask（artifact保存）

### 2) decompose.transform
- In: field + mask + domain
- Out: coeff `a` + `coeff_meta.json`

### 3) coeff_post.fit / transform
- In: train coeff A_train
- Out: z_train（state保存）, z_val, z_test

### 4) train
- In: condition + z_train（または a_train）
- Out: model artifact

### 5) predict
- In: condition + model
- Out: z_hat（または a_hat）

### 6) reconstruct
- In: z_hat + coeff_post state + decomposer state
- Out: field_hat

### 7) eval
- In: field_hat + field
- Out: metrics.json

### 8) viz
- In: run dir（preds/metrics）
- Out: viz images + summary tables

### 9) leaderboard
- In: 複数run dir
- Out: 集計CSV/Markdown

### 10) doctor
- In: config
- Out: 環境/データ/最小smoke結果

---

## P1以降のProcess
- multirun benchmark（method sweep）
  - 例: `python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none,pca`
  - 各runは `outputs/benchmark/<date>/<time>_<tag>_<jobnum>/` に保存
- domain変換（points/mesh）
