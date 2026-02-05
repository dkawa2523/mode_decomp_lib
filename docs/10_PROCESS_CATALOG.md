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
### 1) decomposition
- In: field + mask + domain + preprocess + decomposer
- Out: coeffs.npz（cond + coeff）、preds.npz（recon field）、metrics.json、plots

### 2) preprocessing
- In: decomposition の coeffs.npz
- Out: coeff_post state、coeffs.npz（coeff_z）、metrics.json、plots
- Note: decomposer によって `coeff_post` が自動で無効化される場合がある（例: POD系でのPCA）。強制する場合は `coeff_post.force: true`。

### 3) train
- In: preprocessing の coeffs.npz
- Out: model artifact、metrics.json、plots
- Train 設定（`train.*`）で val split / CV / 簡易チューニング / 可視化を制御

### 4) inference
- In: train model + preprocessing coeff_post + decomposition states
- Out: preds.npz（coeff + field）、metrics.json、plots
- Optuna を使う optimize では `outputs/optuna_trials.csv` / `outputs/optuna_best.json` と可視化を追加

### 5) pipeline
- In: task.decompose_list / task.coeff_post_list / task.model_list
- Out: decomposition/ train leaderboard
- `task.stages` で decomposition/preprocessing/train の範囲を制御

### 6) leaderboard
- In: 複数run dir
- Out: 集計CSV/Markdown

### 7) doctor
- In: config
- Out: 環境/データ/最小smoke結果

---

## 補足: preprocess は内部ステップ
- preprocess（field前処理）は decomposition 時に fit/transform され、inference で inverse が適用される
- state は `outputs/states/preprocess/state.pkl` に保存

## P1以降のProcess
- domain変換（points/mesh）
