# Addon after Task 492: ClearML-ready notes (no integration yet)

## 目的
将来、ClearMLで「データセット管理 / 分解 / 係数後処理 / 学習 / 推論 / 評価」を別Taskとして管理するための“切り口”を壊さない。

## 推奨のTask粒度（将来）
- prepare_dataset（manifest検証、hash付与）
- fit_decomposer（POD等のfitが必要なもの）
- fit_coeff_post（Quantile/Power/PCAなど train-only fit）
- train_model
- inference_coeff
- inference_field
- evaluate
- visualize
- pipeline_sweep

## v1でやること（設計のみ）
- artifact契約により、各stepが I/O だけで再実行可能な状態を維持
- state（decomposer/coeff_post）を `outputs/states/` に保存し、次stepが参照できるようにする
- `options.*` は、将来 step を分離しても同じ意味で解釈されるように命名を固定する

TODO(Task493): 現行 run dir は `runs/<name>/<process>/`（`RunDirManager`）で固定され、
`configuration/run.yaml` が保存される（`src/mode_decomp_ml/run.py` / `cli/run.py`）。
ClearML task分割時は `outputs/states/` と `outputs/preds.npz` を境界にI/Oを切る想定で整合させる。
