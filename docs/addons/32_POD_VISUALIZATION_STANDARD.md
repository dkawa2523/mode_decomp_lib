# Addon after Task 492: Standard Visualizations for Data-driven Decomposition (POD)

POD系を追加すると「何が良くなったか」が分からなくなりがちです。
レビューと現場意思決定のため、以下を標準図として固定します。

## 必須図（figures/ に固定名で保存）
1) `scree.png`：固有値（eigs）/ 特異値スペクトル
2) `energy_cum.png`：累積寄与率（K選定根拠）
3) `recon_error_vs_k.png`：再構成誤差 vs K（train/val/test）
4) `modes_gallery.png`：上位Kモードの可視化（例：k=1..9）
5) `coeff_hist.png`：主要係数の分布（歪み・外れ値の確認）
6) `cond_coeff_corr.png`：condとcoeffの相関ヒートマップ（回帰の見込み）

## Gappy POD 追加図
7) `gappy_mask_coverage.png`：観測点（mask）分布の概要
8) `gappy_recon_compare.png`：観測部分/欠損部分での誤差を分けた比較

## 実装上の注意
- “可視化が増えるほど設定が増える”を避けるため、図は `viz.enable=true` で一括ON/OFFにする
- 図のサンプル数は固定（例：最大8サンプル）にして出力肥大を避ける
- 既存の特殊関数 decomposer でも同じ図が出せるものは共通化する

TODO(Task493): 現行 `src/processes/viz.py` の出力は
`coeff_spectrum.png`, `coeff_hist.png`, `coeff_topk_energy.png`, `field_compare.png`,
`error_map.png`, `recon_sequence.png` などで、本標準の
`scree.png` / `energy_cum.png` / `recon_error_vs_k.png` / `modes_gallery.png` は未実装（Task508）。

---
