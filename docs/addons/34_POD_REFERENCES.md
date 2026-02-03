# References & Research Notes (POD suite addon)

このファイルは実装の根拠・参照先をまとめたものです（Task 493 の調査メモ用）。

## ライブラリ（一次情報）
- scikit-learn `PCA` / `IncrementalPCA`
  - PCA: `svd_solver="full"|"randomized"`、`whiten`、`components_`、`explained_variance_`
  - IncrementalPCA: out-of-core、batch_size指定
- scikit-learn `randomized_svd`（高速SVD）
- modred `pod.compute_POD_arrays_snaps_method(...)`
  - `inner_product_weights` により重み付き内積を実現
  - `atol` などのtruncation指定

## 代表的な文献
- Randomized SVD / Randomized low-rank approximation
  - N. Halko, P.-G. Martinsson, J. A. Tropp, *Finding structure with randomness*, SIAM Review, 2011.
- Gappy POD（欠損/部分観測）
  - K. Willcox, *Unsteady Flow Sensing and Estimation via the Gappy POD*, 2004（MIT資料）
  - T. Bui-Thanh, K. Willcox ほか、gappy PODの応用（逆設計/再構成）
- POD / ROMの基礎
  - POD（=PCA）およびスナップショット法の標準的解説（多数）

## 実装上の注意（失敗しやすい点）
### A) Weighted POD の「重み」の入れ方
- 目標は、離散内積 `<u,v>_W = u^T W v` を domain が決めること
- rectangle/disk/mask：点重み（Wの対角）で十分なことが多い
- mesh：質量行列M（疎行列）を想定。v1は hook と最小対応でOK

### B) Gappy POD の安定性
- 観測点配置（mask）が悪いと最小二乗が不安定
- 正則化λ（Tikhonov）を options でON/OFFできるようにする
- 観測領域と欠損領域の誤差を分けて評価し、問題箇所が分かるようにする

### C) Randomized SVD の再現性
- `random_state` を state と metrics に必ず保存
- oversampling / power iteration などを増やすと精度は上がるが設定爆発しやすい
  - v1ではデフォルトで十分、必要になったら advanced option として露出

### D) Incremental POD の差異
- full PCA と一致しない（近似）
- `fit` と `partial_fit` は更新規約が異なり、同一結果にならない可能性
- batch_size とデータ順序に依存する可能性 → seed とシャッフル規約を固定

---
