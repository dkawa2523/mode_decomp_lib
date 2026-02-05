# P2手法の artifacts 方針（幾何・DL・vector）

docs/04（artifact契約）を崩さず、P2の手法で増える保存対象を整理する。

## 共通（必須）
- config: `configuration/run.yaml`
- meta: 実行環境、seed、git hash、データ識別子、domain種別
- metrics: coeff/field + 追加指標（div/curl等）
- preds: 予測係数、必要なら std
- model: 回帰モデル（`model/`） + decomposer/coeff_post state（`outputs/states/`）

## Graph/LB decomposer
- outputs/states/decomposer/ に保存（推奨）:
  - graph_laplacian: eigenvalues.npy, eigenvectors.npy, (optional) node_index.npy
  - laplace_beltrami: eigenvalues.npy, eigenvectors.npy, mesh_signature.json
- mesh自体（V,F）はデータ側にある想定。
  - ただし再現性が重要なら、toy mesh は outputs にコピーして良い（サイズに注意）

## Autoencoder decomposer (VAE TBD)
- outputs/states/decomposer/:
  - torch weights（.pt）
  - architecture params（json）
  - training curve（metricsに）

## Dictionary Learning（coeff_post）
- outputs/states/coeff_post/:
  - dictionary.npy
  - sparse_code_params.json

## Helmholtz / vector metrics
- metrics:
  - div_norm, curl_norm（true/pred両方）
- plots:
  - div_map, curl_map（必要なら）
