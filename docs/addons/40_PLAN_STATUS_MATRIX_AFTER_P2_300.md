# 計画→残件マトリクス（After P2-300）

現在の P2 queue の進捗を、設計・計画（PRロードマップ）に対応づけて整理します。
この表は “何が残っているか” をレビュー/計画会議で即座に共有するためのものです。

## P2タスク進捗（あなたの提示ログベース）

|ID|Priority|Status|Title|
|---|---|---|---|
|299|P2|done|P2 Preflight（依存/計算コスト/範囲合意）|
|300|P2|done|Decomposer: Graph Laplacian eigenbasis（任意mask）|
|310|P2|in_progress|Domain+Decomposer: Mesh + Laplace–Beltrami（曲面）|
|320|P2|todo|Decomposer: Autoencoder/VAE（Torch）|
|330|P2|todo|Decomposer/CoeffPost: Dictionary Learning（疎表現）|
|340|P2|todo|Vector Field: Helmholtz (div/curl) 分解|
|350|P2|todo|Tracking: ClearML統合（準備/最小）|
|390|P2|todo|P2 Cleanup（整理・docs整合・実行例）|

## 残件の最短順序（推奨）
1. 310（Laplace–Beltrami）を完了（mesh domain + decomposer の “最小” を固定）
2. 320（AE/VAE）: latent decomposer を導入（toyでOK、過剰最適化しない）
3. 330（Dictionary Learning）: **coeff_post** として導入（train-only fit / inverse）
4. 340（Helmholtz）: vector場の物理解釈指標（div/curl）を評価・可視化へ統合
5. 350（ClearML）: 最小統合（無ければ落ちない）
6. 395（artifact validator）: 契約逸脱の早期検知
7. 398（release-ready）: 依存/コマンド/再現性を固定
8. 390（cleanup）: 不要物削除＋docs整合＋実行例

> 390 を最後に置く理由：390 は “整理” なので、機能が増えた後に実施した方が
> 余分な削除や再作業が減り、結果的に最短になります。
