# Task 056 (P2): 不規則点：Graph Fourier（グラフラプラシアン固有）分解

## 目的
不規則サンプリング点や任意マスク領域の近似として、点をグラフとして扱い、
Graph Laplacian の固有ベクトルで分解（Graph Fourier Transform）する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md
- depends_on: work/tasks/030_data_domain_io.md

## スコープ
### In
- `graph_laplacian` decomposer を実装（points-domain優先）
- グラフ構築（kNN / radius / grid-neighbor）を config 化
- 固有分解（上位K）を行い、係数 = U^T f を返す
- state に eigenvectors/eigenvalues と node ordering を保存

### Out
- 巨大Nの高速固有分解はP3
- mesh上のcotan Laplacianは別タスク（Laplace–Beltrami）

## 実装方針（Codex向け）
### 1) 入力を points にする
- domain_spec.type == points を基本にする（N,2）
- rect/mask を points に変換する adapter を用意（P2）

### 2) グラフ構築
- kNN: 距離でk近傍を結ぶ
- 重み: gaussian weight `w_ij = exp(-||xi-xj||^2 / sigma^2)`
- config: k, sigma, symmetrize

### 3) ラプラシアン
- L = D - W（unnormalized） or L_sym（normalized）
- eigen decomposition: smallest K eigenvectors（低周波成分）

### 4) 係数・逆変換
- a = U^T f（Uは(N,K)）
- f_hat = U a
- mask/欠損がある場合は有効点のみで計算

### 5) 保存
- node ordering（sample内の点順）が係数の意味を決めるので必ず保存

## ライブラリ候補
- PyGSP（推奨：graph laplacian/eigsが揃う）
- scipy.sparse + scipy.sparse.linalg.eigsh（自前実装案）
- scikit-learn NearestNeighbors（kNN）

## Acceptance Criteria（完了条件）
- [ ] graph_laplacian decomposer が registry に登録される
- [ ] state に eigenvectors/eigenvalues が保存され、inverseが可能
- [ ] 合成の滑らかな場で低周波成分の再構成が妥当

## Verification（検証手順）
- [ ] 小さな点群で固有分解が通り、round-trip が動く
- [ ] kやKを変えると表現が変わることを確認
