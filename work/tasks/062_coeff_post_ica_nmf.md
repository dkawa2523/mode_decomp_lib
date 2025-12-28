# Task 062 (P2): CoeffPost：ICA / FactorAnalysis / NMF（探索的）

## 目的
PCA以外の“主要成分抽出”として、独立成分（ICA）や非負成分（NMF）などを
係数後処理として追加し、状況に応じた表現を比較できるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/060_coeff_post_registry.md

## スコープ
### In
- `ica` coeff_post（FastICA）を実装（inverse可能）
- `factor_analysis` coeff_post を実装（inverseは近似）
- `nmf` coeff_post を実装（非負制約。入力整形が必要）
- 各手法の前提（非負、中心化など）を docs/20 に追記

### Out
- 非線形次元削減（UMAP等）はP3

## 実装方針（Codex向け）
### 1) ICA
- `sklearn.decomposition.FastICA`
- fitで unmixing/mixing を得る
- transform: Z = ica.transform(A_centered)
- inverse: A_hat = ica.inverse_transform(Z)

### 2) Factor Analysis
- `sklearn.decomposition.FactorAnalysis`
- inverse_transform は無いので、`Z @ components_ + mean_` のような近似を定義
- “可逆性”が弱いので、reconstruct用途なら注意（metricで検証）

### 3) NMF
- `sklearn.decomposition.NMF`
- Aが非負である必要がある
  - `A_shifted = A - A.min(axis=0) + eps` 等のshiftを入れる場合は、そのshiftを state に保存
- inverse: `Z @ W`（components）

### 4) 比較の注意
- ICA/NMFは初期値依存がある → seed固定を必須化
- PCAほど安定でないため P2 扱い

## ライブラリ候補
- scikit-learn（FastICA, FactorAnalysis, NMF）
- numpy

## Acceptance Criteria（完了条件）
- [ ] ica/nmf などが coeff_post registry に登録される
- [ ] seed固定で同じ結果が再現できる（少なくとも統計的に近い）
- [ ] inverse_transform（または近似inverse）が実装され、再構成誤差を評価できる

## Verification（検証手順）
- [ ] 合成データで ICA が独立成分をある程度回収できる
- [ ] NMFで非負制約が守られ、inverseが動く
