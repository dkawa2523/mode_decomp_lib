# Task 061 (P0): CoeffPost：PCA / TruncatedSVD / Whitening の導入

## 目的
分解係数の次元を圧縮し、学習の安定化と主要成分抽出を行うため、
PCA/TruncatedSVD（必要ならwhitening）を CoeffPost として実装する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/060_coeff_post_registry.md

## スコープ
### In
- `pca` coeff_post を実装（fit/transform/inverse_transform）
- （任意）`truncated_svd` coeff_post を実装（大規模向け）
- n_components の指定と、explained_variance による自動決定（energy_threshold）を実装
- state 保存/ロードを artifact 契約に従って実装

### Out
- ICA/NMF（次タスク）

## 実装方針（Codex向け）
### 1) PCA
- 学習データ `A_train`（N,K）で `sklearn.decomposition.PCA` を fit
- `Z = pca.transform(A)`
- inverse: `A_hat = pca.inverse_transform(Z)`
- state: pca object + 係数名/meta（Kの意味）は別途 coeff_meta で保持

### 2) Whitening
- PCAの `whiten=True` を使うか、`Z` を分散1にスケール
- 注意：whitenすると inverse で元分散に戻るが、数値誤差が増える場合あり

### 3) TruncatedSVD
- 平均を引かないSVD。大規模Kに向く
- 係数の平均を別で引く場合は state に mean を保存して整合させる

### 4) 自動次元選択
- `n_components` を固定できる
- または `energy_threshold=0.99` などで explained variance ratio の累積で決める
- ただし比較可能性のため、選ばれた次元K2を meta に保存する

### 5) 注意
- 入力が複素（FFT）なら、先に `complex_to_mag_phase` などで実数化してから PCA

## ライブラリ候補
- scikit-learn（PCA, TruncatedSVD）
- numpy
- joblib（state保存）

## Acceptance Criteria（完了条件）
- [ ] pca coeff_post が registry に登録される
- [ ] fit/transform/inverse_transform が動作する
- [ ] energy_threshold による自動次元選択ができ、選択結果が保存される

## Verification（検証手順）
- [ ] 合成Aで round-trip 誤差が小さい（unit test）
- [ ] tinyデータで coeff_post.fit -> transform が動く
