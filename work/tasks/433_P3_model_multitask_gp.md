# Task: 433 Add: Model Multi-task GP（相関学習, optional dependency, P2枠）

- Priority: P2
- Status: todo
- Depends on: 430, 410
- Unblocks: 440, 490

## Intent
出力（latent）の相関を学習できる Multi-task GP を P2相当の高度枠として追加する。
ただし計算コスト・依存が大きいので optional dependency とし、最小構成で導入する。

## Context / Constraints
- pipeline は (N,L) を前提（タスク430）
- 依存候補: GPyTorch（推奨） or 他GPライブラリ
- heavy になりやすいので、まずは latent 次元を小さくする前提（PCA/PLSなど）
- CI ではスキップできるように設計（pytest marker）

## Plan
- [ ] `model=mtgp` を追加（optional）
- [ ] 最小の train/predict 実装（toyで動くこと）
- [ ] `predict_std` を提供（任意）し、artifactに保存する設計にする
- [ ] docs: “使いどころ” と “コスト注意” を明記
- [ ] examples: small latent（例: pca(n=8)）前提のrun.yaml

## Acceptance Criteria
- [ ] 依存あり環境で mtgp が選べて動く（toy/小規模）
- [ ] 依存なし環境で明確にガイドされる
- [ ] predict_std が出る場合は preds.npz に保存される

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_disk_zernike_pca_mtgp.yaml --dry-run`
- Expected:
  - 実行可 or 明確な依存ガイド
