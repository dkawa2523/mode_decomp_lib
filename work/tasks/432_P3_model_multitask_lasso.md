# Task: 432 Add: Model MultiTaskLasso（native multi-output, sparsity）

- Priority: P2
- Status: done
- Depends on: 430
- Unblocks: 440, 490

## Intent
MultiTaskLasso を追加し、係数/潜在の “疎な共通因子” を仮定した説明性のある回帰を提供する。

## Context / Constraints
- Native multi-output モデルとして実装（wrapper不要）
- Ridgeとの差分（疎性/特徴選択）を docs に明記する
- デフォルトは過度に強くしない（alphaは小さめ、CVは後回し）

## Plan
- [x] `model=multitask_lasso` を追加（scikit-learn）
- [x] params: alpha, max_iter 等を最小化
- [x] tests: toyデータで疎性が効く簡単例 + train/predict shape
- [x] examples: rect+fft+pca+multitask_lasso の run.yaml

## Acceptance Criteria
- [x] registry に追加され、run.yaml から選べる
- [x] L>1 の多出力で学習・推論が通る
- [x] docs に「いつ有効か」が書かれている

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_pca_mtlasso.yaml`
- Expected:
  - metrics.json が生成される
