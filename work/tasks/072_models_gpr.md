# Task 072 (P1): 学習モデル：Gaussian Process Regression（小データ・非線形）

## 目的
小データ領域や非線形性が強い条件→係数（latent）関係に対応するため、GPRを追加する。
推奨は `cond -> z`（PCA後の低次元latent）を予測し、inverseで `a_hat`→再構成へ戻す運用。

## 依存関係
- depends_on: work/tasks/061_coeff_post_pca.md
- depends_on: work/tasks/070_models_sklearn_baseline.md

## ライブラリ候補
- scikit-learn GaussianProcessRegressor
- kernels: RBF, Matern, WhiteKernel

## Acceptance Criteria（完了条件）
- [ ] `gpr` model が registry に登録される
- [ ] kernelをHydraで切替できる
- [ ] 多出力は latent 次元ごとに独立GP（P1）で動く
- [ ] 予測meanを保存し、可能ならstdもartifactに残す

## Verification（検証手順）
- [ ] tiny dataset で train/predict が完走する
- [ ] Ridgeより良い例（合成非線形）で改善を確認できる（可能なら）

## Autopilotルール（重要）
**DO NOT CONTINUE**: 受け入れ条件を満たすまで `done` にしない。
