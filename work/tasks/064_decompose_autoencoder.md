# Task 064 (P2): 非線形分解：AutoEncoder/VAE（PyTorch）

## 目的
線形基底では表現しづらい非線形構造を扱うため、AutoEncoder（必要ならVAE）を「分解器」として導入する。
AEのlatent `z` を係数の代替として扱い、cond→z 回帰と decoder による再構成で評価できるようにする。

## 依存関係
- depends_on: work/tasks/070_models_sklearn_baseline.md
- depends_on: work/tasks/020_hydra_mvp.md

## Acceptance Criteria（完了条件）
- [ ] AEが学習でき、latent_dimが設定化される
- [ ] AE encoder/decoder を artifact として保存できる
- [ ] cond→z（回帰）+ decoder による再構成評価が可能

## Verification（検証手順）
- [ ] tiny datasetでAE単体の再構成が改善する
- [ ] cond→z の回帰と組み合わせても end-to-end が完走する

## Autopilotルール（重要）
**DO NOT CONTINUE**: 受け入れ条件を満たすまで `done` にしない。
