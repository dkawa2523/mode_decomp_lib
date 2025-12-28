# Task 110 (P1): Benchmark Runner（Hydra multirunで組合せ比較）

## 目的
複数の `decompose × coeff_post × model × domain × preprocess` を自動で回し、
artifact（metrics/preds/config）を統一形式で出力して leaderboard で比較できるようにする。

## 依存関係
- depends_on: work/tasks/080_process_e2e.md
- depends_on: work/tasks/100_leaderboard.md

## スコープ
- `task=benchmark`（Hydra multirun前提）のProcessを追加
- 不適合組合せ（例：diskが必要なdecomposeにrectangle等）は明示skipし、理由を記録する
- 比較の真実は run dir の config + metrics（docs/04）

## Acceptance Criteria（完了条件）
- [ ] `python -m processes.benchmark -m ...` で複数runが生成される
- [ ] 各runで metrics が保存され、leaderboard が集計できる
- [ ] skip した組合せがログ/metricsに理由付きで残る

## Verification（検証手順）
- [ ] 2×2×1 程度の小さなsweepを回して run dir が増えることを確認
- [ ] leaderboard が複数runを読み込み表にできることを確認

## Autopilotルール（重要）
**DO NOT CONTINUE**: benchmark と leaderboard が実際に動くまで `done` にしない。
