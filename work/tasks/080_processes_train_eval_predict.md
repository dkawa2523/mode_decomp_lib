# Process: train/predict/reconstruct/eval（artifact契約）

**ID:** 080  
**Priority:** P0  
**Status:** todo  
**Depends on:** 070  
**Unblocks:** 090  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
プロジェクトのCLI入口（process）を揃え、artifact契約どおりに保存する。

必要process:
- train: model学習（必要ならcoeff_post/decomposerもfit）
- predict: cond→(a|z)予測
- reconstruct: 予測係数→field_hat
- eval: metrics出力（coeff/field）
- doctor: 環境/最小実行の健全性チェック

artifact（docs/04）:
- config（.hydra）
- meta（実行条件、git hash等）
- metrics（json/csv）
- preds（npy/csv）
- model（pickle/pt）

## Acceptance Criteria
- [ ] `task=train` → `task=predict` → `task=reconstruct` → `task=eval` が一連で動く
- [ ] outputs に artifact が契約どおり保存される
- [ ] 再構成ができる（field_hatが生成される）

## Verification
- [ ] synthetic dataset で一連を実行し、outputs配下に成果物が揃う
