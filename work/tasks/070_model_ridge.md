# Model: Ridge多出力（cond→(a|z)）

**ID:** 070  
**Priority:** P0  
**Status:** todo  
**Depends on:** 060  
**Unblocks:** 080  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
回帰モデルとして Ridge を多出力で実装する。
- 入力: cond [N,D]
- 出力: target [N,K]（a または z）

設定で以下を切替:
- target_space: a or z
- alpha, fit_intercept, normalize（必要なら）
- cond_scaler（standardize等、最小）

## Acceptance Criteria
- [ ] Ridge が cond→target の多出力回帰として動く
- [ ] target_space を設定で切替できる
- [ ] seed固定で再現できる

## Verification
- [ ] 合成データで train→predict が通り、shapeが一致する
