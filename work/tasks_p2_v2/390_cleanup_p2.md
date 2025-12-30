# Task 390 (P2): P2 Cleanup（整理・docs整合・実行例）

**ID:** 390  
**Priority:** P2  
**Status:** todo  
**Depends on:** 398  
**Unblocks:** None  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
P2完了時点で、不要コード・試行ファイルを残さず、docsと実装を一致させる。

## Scope（最小）
- 未使用ファイル/重複コード削除（docs/16）
- docs/20_METHOD_CATALOG.md を更新（P2手法のImplemented明記）
- examples config を最小で整備（2〜3個）
- doctor/benchmark を再実行して健全性確認

## Acceptance Criteria
- [ ] `task=doctor` が no issue
- [ ] `task=benchmark`（軽量版）が成功
- [ ] 使っていないファイル/コードが残っていない（意図があるならTODOで説明）

## Verification
- [ ] 代表runの outputs を validator（Task395）でPASSする
