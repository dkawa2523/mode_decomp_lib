# Task 350 (P2): Tracking: ClearML統合（準備/最小）

**ID:** 350  
**Priority:** P2  
**Status:** todo  
**Depends on:** 340  
**Unblocks:** 395  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
ClearML 統合を “最小で安全” に導入する。
ClearML が入っていない/使わない環境では **絶対に落ちない** のが要件。

## Scope（最小）
- config: clearml.enabled: true/false
- enabled のときだけ import/初期化する
- artifacts（docs/04）を ClearML に紐付ける
  - config/meta/metrics/preds/model/plots

## Recommended implementation
- add: `src/mode_decomp_ml/tracking/clearml_adapter.py`
- update: process入口で `maybe_init_clearml(cfg)` を呼ぶ
- benchmark（sweep）は parent/child は “将来”
  - P2最小では単発 run のログでOK

## Acceptance Criteria
- [ ] clearml.enabled=false で何も変わらず動く
- [ ] clearml.enabled=true で metrics と plots がログされる（可能なら）
- [ ] clearmlが無い環境では警告してスキップ（落ちない）

## Verification
- [ ] clearml未インストール環境で doctor/benchmark が成功する
