# Task 290 (P1): P1 Cleanup（不要コード削除・docs整合・再現性確認）

**ID:** 290  
**Priority:** P1  
**Status:** done  
**Depends on:** 270  
**Unblocks:** None  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
P1完了時点の“きりのいい完成形”にする。
中途半端な試行コードや未使用ファイルを残して、後から大規模リファクタになる状況を避ける。

## Checklist
- 未使用/重複コード・ファイル・ディレクトリを削除（docs/16）
- doctor が no issue
- benchmark/leaderboard が動く
- 再現性（seed, config保存）が守られている

## Acceptance Criteria
- [x] `task=doctor` が no issue
- [x] `task=benchmark` が成功する（最小sweep）
- [x] `work/tasks_p1_tail/*` の Review Map が埋まっている（Files touched/Verification等）

## Verification
- [x] `python -m mode_decomp_ml.cli.run task=doctor`（dataset shapes 出力を確認）
- [x] `python -m mode_decomp_ml.cli.run task=benchmark task.decompose_list=fft2 task.coeff_post_list=none task.domain=rectangle model=ridge`
- [x] `outputs/doctor/*` と `outputs/benchmark/*` の生成を確認

## Cleanup Checklist
- [x] “将来使うかも” コードを削除（必要ならRFCへ）

---
## Review Map
- **変更ファイル一覧**: `.gitignore`, `work/tasks_p1_tail/290_cleanup_p1_tail.md`, `work/queue.json`
- **削除一覧**: `**/__pycache__/`, `*.pyc`, `.DS_Store`, `.pytest_cache/`
- **検証コマンドと結果**: `python -m mode_decomp_ml.cli.run task=doctor`（no issue）, `python -m mode_decomp_ml.cli.run task=benchmark task.decompose_list=fft2 task.coeff_post_list=none task.domain=rectangle model=ridge`（1 sweep 成功）
