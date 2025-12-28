# クリーンアップ（不要コード/ファイル削除、docs整合）

**ID:** 120  
**Priority:** P0  
**Status:** todo  
**Depends on:** 110  
**Unblocks:** None  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
P0完了時点で、不要なコード/未使用ファイルを残さない。
また docs と実装の不整合（参照パス、説明）が無いよう整える。

## Acceptance Criteria
- [ ] 未使用ファイル/ディレクトリを削除 or TODOで理由を残す
- [ ] docs の reading order と実際の入口/パスが一致する
- [ ] `task=doctor` が引き続き no issue

## Verification
- [ ] `python -m mode_decomp_ml.cli.run task=doctor` が通る
