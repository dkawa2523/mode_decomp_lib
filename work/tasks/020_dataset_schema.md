# データスキーマとDataset実装（scalar/vector, mask）

**ID:** 020  
**Priority:** P0  
**Status:** todo  
**Depends on:** 010  
**Unblocks:** 030  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
データを “cond→field” 学習に統一するため、データスキーマを固定し Dataset を実装する。

必須スキーマ（1サンプル）:
- cond: shape [D]
- field: shape [H,W,C]  （scalar: C=1, vector: C=2）
- mask: shape [H,W]（None可、mask外は評価対象外）
- meta: dict（domain種、スケールなど）

P0では以下を用意:
- synthetic dataset（合成波/ガウス等で検証用）
- npy_dir dataset（cond.npy + field.npy + mask.npy を読む等）

## Acceptance Criteria
- [ ] dataset が上記スキーマで sample を返す
- [ ] scalar/vector を同じコードで扱える（C次元で分岐しない）
- [ ] maskがある場合、mask外を評価から除外できるよう meta が残る

## Verification
- [ ] `task=doctor` で synthetic dataset 1件がロードできる（shapeがログに出る）
