# CoeffPost: standardize + PCA（train-only fit, inverse）

**ID:** 060  
**Priority:** P0  
**Status:** todo  
**Depends on:** 040, 050  
**Unblocks:** 070  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
係数後処理を統一インターフェースで実装する。
P0では以下:
- standardize（平均0/分散1。per-dim。train統計を保存）
- PCA（energy_threshold or n_components、whiten、inverse_transform）

重要:
- fitは train のみ（skew禁止）
- reconstruct/eval のため inverse_transform を必ず提供

## Acceptance Criteria
- [ ] standardize が fit/transform/inverse を持つ
- [ ] PCA が fit/transform/inverse を持つ
- [ ] train-only fit であることが保証される（ログ/コード）

## Verification
- [ ] PCA適用時に latent_dim が記録され、inverse で a_hat に戻せる
