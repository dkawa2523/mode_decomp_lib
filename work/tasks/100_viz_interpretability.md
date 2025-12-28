# 可視化（再構成比較・係数スペクトル・解釈性）

**ID:** 100  
**Priority:** P0  
**Status:** todo  
**Depends on:** 090  
**Unblocks:** 110  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
研究開発で“見て判断”できる最低限の可視化を実装する。
- field_true vs field_hat の画像
- error map（mask内）
- 係数スペクトル（次数/周波数 vs energy）
- 逐次再構成（k=1,2,4,8...）

## Acceptance Criteria
- [ ] png が outputs に保存される
- [ ] 手法別に比較しやすいファイル名/構造になっている

## Verification
- [ ] 2 run の可視化を並べて違いが分かる
