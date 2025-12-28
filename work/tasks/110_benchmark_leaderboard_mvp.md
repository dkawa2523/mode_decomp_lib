# 比較Runner + Leaderboard（最小）

**ID:** 110  
**Priority:** P0  
**Status:** todo  
**Depends on:** 100  
**Unblocks:** 120  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
複数手法を同一条件で比較し、leaderboard を出す最小実装を作る。

P0では:
- 2×2 程度の組合せ（decompose×coeff_post）を回せればOK
- `tools/leaderboard.py` で outputs を集計し、CSVを出す

## Acceptance Criteria
- [ ] benchmarkが複数runを生成し、metricsが揃う
- [ ] leaderboard が CSV を出力できる

## Verification
- [ ] FFT+PCA vs Zernike+PCA で leaderboard に差が出る
