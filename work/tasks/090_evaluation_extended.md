# 評価拡張（coeff誤差 + field誤差 + スペクトル診断）

**ID:** 090  
**Priority:** P0  
**Status:** todo  
**Depends on:** 080  
**Unblocks:** 100  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
比較可能性のため、評価指標を固定する。
最低限:
- coeff_rmse（a空間、z空間）
- field_rmse（mask内）
- energy_cumsum（スペクトル/次数ごとの累積）

## Acceptance Criteria
- [ ] metrics が json で保存され、leaderboardが読める
- [ ] mask がある場合、mask内だけ評価する
- [ ] スペクトル診断（累積エネルギー）が保存される

## Verification
- [ ] 2手法（FFT vs Zernike）で metrics を比較できる
