# Decomposer: Zernike（legacy移植 or 最小実装）

**ID:** 050  
**Priority:** P0  
**Status:** todo  
**Depends on:** 030  
**Unblocks:** 060  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
disk 領域向けの主力分解として Zernike を実装する。

優先:
1) `legacy/` に旧Zernike実装がある場合は **必要最小限だけ** `src/mode_decomp_ml/decompose/zernike.py` へ移植する
2) 旧実装が無い場合は、SciPy/既存依存のみで成立する最小実装を作る（高次最適化はしない）

必須:
- transform: field→a
- inverse: a→field_hat
- coeff_meta: (n,m) など次数情報、正規化、mask扱い

## Acceptance Criteria
- [ ] disk domain で Zernike transform/inverse が動作する
- [ ] 係数次元が設定で決められる（n_max 等）
- [ ] coeff_meta に (n,m) の対応が残る

## Verification
- [ ] 低次モードのみでの逐次再構成が可視化できる（k=1,2,4...）
