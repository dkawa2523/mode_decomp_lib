# Task 340 (P2): Vector Field: Helmholtz (div/curl) 分解（評価・可視化統合）

**ID:** 340  
**Priority:** P2  
**Status:** todo  
**Depends on:** 330  
**Unblocks:** 350  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
ベクトル場（C=2）の物理解釈性を高めるため、div/curl を計算し、
必要なら Helmholtz 分解（toy）を導入する。

## Scope（最小）
- rectangle grid の vector field v=(u,v) を対象
- div, curl を有限差分で計算
- metrics に追加:
  - div_l2_true/pred
  - curl_l2_true/pred
- viz に追加（任意）:
  - div_map, curl_map

> Helmholtz分解（solenoidal/irrotational分離）は toy でOK。
> まずは指標と可視化を入れて、ベクトル場の評価を成立させる。

## Recommended file plan（目安）
- add: `src/mode_decomp_ml/vector_ops/derivatives.py`（div/curl）
- add: `src/mode_decomp_ml/vector_ops/helmholtz.py`（toy）
- update: evaluate pipeline（vectorの場合に指標を計算）
- update: viz pipeline（vector指標のmap出力）

## Acceptance Criteria
- [ ] vector field の評価で div/curl 指標が出る
- [ ] scalar field には影響しない（分岐は評価内だけ）
- [ ] 代表サンプルで div/curl の可視化ができる（任意でもOK）

## Verification
- [ ] toyの回転流/発散流で div/curl が期待通りになることを確認
