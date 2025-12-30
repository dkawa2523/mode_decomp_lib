# Task 330 (P2): CoeffPost: Dictionary Learning（疎表現）

**ID:** 330  
**Priority:** P2  
**Status:** todo  
**Depends on:** 320  
**Unblocks:** 340  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal（推奨：coeff_post として実装）
Dictionary Learning を **coeff_post** として導入し、
特殊関数展開・POD・AE などの係数ベクトルに対して “疎表現” を得る。

> decomposer として場を直接辞書化するより、coeff_post の方が
> 既存パイプラインを壊さず、比較しやすい（docs/00,11）。

## Scope（最小）
- fit(train_coeff): 辞書 D を学習
- transform: sparse code z を推定（L1）
- inverse_transform: z→coeff_hat（z @ D）
- artifacts: dictionary.npy と params を保存

## Implementation notes
- sklearn `DictionaryLearning` または `MiniBatchDictionaryLearning`
- 係数次元が大きい場合に備え、MiniBatch を優先（設定で切替）
- “fitはtrainのみ” をログで明確化（skew禁止）

## Acceptance Criteria
- [ ] `coeff_post=dict_learning` が選択できる（configs追加）
- [ ] fit/transform/inverse が揃う（reconstructに必要）
- [ ] sparsity 指標（平均|z|0近似等）を metrics に残せる

## Verification
- [ ] 既存の1手法（例: POD→Ridge）に dict_learning を挟んで動く

## Cleanup Checklist
- [ ] 実装を最小に保ち、過剰な派生オプションを増やさない
