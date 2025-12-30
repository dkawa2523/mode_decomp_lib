# Task 320 (P2): Decomposer: Autoencoder/VAE（Torch）

**ID:** 320  
**Priority:** P2  
**Status:** todo  
**Depends on:** 310  
**Unblocks:** 330  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
非線形分解（AE/VAE）を decomposer として追加し、cond→latent→field の流れを作る。
ただし P2 では **toyで動く最小** を優先し、複雑なモデル探索はしない。

## Scope（最小）
- rectangle/disk の grid field を入力（H,W,C）
- 小さな Conv Autoencoder（VAEは optional）
- fit(train-only) で AE を学習し、encoder 出力を latent とする
- transform: field→latent
- inverse: latent→field_hat
- artifacts に torch weights を保存

## Recommended integration
- decomposer plugin として実装（PODと同様に fit が必要）
- 回帰モデル（Ridge/GPR等）は latent を target に学習する
  - 既存の pipeline が “decomposer.fit → coeff → model.fit” を想定しているため

## Config（例）
- latent_dim: 8/16/32
- epochs: 10（toy）
- batch_size: 16
- lr: 1e-3
- use_amp: false（最小）

## Acceptance Criteria
- [ ] `decompose=autoencoder` が選択できる（configs追加）
- [ ] toyデータで学習が走り、lossが下がる
- [ ] inverseで field_hat が生成され、可視化できる
- [ ] weights（.pt）が artifacts に保存される

## Verification
- [ ] 1 run で outputs に reconstruction png と weights が生成される

## Cleanup Checklist
- [ ] 過剰なモデル種類を増やさない（最小AEのみ）
- [ ] デバッグ専用の巨大ログ/printを削除
