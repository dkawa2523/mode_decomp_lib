# CoeffPost: standardize + PCA（train-only fit, inverse）

**ID:** 060  
**Priority:** P0  
**Status:** done  
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
- [x] standardize が fit/transform/inverse を持つ
- [x] PCA が fit/transform/inverse を持つ
- [x] train-only fit であることが保証される（ログ/コード）

## Verification
- [x] PCA適用時に latent_dim が記録され、inverse で a_hat に戻せる

## Review Map（必須）
- 変更ファイル一覧（追加/変更/削除）: `src/mode_decomp_ml/coeff_post/__init__.py`（更新）, `configs/coeff_post/standardize.yaml`（追加）, `configs/coeff_post/pca.yaml`（更新）, `tests/test_coeff_post.py`（追加）
- 重要な関数/クラス: `BaseCoeffPost`, `NoOpCoeffPost`, `StandardizeCoeffPost`, `PCACoeffPost`, `build_coeff_post`
- 設計判断: train/serve skew防止のため`fit(split="train")`を必須化し、PCAは`energy_threshold`か`n_components`のみ許可。stateは`state.pkl`にpickle保存できる形で保持し、latent_dimはfit結果から記録。
- リスク/注意点: PCAの逆変換は情報損失があるため完全再構成ではない。`n_components`と`energy_threshold`の同時指定はエラーになる。
- 検証コマンドと結果: `pytest -q tests/test_coeff_post.py`（pass）
- 削除一覧: なし
