# P1: Model GPR（小データ向け）

**ID:** 230  
**Priority:** P1  
**Status:** done  
**Depends on:** 060  
**Unblocks:** 240  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
Gaussian Process Regressor を追加し、cond→(a|z) の多出力回帰に対応する。

## Acceptance Criteria
- [x] GPR が cond→target の多出力回帰として動く
- [x] target_space を設定で切替できる
- [x] kernel/scale/seed を config で管理できる

## Verification
- [x] `pytest tests/test_models_gpr.py`

## Review Map
- **変更ファイル一覧**
  - 変更: `src/mode_decomp_ml/models/__init__.py`, `configs/model/gpr.yaml`
  - 追加: `tests/test_models_gpr.py`
- **重要な関数/クラス**
  - `src/mode_decomp_ml/models/__init__.py`: `GPRRegressor`, `_build_gpr_kernel`
- **設計判断**
  - sklearn の `GaussianProcessRegressor` を使い、RBF + WhiteKernel を最小構成で設定可能にした。
  - `target_space` と kernel/scale を config で明示し、比較可能性と再現性を担保した。
- **リスク/注意点**
  - GPR は計算量が重いので小データ向け（cond→latent を推奨）。
  - kernel 設定を変えると比較不能になるため config の固定が必要。
- **検証コマンドと結果**
  - `pytest tests/test_models_gpr.py`（PASS）
