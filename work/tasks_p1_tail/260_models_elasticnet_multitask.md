# Task 260 (P1): Model拡張: ElasticNet / MultiTask（比較用）

**ID:** 260  
**Priority:** P1  
**Status:** done  
**Depends on:** 240, 250  
**Unblocks:** 270  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
“モデルが頻繁に増える” 要件に合わせて、比較用の追加モデルを最小で導入する。
Ridge/GPRの間を埋める、疎性・安定性の比較用。

## Scope（最小）
- ElasticNet（multi-output対応）
- （任意）MultiTaskElasticNet（共通の疎性パターンがある場合）

## Implementation notes
- sklearn を使用（新規重依存は増やさない）
- config:
  - `configs/model/elasticnet.yaml`（alpha, l1_ratio, max_iter 等）
  - （任意）`configs/model/multitask_elasticnet.yaml`
- registry:
  - model plugin registry に追加
- benchmark:
  - sweep で `model=elasticnet` が回せること

## Acceptance Criteria
- [x] `model=elasticnet` が train→predict→eval で動く
- [x] leaderboard で ridge / elasticnet / gpr が比較できる（同一metric）

## Verification
- [x] 小さな sweep（2〜5 run）で elasticnet を含めて実行し、metricsが揃うこと

## Cleanup Checklist
- [x] 追加モデルの“使われない便利関数”を作らない

## Review Map（必須）
- **変更ファイル一覧**: `src/mode_decomp_ml/models/__init__.py`, `configs/model/elasticnet.yaml`, `configs/model/multitask_elasticnet.yaml`
- **重要な関数/クラス**: `ElasticNetRegressor`, `MultiTaskElasticNetRegressor` in `src/mode_decomp_ml/models/__init__.py`
- **設計判断**: 既存のモデル実装が `src/mode_decomp_ml/models/__init__.py` に集約されているため、同一ファイルに追加。ElasticNetはmulti-output対応のsklearn実装を直接利用し、MultiTask版は共通疎性比較用に最小追加。
- **リスク/注意点**: 少数サンプルのため収束警告が出る場合がある（ElasticNet/GPRのConvergenceWarning）。
- **検証コマンドと結果**: `python -m mode_decomp_ml.cli.run -m task=benchmark model=ridge,elasticnet,gpr task.decompose_list=fft2 task.coeff_post_list=none task.domain=rectangle`（3 runs完了、警告のみ）。`python -m mode_decomp_ml.cli.run task=leaderboard task.runs='outputs/benchmark/**/eval'`（leaderboard生成、elasticnet/ridge/gpr行を確認）。
