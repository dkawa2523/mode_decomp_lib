# Task 395 (P2): Artifact validator（契約逸脱の早期検知）

**ID:** 395  
**Priority:** P2  
**Status:** done  
**Depends on:** 350  
**Unblocks:** 398  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
手法が増えるほど artifact が崩れやすくなるため、
docs/04 の契約を “自動で検査” する validator を追加する。

## Scope（最小）
- `tools/validate_artifacts.py` を追加
- 入力: outputs run dir
- 検査:
  - config/.hydra が存在
  - meta.json が存在（git hash/seed など最低限）
  - metrics.json が存在
  - model/ が存在
  - 予測系ファイル（preds）が存在（taskにより条件分岐OK）

## Acceptance Criteria
- [x] validator が1 run を検査して PASS/FAIL を返す
- [ ] doctor から “最近のrun” を検査できる（任意）
- [x] CIが無くてもローカルで使える

## Verification
- [x] 既存のbenchmark runに validator をかけてPASSする

---
### Review Map
- **変更ファイル一覧**: `tools/validate_artifacts.py` `tests/test_validate_artifacts.py` `work/tasks_p2_v2/395_artifact_validator.md`
- **重要な関数/クラス**: `tools/validate_artifacts.py:main` `tools/validate_artifacts.py:_find_config` `tools/validate_artifacts.py:_check_meta` `tools/validate_artifacts.py:_check_pred_files`
- **設計判断**: run dirのtask種別に応じて必要artifactを最小判定し、evalは上流run_dirを検査してmodel/predsの存在を担保。Hydra configは`.hydra/config.yaml`と`hydra/config.yaml`の両対応で最小互換。
- **リスク/注意点**: benchmark内のtrain/predict/reconstructは`.hydra`がないため、このvalidatorを直接当てるとFAILになる（evalから検査する運用）。
- **検証コマンドと結果**: `python3 tools/validate_artifacts.py outputs/benchmark/2025-12-29/13-54-09_ex_pod_ridge_None/run_000__pod_svd__none/eval` -> `[PASS] Artifact validation passed.`
