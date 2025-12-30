# Task 270 (P1): Docs/Examples更新（P1 tail反映）

**ID:** 270  
**Priority:** P1  
**Status:** done  
**Depends on:** 250, 260  
**Unblocks:** 290  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
P1で増えた機能を docs と examples に反映し、あとから読み返しても迷わない状態にする。

## Deliverables（最小）
- docs/20_METHOD_CATALOG.md 更新
  - Fourier–Bessel / POD / GPR / uncertainty / elasticnet の “Implemented” を明記
- docs/09_EVALUATION_PROTOCOL.md 追記（uncertainty の扱い：任意、比較には含めない等）
- configs/examples/ に短い実行例を2つ追加（例）
  1) POD + Ridge（高速）
  2) POD + GPR + uncertainty（小データ向け）

## Acceptance Criteria
- [x] docs が実装と整合している（実行例が現実に動く）
- [x] 例が短い（最小）・比較可能性が壊れない（seed/split固定）

## Verification
- [x] docsに書いたコマンドを実際に実行し、artifactが揃うこと

## Cleanup Checklist
- [x] 古い説明・重複説明を削除

## Review Map
- 変更ファイル一覧: `docs/20_METHOD_CATALOG.md`, `docs/09_EVALUATION_PROTOCOL.md`, `configs/examples/pod_ridge.yaml`, `configs/examples/pod_gpr_uncertainty.yaml`
- 重要な入口/参照: `configs/examples/pod_ridge.yaml`, `configs/examples/pod_gpr_uncertainty.yaml`（Hydra root config）, `docs/20_METHOD_CATALOG.md`, `docs/09_EVALUATION_PROTOCOL.md`
- 設計判断: examples は task=benchmark の単一コマンドに統一し、`# @package _global_` で root merge; 小さな synthetic 設定で高速化
- リスク/注意点: GPR 例は sklearn の ConvergenceWarning が出る（学習は成功）; uncertainty は比較指標に含めない
- 検証コマンドと結果: `python -m mode_decomp_ml.cli.run --config-name examples/pod_ridge`（成功）, `python -m mode_decomp_ml.cli.run --config-name examples/pod_gpr_uncertainty`（成功, ConvergenceWarning 2件）
- 削除一覧: `docs/20_METHOD_CATALOG.md` の未存在ファイル参照（`mode_decomposition.md`）を削除
