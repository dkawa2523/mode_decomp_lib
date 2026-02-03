# クリーンアップ（不要コード/ファイル削除、docs整合）

**ID:** 120  
**Priority:** P0  
**Status:** done  
**Depends on:** 110  
**Unblocks:** None  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
P0完了時点で、不要なコード/未使用ファイルを残さない。
また docs と実装の不整合（参照パス、説明）が無いよう整える。

## Acceptance Criteria
- [x] 未使用ファイル/ディレクトリを削除 or TODOで理由を残す
- [x] docs の reading order と実際の入口/パスが一致する
- [x] `task=doctor` が引き続き no issue

## Verification
- [x] `python -m mode_decomp_ml.cli.run task=doctor` が通る

## Review Map
- **変更ファイル一覧**
  - 変更: `docs/README.md`, `docs/17_EXTENSION_PLAYBOOK.md`, `work/tasks/120_cleanup_prune.md`, `work/queue.json`
  - 削除: `work/BACKLOG.md`, `examples/PROMPT_PACK.md`, `examples/REPO_CONTEXT_TEMPLATE.md`, `legacy/README.md`, `scripts/.keep`, `work/tasks/000_setup.md`, `work/tasks/010_skeleton.md`, `work/tasks/020_hydra_mvp.md`, `work/tasks/030_data_domain_io.md`, `work/tasks/040_preprocess_pipeline.md`, `work/tasks/045_vector_transforms.md`, `work/tasks/050_decomposer_registry.md`, `work/tasks/051_decompose_zernike.md`, `work/tasks/052_decompose_fft_dct.md`, `work/tasks/053_decompose_fourier_bessel.md`, `work/tasks/054_decompose_rbf.md`, `work/tasks/055_decompose_wavelet.md`, `work/tasks/056_decompose_graph_fourier.md`, `work/tasks/057_decompose_laplace_fem.md`, `work/tasks/058_decompose_laplace_beltrami.md`, `work/tasks/059_decompose_pod_svd.md`, `work/tasks/060_coeff_post_registry.md`, `work/tasks/061_coeff_post_pca.md`, `work/tasks/062_coeff_post_ica_nmf.md`, `work/tasks/063_decompose_dict_learning.md`, `work/tasks/064_decompose_autoencoder.md`, `work/tasks/070_models_sklearn_baseline.md`, `work/tasks/071_models_torch_mlp.md`, `work/tasks/072_models_gpr.md`, `work/tasks/080_process_e2e.md`, `work/tasks/085_metrics_module.md`, `work/tasks/090_viz_reports.md`, `work/tasks/095_doctor_tests.md`, `work/tasks/100_leaderboard.md`, `work/tasks/110_benchmark_runner.md`, `work/tasks/115_basis_cache.md`, `work/tasks/116_interpretability_report.md`, `work/tasks/120_tracking_hooks.md`, `work/tasks/125_prune_unused.md`
- **重要な関数/クラス**
  - なし（ドキュメント整合と不要ファイル削除のみ）
- **設計判断**
  - P0完了後に参照されないテンプレ/バックログ類は削除して、queueとtask実態の不一致を解消。
  - docsの読む順番と入口参照を、実在ファイルに合わせて整理。
- **リスク/注意点**
  - 旧テンプレ由来のタスクは削除したため、必要になれば `work/templates/TASK.md` から新規起票する。
- **検証コマンドと結果**
  - `python -m mode_decomp_ml.cli.run task=doctor` → dataset/cond/field/mask出力を確認（no issue）
