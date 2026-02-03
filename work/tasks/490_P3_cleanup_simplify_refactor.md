# Task: 490 Cleanup: YAML/出力/コードの整理（削除・共通化・docs整合）

- Priority: P0
- Status: done
- Depends on: 402, 440
- Unblocks: 491

## Intent
このP3追加・リファクタ後に、不要コード/未使用config/死んだ例を削除し、
第三者が読める最小構成に整える（スパゲッティ化・負債増殖を止める）。

## Context / Constraints
- Deletion policy に従い “使っていないものは削除”
- docs と実装の整合を取る（読む順が壊れない）
- 既存のタスクが途中のまま次へ行かない（完了条件を満たす）

## Plan
- [ ] 未使用の configs/ を削除（deprecated期限が過ぎたもの）
- [ ] 重複コードを baseclass/utility へ統合し、呼び出し側を更新
- [ ] examples/ を 5〜8 本程度に整理（多すぎる例は削除）
- [ ] docs/README（読む順）を更新
- [ ] `python -m pytest` と bench quick を実行し、最低限の健全性を確認
- [ ] `tools/lint_unused.py` 等があれば整備（無ければTODOに）

## Acceptance Criteria
- [ ] configs/ の総数が減り、run.yaml 入口中心になっている
- [ ] 出力レイアウトが統一され、第三者が `runs/` を見て理解できる
- [ ] 主要テストが通る（少なくとも unit + e2e smoke）
- [ ] “不要ファイル削除” がPRに含まれている（放置しない）

## Verification
- Command:
  - `python -m pytest`
  - `bash scripts/bench/run_p0p1_p2ready.sh`
- Expected:
  - 失敗なく完走（skipがある場合は理由がログに出る）

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/pipeline/loaders.py`, `src/processes/eval.py`, `src/processes/reconstruct.py`, `src/processes/viz.py`, `src/processes/predict.py`, `src/processes/leaderboard.py`, `configs/task/leaderboard.yaml`, `examples/run_scalar_rect_fft2_ridge.yaml`, `examples/run_scalar_disk_zernike.yaml`, `docs/README.md`, `docs/03_CONFIG_CONVENTIONS.md`, `docs/20_METHOD_CATALOG.md`, `docs/USER_QUICKSTART.md`, `README.md`
- 重要な入口/関数: `src/mode_decomp_ml/pipeline/loaders.py` の `load_train_artifacts` / `load_preprocess_state_from_run` / `load_model_state`, `src/processes/eval.py`, `src/processes/reconstruct.py`, `src/processes/viz.py`, `src/processes/predict.py`
- 設計判断: 反復していた artifact/state 読み込み処理を `pipeline/loaders.py` に集約し、process 側は API 呼び出しに統一。leaderboard の既定パターンは `runs/**/metrics.json` のみにして出力レイアウトを一本化。
- リスク/注意点: 旧 `outputs/**` は leaderboard の既定対象から外れたため、必要なら `task.runs` で明示する。FFT2 は複素係数のため run.yaml 例は codec を必須化。
- 検証コマンドと結果: `python -m pytest` は `python` 不在のため `python3 -m pytest` で実行（65 passed, 3 skipped）。`bash scripts/bench/run_p0p1_p2ready.sh` 完走。
- 削除一覧: `configs/_deprecated/**`, `examples/run_scalar_disk_zernike_pca_xgb.yaml`, `examples/run_scalar_rect_fft_pca_mtlasso.yaml`, `examples/run_scalar_rect_fft_power_ridge.yaml`, `examples/run_scalar_rect_fft_quantile_ridge.yaml`, `examples/run_scalar_rect_fft_ridge.yaml`
- TODO: `tools/lint_unused.py` が存在しないため、後続タスクで追加・整備する
