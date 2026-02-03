# Task: 401 Refactor: 出力フラット化（runs/<tag>/<run_id>/ 固定 + artifact統一）

- Priority: P0
- Status: todo
- Depends on: 398, 399
- Unblocks: 402, 490

## Intent
出力ディレクトリ/ファイルを **固定の浅い構造**に統一し、結果確認・レビュー・比較を容易にする。
Hydraの深い階層を廃止/縮退し、`runs/<tag>/<run_id>/` へ集約する。

## Context / Constraints
- artifact契約（metrics/preds/model/meta/viz 等）は維持する
- すべての Process（train/predict/reconstruct/eval/viz/bench）で同じ構造を出す
- 既存の出力参照コード（leaderboard等）があれば更新する

## Plan
- [ ] `RunDirManager`（仮）を導入し、run_id 生成（timestamp + short hash）と run_dir を一元管理
- [ ] `ArtifactWriter` を導入し、固定ファイル名に統一して保存
- [ ] 既存processを更新し、Hydra依存の出力パス構築を削除する
- [ ] docs更新: `docs/04_ARTIFACTS_AND_VERSIONING.md` に新レイアウトを反映
- [ ] 互換: 旧outputs/配下のrunも leaderboard が読める（可能なら）

## Acceptance Criteria
- [ ] `runs/<tag>/<run_id>/` 直下に `run.yaml, manifest_run.json, metrics.json, preds.npz` が生成される
- [ ] `model/`, `states/`, `figures/`, `tables/` のサブフォルダが固定で作成される
- [ ] bench/leaderboard が新レイアウトで動く（最低1ケース）
- [ ] docs/04 が更新され、第三者が artifact を解釈できる

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml`
- Expected:
  - `runs/.../metrics.json` が存在し、`figures/` にpngが出る

---

## Review Map
- 変更ファイル一覧:
  - 追加: src/mode_decomp_ml/pipeline/artifacts.py
  - 変更: src/mode_decomp_ml/pipeline/utils.py, src/mode_decomp_ml/pipeline/__init__.py
  - 変更: src/mode_decomp_ml/run.py, src/mode_decomp_ml/cli/run.py
  - 変更: src/processes/train.py, src/processes/predict.py, src/processes/reconstruct.py, src/processes/eval.py, src/processes/viz.py, src/processes/benchmark.py, src/processes/leaderboard.py
  - 変更: src/mode_decomp_ml/decompose/__init__.py, src/mode_decomp_ml/decompose/autoencoder.py, src/mode_decomp_ml/coeff_post/__init__.py, src/mode_decomp_ml/preprocess/__init__.py, src/mode_decomp_ml/models/__init__.py
  - 変更: src/mode_decomp_ml/tracking/leaderboard.py, src/mode_decomp_ml/tracking/clearml.py
  - 変更: tools/validate_artifacts.py, tools/leaderboard.py
  - 変更: configs/config.yaml, configs/examples/pod_ridge.yaml, configs/examples/pod_gpr_uncertainty.yaml, configs/task/leaderboard.yaml
  - 変更: docs/04_ARTIFACTS_AND_VERSIONING.md, docs/10_PROCESS_CATALOG.md
  - 変更: tests/test_processes_e2e.py, tests/test_validate_artifacts.py
- 重要な関数/クラス:
  - src/mode_decomp_ml/pipeline/utils.py: RunDirManager, make_run_id
  - src/mode_decomp_ml/pipeline/artifacts.py: ArtifactWriter, load_* helpers
  - src/processes/*: 新しいpreds.npz/metrics.json/manifest_run.json出力と読み込み
- 設計判断:
  - ルート直下の固定ファイル名（run.yaml/manifest_run.json/metrics.json/preds.npz）に統一し、model/states/figures/tables配下へ整理
  - 旧レイアウトは読み取り側でfallback対応（leaderboard/validate/predict系のロード）
- リスク/注意点:
  - preds_meta/field_std_metaはmanifest_run.jsonへ移動したため、外部ツールは参照先更新が必要
  - Hydra run_idはconfig側のjob.numベース、run.yamlは短縮hashベースでrun_idが異なる
- 検証コマンドと結果:
  - python3 -m pytest tests/test_validate_artifacts.py -q（pass）
- 削除一覧:
  - src/processes/benchmark.py: _write_config_snapshot 関数を削除
