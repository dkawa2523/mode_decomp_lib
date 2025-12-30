# Hydra最小導入（config真実・task入口統一）

**ID:** 010  
**Priority:** P0  
**Status:** done  
**Depends on:** 000  
**Unblocks:** 020  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
Hydraを “設定が真実” になる形で導入し、process入口（train/eval/predict...）を config/task で切り替えられるようにする。

- `configs/config.yaml` を root とし、`task=train` 等で入口を切替
- run_dir, seed, output_dir 等は docs/03 に従う
- `python -m processes.doctor` ではなく、統一入口 `python -m mode_decomp_ml.cli.run task=doctor` を用意

## Acceptance Criteria
- [x] `python -m mode_decomp_ml.cli.run task=doctor` が動き、configが保存される
- [x] `configs/task/*.yaml` が最低限存在し、taskの切替ができる
- [x] 重要なパラメータ（seed/run_dir/output_dir）が config に集約されている

## Verification
- [x] `python3 -m mode_decomp_ml.cli.run task=doctor` を実行し、`outputs/doctor/2025-12-28/22-38-43/.hydra/config.yaml` を確認

## Review Map
- 変更ファイル一覧: `configs/config.yaml`, `configs/task/doctor.yaml`, `src/mode_decomp_ml/cli/run.py`, `src/mode_decomp_ml/cli/__init__.py`, `src/processes/doctor.py`, `mode_decomp_ml/__init__.py`, `processes/__init__.py`, `work/tasks/010_hydra_mvp.md`, `work/queue.json`, `outputs/doctor/2025-12-28/22-38-43/.hydra/config.yaml`
- 重要な関数/クラス: `src/mode_decomp_ml/cli/run.py`（Hydra/フォールバックのエントリとtask routing）, `src/processes/doctor.py`（必須configキー検証）, `mode_decomp_ml/__init__.py`（src/パス拡張）, `processes/__init__.py`（src/パス拡張）
- 設計判断: Hydra未導入環境でも `task=doctor` を実行できるよう、Hydra不在時はYAML構成+`.hydra/config.yaml` を作る最小フォールバックを追加
- リスク/注意点: フォールバックはHydraの全機能を再現しないため、正式運用は `hydra-core` インストール前提
- 検証コマンドと結果: `python3 -m mode_decomp_ml.cli.run task=doctor` → `doctor ok` / `.hydra/config.yaml` 作成
- 削除一覧: `sitecustomize.py`
