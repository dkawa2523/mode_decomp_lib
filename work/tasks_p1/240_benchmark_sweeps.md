# P1: Benchmark sweeps（Hydra multirun）

**ID:** 240  
**Priority:** P1  
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
Hydra multirun で benchmark sweeps を実行できるようにし、run_dir 衝突を避ける。

## Acceptance Criteria
- [x] multirun 実行時に run_dir が job num で一意化される
- [x] benchmark multirun の実行例と出力先が docs に明記される
- [x] `task.decompose_list` / `task.coeff_post_list` の override で sweeps が動く

## Verification
- [x] `PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none`

## Review Map
- **変更ファイル一覧**
  - 変更: `configs/config.yaml`, `docs/03_CONFIG_CONVENTIONS.md`, `docs/04_ARTIFACTS_AND_VERSIONING.md`, `docs/10_PROCESS_CATALOG.md`
  - 変更: `work/tasks_p1/240_benchmark_sweeps.md`
- **重要な入口/関数/クラス**
  - `configs/config.yaml`: `run_dir`, `hydra.sweep.subdir`
  - `src/processes/benchmark.py`: `main`（benchmark 実行入口）
- **設計判断**
  - run_dir に `hydra:job.num` を追加し、Hydra multirun での衝突を避ける方針に統一。
  - 実行例は docs に最短のコマンドと出力先だけを追記し、拡張は最小に留めた。
- **リスク/注意点**
  - run_dir の末尾に job num が付くため、固定パス前提のスクリプトがあれば更新が必要。
- **検証コマンドと結果**
  - `PYTHONPATH=src python -m mode_decomp_ml.cli.run -m task=benchmark task.decompose_list=fft2,zernike task.coeff_post_list=none`
  - 結果: HYDRA が 2 jobs を launch（exit 0）、`outputs/benchmark/2025-12-29/09-21-05_0` と `_1` を生成
