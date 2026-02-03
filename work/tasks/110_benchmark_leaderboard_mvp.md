# 比較Runner + Leaderboard（最小）

**ID:** 110  
**Priority:** P0  
**Status:** done  
**Depends on:** 100  
**Unblocks:** 120  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
複数手法を同一条件で比較し、leaderboard を出す最小実装を作る。

P0では:
- 2×2 程度の組合せ（decompose×coeff_post）を回せればOK
- `tools/leaderboard.py` で outputs を集計し、CSVを出す

## Acceptance Criteria
- [x] benchmarkが複数runを生成し、metricsが揃う
- [x] leaderboard が CSV を出力できる

## Verification
- [x] FFT+PCA vs Zernike+PCA で leaderboard に差が出る

---

### Review Map（必須）
- **変更ファイル一覧**: `src/processes/benchmark.py`, `src/processes/leaderboard.py`, `src/mode_decomp_ml/tracking/leaderboard.py`, `tools/leaderboard.py`, `configs/task/benchmark.yaml`, `configs/task/leaderboard.yaml`, `src/mode_decomp_ml/pipeline/utils.py`
- **重要な関数/クラス**: `src/processes/benchmark.py:main`, `src/mode_decomp_ml/tracking/leaderboard.py:collect_rows`, `src/mode_decomp_ml/tracking/leaderboard.py:write_leaderboard`, `src/processes/leaderboard.py:main`, `tools/leaderboard.py:main`
- **設計判断**: benchmark は組合せごとに train/predict/reconstruct/eval を run_dir 配下へまとめ、eval の `metrics.json` と `.hydra/config.yaml` を leaderboard で拾えるようにした。leaderboard は指標を 1 行に集約し、配列指標は JSON 文字列として保持して列爆発を避けた。disk domain を benchmark で固定し FFT は mask_zero_fill に統一。
- **リスク/注意点**: synthetic の num_samples=1 だと PCA が RuntimeWarning を出す（n_samples-1）。必要なら task 側で dataset overrides を増やす。energy_cumsum は配列として JSON 文字列で出力されるため数値列としては並び替え不可。
- **検証コマンドと結果**: `python -m mode_decomp_ml.cli.run task=benchmark`（成功、PCA Warning あり）; `python tools/leaderboard.py outputs/benchmark/2025-12-29/02-45-27/**/eval --out outputs/benchmark/2025-12-29/02-45-27/leaderboard.csv --md outputs/benchmark/2025-12-29/02-45-27/leaderboard.md`（4行出力）
- **削除一覧**: なし
