# 評価拡張（coeff誤差 + field誤差 + スペクトル診断）

**ID:** 090  
**Priority:** P0  
**Status:** done  
**Depends on:** 080  
**Unblocks:** 100  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
比較可能性のため、評価指標を固定する。
最低限:
- coeff_rmse（a空間、z空間）
- field_rmse（mask内）
- energy_cumsum（スペクトル/次数ごとの累積）

## Acceptance Criteria
- [x] metrics が json で保存され、leaderboardが読める
- [x] mask がある場合、mask内だけ評価する
- [x] スペクトル診断（累積エネルギー）が保存される

## Verification
- [x] 2手法（FFT vs Zernike）で metrics を比較できる

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/evaluate/__init__.py`, `src/processes/eval.py`, `configs/eval/basic.yaml`, `tests/test_processes_e2e.py`, `docs/09_EVALUATION_PROTOCOL.md`
- 重要な関数/クラス: `src/mode_decomp_ml/evaluate/__init__.py` の `coeff_energy_cumsum`/`compute_metrics`, `src/processes/eval.py` の `main`
- 設計判断: energy_cumsum は a-space 係数から算出し、Zernikeは次数ごとに累積、それ以外は係数順で累積する。
- リスク/注意点: FFTの複素係数は real/imag を合成してエネルギー化するため、周波数順の解釈は viz 側で補助が必要。
- 検証コマンドと結果: `pytest tests/test_processes_e2e.py`（PASS）、`python - <<'PY' ...`（FFT vs Zernike パイプライン実行で metrics 互換性確認 / PASS）
- 削除一覧: なし
