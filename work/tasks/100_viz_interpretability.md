# 可視化（再構成比較・係数スペクトル・解釈性）

**ID:** 100  
**Priority:** P0  
**Status:** done  
**Depends on:** 090  
**Unblocks:** 110  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
研究開発で“見て判断”できる最低限の可視化を実装する。
- field_true vs field_hat の画像
- error map（mask内）
- 係数スペクトル（次数/周波数 vs energy）
- 逐次再構成（k=1,2,4,8...）

## Acceptance Criteria
- [x] png が outputs に保存される
- [x] 手法別に比較しやすいファイル名/構造になっている

## Verification
- [x] 2 run の可視化を並べて違いが分かる

---

### Review Map（必須）
- **変更ファイル一覧**: `src/mode_decomp_ml/viz/__init__.py`, `src/processes/viz.py`, `configs/task/viz.yaml`, `configs/viz/basic.yaml`, `configs/config.yaml`, `tests/test_processes_e2e.py`
- **重要な関数/クラス**: `src/processes/viz.py:main`, `src/mode_decomp_ml/viz/__init__.py:plot_field_grid`, `src/mode_decomp_ml/viz/__init__.py:plot_error_map`, `src/mode_decomp_ml/viz/__init__.py:coeff_energy_spectrum`
- **設計判断**: 係数スペクトルは `coeff_meta` に基づき Zernike は次数集計、FFT/DCT は周波数平面ヒートマップで最小限に統一。逐次再構成は `coeff_meta` のフラット順で切り詰め、複素係数は偶数境界に合わせて比較可能性を維持。出力は `viz/sample_XXXX/` 配下の固定ファイル名にして run 間比較を容易化。
- **リスク/注意点**: FFT/DCT の逐次再構成はフラット順なので周波数順ではない。`viz.sample_index` の単一サンプル前提のため、比較は同じ index を使うこと。mask 外は NaN 表示で除外。
- **検証コマンドと結果**: `pytest tests/test_processes_e2e.py`（未実行）
- **削除一覧**: なし
