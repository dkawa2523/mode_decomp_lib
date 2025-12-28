# Task 095 (P0): Doctor + smoke test + 最小pytest整備

## 目的
多人数開発で壊れやすいポイント（I/O、mask、係数meta、再現性）を早期に検知するため、
doctorコマンドと最小のpytestを整備する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/080_process_e2e.md

## スコープ
### In
- `processes/doctor.py` を実装（環境/依存/データ存在/設定の妥当性チェック）
- tiny dataset を使った end-to-end smoke を1本用意
- unit test: Zernike round-trip, PCA round-trip, FFT round-trip を追加（可能な範囲で）

### Out
- CI導入はP1（GitHub Actions等）

## 実装方針（Codex向け）
### 1) doctorが見るもの
- python version / 주요ライブラリ versions
- GPU有無（torch）
- data path の存在
- domain config の妥当性（diskなら中心/半径など）
- methodのregistry key が存在するか

### 2) smoke test
- `python -m processes.pipeline_run data=tiny decompose=zernike coeff_post=pca model=ridge`
- 成功したら metrics.json が出ていることを確認

### 3) pytest
- `tests/test_zernike_roundtrip.py`
- `tests/test_fft_roundtrip.py`
- `tests/test_pca_roundtrip.py`
- 大きなデータは使わず合成で

## ライブラリ候補
- pytest
- numpy
- scipy（必要なら）

## Acceptance Criteria（完了条件）
- [ ] doctor が失敗原因を明確に出す
- [ ] smokeが1コマンドで通る
- [ ] pytestが最低3本ある

## Verification（検証手順）
- [ ] `python -m processes.doctor` が成功する
- [ ] `pytest -q` が通る（環境依存はskipで対応）
