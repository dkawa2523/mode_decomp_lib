# Task 120 (P2): ClearML統合に備えたtracking hook（導入はしない）

## 目的
将来 ClearML を導入するときに変更範囲を最小化するため、
“今は導入しない” 前提で、Process入口に tracking hook を入れられる設計を用意する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/020_hydra_mvp.md

## スコープ
### In
- config に `tracking.enabled` を追加
- tracking有効時に呼ばれる `core/tracking.py` を用意（現状はno-op）
- artifactのアップロード対象（model/metrics/viz）をリスト化

### Out
- ClearML依存追加・認証設定

## 実装方針（Codex向け）
### 1) hookの位置
- すべての Process の `main()` 先頭で `tracking.init(cfg)` を呼ぶ
- enabled=false なら no-op

### 2) 将来の差分
- ClearMLを入れる時は `core/tracking_clearml.py` を追加して差し替えるだけにする

## ライブラリ候補
- （今は追加しない）clearml
- python標準（no-op）

## Acceptance Criteria（完了条件）
- [ ] tracking.enabled が存在し、falseなら挙動が変わらない
- [ ] enabled=true にしても no-op で動く

## Verification（検証手順）
- [ ] tiny run を tracking.enabled=true で実行しても壊れない
