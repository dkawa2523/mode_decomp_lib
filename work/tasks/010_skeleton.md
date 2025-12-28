# Task 010 (P0): プロジェクト骨格の導入（Process中心の構成へ）

## 目的
分解手法やモデルが増えても仕様がぶれないよう、`Process` 中心のディレクトリ構成と依存方向を導入する。
以降のタスクはこの骨格に新規プラグインを追加していく。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/000_setup.md

## スコープ
### In
- `src/` 配下に core/data/preprocess/vector/decompose/coeff_post/models/metrics/processes を作る
- Process entrypoint（CLI）を統一する（例: `python -m processes.train`）
- 既存コード（Zernike分解・回帰など）を新構成へ移すか、ラッパで呼べるようにする
- ログ/シード固定の共通ユーティリティを core に置く

### Out
- Hydra導入（次タスク）
- 新しい分解手法の追加（Task 050〜）

## 実装方針（Codex向け）
### 1) 依存方向を守る
- `core` は下位層、他は `core` に依存して良い
- `processes` は上位層で、必要な機能を呼び出すだけ（ロジックは各モジュール）

### 2) “入口” を揃える
- `processes/` に `train.py`, `eval.py`, `predict.py`, `reconstruct.py`, `viz.py`, `leaderboard.py`, `doctor.py` を用意
- まだ中身が薄くても良いので、呼び出しとI/Oだけ先に固定

### 3) 既存コードの移植戦略
- 直接移すのが難しい場合：
  - `legacy/` に既存モジュールを置き、`decompose/zernike.py` から呼ぶ
  - 後で段階的に置換する（TODOを残す）

### 4) 最小のログ/seed
- `core/seed.py` に `set_global_seed(seed)`（numpy/random/torch対応）
- `core/logging.py` に `get_logger(name)`

## ライブラリ候補
- python標準（argparse / logging） ※Hydraは次タスク
- numpy
- torch（ある場合：seed固定のみ先に）

## Acceptance Criteria（完了条件）
- [ ] 新しいディレクトリ構造が追加され、importの循環がない
- [ ] Process入口が最低限動く（tinyデータで no-op でもOK）
- [ ] 既存Zernike分解が新構造から呼べる（暫定ラッパでもOK）

## Verification（検証手順）
- [ ] `python -m processes.doctor` が起動して環境情報を表示できる
- [ ] tinyデータで `python -m processes.reconstruct` が最後まで走る（暫定でも）
