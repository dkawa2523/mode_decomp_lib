# Task 020 (P0): Hydra導入とconfigグループ設計（MVP）

## 目的
本プロジェクトの“設定が真実”を実現するため Hydra を導入し、
分解・係数後処理・モデルを切り替え可能な config グループ構造を作る。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/010_skeleton.md

## スコープ
### In
- Hydraを導入し、`conf/` を作成する（docs/03に準拠）
- 最低限の defaults 構成（data/domain/preprocess/decompose/coeff_post/model/process）を用意
- run dir（outputs/<process>/...）の統一
- config snapshot と meta.json を必ず保存

### Out
- 全手法のconfig作成（必要最小のみ）
- ClearML連携（後タスク）

## 実装方針（Codex向け）
### 1) 入口をHydra化する
- `processes/*.py` の `main()` を Hydra でラップ
- `@hydra.main(version_base=None, config_path="../../conf", config_name="config")` のように固定
- processごとに `conf/process/<name>.yaml` を用意し defaults で切替

### 2) configグループ（まずP0のみ）
- `conf/decompose/zernike.yaml`
- `conf/decompose/fft2.yaml`
- `conf/decompose/dct2.yaml`
- `conf/coeff_post/pca.yaml`（fit必要）
- `conf/model/ridge.yaml`（baseline）
- `conf/preprocess/basic.yaml`（scale + missing + detrend）

### 3) outputsとartifact
- Hydraのrun dir を `outputs/${process.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}_${tag}` に
- `meta.json` を必ず生成（seed, git hash, versions, dataset hash）

### 4) 禁止事項
- config に載っていないパラメータをコードで勝手に決めない
- 例外：テスト用の固定値（tests内）は可

## ライブラリ候補
- hydra-core
- omegaconf
- python-dotenv（必要なら環境変数）

## Acceptance Criteria（完了条件）
- [ ] Hydraで `process=train` 等の切替ができる
- [ ] run dir に config snapshot と meta.json が必ず残る
- [ ] decompose/method と coeff_post/method と model/method を config で変更できる

## Verification（検証手順）
- [ ] `python -m processes.train process=train decompose=fft2 coeff_post=pca model=ridge` が起動する
- [ ] run dir に `hydra/` と `meta.json` が存在する
