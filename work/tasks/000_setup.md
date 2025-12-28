# Task 000 (P0): Repo棚卸しと最小I/Oの確定

## 目的
既存のモード分解コード（Zernike展開）とデータセットの入出力を棚卸しし、
このdevkitの Domain Model（docs/02）に合わせて **最小のI/O契約** を確定する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## スコープ
### In
- 既存repoの入口スクリプト/ノートブック/関数を列挙し、実行フローを図にする
- 入力データの形式（field/mask/condition/meta）を整理し、サンプルIDの定義を確定
- 現状のZernike分解の係数の並び（(n,m)順序、実装上の規約）を `coeff_meta` として言語化
- tiny dataset（最小サンプル数）を作って smoke 実行できる状態にする

### Out
- 大規模リファクタやHydra導入（次タスクで実施）
- 新規手法の追加（次タスク以降）

## 実装方針（Codex向け）
### 1) 棚卸し成果物を作る
- `docs/` ではなく repo 側に `project_docs/REPO_CONTEXT.md` を追加（または既存に追記）
  - 入口（train/eval相当）  
  - 主要モジュール  
  - データI/O（どこで読み、どう前処理し、どこに保存するか）  
  - 係数定義（Zernikeのindex、正規化、mask扱い）  

### 2) I/O契約を確定する（最小）
- `FieldSample` 相当の dataclass を1つ作る（後で拡張）
- `field: (H,W,C)` `mask: (H,W)` `condition: (D,)` を最低限揃える
- 既存コードの内部表現が違う場合は adapter を作って合わせる

### 3) tiny dataset fixture
- 実データの一部を `data/tiny/` にコピーするか、合成データ生成で代替
- `python -m <entry> --config-name=tiny` で最後まで動く経路を作る

## ライブラリ候補
- numpy（配列）
- pydantic または dataclasses（型）
- pytest（smoke）

## Acceptance Criteria（完了条件）
- [ ] 現状フローが `project_docs/REPO_CONTEXT.md` にまとまっている（TODO可、捏造禁止）
- [ ] FieldSample相当の最小I/Oが定義され、tiny datasetで1回走る
- [ ] `coeff_meta`（Zernikeの係数並び/正規化）が文章で説明され、保存できる

## Verification（検証手順）
- [ ] `python -m <entry> ...` で tiny を最後まで実行できる（成功ログ）
- [ ] （可能なら）`pytest -q` で smoke が1本通る
