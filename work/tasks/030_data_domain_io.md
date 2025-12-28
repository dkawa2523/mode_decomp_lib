# Task 030 (P0): データI/O・DomainSpec・splitの共通化

## 目的
スカラー/ベクトル場・mask・domain情報・condition を統一的に扱えるI/O層を作り、
分解/学習/評価の全工程で同じデータモデルを使えるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/020_hydra_mvp.md

## スコープ
### In
- `FieldSample` / `DomainSpec` を `src/data` で実装（docs/02）
- dataset loader（raw -> FieldSample）を実装
- split戦略（random/group）を実装し、seedで固定
- dataset hash / split meta を artifact として保存

### Out
- 大量データ最適化（memmap等）は後回し
- mesh-domainの本格対応（P2）

## 実装方針（Codex向け）
### 1) DomainSpec を最小実装
- `rect`: x,y grid（均一格子前提でOK）
- `disk`: disk mask + 正規化座標（r<=1）を提供できること
- `mask`: 任意mask（x,yはrectと同等扱いでも可）
- 将来の `points/mesh` は placeholder を作って TODO

### 2) dataset loader
- 現状データセットのフォーマットに合わせて `Dataset` クラスを作る
- 返すのは `FieldSample`
- maskが無い場合は全Trueで生成（明示）

### 3) split
- `split.py` に `make_split(sample_ids, strategy, seed, groups=None)` を実装
- split結果（train/val/test の sample_idリスト）を保存

### 4) 小さな便利機能
- `to_numpy()`、`validate_shapes()`、`apply_mask()` などのユーティリティ

## ライブラリ候補
- numpy
- pandas（条件テーブルがCSVなら）
- scikit-learn（train_test_split / GroupKFold など）

## Acceptance Criteria（完了条件）
- [ ] FieldSample/DomainSpec が実装され、全Processで使える
- [ ] splitがseedで再現し、split meta が保存される
- [ ] dataset hash が保存され、比較可能性の根拠になる

## Verification（検証手順）
- [ ] tiny dataset を loader で読み、FieldSampleのshape検証が通る
- [ ] 同じseedでsplitが同一になることを確認（テスト or ログ）
