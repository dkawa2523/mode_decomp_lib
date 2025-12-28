# Task 100 (P1): Leaderboard集計（run dirから自動で比較表生成）

## 目的
複数手法・複数runの比較評価を楽にするため、
run dir を走査して metrics/config を集計する leaderboard を実装する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/085_metrics_module.md

## スコープ
### In
- `processes/leaderboard.py` を実装（複数run dir入力）
- metrics.json と hydra config を読み、1行=1runの表を作る
- キー（decompose/coeff_post/model/seed/dataset hash）を列として出力
- CSV + Markdown を生成

### Out
- Webダッシュボード（P3）

## 実装方針（Codex向け）
### 1) 入力
- `--runs outputs/**/` のように glob で受け取る
- metrics.json が無いrunはスキップ/警告

### 2) 出力
- `leaderboard.csv`
- `leaderboard.md`
- sorting: primary metric（config指定）

### 3) 比較不能の扱い
- dataset hash が違うものは別グループに分ける
- coeff_meta mismatch は flag を立てる

## ライブラリ候補
- pandas
- pyyaml / omegaconf（config読み）
- glob

## Acceptance Criteria（完了条件）
- [ ] 複数run dirから集計できる
- [ ] 比較キーが表に含まれている
- [ ] CSV/Markdownが生成される

## Verification（検証手順）
- [ ] tinyで2種類のrunを作り、leaderboardで2行出ることを確認
