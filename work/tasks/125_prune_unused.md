# Task 125 (P1): 不要コード/不要ファイルの棚卸しと削除（定期クリーンアップ）

> **⚠️ DO NOT CONTINUE**: このタスクの Acceptance Criteria / Verification をすべて満たすまで、次のタスクに進まないこと。

## 目的
- 置き換えや試行で発生した **未使用コード/未使用ファイル/未使用ディレクトリ** を削除し、保守コストを抑える
- 「どこが重要か分からない」状態を避け、レビュー容易性を上げる（docs/15）

## 背景 / 根拠（必読）
- docs/14_COMPACT_CODE_POLICY.md
- docs/15_REVIEWER_GUIDE.md
- docs/16_DELETION_AND_PRUNING.md
- docs/07_TESTING_CI.md

## スコープ
### In
- src/ 配下の未使用モジュール/関数/分岐の削除
- configs/ の未使用設定ファイルの削除（参照されないもの）
- tools/ の重複スクリプト、未使用テンプレの削除
- docs の読み順や参照の整理（不要になった記述の削除）
- import整理・型/フォーマットの軽微な修正（削除に伴うもの）

### Out
- 大規模リネームやアーキテクチャ刷新（必要ならRFC/ADRへ）

## 手順（推奨）
1. **現状把握**
   - `git status` がクリーンであること
   - `python tools/codex_prompt.py list` で現在のWIP確認（WIP=1）
2. **未使用物の候補抽出**
   - `git grep` / `rg` で参照されないファイルや死んだ分岐を探す
   - “使ってなさそう” を根拠（参照0、設定未参照、Process未登録など）として記録
3. **削除**
   - ファイル削除 → import修正 → テスト/スモーク
4. **レビュー用メモ**
   - 何を消したか（削除リスト）と理由（未参照/置換済み）を残す

## Acceptance Criteria（完了条件）
- [ ] 未使用ファイル/未使用ディレクトリが削除されている（根拠付きの削除リストあり）
- [ ] `python tools/codex_prompt.py doctor` が 0 issue
- [ ] 最低限の検証（pytest or smoke）が通る
- [ ] docs/README.md の読む順・参照が最新になっている

## Verification（検証手順）
- [ ] `python tools/codex_prompt.py doctor`
- [ ] `python -m compileall src`
- [ ] `pytest -q`（pytest未導入なら最小smokeに置換し、理由を記録）

## Review Map（必須：レビュワー向け）
- 削除したファイル一覧（削除理由と参照確認方法）
- 重要な入口（process/pipeline）が変わっていないことの確認
- 検証コマンドと結果（要点）
