# Task <ID> (<Priority>): <Title>

> **⚠️ DO NOT CONTINUE**: このタスクの Acceptance Criteria / Verification をすべて満たすまで、次のタスクに進まないこと。  
> （依存タスクの都合で後戻りが必要なら、queueのstatusを `in_progress` に戻し、Reopen欄に追記する）

## 目的
- （このタスクで達成したいこと）

## 背景 / 根拠（必読）
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/02_DOMAIN_MODEL.md
- docs/10_PROCESS_CATALOG.md
- （必要なら）docs/20_METHOD_CATALOG.md

## コンパクト実装ポリシー（必須）
- docs/14_COMPACT_CODE_POLICY.md に従う
  - “将来のため”の過剰抽象化をしない
  - 既存ライブラリ/既存実装で代替できるなら新規実装しない
  - 置き換えで不要になったコード/ファイルは削除（docs/16）

## スコープ
### In
- （やること）

### Out
- （やらないこと：将来の拡張は別タスクへ）

## 変更計画（Plan）
- 追加/変更/削除するファイルを列挙
- 入口（process/pipeline）から辿れる導線を明記

## 実装メモ（任意）
- 重要な設計判断、トレードオフ、注意点

## 削除・整理（必須）
- このタスクで不要になったものを列挙し、**削除する**
  - 例：未使用関数、古いconfig、使われないスクリプト、不要な中間ファイル生成
- 削除できない場合は理由と削除期限タスク（unblocks）を起票

## Acceptance Criteria（完了条件）
- [ ] ...
- [ ] ...

## Verification（検証手順）
- [ ] コマンド例: `python -m ...`
- [ ] 期待する出力/ログ

## Review Map（必須：レビュワー向け）
- 変更ファイル一覧（追加/変更/削除）
- 重要な入口/関数/クラス（どこを見れば良いか）
- 設計判断（2〜5行）
- リスク/注意点（skew/coeff_meta/mask/boundary等）
- 検証コマンドと結果（要点）

## Reopen（後戻りする場合に追記）
- Reopen Reason:
- Affected Tasks:
- Fix Plan:
