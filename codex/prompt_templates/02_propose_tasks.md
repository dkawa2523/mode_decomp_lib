# Propose Tasks Prompt（拡張要求→タスク化）

必ず読む：
- docs/00_INVARIANTS.md
- docs/09_EVALUATION_PROTOCOL.md
- docs/10_PROCESS_CATALOG.md
- docs/11_PLUGIN_REGISTRY.md
- docs/14〜17（コンパクト/レビュー/削除/拡張）
- work/queue.json

目的：
- ユーザー要求（新しい特徴量化、モデル追加、I/O対応、可視化拡充、ClearML、リファクタ等）を、
  “比較基盤を壊さず” 実装可能なタスク列へ分解する。

要求：
- 10〜20タスク程度（P0/P1中心、必要ならP2）
- 各タスクに必ず：
  - Acceptance Criteria（完了条件）
  - Verification（検証）
  - depends_on / unblocks（必要なら）
- コードはコンパクトに（docs/14）
- レビュー容易性（docs/15）：Review Mapを残せる粒度に分割
- 不要物削除（docs/16）：置き換えで死ぬ物は削除するタスクも含める

出力：
- work/tasks/<id>_<slug>.md を作成
- work/queue.json に追記
- `python tools/codex_prompt.py doctor` が通る状態にする
