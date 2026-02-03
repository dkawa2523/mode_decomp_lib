# Execute Task Prompt

必ず読む：
- codex/SESSION_CONTEXT.md
- docs/00_INVARIANTS.md
- work/queue.json
- work/tasks/<task>.md
- 指定skills

やること：
- タスクmdのPlan/Acceptanceに従って実装
- テストを追加/更新
- 検証コマンドを提示
- 完了したら queue status を done に更新

実装不能なら：
- blocked + reason/unblock_condition/next_action を追記し、
  解除子タスクを作って queue に追加（unblocks必須）

禁止：
- “確認してください”で止めない
