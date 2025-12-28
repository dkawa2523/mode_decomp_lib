# Bootstrap Prompt

あなたはこのリポジトリに devkit（docs/work/agentskills/codex/tools）を導入します。

必ず読む：
- docs/00_INVARIANTS.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md
- agentskills/ROUTER.md

やること：
1) 現状の入口（train/eval/predict）とデータI/Oを棚卸し
2) Processカタログを現実に合わせて更新（TODO明記）
3) P0タスクを queue に起票（depends_on / unblocks を使う）

禁止：
- いきなり大規模リファクタしない（必ずタスク化）
