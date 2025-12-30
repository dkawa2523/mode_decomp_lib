# docs/ README

## 読む順番（推奨）
1. 00_INVARIANTS.md（最重要：不変条件）
2. 02_DOMAIN_MODEL.md（データ/係数/Processの概念）
3. 01_ARCHITECTURE.md（構成と依存方向）
4. 03_CONFIG_CONVENTIONS.md（Hydra前提：group/override/run_dir/seed）
5. 04_ARTIFACTS_AND_VERSIONING.md（artifact契約）
6. 05_SECURITY_SECRETS.md（セキュリティ/秘密情報）
7. 09_EVALUATION_PROTOCOL.md（比較・評価のルール）
8. 10_PROCESS_CATALOG.md（Process一覧とI/O）
9. 11_PLUGIN_REGISTRY.md（分解/後処理/モデルの拡張方法）
10. 07_TESTING_CI.md（テスト方針）
11. 12_INTEGRATIONS_READY.md（ClearML等を見越す）
12. 13_TASK_FLOW.md（タスク運用ルール：止まらず、かつ中途半端に進めない）
13. 14_COMPACT_CODE_POLICY.md（コード量を増やさない：コンパクト実装）
14. 15_REVIEWER_GUIDE.md（レビューしやすさ：重要箇所の特定）
15. 16_DELETION_AND_PRUNING.md（不要コード/不要ファイルを残さない）
16. 17_EXTENSION_PLAYBOOK.md（拡張の起票・実装・比較の手順）
17. 18_SCOPE_LOCK.md（タスクが増え続けない運用）
18. 19_BACKLOG_QUEUES.md（P1/P2は別queueで管理）
19. 20_METHOD_CATALOG.md（手法カタログ：一次分解 + 係数後処理）
20. 98_CODEX_AUTOMATION_LESSONS.md（Codex運用のコツ）
21. 99_AUTOPILOT_TROUBLESHOOTING.md（詰まった時）

## 原則
- `docs/` は「不変の契約」です。破る場合は必ず `work/rfcs/` を起票してください。
- 実装の単位は `Process`（単独実行可能）です。各ProcessはI/Oとartifactが明確であること。
- コード量を増やさない / 読みやすさ / 削除徹底は **契約** として扱います（docs/14〜16）。
