# docs/ README

## 読む順番（推奨）
1. 00_INVARIANTS.md（最重要：不変条件）
2. USER_QUICKSTART.md（run.yaml 入口の最短手順）
3. 02_DOMAIN_MODEL.md（データ/係数/Processの概念）
4. 01_ARCHITECTURE.md（構成と依存方向）
5. 03_CONFIG_CONVENTIONS.md（Hydra + run.yaml の運用ルール）
6. 04_ARTIFACTS_AND_VERSIONING.md（artifact契約）
7. 05_SECURITY_SECRETS.md（セキュリティ/秘密情報）
8. 09_EVALUATION_PROTOCOL.md（比較・評価のルール）
9. 10_PROCESS_CATALOG.md（Process一覧とI/O）
10. 11_PLUGIN_REGISTRY.md（分解/後処理/モデルの拡張方法）
11. 07_TESTING_CI.md（テスト方針）
12. 12_INTEGRATIONS_READY.md（ClearML等を見越す）
13. 13_TASK_FLOW.md（タスク運用ルール：止まらず、かつ中途半端に進めない）
14. 14_COMPACT_CODE_POLICY.md（コード量を増やさない：コンパクト実装）
15. 15_REVIEWER_GUIDE.md（レビューしやすさ：重要箇所の特定）
16. 16_DELETION_AND_PRUNING.md（不要コード/不要ファイルを残さない）
17. 17_EXTENSION_PLAYBOOK.md（拡張の起票・実装・比較の手順）
18. 18_SCOPE_LOCK.md（タスクが増え続けない運用）
19. 19_BACKLOG_QUEUES.md（P1/P2は別queueで管理）
20. 20_METHOD_CATALOG.md（手法カタログ：一次分解 + 係数後処理）
21. 14_OPTIONAL_DEPENDENCIES.md（optional依存一覧）
22. 24_DECOMPOSER_RECOMMENDATIONS.md（decomposer推奨設定）
23. 25_AUTOENCODER_GUIDE.md（autoencoder運用ガイド）
24. 26_LOGGING_POLICY.md（loggingの運用方針）
25. 27_CODE_MAP.md（コード構成と処理動線）
26. 28_COEFF_META_CONTRACT.md（coeff_metaの最低限契約）
27. 98_CODEX_AUTOMATION_LESSONS.md（Codex運用のコツ）
28. 99_AUTOPILOT_TROUBLESHOOTING.md（詰まった時）

## docs/addons 目次
- 25_AFTER_240_NEXT_STEPS.md（240以降の差分メモ）
- 26_FILE_ADDITIONS_AFTER_240.md（追加ファイル一覧）
- 30_POD_SUITE_SPEC.md（POD評価スイート仕様）
- 31_POD_CONFIG_MINIMAL.md（POD最小設定）
- 32_POD_VISUALIZATION_STANDARD.md（POD可視化標準）
- 32_SPECIAL_FUNCTION_SUITE.md（特殊関数スイート）
- 33_CLEARML_READY_NOTES.md（ClearML対応メモ）
- 34_POD_REFERENCES.md（POD参考）
- 35_DATASET_TEMPLATE_SAMPLES.md（dataset テンプレ）
- 40_PLAN_STATUS_MATRIX_AFTER_P2_300.md（P2計画状況）
- 41_MESH_LB_IMPLEMENTATION_GUIDE.md（Mesh/LB実装）
- 42_ARTIFACTS_FOR_GEOMETRY_AND_DL.md（幾何/DL artifact）
- README_AFTER_240.md（240以降のREADME）
- README_AFTER_P2_300.md（P2_300以降のREADME）

## 原則
- `docs/` は「不変の契約」です。破る場合は必ず `work/rfcs/` を起票してください。
- 実装の単位は `Process`（単独実行可能）です。各ProcessはI/Oとartifactが明確であること。
- コード量を増やさない / 読みやすさ / 削除徹底は **契約** として扱います（docs/14〜16）。
- 旧プリセットは cleanup で削除済み。新規 YAML を増やさず `run.yaml` + `params` を優先する。
