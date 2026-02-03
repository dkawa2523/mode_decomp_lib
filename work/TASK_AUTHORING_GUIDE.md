# Task起票ガイド（拡張をぶらさずに増やす）

このガイドは、将来の拡張（特徴量化、モデル追加、別I/O、可視化、ClearML、リファクタ等）を
**Codexが自律的にTask化**できるようにするための手順です。

## 不変（まず読む）
- docs/00_INVARIANTS.md
- docs/09_EVALUATION_PROTOCOL.md
- docs/10_PROCESS_CATALOG.md
- docs/11_PLUGIN_REGISTRY.md
- docs/14〜17（コンパクト/レビュー/削除/拡張）

## ルール
- 1Task = 1実装単位（E2Eで検証できる最小粒度）
- 依存関係は必ず `depends_on` で明示
- blockedにする場合は解除子タスクを起票し、`unblocks` を必ず付ける
- WIPは1（in_progressは同時に1つまで）

## 新規Taskの作り方
### 方法A: 手動（確実）
1. `work/templates/TASK.md` をコピーして `work/tasks/<id>_<slug>.md` を作成
2. `work/queue.json` の `tasks[]` に追記
   - fields: id, priority, status(todo), title, path, skills, contracts, depends_on, unblocks
3. `python tools/codex_prompt.py doctor` を実行し整合性チェック

### 方法B: 自動生成（推奨）
`tools/taskgen.py` を使うと、mdとqueue追記を自動化できます。

例：
```bash
python tools/taskgen.py   --id 130   --priority P1   --title "不要コード/不要ファイルの棚卸しと削除"   --path work/tasks/130_prune_unused.md   --skills S95_tests_ci,S00_repo_onboarding   --depends_on 080,095   --contracts docs/00_INVARIANTS.md,docs/16_DELETION_AND_PRUNING.md,docs/07_TESTING_CI.md
```

## Task IDの目安
- 既存の流れを崩さないため、基本は “現在の最大IDより大きい番号” を使う
- 連番でも良いが、カテゴリで固めたい場合は 130/140/150… のようにブロックを取る

## Codexに依頼する時のコツ
- 「質問で止まるな」「TODOで前進」「Review Mapを残せ」「不要物は削除」を必ず入れる
- 実装が大きい場合は RFC/ADR に分割（work/templates）
