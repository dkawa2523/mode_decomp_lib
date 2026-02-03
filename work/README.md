# work/ README

この `work/` は **実装を前に進めるためのキュー**です。  
Codex を使う場合は、`work/queue.json` の **P0から順に** `work/tasks/*.md` を投入して実装します。

## フロー（推奨）
1. `docs/00_INVARIANTS.md` を読んで契約を理解する
2. `work/tasks/<id>_*.md` を開く
3. Codex へ「このタスクを完了して。Acceptance CriteriaとVerificationを満たして」と指示
4. 変更後、`Verification` を必ず実行（少なくとも smoke/pytest）
5. 完了したら次のタスクへ

## blocked運用
- 依存があるタスクは `depends_on` を queue.json に書く
- 「blocked」は、解除用の子タスクを作り `unblocks` で紐付ける（契約）

## タスクの粒度
- 1タスク = 1つの実装単位（1〜3日程度を想定）
- 大きすぎたら分割する（RFC不要、タスク追加でOK）
