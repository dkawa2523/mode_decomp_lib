# Autopilot Mode Prompt（質問禁止・必ず前進）

あなたは自動反復（autopilot）モードで動作する。

ルール：
- 質問して止まらない
- 状態の真実は work/queue.json
- task md の Blocked 記述が古い場合は更新し、queueがtodo/in_progressなら実装を進める
- 進められない場合は blocked + 解除子タスク（unblocks必須）で前進する


## 追加ルール（このプロジェクト固有）
- 次のタスクに進む前に、**必ず**現在タスクの Acceptance Criteria / Verification を満たすこと
- ユーザー確認が必要そうでも、質問で止まらない：安全側のデフォルトを選び、TODOとして明記して前進する
- queue 上の **WIPは1**：他タスクに手を付けない（必要なら現在タスクを blocked にし、解除タスクを起票）
- `done` にする前に、検証コマンドを実行し、結果をタスクmdに追記する（証跡）


## NO TASK CREEP（必須）
- 実行中に `work/queue.json` に新規タスクIDを追加してはいけない。
- 追加作業が必要なら、当該タスクの `## TODO (post-P0)` に箇条書きでまとめ、必要なら `work/rfcs/` にRFCを作って止める。
- P1/P2の拡張は `work/queue_p1.json` / `work/queue_p2.json` に起票する（P0を汚染しない）。
