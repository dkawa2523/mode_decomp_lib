# タスク運用ルール（Autopilotで止まらず、かつ中途半端に進めないために）

## 目的
- Autopilot（tools/autopilot.sh）が途中停止しない
- ただし「未完成のまま次へ進む」事故を避ける（後から大規模リファクタが必要になる状態を避ける）

## 不変ルール
- **WIPは常に1**（work/queue.json の `in_progress` は1つまで）
- 現在タスクの **Acceptance Criteria** と **Verification** を満たすまで `status=done` にしない
- 不明点があっても「確認してください」で止めない。次のいずれかを必ず選ぶ：
  1) 実装して `done`
  2) `blocked` にして理由/解除条件を書き、**解除子タスク（unblocks）を必ず起票**
  3) queue/task の状態ズレを修正して 1) に戻る

## タスクをやり直す（後戻り）のルール
- 依存タスクの後で問題が見つかったら、対象タスクの `status` を `in_progress` に戻してよい
- その際、タスクmdに以下を追記する：
  - Reopen Reason（何が見つかったか）
  - Affected Tasks（影響範囲）
  - Fix Plan（今回の修正計画）
- 修正完了し検証が通ったら、再度 `done` に戻す

## Autopilotが止まりそうな典型と対策
- doctor 失敗（task md欠落 / depends_on 欠落 / blockedの解除子なし）
  - → まず doctor を直し、キュー整合を回復する
- Codexが「方針を質問」しそう
  - → TODOを書いて安全側で仮実装し、検証可能な形にする


## 追加：コード量・削除のルール（必須）
- タスク中に不要になったコード/ファイルは **そのタスクで削除** する（docs/16）
- 過剰抽象化や不要機能の追加を避け、コンパクトに保つ（docs/14）
- タスク完了時に Review Map を残す（docs/15）
