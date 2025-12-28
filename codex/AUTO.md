# AUTO Rules（反復運用の不変ルール）

1) WIP制限
- in_progress があれば最優先で完了させる（WIPを増やさない）

2) blocked運用
- 順番待ちは depends_on を使う（blockedにしない）
- blocked にする場合は解除子タスクを作成し、子タスクの queue に `unblocks:["親ID"]` を必ず付ける

3) “確認してください”禁止（autopilot）
- 質問して止まらない
- 次のいずれかを必ず選ぶ：
  A) 実装してdone
  B) blocked + 解除タスク起票
  C) queue/taskの状態ズレ修正→A

4) 出力要件
- 変更計画 → 実装 → テスト → 検証コマンド → 互換性影響 → queue更新
