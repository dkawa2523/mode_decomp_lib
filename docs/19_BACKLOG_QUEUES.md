# backlog queues（P1/P2）

P0を “終わらせる” ため、拡張タスクは別queueで管理します。

- `work/queue_p1.json` : P1（Bessel/POD/GPR、任意mask、sweep等）
- `work/queue_p2.json` : P2（Graph/FEM/LB、曲面/メッシュ等）

## 運用
- P0が完了したら、次に実施したいqueueを `work/queue.json` に置き換えて Autopilot を回してください。
  例）
  ```bash
  cp work/queue_p1.json work/queue.json
  LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 30
  ```

- 進捗を失いたくない場合は、queueは別名のまま保持し、Autopilotの対象queueを切り替える運用でも良いです
  （必要なら tools/autopilot.sh を拡張して queue path を引数化します）。
