# Add-on: P1(残り) + P2（After P1 Task 240）

あなたの進捗（終了タスク）:
- P0: 000〜120 ✅
- P1: 200/210/220/230/240 ✅

このzipは、**P1の残り（250/260/270/290）** と **P2キュー**を追加し、
既存の実装/成果を壊さずに次へ進めるための “差分 overlay” です。

## 1) 適用（上書き展開）
プロジェクトルートで:
```bash
unzip -o mode_decomp_greenfield_addon_after_240_v1_flat.zip -d .
```

## 2) queue を更新（推奨：自動パッチ）
現在の `work/queue.json` に残タスクを安全に追加します（バックアップ付き）:
```bash
python tools/_legacy/apply_addon_after_240.py
```

- これにより、タスク **250/260/270/290** が `work/queue.json` に追加され、
  さらに上記の終了タスクIDが `done` に揃えられます（未整合があった場合の補正）。

## 3) Autopilot を再開
```bash
LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 60
```

## 4) P2 に進む場合
P1が完了したら、P2キューを開始できます:
```bash
cp work/queue_p2_ready.json work/queue.json
LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 60
```

## 重要（終わらせるためのルール）
- 実行中に `work/queue.json` に新規タスクIDを足さない（NO TASK CREEP）
- 追加検討は `work/rfcs/` にRFCとして残し、次キューで実施する
- コード量を増やしすぎない（docs/14）・レビュー地図を残す（docs/15）・不要物は削除（docs/16）

## 追加されるもの（今回のzip）
- docs/addons/25_AFTER_240_NEXT_STEPS.md
- work/tasks_p1_tail/250_*.md など（P1残りタスク）
- work/queue_after_240_with_tail.json（手動で置き換えたい方向け）
- tools/_legacy/apply_addon_after_240.py（queue自動パッチ）
- work/queue_p2_ready.json（P2開始用）
