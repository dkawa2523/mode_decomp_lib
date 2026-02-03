# Automation Rules（キュー運用）

## status の意味
- todo: 未着手
- in_progress: 着手中（WIPを増やさない）
- blocked: 人の判断/外部要因で停止（必ず解除子タスクが必要）
- done: 完了

## blocked運用（必須）
- 順番待ちは depends_on を使う（blockedにしない）
- blockedにする場合は解除タスクを作る：
  - 解除タスクの queue エントリに `unblocks: ["<親ID>"]` を付ける

## タスクの分割
- 大タスクは 020a/020b のように分割して前進できる形にする

## コマンド
- `python tools/codex_prompt.py list`
- `python tools/codex_prompt.py doctor`
- `python tools/codex_prompt.py next`
- `python tools/codex_prompt.py done 020`
- `python tools/codex_prompt.py set 010 blocked`


## 便利ショートカット
- `./autopilot.sh 30`（= ./tools/autopilot.sh 30）
- `./doctor.sh`（queue整合チェック）
