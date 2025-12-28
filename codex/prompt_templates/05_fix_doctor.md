# Fix Doctor Prompt（整合性修復専用）

今回は実装を進めない。`python tools/codex_prompt.py doctor` を通すことだけを目的にする。

やって良いこと：
- work/queue.json の修正（重複ID、invalid status、missing path、unblocks/depends_on の追加）
- missing task md の作成（テンプレでOK）
- blocked親の解除タスク作成（子に unblocks を必ず付ける）

禁止：
- モデル/処理本体の実装を進めない
