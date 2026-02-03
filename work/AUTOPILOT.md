# Autopilot（自動反復）運用

## 目的
- queue駆動で Codex に連続実行させる
- 止まる要因（doctor/flags/確認要求/無限ループ）を最小化する

## 使い方
```bash
chmod +x tools/autopilot.sh
./tools/autopilot.sh 10
```

## 停止条件（デフォルト）
- 全タスク done（=非doneが0）
- todo/in_progressが0で、blockedだけ残る（手動判断へ）
- MAX_ITERS到達
- 同一タスク連続 / 無変更連続のガードで停止

## 重要
- approvalをneverにすると止まりにくいが危険。必ずsandboxを維持し、ブランチ運用/ログ確認をする。
- `work/.autopilot/` にログが残る。機密がある場合はログ方針を見直す。


## トラブルシューティング

### 1) `selected_task=...` 以降、ターミナルに出力が増えない

正常な可能性があります。autopilot は Codex の出力を `work/.autopilot/<timestamp>/` に保存します。
`LOGDIR=...` が表示されているので、そのディレクトリのログを確認してください。

よく使うコマンド（最新ログを自動取得）：

```bash
LOGDIR="$(ls -td work/.autopilot/* | head -1)"
ls -lh "$LOGDIR"

# 実行コマンド（フラグ確認）
cat "$LOGDIR/codex_out_1.txt.cmd"

# stderr を追う（最重要）
tail -n 80 "$LOGDIR/codex_err_1.txt"
tail -f "$LOGDIR/codex_err_1.txt"

# stdout も必要なら
tail -n 80 "$LOGDIR/codex_out_1.txt"
```

ターミナルにも流したい場合：
```bash
LIVE_TEE=1 ./tools/autopilot.sh 30
```

### 2) `doctor failed` で止まる

`AUTO_FIX_DOCTOR=1` の場合、`doctor` の問題（queue/tasks整合）だけを自動修復して再開します。
直り切らない場合は `doctor_initial.txt` / `doctor_iter_*.txt` を確認し、blocked解除タスクや depends_on を修正します。

```bash
python tools/codex_prompt.py doctor
```

### 3) よくある実行エラーと対処

- `unknown variant xhigh ... model_reasoning_effort`  
  → `~/.codex/config.toml` の `model_reasoning_effort` を `high` にするか、Codex CLI を更新する  
  （IDE/CLIでconfigを共有している場合があります）

- `unexpected argument '--ask-for-approval'`  
  → `--ask-for-approval` は global flags なので `codex <global> exec <execflags>` の順で渡す  
  （autopilot はこの順に組み立てる実装になっています）

- `syntax error: unexpected end of file`  
  → `bash -n tools/autopilot.sh` で構文チェック。コピー時の欠損や here-doc/if/fi の不整合を疑う
