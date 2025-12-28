#!/usr/bin/env bash
set -euo pipefail

# Fail-fast diagnostics
trap 'echo "ERROR: autopilot failed at line $LINENO" >&2' ERR

# -------------------------
# Portable Python launcher (macOS互換)
# -------------------------
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: Python が見つかりません。 python3 をインストールするか PYTHON_BIN を指定してください" >&2
    exit 127
  fi
fi

# -------------------------
# Portable sha1 (macOS互換)
# -------------------------
sha1_cmd() {
  # NOTE:
  # - GNU/Linux: sha1sum
  # - macOS: shasum -a 1
  # - どちらも無い場合は python で stdin をsha1化（最後の手段）
  if command -v sha1sum >/dev/null 2>&1; then
    sha1sum
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 1
  else
    "$PYTHON_BIN" - <<'PY'
import sys, hashlib
data = sys.stdin.buffer.read()
print(hashlib.sha1(data).hexdigest() + "  -")
PY
  fi
}

# =========================
# Autopilot: queue駆動の反復実行
# =========================
# 目的:
# - doctor → next(prompt生成) → codex exec → doctor ... を反復
# - “止まりやすい”要因を吸収（flags差/確認要求/無限ループ/doctor失敗）
#
# 使い方:
#   chmod +x tools/autopilot.sh
#   ./tools/autopilot.sh 10
#
# 環境変数:
#   MAX_ITERS=30（引数より優先するなら書き換え）
#   CODEX_GLOBAL_FLAGS="--ask-for-approval never"
#   CODEX_EXEC_FLAGS="--sandbox workspace-write"
#   CODEX_FLAGS="..."（旧形式。自動で分離を試みる）
#   AUTO_FIX_DOCTOR=1
#   STOP_WHEN_ONLY_BLOCKED=1
#   MAX_SAME_TASK=3
#   MAX_NO_CHANGE=2
#   FORCE_PROGRESS_ON_STALL=1
#
# 注意:
# - approval=never は止まりにくいが危険。sandbox維持・ブランチ運用・差分レビューが前提。

MAX_ITERS="${MAX_ITERS:-${1:-20}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/work/.autopilot/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOGDIR"

AUTO_FIX_DOCTOR="${AUTO_FIX_DOCTOR:-1}"
STOP_WHEN_ONLY_BLOCKED="${STOP_WHEN_ONLY_BLOCKED:-1}"
MAX_SAME_TASK="${MAX_SAME_TASK:-3}"
MAX_NO_CHANGE="${MAX_NO_CHANGE:-2}"
FORCE_PROGRESS_ON_STALL="${FORCE_PROGRESS_ON_STALL:-1}"
LIVE_TEE="${LIVE_TEE:-0}"  # 1: codex stdout/stderr をターミナルにも流す（ログにも保存）

# ---- flags (new preferred) ----
CODEX_GLOBAL_FLAGS="${CODEX_GLOBAL_FLAGS:-}"
CODEX_EXEC_FLAGS="${CODEX_EXEC_FLAGS:-}"

# ---- legacy (backward compatible) ----
CODEX_FLAGS="${CODEX_FLAGS:-}"
if [ -n "$CODEX_FLAGS" ] && { [ -z "$CODEX_GLOBAL_FLAGS" ] && [ -z "$CODEX_EXEC_FLAGS" ]; }; then
  # naive split: put ask-for-approval into global; sandbox into exec
  if echo "$CODEX_FLAGS" | grep -q -- "--ask-for-approval"; then
    CODEX_GLOBAL_FLAGS="$(echo "$CODEX_FLAGS" | tr ' ' '\n' | awk '
      BEGIN{p=""}
      /--ask-for-approval/{print; getline; print; next}
      { }' | tr '\n' ' ')"
  fi
  if echo "$CODEX_FLAGS" | grep -q -- "--sandbox"; then
    CODEX_EXEC_FLAGS="$(echo "$CODEX_FLAGS" | tr ' ' '\n' | awk '
      BEGIN{p=""}
      /--sandbox/{print; getline; print; next}
      { }' | tr '\n' ' ')"
  fi
fi

# Defaults if not set
if [ -z "$CODEX_GLOBAL_FLAGS" ]; then
  CODEX_GLOBAL_FLAGS="__AUTO__"
fi
if [ -z "$CODEX_EXEC_FLAGS" ]; then
  CODEX_EXEC_FLAGS="__AUTO__"
fi

echo "LOGDIR=$LOGDIR" | tee "$LOGDIR/_meta.txt"
echo "MAX_ITERS=$MAX_ITERS" | tee -a "$LOGDIR/_meta.txt"
echo "CODEX_GLOBAL_FLAGS=$CODEX_GLOBAL_FLAGS" | tee -a "$LOGDIR/_meta.txt"
echo "CODEX_EXEC_FLAGS=$CODEX_EXEC_FLAGS" | tee -a "$LOGDIR/_meta.txt"
echo "AUTO_FIX_DOCTOR=$AUTO_FIX_DOCTOR STOP_WHEN_ONLY_BLOCKED=$STOP_WHEN_ONLY_BLOCKED" | tee -a "$LOGDIR/_meta.txt"
echo "MAX_SAME_TASK=$MAX_SAME_TASK MAX_NO_CHANGE=$MAX_NO_CHANGE FORCE_PROGRESS_ON_STALL=$FORCE_PROGRESS_ON_STALL LIVE_TEE=$LIVE_TEE" | tee -a "$LOGDIR/_meta.txt"

# -------------------------
# Helpers
# -------------------------
has_cmd() { command -v "$1" >/dev/null 2>&1; }

preflight() {
  set +e
  codex --version > "$LOGDIR/codex_version.txt" 2>&1
  codex --help > "$LOGDIR/codex_help.txt" 2>&1
  codex exec --help > "$LOGDIR/codex_exec_help.txt" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "Preflight failed. See logs:" >&2
    echo "  $LOGDIR/codex_version.txt" >&2
    echo "  $LOGDIR/codex_help.txt" >&2
    echo "  $LOGDIR/codex_exec_help.txt" >&2
    exit 2
  fi
}

# feature detect flags supported by codex exec
supports_exec_flag() {
  local flag="$1"
  grep -q -- "$flag" "$LOGDIR/codex_exec_help.txt"
}


resolve_codex_flags() {
  # Resolve global flags (approval) and exec flags (sandbox) by reading captured help.
  # If user supplied explicit flags, they are kept as-is.
  if [ "$CODEX_GLOBAL_FLAGS" = "__AUTO__" ]; then
    if grep -q -- "--ask-for-approval" "$LOGDIR/codex_help.txt"; then
      CODEX_GLOBAL_FLAGS="--ask-for-approval never"
    elif grep -q -- "--approval-policy" "$LOGDIR/codex_help.txt"; then
      CODEX_GLOBAL_FLAGS="--approval-policy never"
    elif grep -q -- "--approval_policy" "$LOGDIR/codex_help.txt"; then
      CODEX_GLOBAL_FLAGS="--approval_policy never"
    else
      # Unknown CLI. Prefer empty rather than failing hard.
      CODEX_GLOBAL_FLAGS=""
      echo "WARN: could not detect approval flag from codex --help. Proceeding without approval flag." >&2
    fi
  fi

# Git repo / trusted directory check:
# Some Codex builds require running inside a git repo (or trusted dir) unless this flag is specified.
# We append it only when supported to avoid 'unknown option' failures.
if grep -q -- "--skip-git-repo-check" "$LOGDIR/codex_help.txt"; then
  if [ -n "$CODEX_GLOBAL_FLAGS" ]; then
    CODEX_GLOBAL_FLAGS="$CODEX_GLOBAL_FLAGS --skip-git-repo-check"
  else
    CODEX_GLOBAL_FLAGS="--skip-git-repo-check"
  fi
fi

  if [ "$CODEX_EXEC_FLAGS" = "__AUTO__" ]; then
    if supports_exec_flag "--sandbox_mode"; then
      CODEX_EXEC_FLAGS="--sandbox_mode workspace-write"
    elif supports_exec_flag "--sandbox-mode"; then
      CODEX_EXEC_FLAGS="--sandbox-mode workspace-write"
    elif supports_exec_flag "--sandbox"; then
      CODEX_EXEC_FLAGS="--sandbox workspace-write"
    elif supports_exec_flag "-s"; then
      CODEX_EXEC_FLAGS="-s workspace-write"
    else
      CODEX_EXEC_FLAGS=""
      echo "WARN: could not detect sandbox flag from codex exec --help. Proceeding without sandbox flag." >&2
    fi
  fi
# Git repo check bypass (Codex may refuse to run outside a trusted git repo unless this is specified).
# Prefer adding it to `codex exec` flags (it is an exec-specific flag in current Codex CLI docs).
if supports_exec_flag "--skip-git-repo-check"; then
  case " $CODEX_EXEC_FLAGS " in
    *" --skip-git-repo-check "*) : ;;
    *)
      if [ -n "$CODEX_EXEC_FLAGS" ]; then
        CODEX_EXEC_FLAGS="$CODEX_EXEC_FLAGS --skip-git-repo-check"
      else
        CODEX_EXEC_FLAGS="--skip-git-repo-check"
      fi
      ;;
  esac
fi

}


# make a codex exec command line (global flags before 'exec', exec flags after)
run_codex_exec() {
  local prompt_file="$1"
  local out_file="$2"
  local err_file="$3"
  local last_file="$4"
  local cmd_file="$out_file.cmd"

  local extra_exec=""
  if supports_exec_flag "--output-last-message"; then
    extra_exec="$extra_exec --output-last-message \"$last_file\""
  fi
  # save full command for debugging
  echo "codex $CODEX_GLOBAL_FLAGS exec $CODEX_EXEC_FLAGS $extra_exec - < \"$prompt_file\"" > "$cmd_file"

  # shellcheck disable=SC2086
  if [ "$LIVE_TEE" -eq 1 ]; then
    # stdout/stderr をターミナルにも流しつつ、ログにも保存
    if supports_exec_flag "--output-last-message"; then
      codex $CODEX_GLOBAL_FLAGS exec $CODEX_EXEC_FLAGS --output-last-message "$last_file" - < "$prompt_file"             2> >(tee "$err_file" >&2) | tee "$out_file"
    else
      codex $CODEX_GLOBAL_FLAGS exec $CODEX_EXEC_FLAGS - < "$prompt_file"             2> >(tee "$err_file" >&2) | tee "$out_file"
    fi
  else
    # 静音モード：ログファイルにのみ保存（ターミナルは進捗行だけ）
    if supports_exec_flag "--output-last-message"; then
      codex $CODEX_GLOBAL_FLAGS exec $CODEX_EXEC_FLAGS --output-last-message "$last_file" - < "$prompt_file"             > "$out_file" 2> "$err_file"
    else
      codex $CODEX_GLOBAL_FLAGS exec $CODEX_EXEC_FLAGS - < "$prompt_file"             > "$out_file" 2> "$err_file"
    fi
  fi
}

    queue_ids() {
      "$PYTHON_BIN" - <<'PY'
import json
q=json.load(open("work/queue.json"))
ids=sorted(str(t.get("id")) for t in q.get("tasks", []))
print("\n".join(ids))
PY
    }

    queue_ids_hash() {
      queue_ids | sha1_cmd | awk '{print $1}'
    }

queue_counts() {
"$PYTHON_BIN" - <<'PY'
import json
q=json.load(open("work/queue.json"))
c={"todo":0,"in_progress":0,"blocked":0,"done":0}
for t in q.get("tasks",[]):
    s=t.get("status","todo")
    c[s]=c.get(s,0)+1
print(c["todo"], c["in_progress"], c["blocked"], c["done"])
PY
}

non_done_count() {
"$PYTHON_BIN" - <<'PY'
import json
q=json.load(open("work/queue.json"))
print(sum(1 for t in q.get("tasks",[]) if t.get("status")!="done"))
PY
}

only_blocked_left() {
"$PYTHON_BIN" - <<'PY'
import json
q=json.load(open("work/queue.json"))
c={"todo":0,"in_progress":0,"blocked":0,"done":0}
for t in q.get("tasks",[]):
    s=t.get("status","todo")
    c[s]=c.get(s,0)+1
print(1 if (c["todo"]==0 and c["in_progress"]==0 and c["blocked"]>0) else 0)
PY
}

doctor_or_fix() {
  local stage="$1"
  local out="$LOGDIR/doctor_${stage}.txt"
  set +e
  "$PYTHON_BIN" "$ROOT/tools/codex_prompt.py" doctor > "$out" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  echo "doctor failed at $stage. report=$out" >&2
  if [ "$AUTO_FIX_DOCTOR" -ne 1 ]; then
    exit 2
  fi

  local fix_prompt="$LOGDIR/prompt_fix_doctor_${stage}.md"
  cat > "$fix_prompt" <<PROMPT
# ===== AUTOPILOT: FIX DOCTOR ISSUES ONLY =====
あなたは開発実装を進めてはいけません。今回は **work/queue.json と work/tasks/*.md の整合性修復だけ**を行います。

目的：
- \`python tools/codex_prompt.py doctor\` が成功（exit 0）する状態にする。

やってよいこと（必要な範囲のみ）：
- work/queue.json の修正（重複ID、invalid status、missing path、unblocks/depends_on の追加）
- 存在しない work/tasks/*.md を作成（テンプレでOK）
- blocked 親タスクがあるなら、解除用の子タスクを作る/修正する
  - 子タスク側の queue.json エントリに必ず \`unblocks: ["<親ID>"]\` を付ける

やってはいけないこと：
- モデル/処理本体の実装を進める
- 大規模リファクタ
- 新機能追加

以下が doctor の出力です（この内容を解消してください）：
\`\`\`
$(cat "$out")
\`\`\`
PROMPT

  local fix_out="$LOGDIR/codex_out_fix_doctor_${stage}.txt"
  local fix_err="$LOGDIR/codex_err_fix_doctor_${stage}.txt"
  local fix_last="$LOGDIR/last_message_fix_doctor_${stage}.md"
  run_codex_exec "$fix_prompt" "$fix_out" "$fix_err" "$fix_last"

  "$PYTHON_BIN" "$ROOT/tools/codex_prompt.py" doctor > "$LOGDIR/doctor_${stage}_after_fix.txt" 2>&1 || {
    echo "doctor still failing after attempted fix. See $LOGDIR/doctor_${stage}_after_fix.txt" >&2
    exit 2
  }
}

extract_selected_task_id() {
  # parse from prompt file line: "- id: 020"
  local prompt_file="$1"
  grep -E "^- id:\s*[0-9]{3}" "$prompt_file" | head -n 1 | awk '{print $3}'
}

detect_stall() {
  # detect "confirm please"/"差分なし"/"未実装"/"read-only" etc in last output
  local out_file="$1"
  local err_file="${2:-}"
  local patterns=(
    "確認をお願いします"
    "確認:"
    "Approve"
    "未実装"
    "差分なし"
    "blocked"
    "read-only"
    "書き込み不可"
    "Change Plan[[:space:]]*None"
    "Implementation[[:space:]]*None"
  )
  for p in "${patterns[@]}"; do
    if grep -qiE "$p" "$out_file"; then
      return 0
    fi
    if [ -n "$err_file" ] && grep -qiE "$p" "$err_file"; then
      return 0
    fi
  done
  return 1
}

git_snapshot() {
  if has_cmd git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --porcelain 2>/dev/null | sha1_cmd | awk '{print $1}'
  else
    # fallback: hash of queue.json only
    "$PYTHON_BIN" - <<'PY'
import hashlib, pathlib
p=pathlib.Path("work/queue.json")
h=hashlib.sha1(p.read_bytes()).hexdigest()
print(h)
PY
  fi
}

# -------------------------
# Main
# -------------------------
preflight
resolve_codex_flags
echo "RESOLVED_CODEX_GLOBAL_FLAGS=$CODEX_GLOBAL_FLAGS" | tee -a "$LOGDIR/_meta.txt"
echo "RESOLVED_CODEX_EXEC_FLAGS=$CODEX_EXEC_FLAGS" | tee -a "$LOGDIR/_meta.txt"

# Task creep guard: forbid adding new task IDs during a single autopilot run.
TASK_CREEP_GUARD="${TASK_CREEP_GUARD:-1}"
BASE_QUEUE_IDS_HASH="$(queue_ids_hash || true)"
echo "TASK_CREEP_GUARD=$TASK_CREEP_GUARD BASE_QUEUE_IDS_HASH=$BASE_QUEUE_IDS_HASH"
doctor_or_fix "initial"

same_task_streak=0
no_change_streak=0
prev_task=""
prev_snap="$(git_snapshot)"

for ((i=1; i<=MAX_ITERS; i++)); do
  echo "=== iter $i ==="
  read -r todo inprog blocked done <<< "$(queue_counts)"
  echo "todo=$todo in_progress=$inprog blocked=$blocked done=$done" | tee "$LOGDIR/counts_$i.txt"

  non_done="$(non_done_count)"
  if [ "$non_done" -eq 0 ]; then
    echo "All tasks are done. stop."
    break
  fi

  if [ "$STOP_WHEN_ONLY_BLOCKED" -eq 1 ]; then
    if [ "$(only_blocked_left)" -eq 1 ]; then
      echo "Only blocked tasks remain. stop for manual decision."
      break
    fi
  fi

  prompt_file="$LOGDIR/prompt_$i.md"
  "$PYTHON_BIN" "$ROOT/tools/codex_prompt.py" next --mode autopilot > "$prompt_file"

  # Extract selected task id for loop guards
  cur_task="$(extract_selected_task_id "$prompt_file" || true)"
  if [ -n "$cur_task" ] && [ "$cur_task" = "$prev_task" ]; then
    same_task_streak=$((same_task_streak+1))
  else
    same_task_streak=0
  fi
  prev_task="$cur_task"
  echo "selected_task=${cur_task:-unknown}"
  echo "same_task_streak=$same_task_streak"

  if [ "$same_task_streak" -ge "$MAX_SAME_TASK" ]; then
    echo "Same task repeated $same_task_streak times. stop. (task=$cur_task)"
    break
  fi

  out_file="$LOGDIR/codex_out_$i.txt"
  err_file="$LOGDIR/codex_err_$i.txt"
  last_file="$LOGDIR/last_message_$i.md"
  echo "logs: stdout=$out_file stderr=$err_file cmd=$out_file.cmd last=$last_file"
  echo "hint: tail -f \"$err_file\"  # (or set LIVE_TEE=1)"
  run_codex_exec "$prompt_file" "$out_file" "$err_file" "$last_file"

  # If stalled, optionally force progress once
  if [ "$FORCE_PROGRESS_ON_STALL" -eq 1 ] && detect_stall "$out_file" "$err_file"; then
    force_prompt="$LOGDIR/prompt_force_progress_$i.md"
    cat > "$force_prompt" <<'PROMPT'
# ===== AUTOPILOT: FORCE PROGRESS =====
前の応答が「確認要求」「未実装」「差分なし」「書き込み不可」等で止まっています。
質問して止まってはいけません。必ず次のいずれかを選び、実際にファイルを変更してください：

A) 実装してテストして queue を done にする
B) 実装不能なら blocked にし、解除子タスクを作って queue に追加（子に unblocks を必ず付ける）
C) queue/taskの状態ズレ（blocked/depends_on/unblocks）を修正して A に進む

制約：
- 状態の真実は work/queue.json
- sandbox/workspace-write 前提で書き込み可能として行動する
PROMPT
    force_out="$LOGDIR/codex_out_force_progress_$i.txt"
    force_err="$LOGDIR/codex_err_force_progress_$i.txt"
    force_last="$LOGDIR/last_message_force_progress_$i.md"
    run_codex_exec "$force_prompt" "$force_out" "$force_err" "$force_last"
  fi

  doctor_or_fix "iter_${i}"

  # No-change guard
  cur_snap="$(git_snapshot)"
  if [ "$cur_snap" = "$prev_snap" ]; then
    no_change_streak=$((no_change_streak+1))
  else
    no_change_streak=0
  fi
  prev_snap="$cur_snap"
  if [ "$no_change_streak" -ge "$MAX_NO_CHANGE" ]; then
    echo "No-change streak $no_change_streak. stop."
    break
  fi
done

echo "Autopilot finished. Logs: $LOGDIR"
