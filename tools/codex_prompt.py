#!/usr/bin/env python3
"""
tools/codex_prompt.py

queue.json から “次にやるべきタスク” を選び、Codex に貼る統合プロンプトを生成します。

特徴:
- 選択順: in_progress → blocked(解除タスク優先) → todo(depends_on尊重)
- blocked は放置しない: 親blockedに解除タスクが無いと doctor がエラー
- doctor: キュー整合性（欠落md/重複ID/孤立blocked/欠落depends_on）を検査
- タスクmdが欠落/空でも止まらず、テンプレを埋め込んで “まずmdを作れ” を指示
- autopilotモード: 質問禁止/前進強制の文言をプロンプトへ注入

使い方:
  python tools/codex_prompt.py list
  python tools/codex_prompt.py doctor
  python tools/codex_prompt.py next
  python tools/codex_prompt.py next --dry-run
  python tools/codex_prompt.py next --mode autopilot
  python tools/codex_prompt.py done 020
  python tools/codex_prompt.py set 010 blocked
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
VALID_STATUS = {"todo", "in_progress", "blocked", "done"}

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "work" / "queue.json").exists() and (cur / "codex").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("repo root not found (expected work/queue.json and codex/)")

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

def normalize_task_id(task_id: Any) -> str:
    s = str(task_id).strip()
    if re.fullmatch(r"\d+", s):
        return s.zfill(3)
    return s

def now_utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def task_sort_key(task: Dict[str, Any]) -> Tuple[int, int]:
    pri = PRIORITY_RANK.get(task.get("priority", "P9"), 99)
    tid = normalize_task_id(task.get("id", "999"))
    try:
        tid_int = int(tid) if re.fullmatch(r"\d+", tid) else 999
    except ValueError:
        tid_int = 999
    return pri, tid_int

def index_tasks(queue: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {normalize_task_id(t.get("id")): t for t in queue.get("tasks", [])}

def _as_id_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = re.split(r"[,\s]+", v.strip())
        return [normalize_task_id(p) for p in parts if p]
    if isinstance(v, list):
        return [normalize_task_id(x) for x in v if x is not None]
    return []

def deps_satisfied(task: Dict[str, Any], id2task: Dict[str, Dict[str, Any]]) -> bool:
    deps = _as_id_list(task.get("depends_on"))
    for dep_id in deps:
        dep = id2task.get(dep_id)
        if not dep or str(dep.get("status")) != "done":
            return False
    return True

def extract_blocked_section(md: str) -> str:
    if not md:
        return ""
    m = re.search(r"(?im)^\s*##\s+Blocked\s*$", md)
    if not m:
        m = re.search(r"(?im)^\s*##\s+ブロック\s*$", md)
    if not m:
        return ""
    tail = md[m.end():]
    m2 = re.search(r"(?im)^\s*##\s+", tail)
    return (tail[:m2.start()] if m2 else tail).strip()

def parse_unblock_task_ids_from_text(text: str, parent_id: str) -> List[str]:
    if not text:
        return []
    scores: Dict[str, int] = {}

    # YAML-ish: - unblock_tasks: [011, 012]
    for group in re.findall(r"(?im)^\s*-\s*unblock_tasks\s*:\s*\[([^\]]+)\]\s*$", text):
        for tid in re.findall(r"\b(\d{1,4})\b", group):
            tidn = normalize_task_id(tid)
            if tidn != parent_id:
                scores[tidn] = scores.get(tidn, 0) + 10

    # filename references: 011_xxx.md
    for fn in re.findall(r"\b(\d{3}_[A-Za-z0-9_\-]+\.md)\b", text):
        tid = fn.split("_", 1)[0]
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 6

    # explicit unblock 011 / 解除 011 / 次 011
    for tid in re.findall(r"(?i)\bunblock\b[^\n]*?\b(\d{1,4})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 8
    for tid in re.findall(r"解除[^\n]*?\b(\d{1,4})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 8
    for tid in re.findall(r"(?i)(?:next|step|task|次)[^\n]*?\b(\d{1,4})\b", text):
        tidn = normalize_task_id(tid)
        if tidn != parent_id:
            scores[tidn] = scores.get(tidn, 0) + 4

    candidates = [k for k, v in scores.items() if re.fullmatch(r"\d{3}", k) and v > 0]
    candidates.sort(key=lambda x: (-scores[x], x))
    return candidates

def find_unblockers_by_field(parent_id: str, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits = []
    for t in tasks:
        if str(t.get("status")) not in ("todo", "in_progress"):
            continue
        if parent_id in _as_id_list(t.get("unblocks")):
            hits.append(t)
    hits.sort(key=task_sort_key)
    hits.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
    return hits

def resolve_unblocker_task(repo_root: Path, queue: Dict[str, Any], parent_task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tasks = queue.get("tasks", [])
    id2task = index_tasks(queue)
    parent_id = normalize_task_id(parent_task.get("id"))

    # 1) queue.json field `unblocks`
    hits = find_unblockers_by_field(parent_id, tasks)
    if hits:
        return hits[0]

    # 2) parent's md blocked section hints
    md_path = repo_root / str(parent_task.get("path", ""))
    md = _read_text(md_path)
    blocked_text = extract_blocked_section(md)
    candidates = parse_unblock_task_ids_from_text(blocked_text, parent_id)
    live = []
    for cid in candidates:
        t = id2task.get(cid)
        if t and str(t.get("status")) in ("todo", "in_progress"):
            live.append(t)
    if live:
        live.sort(key=task_sort_key)
        live.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
        return live[0]

    # 3) search other task md for "Unblocks: parent"
    patterns = [
        re.compile(rf"(?im)^\s*Unblocks\s*:\s*{re.escape(parent_id)}\s*$"),
        re.compile(rf"(?i)\bunblock(?:s|ing)?\b[^\n]*?\b{re.escape(parent_id)}\b"),
        re.compile(rf"解除[^\n]*?\b{re.escape(parent_id)}\b"),
    ]
    hits = []
    for t in tasks:
        if str(t.get("status")) not in ("todo", "in_progress"):
            continue
        p = repo_root / str(t.get("path", ""))
        txt = _read_text(p)
        if txt and any(pat.search(txt) for pat in patterns):
            hits.append(t)
    if hits:
        hits.sort(key=task_sort_key)
        hits.sort(key=lambda x: 0 if str(x.get("status")) == "in_progress" else 1)
        return hits[0]

    return None

def pick_next_task(repo_root: Path, queue: Dict[str, Any], *, skip_blocked: bool) -> Optional[Dict[str, Any]]:
    tasks = queue.get("tasks", [])
    id2task = index_tasks(queue)

    # 1) in_progress
    inprog = [t for t in tasks if str(t.get("status")) == "in_progress"]
    if inprog:
        inprog.sort(key=lambda t: (t.get("started_at") or "9999",) + task_sort_key(t))
        return inprog[0]

    # 2) blocked -> unblocker
    blocked = [t for t in tasks if str(t.get("status")) == "blocked"]
    if blocked and not skip_blocked:
        blocked.sort(key=task_sort_key)
        for parent in blocked:
            u = resolve_unblocker_task(repo_root, queue, parent)
            if u:
                return u
        return blocked[0]

    # 3) todo eligible (deps satisfied)
    todo = [t for t in tasks if str(t.get("status")) == "todo"]
    if not todo:
        return None
    eligible = [t for t in todo if deps_satisfied(t, id2task)]
    if eligible:
        eligible.sort(key=task_sort_key)
        return eligible[0]

    # 4) if none eligible, pull unmet dependency to guide
    unmet = []
    for t in todo:
        for dep_id in _as_id_list(t.get("depends_on")):
            dep = id2task.get(dep_id)
            if dep and str(dep.get("status")) != "done":
                unmet.append(dep)
    if unmet:
        uniq = {normalize_task_id(u.get("id")): u for u in unmet}
        unmet = list(uniq.values())
        # prioritize in_progress then blocked then todo
        in2 = [t for t in unmet if str(t.get("status")) == "in_progress"]
        if in2:
            in2.sort(key=lambda t: (t.get("started_at") or "9999",) + task_sort_key(t))
            return in2[0]
        if not skip_blocked:
            b2 = [t for t in unmet if str(t.get("status")) == "blocked"]
            if b2:
                b2.sort(key=task_sort_key)
                for parent in b2:
                    u = resolve_unblocker_task(repo_root, queue, parent)
                    if u:
                        return u
                return b2[0]
        t2 = [t for t in unmet if str(t.get("status")) == "todo"]
        if t2:
            t2.sort(key=task_sort_key)
            return t2[0]

    todo.sort(key=task_sort_key)
    return todo[0]

def get_skill_paths(repo_root: Path, skill_ids: List[str]) -> List[Path]:
    reg_path = repo_root / "agentskills" / "skill_registry.json"
    if not reg_path.exists():
        return []
    reg = load_json(reg_path)
    id2path = {s.get("id"): s.get("path") for s in reg.get("skills", [])}
    out = []
    for sid in skill_ids:
        p = id2path.get(sid)
        if p:
            out.append(repo_root / p)
    return out

def build_prompt(repo_root: Path, task: Dict[str, Any], *, mode: str, note: str = "") -> str:
    session = _read_text(repo_root / "codex" / "SESSION_CONTEXT.md").strip()

    task_id = normalize_task_id(task.get("id"))
    task_title = str(task.get("title", "")).strip()
    task_priority = str(task.get("priority", "")).strip()
    task_path = str(task.get("path", "")).strip()
    status = str(task.get("status", "todo"))

    contract_paths = task.get("contracts") or ["docs/00_INVARIANTS.md"]
    contracts_list = "\n".join([f"- {p}" for p in contract_paths])

    skill_ids = task.get("skills") or []
    skill_paths = [str(p.relative_to(repo_root)) for p in get_skill_paths(repo_root, list(skill_ids))]
    skills_list = "\n".join([f"- {p}" for p in skill_paths]) if skill_paths else "- agentskills/ROUTER.md"

    md_path = repo_root / task_path if task_path else None
    task_text = _read_text(md_path).strip() if md_path else ""

    template_text = _read_text(repo_root / "work" / "templates" / "TASK.md").strip()
    if not task_text:
        task_text = (
            "## ⚠️ Task file is missing or empty\n"
            f"- expected: `{task_path}`\n"
            "- まずこのファイルを作成/記入してから実装に着手してください。\n\n"
            + (template_text or "(TASK template missing)")
        )

    extra_mode = ""
    if mode == "autopilot":
        extra_mode = (
            "\n# ===== AUTOPILOT MODE (mandatory) =====\n"
            "- 質問して止まらない（“確認してください”禁止）\n"
            "- 状態の真実は work/queue.json\n"
            "- 次のタスクに進む前に、現在タスクの Acceptance Criteria / Verification を満たす\n"
            "- コードはコンパクトに（docs/14）：過剰抽象化・不要機能の追加は禁止\n"
            "- レビュー容易性（docs/15）：task md に Review Map（重要ファイル/入口/判断/検証）を追記\n"
            "- 不要物削除（docs/16）：置き換えで不要になったコード/ファイルを残さない\n"
            "- 進められない場合は blocked + 解除子タスク（unblocks必須）で前進する\n"
        )

    if status == "blocked":
        extra_mode += (
            "\n# ===== BLOCKED MODE (mandatory) =====\n"
            "このタスクは blocked です。次を必ず実施：\n"
            "1) `## Blocked` に reason/unblock_condition/next_action を明記\n"
            "2) 解除子タスクを作る場合、queue.json 子タスクに `unblocks:[\"<parent_id>\"]` を付ける\n"
        )

    prompt = f"""# ===== SESSION CONTEXT =====
{session}

# ===== SELECTED TASK =====
- id: {task_id}
- title: {task_title}
- priority: {task_priority}
- status: {status}
- path: {task_path}

# ===== CONTRACTS TO FOLLOW (open & read) =====
{contracts_list}

# ===== SKILLS TO FOLLOW (open & follow) =====
{skills_list}

{note}{extra_mode}
# ===== TASK FILE (single source of truth) =====
{task_text}

# ===== OUTPUT REQUIREMENTS (mandatory) =====
1) 変更計画（ファイル単位）
2) 実装（差分が分かるように）
3) 追加/更新したテスト
4) 検証コマンド
5) 互換性影響（config/CLI/artifact）
6) タスク完了時：work/queue.json の status を done に更新（満たせない場合は blocked + 解除タスク起票）
7) レビュー用メモ（Review Map）を task md 末尾に追記（重要ファイル/入口/判断/検証結果/削除一覧）
"""
    return prompt

def cmd_list(repo_root: Path) -> int:
    queue = load_json(repo_root / "work" / "queue.json")
    tasks = sorted(queue.get("tasks", []), key=task_sort_key)
    print("id   pri  status         title")
    print("---- ---- -------------- ------------------------------")
    for t in tasks:
        tid = normalize_task_id(t.get("id"))
        print(f"{tid:>4} {t.get('priority',''):>4} {t.get('status',''):>14} {t.get('title','')}")
    return 0

def cmd_doctor(repo_root: Path) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tasks = queue.get("tasks", [])
    id2task = index_tasks(queue)

    issues = 0

    # duplicate ids
    seen = {}
    for t in tasks:
        tid = normalize_task_id(t.get("id"))
        seen[tid] = seen.get(tid, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        issues += len(dups)
        print("❌ Duplicate task IDs (normalized):")
        for k, v in sorted(dups.items()):
            print(f"  - {k}: {v} entries")

    # invalid status
    bad = [(normalize_task_id(t.get("id")), t.get("status")) for t in tasks if str(t.get("status")) not in VALID_STATUS]
    if bad:
        issues += len(bad)
        print("❌ Invalid status values:")
        for tid, st in bad:
            print(f"  - {tid}: {st} (expected one of {sorted(VALID_STATUS)})")

    # missing task files
    missing = []
    for t in tasks:
        p = str(t.get("path", "")).strip()
        if not p or not (repo_root / p).exists():
            missing.append((normalize_task_id(t.get("id")), p or "(no path)"))
    if missing:
        issues += len(missing)
        print("⚠️ Missing task markdown files:")
        for tid, p in missing:
            print(f"  - {tid}: {p}")

    # blocked without unblocker
    blocked = [t for t in tasks if str(t.get("status")) == "blocked"]
    if blocked:
        print("\nBlocked analysis:")
        for parent in sorted(blocked, key=task_sort_key):
            pid = normalize_task_id(parent.get("id"))
            u = resolve_unblocker_task(repo_root, queue, parent)
            if u is None:
                issues += 1
                print(f"  ❌ {pid} has no unblocker task detected. Add one with queue.json field `unblocks: [\"{pid}\"]`.")
            else:
                uid = normalize_task_id(u.get("id"))
                print(f"  ✅ {pid} -> unblocker candidate: {uid} ({u.get('status')})")

    # depends_on sanity
    for t in tasks:
        deps = _as_id_list(t.get("depends_on"))
        for dep in deps:
            if dep not in id2task:
                issues += 1
                print(f"❌ {normalize_task_id(t.get('id'))} depends_on missing task id: {dep}")

    if issues == 0:
        print("✅ doctor: no issues found.")
        return 0
    print(f"\ndoctor: found {issues} issue(s).")
    return 2

def cmd_next(repo_root: Path, *, dry_run: bool, skip_blocked: bool, mode: str) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    task = pick_next_task(repo_root, queue, skip_blocked=skip_blocked)
    if task is None:
        print("# No tasks available.")
        return 0

    if not dry_run:
        if str(task.get("status")) == "todo" and queue.get("policy", {}).get("auto_set_in_progress_on_next", True):
            task["status"] = "in_progress"
            task["started_at"] = task.get("started_at") or now_utc_iso()
        task["last_presented_at"] = now_utc_iso()
        queue["updated_at"] = now_utc_iso()
        save_json(queue_path, queue)

    note = ""
    id2task = index_tasks(queue)
    if str(task.get("status")) in ("todo", "in_progress") and not deps_satisfied(task, id2task):
        note = (
            "# ===== NOTE =====\n"
            "⚠️ このタスクは depends_on が未完了の可能性があります。依存を解消してから進めてください。\n\n"
        )
    print(build_prompt(repo_root, task, mode=mode, note=note))
    return 0

def cmd_done(repo_root: Path, task_id: str) -> int:
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tid = normalize_task_id(task_id)
    found = False
    for t in queue.get("tasks", []):
        if normalize_task_id(t.get("id")) == tid:
            t["status"] = "done"
            t["done_at"] = now_utc_iso()
            found = True
    if not found:
        print(f"Task id {tid} not found")
        return 1
    queue["updated_at"] = now_utc_iso()
    save_json(queue_path, queue)
    print(f"Marked {tid} as done")
    return 0

def cmd_set(repo_root: Path, task_id: str, new_status: str) -> int:
    if new_status not in VALID_STATUS:
        print(f"Invalid status: {new_status} (expected one of {sorted(VALID_STATUS)})")
        return 2
    queue_path = repo_root / "work" / "queue.json"
    queue = load_json(queue_path)
    tid = normalize_task_id(task_id)
    found = False
    for t in queue.get("tasks", []):
        if normalize_task_id(t.get("id")) == tid:
            t["status"] = new_status
            if new_status == "in_progress":
                t["started_at"] = t.get("started_at") or now_utc_iso()
            found = True
    if not found:
        print(f"Task id {tid} not found")
        return 1
    queue["updated_at"] = now_utc_iso()
    save_json(queue_path, queue)
    print(f"Set {tid} -> {new_status}")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "doctor", "next", "done", "set"])
    ap.add_argument("task_id", nargs="?")
    ap.add_argument("status", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-blocked", action="store_true")
    ap.add_argument("--mode", choices=["normal","autopilot"], default="normal")

    args = ap.parse_args()
    repo_root = find_repo_root(Path(__file__).parent)

    if args.cmd == "list":
        return cmd_list(repo_root)
    if args.cmd == "doctor":
        return cmd_doctor(repo_root)
    if args.cmd == "next":
        return cmd_next(repo_root, dry_run=args.dry_run, skip_blocked=args.skip_blocked, mode=args.mode)
    if args.cmd == "done":
        if not args.task_id:
            raise SystemExit("done requires task_id")
        return cmd_done(repo_root, args.task_id)
    if args.cmd == "set":
        if not args.task_id or not args.status:
            raise SystemExit("set requires task_id and status")
        return cmd_set(repo_root, args.task_id, args.status)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
