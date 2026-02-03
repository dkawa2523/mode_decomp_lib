#!/usr/bin/env python3
"""tools/taskgen.py

Task md の作成と work/queue.json への追記を支援する小ツール。

目的:
- Codexが “拡張タスク” を起票しやすくする
- queueスキーマを壊さず、doctorで検出できる形を維持する

使い方:
  python tools/taskgen.py --id 130 --priority P1 --title "..." --path work/tasks/130_xxx.md \
    --skills S10_config_hydra,S95_tests_ci \
    --contracts docs/00_INVARIANTS.md,docs/11_PLUGIN_REGISTRY.md \
    --depends_on 080,095

注意:
- 既存IDと衝突したら失敗します。
- md作成後、`python tools/codex_prompt.py doctor` で整合性確認してください。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_STATUS = {"todo", "in_progress", "blocked", "done"}

def normalize_task_id(task_id: str) -> str:
    s = str(task_id).strip()
    if re.fullmatch(r"\d+", s):
        return s.zfill(3)
    return s

def split_csv(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def render_task_md(template_text: str, task_id: str, priority: str, title: str) -> str:
    out = template_text
    out = out.replace("<ID>", task_id)
    out = out.replace("<Priority>", priority)
    out = out.replace("<Title>", title)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--priority", required=True, help="P0/P1/P2")
    ap.add_argument("--title", required=True)
    ap.add_argument("--path", required=True, help="work/tasks/...md")
    ap.add_argument("--status", default="todo")
    ap.add_argument("--skills", default="")
    ap.add_argument("--contracts", default="")
    ap.add_argument("--depends_on", default="")
    ap.add_argument("--unblocks", default="")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    queue_path = repo_root / "work" / "queue.json"
    template_path = repo_root / "work" / "templates" / "TASK.md"
    md_path = repo_root / args.path

    task_id = normalize_task_id(args.id)
    status = args.status.strip()
    if status not in VALID_STATUS:
        raise SystemExit(f"Invalid --status={status}. Must be one of {sorted(VALID_STATUS)}")

    queue = load_json(queue_path)
    tasks: List[Dict[str, Any]] = list(queue.get("tasks", []))

    # ID collision check
    for t in tasks:
        if normalize_task_id(str(t.get("id"))) == task_id:
            raise SystemExit(f"Task id already exists: {task_id}")

    # Write md if missing
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if not md_path.exists():
        template = template_path.read_text(encoding="utf-8")
        md_path.write_text(render_task_md(template, task_id, args.priority, args.title), encoding="utf-8")

    task_obj: Dict[str, Any] = {
        "id": task_id,
        "priority": args.priority.strip(),
        "status": status,
        "title": args.title.strip(),
        "path": args.path.strip(),
        "skills": split_csv(args.skills),
        "contracts": split_csv(args.contracts),
        "depends_on": [normalize_task_id(x) for x in split_csv(args.depends_on)],
        "unblocks": [normalize_task_id(x) for x in split_csv(args.unblocks)],
    }
    tasks.append(task_obj)
    queue["tasks"] = tasks
    queue["updated_at"] = queue.get("updated_at") or ""
    save_json(queue_path, queue)

    print(f"✅ created: {args.path}")
    print(f"✅ appended to: work/queue.json (id={task_id})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
