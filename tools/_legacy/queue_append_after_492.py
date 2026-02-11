#!/usr/bin/env python3

"""
Safely append tasks into an existing work/queue.json.

- Creates a timestamped backup of the original queue.
- Refuses to append if any task id already exists.
- Keeps stable ordering: existing tasks first, then appended tasks in id order.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, type=str, help="Path to existing work/queue.json")
    ap.add_argument("--append", required=True, type=str, help="Path to addon json (list of tasks)")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    addon_path = Path(args.append)

    if not queue_path.exists():
        raise SystemExit(f"queue not found: {queue_path}")
    if not addon_path.exists():
        raise SystemExit(f"append not found: {addon_path}")

    queue = load_json(queue_path)
    addon = load_json(addon_path)

    if isinstance(queue, list):
        queue_tasks = queue
        queue_is_list = True
    elif isinstance(queue, dict) and isinstance(queue.get("tasks"), list):
        queue_tasks = queue["tasks"]
        queue_is_list = False
    else:
        raise SystemExit("queue.json must be a JSON list of task objects or an object with tasks[]")
    if not isinstance(addon, list):
        raise SystemExit("addon json must be a JSON list of task objects")

    existing_ids = set()
    for t in queue_tasks:
        if isinstance(t, dict) and "id" in t:
            existing_ids.add(str(t["id"]))

    addon_ids = []
    for t in addon:
        if not isinstance(t, dict) or "id" not in t:
            raise SystemExit("addon tasks must be objects with an 'id' field")
        addon_ids.append(str(t["id"]))

    dup = sorted(set(addon_ids) & existing_ids)
    if dup:
        raise SystemExit(f"refusing to append: duplicate ids detected: {dup}")

    # Backup
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = queue_path.with_suffix(queue_path.suffix + f".bak.{ts}")
    shutil.copy2(queue_path, backup_path)

    # Stable sort new tasks by numeric-ish id if possible
    def key(task: Dict[str, Any]):
        tid = str(task.get("id", ""))
        try:
            return (0, int(tid))
        except Exception:
            return (1, tid)

    addon_sorted = sorted(list(addon), key=key)
    merged = list(queue_tasks) + addon_sorted

    if queue_is_list:
        output = merged
    else:
        queue["tasks"] = merged
        queue["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        output = queue

    dump_json(queue_path, output)
    print(f"Appended {len(addon)} tasks into {queue_path}")
    print(f"Backup saved to {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
