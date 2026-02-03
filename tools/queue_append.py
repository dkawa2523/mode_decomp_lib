#!/usr/bin/env python3
"""
Append tasks to an existing work/queue.json without breaking history.

Usage:
  python tools/queue_append.py --queue work/queue.json --append work/queue_addon_398.json

Rules:
- task id must be unique (string or int is normalized to string)
- new tasks are appended; existing tasks untouched
- writes a backup: <queue>.bak.<timestamp>
"""
import argparse, json, os, shutil, time
from pathlib import Path

def norm_id(x):
    # keep as string for stability; preserve leading zeros if present
    if isinstance(x, str):
        return x
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--append", required=True)
    args = ap.parse_args()

    qpath = Path(args.queue)
    apath = Path(args.append)
    if not qpath.exists():
        raise SystemExit(f"queue not found: {qpath}")
    if not apath.exists():
        raise SystemExit(f"append file not found: {apath}")

    queue = json.loads(qpath.read_text(encoding="utf-8"))
    addon = json.loads(apath.read_text(encoding="utf-8"))

    queue_is_list = isinstance(queue, list)
    if queue_is_list:
        queue_tasks = queue
    elif isinstance(queue, dict) and isinstance(queue.get("tasks"), list):
        queue_tasks = queue["tasks"]
    else:
        raise SystemExit("queue.json must be a JSON list or an object with a tasks list")
    if not isinstance(addon, list):
        raise SystemExit("append json must be a JSON list")

    existing_ids = {norm_id(t.get("id")) for t in queue_tasks if isinstance(t, dict) and "id" in t}
    new_ids = []
    for t in addon:
        if not isinstance(t, dict):
            raise SystemExit("append contains non-dict task")
        if "id" not in t:
            raise SystemExit("append task missing id")
        tid = norm_id(t["id"])
        if tid in existing_ids:
            raise SystemExit(f"duplicate task id: {tid}")
        new_ids.append(tid)

    # backup
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bak = qpath.with_suffix(qpath.suffix + f".bak.{ts}")
    shutil.copy2(qpath, bak)

    # append + write
    queue2 = queue_tasks + addon
    if queue_is_list:
        out = queue2
    else:
        queue["tasks"] = queue2
        queue["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = queue
    qpath.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: appended {len(addon)} tasks -> {qpath}")
    print(f"backup: {bak}")
    print(f"new ids: {new_ids[0]}..{new_ids[-1]}" if new_ids else "no tasks appended")

if __name__ == "__main__":
    main()
