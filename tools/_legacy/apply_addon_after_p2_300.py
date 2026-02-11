#!/usr/bin/env python3
"""Apply add-on after completing P2 tasks 299/300.

- Backup work/queue.json
- Ensure task 299,300 are marked done if present
- Ensure tasks 350 depends_on includes 395 (if user uses this add-on)
- Add new tasks 395 and 398 if missing
- Optionally adjust 390 to depend on 398 (to keep a clean finish)

Run:
  python tools/_legacy/apply_addon_after_p2_300.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

DONE_IDS = {"299", "300"}

NEW_TASKS = [
  {
    "id": "395",
    "priority": "P2",
    "status": "todo",
    "title": "Artifact validator（契約逸脱の早期検知）",
    "path": "work/tasks_p2_v2/395_artifact_validator.md",
    "skills": ["S95_tests_ci", "S70_pipeline"],
    "contracts": [
      "docs/00_INVARIANTS.md",
      "docs/04_ARTIFACTS_AND_VERSIONING.md",
      "docs/13_TASK_FLOW.md",
      "docs/14_COMPACT_CODE_POLICY.md",
      "docs/15_REVIEWER_GUIDE.md",
      "docs/16_DELETION_AND_PRUNING.md",
    ],
    "depends_on": ["350"],
    "unblocks": ["398"],
  },
  {
    "id": "398",
    "priority": "P2",
    "status": "todo",
    "title": "Release-ready packaging（依存・実行・再現性の固定）",
    "path": "work/tasks_p2_v2/398_release_ready_packaging.md",
    "skills": ["S10_docs_update", "S98_refactor_cleanup"],
    "contracts": [
      "docs/00_INVARIANTS.md",
      "docs/03_CONFIG_CONVENTIONS.md",
      "docs/04_ARTIFACTS_AND_VERSIONING.md",
      "docs/13_TASK_FLOW.md",
      "docs/14_COMPACT_CODE_POLICY.md",
      "docs/15_REVIEWER_GUIDE.md",
      "docs/16_DELETION_AND_PRUNING.md",
    ],
    "depends_on": ["395"],
    "unblocks": ["390"],
  },
]

def main() -> None:
    root = Path(".").resolve()
    qpath = root / "work" / "queue.json"
    if not qpath.exists():
        raise SystemExit(f"queue.json not found: {qpath}")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = qpath.with_suffix(".json.bak." + ts)
    shutil.copy2(qpath, bak)
    print(f"[backup] {qpath} -> {bak}")

    q = json.loads(qpath.read_text(encoding="utf-8"))
    tasks = q.get("tasks", [])
    if not isinstance(tasks, list):
        raise SystemExit("queue.json: tasks must be a list")

    by_id = {str(t.get("id")): t for t in tasks}

    # mark done (if exists)
    for tid in DONE_IDS:
        if tid in by_id:
            by_id[tid]["status"] = "done"

    # add new tasks if missing
    added = 0
    for t in NEW_TASKS:
        tid = t["id"]
        if tid not in by_id:
            tasks.append(t)
            by_id[tid] = t
            added += 1

    # adjust dependencies if tasks exist
    # Make 350 unblock 395, and 390 depend on 398 if present
    if "350" in by_id:
        unb = by_id["350"].get("unblocks") or []
        if "395" not in unb:
            unb.append("395")
            by_id["350"]["unblocks"] = unb

    if "390" in by_id:
        deps = by_id["390"].get("depends_on") or []
        if "398" not in deps and "395" in by_id and "398" in by_id:
            # keep existing deps (e.g., 350), but ensure 398 is included
            deps.append("398")
            by_id["390"]["depends_on"] = sorted(set(deps), key=str)

    q["tasks"] = tasks
    q["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    qpath.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] updated queue.json (added_new_tasks={added})")
    print("[next] Run autopilot: LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 80")

if __name__ == "__main__":
    main()
