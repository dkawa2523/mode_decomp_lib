#!/usr/bin/env python3
"""
Apply add-on after completing tasks up to P1-240.

- Backups current work/queue.json
- Ensures known completed tasks are marked done
- Adds P1 tail tasks: 250/260/270/290 if missing
- Writes updated work/queue.json

Run:
  python tools/_legacy/apply_addon_after_240.py
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

DONE_IDS = ["000", "010", "020", "030", "040", "050", "060", "070", "080", "090", "100", "110", "120", "200", "210", "220", "230", "240"]

TAIL_TASKS = [
  {
    "id": "250",
    "priority": "P1",
    "status": "todo",
    "title": "Uncertainty（係数→場）伝播と可視化（GPR）",
    "path": "work/tasks_p1_tail/250_uncertainty_viz.md",
    "skills": ["S60_evaluation","S96_benchmark_reports","S72_numerical_stability"],
    "contracts": ["docs/00_INVARIANTS.md", "docs/03_CONFIG_CONVENTIONS.md", "docs/04_ARTIFACTS_AND_VERSIONING.md", "docs/09_EVALUATION_PROTOCOL.md", "docs/10_PROCESS_CATALOG.md", "docs/11_PLUGIN_REGISTRY.md", "docs/13_TASK_FLOW.md", "docs/14_COMPACT_CODE_POLICY.md", "docs/15_REVIEWER_GUIDE.md", "docs/16_DELETION_AND_PRUNING.md", "docs/18_SCOPE_LOCK.md"],
    "depends_on": ["240"],
    "unblocks": ["260","270"],
  },
  {
    "id": "260",
    "priority": "P1",
    "status": "todo",
    "title": "Model拡張: ElasticNet / MultiTask（比較用）",
    "path": "work/tasks_p1_tail/260_models_elasticnet_multitask.md",
    "skills": ["S52_regression_models"],
    "contracts": ["docs/00_INVARIANTS.md", "docs/03_CONFIG_CONVENTIONS.md", "docs/04_ARTIFACTS_AND_VERSIONING.md", "docs/09_EVALUATION_PROTOCOL.md", "docs/10_PROCESS_CATALOG.md", "docs/11_PLUGIN_REGISTRY.md", "docs/13_TASK_FLOW.md", "docs/14_COMPACT_CODE_POLICY.md", "docs/15_REVIEWER_GUIDE.md", "docs/16_DELETION_AND_PRUNING.md", "docs/18_SCOPE_LOCK.md"],
    "depends_on": ["240","250"],
    "unblocks": ["270"],
  },
  {
    "id": "270",
    "priority": "P1",
    "status": "todo",
    "title": "Docs/Examples更新（P1 tail反映）",
    "path": "work/tasks_p1_tail/270_docs_update_p1_tail.md",
    "skills": ["S10_docs_update"],
    "contracts": ["docs/00_INVARIANTS.md", "docs/03_CONFIG_CONVENTIONS.md", "docs/04_ARTIFACTS_AND_VERSIONING.md", "docs/09_EVALUATION_PROTOCOL.md", "docs/10_PROCESS_CATALOG.md", "docs/11_PLUGIN_REGISTRY.md", "docs/13_TASK_FLOW.md", "docs/14_COMPACT_CODE_POLICY.md", "docs/15_REVIEWER_GUIDE.md", "docs/16_DELETION_AND_PRUNING.md", "docs/18_SCOPE_LOCK.md"],
    "depends_on": ["250","260"],
    "unblocks": ["290"],
  },
  {
    "id": "290",
    "priority": "P1",
    "status": "todo",
    "title": "P1 Cleanup（不要コード削除・docs整合・再現性確認）",
    "path": "work/tasks_p1_tail/290_cleanup_p1_tail.md",
    "skills": ["S98_refactor_cleanup","S95_tests_ci"],
    "contracts": ["docs/00_INVARIANTS.md", "docs/03_CONFIG_CONVENTIONS.md", "docs/04_ARTIFACTS_AND_VERSIONING.md", "docs/09_EVALUATION_PROTOCOL.md", "docs/10_PROCESS_CATALOG.md", "docs/11_PLUGIN_REGISTRY.md", "docs/13_TASK_FLOW.md", "docs/14_COMPACT_CODE_POLICY.md", "docs/15_REVIEWER_GUIDE.md", "docs/16_DELETION_AND_PRUNING.md", "docs/18_SCOPE_LOCK.md"],
    "depends_on": ["270"],
    "unblocks": [],
  },
]

def main() -> None:
    root = Path(".").resolve()
    qpath = root / "work" / "queue.json"
    if not qpath.exists():
        raise SystemExit(f"queue.json not found: {qpath}")

    # backup
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bak = qpath.with_suffix(".json.bak." + ts)
    shutil.copy2(qpath, bak)
    print(f"[backup] {qpath} -> {bak}")

    q = json.loads(qpath.read_text(encoding="utf-8"))
    tasks = q.get("tasks", [])
    if not isinstance(tasks, list):
        raise SystemExit("queue.json: tasks must be a list")

    by_id = {str(t.get("id")): t for t in tasks}

    # mark done
    changed = 0
    for tid in DONE_IDS:
        if tid in by_id and by_id[tid].get("status") != "done":
            by_id[tid]["status"] = "done"
            changed += 1

    # add tail tasks if missing
    added = 0
    for t in TAIL_TASKS:
        tid = t["id"]
        if tid not in by_id:
            tasks.append(t)
            by_id[tid] = t
            added += 1

    q["tasks"] = tasks
    q["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    qpath.write_text(json.dumps(q, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] updated queue.json (marked_done={changed}, added_tail_tasks={added})")
    print("[next] Run autopilot: LIVE_TEE=1 PYTHON_BIN=/usr/bin/python3 ./autopilot.sh 60")

if __name__ == "__main__":
    main()
