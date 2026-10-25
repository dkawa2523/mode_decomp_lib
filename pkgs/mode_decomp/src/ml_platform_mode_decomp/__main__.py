"""Run one local YAML; environment profiles and platform-style --task are optional."""

import argparse
import json

from .config import load_config
from .runners import run_task


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run a field representation or surrogate task")
    parser.add_argument("config", nargs="?", help="Run YAML path")
    parser.add_argument("--task", dest="legacy_task", help="Run YAML path (alternative to the positional path)")
    parser.add_argument("--profile", help="Optional environment profile YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    args = parser.parse_args(argv)
    if args.config and args.legacy_task:
        parser.error("Supply either CONFIG or --task, not both")
    config_path = args.config or args.legacy_task
    if config_path is None:
        parser.error("A run YAML is required: CONFIG or --task PATH")
    result = run_task(load_config(config_path, args.profile, overrides=args.overrides))
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "metrics": result.metrics,
                "artifacts": {key: str(path) for key, path in result.artifacts.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
