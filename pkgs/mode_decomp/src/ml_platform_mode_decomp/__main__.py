"""Run one local YAML; environment profiles and platform-style --task are optional."""

import argparse
import json

from ml_platform_core.result import RunError
from yaml import YAMLError

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
    try:
        result = run_task(load_config(config_path, args.profile, overrides=args.overrides))
    except RunError as exc:
        _print_result(exc.result, error=str(exc))
        return 1
    except (ValueError, TypeError, OSError, ImportError, YAMLError) as exc:
        _print_result(None, error=str(exc))
        return 1
    _print_result(result)
    return 0


def _print_result(result, *, error=None):
    status = "failed" if error is not None else "ok"
    if error is None and result is not None and result.extra.get("analysis_complete") is False:
        status = "partial"
    print(
        json.dumps(
            {
                "status": status,
                "run_dir": None if result is None else str(result.run_dir),
                "metrics": {} if result is None else result.metrics,
                **{
                    name: {} if result is None else {key: str(path) for key, path in getattr(result, name).items()}
                    for name in ("artifacts", "tables", "plots")
                },
                **(
                    {
                        "error": error,
                        "failed_phase": "configuration" if result is None else result.extra.get("failed_phase"),
                    }
                    if error is not None
                    else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
