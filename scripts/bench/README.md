# Legacy Benchmark Sweeps (Deprecated)

This folder contains older benchmark sweep scripts that were used before `tools/bench/` became the canonical
benchmark entrypoint.

## Status

- This directory is **legacy**.
- Prefer using `tools/bench/` for new benchmarking runs and reports.

## Recommended (Canonical) Workflow

```bash
python3 tools/bench/run_benchmark_v1.py --cases rectangle_scalar --stages decomposition
python3 tools/bench/generate_summary_benchmark_md.py --out summary_benchmark.md
```

## Notes

- These scripts may be removed after a deprecation window.
- If you rely on these scripts, migrate to `tools/bench/` first.

