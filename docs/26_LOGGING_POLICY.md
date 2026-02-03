# Logging policy

This project uses Python's standard `logging` module. Avoid `print` in runtime code; use a module-level logger.

## Levels
- `DEBUG`: detailed internal state, shapes, intermediate values.
- `INFO`: major steps (fit/transform/inverse) and high-level summaries.
- `WARNING`: non-fatal fallbacks (e.g., algorithm substitution), numerical instability warnings.
- `ERROR`: unrecoverable failures before raising.

## Format (recommended)
Configure logging in your entrypoint (CLI or script):
```
%(asctime)s %(levelname)s %(name)s: %(message)s
```

## Configuration example
```
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
```

## Rules
- Do not log large arrays; log shapes and summary stats instead.
- Prefer one warning per fallback event to avoid noisy logs.
- If a warning affects reproducibility, also record it in `coeff_meta` or artifacts when applicable.
