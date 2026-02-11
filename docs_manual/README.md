# docs_manual (MkDocs manual)

`docs_manual/` is a MkDocs site that summarizes this repository in a structured, third-party friendly format.

## Canonical vs manual

- Canonical docs live under `docs/` (numbered docs).
- This MkDocs manual (`docs_manual/`) is a curated view and may lag behind `docs/` if not regenerated/updated.

## Terminology rules

- Use **"モード分解"** consistently. Avoid alternative terms like "特徴量化" or "featureization" in the docs.
- Always distinguish:
  - `field`: original spatial field (scalar or vector)
  - `coeff`: coefficients after mode decomposition
  - `coeff(a)`: codec-encoded coefficient vector (model default target)
  - `coeff(z)`: post-processed coefficient vector (e.g. PCA)

## Build / serve locally

```bash
python3 -m pip install mkdocs
cd docs_manual
mkdocs serve -f mkdocs.yml
```

## Mermaid diagrams

This manual uses Mermaid without a MkDocs plugin:
- Use `<div class="mermaid">...</div>` blocks in Markdown.
- Mermaid is loaded via `extra_javascript` in `mkdocs.yml`.
- Mermaid is vendored under `docs_manual/docs/assets/js/mermaid.min.js` so diagrams work offline.
