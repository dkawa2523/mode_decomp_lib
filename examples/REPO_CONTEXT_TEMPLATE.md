# REPO_CONTEXT.md（テンプレ）

このファイルは “対象repoの事実” をChatGPT/Codexに渡すための入力パケットです。
捏造を防ぎ、docs/work/tasksを高密度に生成するために必須です。

## 最低限の内容
- tree -L 4
- 入口（train/eval/predict等）
- configs構造
- データI/O（入力形式、主キー、目的変数）
- 評価（split/seed/metrics）
- 依存（requirements/pyproject）

## 生成コマンド例（Mac/Linux）
```bash
OUT=REPO_CONTEXT.md
{
  echo "# Repo Context"
  echo "## tree"
  tree -L 4
  echo
  echo "## entrypoints"
  ls -la scripts 2>/dev/null || true
  echo
  echo "## configs"
  tree -L 4 configs 2>/dev/null || true
  echo
  echo "## deps"
  cat pyproject.toml 2>/dev/null || true
  cat requirements.txt 2>/dev/null || true
  echo
  echo "## grep: train/eval/predict"
  rg -n "train|evaluate|predict|metrics|r2|rmse|mae" -S src scripts 2>/dev/null || true
} > "$OUT"
```
