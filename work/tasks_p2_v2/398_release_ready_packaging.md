# Task 398 (P2): Release-ready packaging（依存・実行・再現性の固定）

**ID:** 398  
**Priority:** P2  
**Status:** done  
**Depends on:** 395  
**Unblocks:** 390  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Goal
このプロジェクトを「基盤」として他者が使える状態にするため、
依存・実行コマンド・再現性を最小で固定する。

## Scope（最小）
- `README.md` に “最短実行例” を追加/更新
- `requirements.txt` or `pyproject.toml` の整合（どちらかに寄せる）
- version をどこかに固定（例: `src/mode_decomp_ml/__init__.py`）
- examples config を 2〜3 個に絞って動作確認

## Acceptance Criteria
- [x] 新規環境で `pip install -r requirements.txt` で動く
- [x] `task=doctor` と `task=benchmark` のコマンドが README通りに動く
- [x] 依存が過剰に増えていない（最小）

## Verification
- [x] 新しいvenvで実行確認（可能なら）

## Review Map
- 変更ファイル一覧: `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`, `src/mode_decomp_ml/__init__.py`, `work/tasks_p2_v2/398_release_ready_packaging.md`, `work/queue.json`
- 重要な関数/クラス: `src/mode_decomp_ml/__init__.py` の `__version__`（パッケージ版の固定）
- 設計判断: 依存は `requirements.txt`/`pyproject.toml` を同期し、編集インストールに必要な最小の build-system と src 配置だけを追加
- リスク/注意点: 小規模データの PCA/GPR で警告が出るが、実行自体は完了（警告は README の最短実行例にも出る可能性）
- 検証コマンドと結果: `.venv_release_copy/bin/python -m pip install -r requirements.txt --no-index` 成功; `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=doctor` 成功; `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=benchmark` 成功（PCA warning）; `PYTHONPATH=src python -m mode_decomp_ml.cli.run --config-name examples/pod_ridge` 成功; `PYTHONPATH=src python -m mode_decomp_ml.cli.run --config-name examples/pod_gpr_uncertainty` 成功（GPR warning）
