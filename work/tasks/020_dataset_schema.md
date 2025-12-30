# データスキーマとDataset実装（scalar/vector, mask）

**ID:** 020  
**Priority:** P0  
**Status:** done  
**Depends on:** 010  
**Unblocks:** 030  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
データを “cond→field” 学習に統一するため、データスキーマを固定し Dataset を実装する。

必須スキーマ（1サンプル）:
- cond: shape [D]
- field: shape [H,W,C]  （scalar: C=1, vector: C=2）
- mask: shape [H,W]（None可、mask外は評価対象外）
- meta: dict（domain種、スケールなど）

P0では以下を用意:
- synthetic dataset（合成波/ガウス等で検証用）
- npy_dir dataset（cond.npy + field.npy + mask.npy を読む等）

## Acceptance Criteria
- [x] dataset が上記スキーマで sample を返す
- [x] scalar/vector を同じコードで扱える（C次元で分岐しない）
- [x] maskがある場合、mask外を評価から除外できるよう meta が残る

## Verification
- [x] `task=doctor` で synthetic dataset 1件がロードできる（shapeがログに出る）

---

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/data/datasets.py` `src/mode_decomp_ml/data/__init__.py` `src/processes/doctor.py` `configs/config.yaml` `configs/dataset/synthetic.yaml` `configs/dataset/npy_dir.yaml` `docs/02_DOMAIN_MODEL.md` `docs/10_PROCESS_CATALOG.md` `tests/conftest.py` `tests/test_datasets.py` `configs/dataset/placeholder.yaml`(削除)
- 重要な関数/クラス: `src/mode_decomp_ml/data/datasets.py` の `FieldSample`, `build_dataset`, `SyntheticDataset`, `NpyDirDataset`、`src/processes/doctor.py` の `main`
- 設計判断: datasetは registry で name から生成し、cond/field/mask/meta の固定スキーマを `_validate_sample` で検証。mask扱いは `mask_policy` を config で明示し、missing時は早期エラーにした。
- リスク/注意点: `npy_dir` は field の入力次元が曖昧な場合にエラーになり得るため、4D (N,H,W,C) または 3D (H,W,C) を想定。maskが無い場合は `mask_policy` を `allow_none` にする必要がある。
- 検証コマンドと結果: `PYTHONPATH=src python -m mode_decomp_ml.cli.run task=doctor` を実行し、cond/field/mask shape を出力確認（dataset=synthetic）。
- 削除一覧: `configs/dataset/placeholder.yaml`
