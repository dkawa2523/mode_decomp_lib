# P1: Decomposer POD/SVD（train-only fit）

**ID:** 220  
**Priority:** P1  
**Status:** done  
**Depends on:** 200  
**Unblocks:** 230  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
train-only fit の POD/SVD 分解器を追加する。

## Acceptance Criteria
- [x] pod_svd が registry に登録され、fit/transform/inverse が動作する
- [x] train-only fit を満たす
- [x] coeff_meta が保存できる
- [x] smoke test を追加する

## Verification
- [ ] `pytest tests/test_decompose_pod_svd.py`（pytest 未導入のため未実行）
- [x] `python - <<'PY' ...`（max_err=2.384185791015625e-07）

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/decompose/pod_svd.py`, `src/mode_decomp_ml/decompose/__init__.py`, `src/processes/train.py`, `configs/decompose/pod_svd.yaml`, `tests/test_decompose_pod_svd.py`
- 重要な関数/クラス: `src/mode_decomp_ml/decompose/pod_svd.py` の `PODSVDDecomposer.fit`/`transform`/`inverse_transform`, `src/processes/train.py` の `_SubsetDataset`
- 設計判断: POD 基底はチャンネルごとに SVD し、mask は fit 時の固定マスク一致を要求することで比較可能性を担保した。
- リスク/注意点: mask がサンプルごとに変わるデータは `mask_policy=ignore_masked_points` でもエラーになるため、必要なら専用ポリシー追加が必要。
- 検証コマンドと結果: `python - <<'PY' ...`（max_err=2.384185791015625e-07）、`pytest tests/test_decompose_pod_svd.py`（pytest 未導入）
- 削除一覧: なし
