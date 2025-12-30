# Model: Ridge多出力（cond→(a|z)）

**ID:** 070  
**Priority:** P0  
**Status:** done  
**Depends on:** 060  
**Unblocks:** 080  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
回帰モデルとして Ridge を多出力で実装する。
- 入力: cond [N,D]
- 出力: target [N,K]（a または z）

設定で以下を切替:
- target_space: a or z
- alpha, fit_intercept, normalize（必要なら）
- cond_scaler（standardize等、最小）

## Acceptance Criteria
- [x] Ridge が cond→target の多出力回帰として動く
- [x] target_space を設定で切替できる
- [x] seed固定で再現できる

## Verification
- [x] 合成データで train→predict が通り、shapeが一致する

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/models/__init__.py`, `configs/model/ridge.yaml`, `tests/test_models_ridge.py`
- 重要な関数/クラス: `src/mode_decomp_ml/models/__init__.py` の `RidgeRegressor`, `build_regressor`
- 設計判断: Ridge本体とcond標準化を最小構成で実装し、`target_space` を明示設定にして比較可能性を維持。保存は既存artifact契約に合わせて `model.pkl` に統一。
- リスク/注意点: `target_space` はモデルのメタ情報であり、誤設定すると downstream の解釈がズレる。
- 検証コマンドと結果: `pytest tests/test_models_ridge.py`（PASS）
