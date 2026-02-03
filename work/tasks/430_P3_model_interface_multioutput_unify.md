# Task: 430 Refactor: Model I/F 統一（常に多出力、wrapperで単一/複数を吸収）

- Priority: P1
- Status: done
- Depends on: 398
- Unblocks: 431, 432, 433, 440, 490

## Intent
単一目的変数/多目的変数で処理が分岐して複雑化しないよう、
モデルI/Fを **常に (N,L)** の多出力として扱う設計に統一する。
単一目的は L=1 として扱い、分岐はモデル内部の wrapper に閉じ込める。

## Context / Constraints
- pipeline側に `if L==1` の分岐を入れない
- 2系統:
  - NativeMultiOutputModel（MultiTaskLasso, MTGP等）
  - IndependentMultiOutputWrapper（XGB/LGBM/CatBoost等）
- 既存 Ridge/GPR との互換を維持する

## Plan
- [ ] `BaseRegressor`（fit/predict/predict_std(optional)）を統一
- [ ] `IndependentMultiOutputWrapper` を実装（1D regressor を L 回回す）
- [ ] 既存 Ridge/GPR を新I/Fに合わせる（破壊的変更は避け、Adapter可）
- [ ] tests: 1出力(L=1)と多出力(L>1)の両方でtrain/predictが通る

## Acceptance Criteria
- [ ] モデル呼び出し側（pipeline/processes）に “単一/複数” の分岐が無い
- [ ] Ridge/GPR が新I/Fで動く
- [ ] wrapper が導入され、GBDT等の追加に備えられる

## Verification
- unit tests が通る
- `model=ridge` 既存の例が壊れない

## Review Map
- 変更ファイル一覧（追加/変更/削除）
  - 追加: `src/mode_decomp_ml/plugins/models/base.py`, `tests/test_models_interface_multioutput.py`
  - 変更: `src/mode_decomp_ml/plugins/models/sklearn.py`, `src/mode_decomp_ml/plugins/models/__init__.py`, `src/mode_decomp_ml/plugins/registry.py`
  - 削除: なし
- 重要な関数/クラス
  - `BaseRegressor`, `IndependentMultiOutputWrapper` (`src/mode_decomp_ml/plugins/models/base.py`)
  - `ElasticNetRegressor`, `GPRRegressor` の多出力統一 (`src/mode_decomp_ml/plugins/models/sklearn.py`)
- 設計判断
  - BaseRegressor に `predict_std` を追加し、常に (N, L) へ整形する共通ヘルパーを導入。
  - 単一出力モデルは `IndependentMultiOutputWrapper` で L 回学習し、多出力I/Fへ吸収。
  - wrapper 内の factory は fit 後に破棄し、pickle 互換を保つ。
- リスク/注意点
  - wrapper は fit 後に再学習できない設計（必要なら新インスタンスを作成）。
  - base モデルは 1D 出力を前提（2D を返す場合はエラー）。
  - GPR の std 出力 shape 仕様に依存するため、sklearn 変更時は注意。
- 検証コマンドと結果
  - `pytest tests/test_models_ridge.py tests/test_models_gpr.py tests/test_models_interface_multioutput.py`（10 passed）
  - `pytest tests/test_processes_e2e.py`（1 passed）
