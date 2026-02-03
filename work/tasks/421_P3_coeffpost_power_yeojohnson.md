# Task: 421 Add: CoeffPost PowerTransform（Yeo-Johnson）

- Priority: P1
- Status: done
- Depends on: 410
- Unblocks: 440, 490

## Intent
係数後処理として PowerTransform（Yeo-Johnson）を追加し、歪度の大きい係数分布をガウスに寄せて
線形モデルやGPRの効きを改善する。

## Context / Constraints
- train-only fit（リーク禁止）
- inverse 必須（再構成で戻す）
- standardize有無をパラメータ化

## Plan
- [ ] coeff_post: `power_yeojohnson` を追加（scikit-learn PowerTransformer）
- [ ] state 保存
- [ ] tests: fit/transform/inverse + 数値安定性
- [ ] docs: Quantileとの使い分け（外れ値 vs 歪度）を短く記述
- [ ] examples: power+ridge の run.yaml

## Acceptance Criteria
- [ ] `coeff_post=power_yeojohnson` が選べる
- [ ] train/predict/reconstruct が通る
- [ ] stateが保存される

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_power_ridge.yaml`
- Expected:
  - metrics.json が生成される

## Review Map
- 変更ファイル一覧（追加/変更/削除）
  - 追加: `configs/coeff_post/power_yeojohnson.yaml`, `examples/run_scalar_rect_fft_power_ridge.yaml`
  - 変更: `src/mode_decomp_ml/plugins/coeff_post/basic.py`, `tests/test_coeff_post.py`, `docs/03_CONFIG_CONVENTIONS.md`, `docs/20_METHOD_CATALOG.md`, `docs/USER_QUICKSTART.md`
  - 削除: なし
- 重要な関数/クラス
  - `PowerYeoJohnsonCoeffPost` (`src/mode_decomp_ml/plugins/coeff_post/basic.py`)
  - `build_coeff_post` registry 経由で `coeff_post=power_yeojohnson` を解決
- 設計判断
  - sklearn `PowerTransformer(method="yeo-johnson")` をラップし、train-only fit を強制。
  - `standardize`/`copy` を config で切替可能にし、transformer と lambdas を state に保存。
  - 既存 coeff_post の shape/feature validation を踏襲して比較可能性を維持。
- リスク/注意点
  - train/serve skew 回避のため、fit は必ず train split のみ。
  - 特徴量次元が一致しない場合はエラーで停止（silent failure 回避）。
  - 分布の極端なシフトや定数列が多い場合は数値安定性に注意。
- 検証コマンドと結果
  - `python3 -m pytest tests/test_coeff_post.py`（7 passed）
