# Task: 420 Add: CoeffPost QuantileTransform（ロバスト化、train-only fit）

- Priority: P1
- Status: todo
- Depends on: 410
- Unblocks: 440, 490

## Intent
係数後処理として QuantileTransform を追加し、外れ値・歪度が強い係数分布をロバスト化して
学習（Ridge/GBDT/GPR）の安定性を向上させる。

## Context / Constraints
- train-only fit（リーク禁止）
- inverse 可能だが端部は近似になる可能性がある → meta に記録
- PCAと併用するかは設計次第（当面は “どちらか1つ” を基本）

## Plan
- [ ] coeff_post: `quantile` を追加（scikit-learn QuantileTransformer）
- [ ] state 保存（quantiles等）
- [ ] tests: fit/transform/inverse が動く + 形状契約
- [ ] docs: いつ有効か（外れ値/歪度）を短く記述
- [ ] examples: quantile+ridge の run.yaml

## Acceptance Criteria
- [ ] `coeff_post=quantile` が registry で選べる
- [ ] train/predict/reconstruct が通る
- [ ] state が `states/coeff_post_quantile.pkl` 等で保存される

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_quantile_ridge.yaml`
- Expected:
  - metrics.json が出力される
