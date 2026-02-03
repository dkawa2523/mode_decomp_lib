# TASK 512: 設計メモ: ClearML Task化を見越した境界確認（I/O契約維持、実装は最小）

## 目的
将来ClearMLでstepをTask化することを見越し、今の時点で“分割できる形”を維持します（実装はしない）。

## 作業内容
1. `fit_decomposer` / `fit_coeff_post` / `train_model` / `predict_latent` / `reconstruct` / `eval` / `viz` の関数境界を確認
2. stateやartifactのI/Oがそれぞれの境界で完結しているか確認
3. docs/addons/33 に、現状の境界とTODOを追記
4. コード側の境界が曖昧なら最小のリファクタリング（ただしスコープ増やさない）

## 受け入れ条件
- docsに将来のClearML Task分割の設計メモが反映されている
- 今の実装が境界を壊していない（I/Oが保たれる）

## 検証
- train→predict→reconstruct→eval が state/artefact 経由で通る
