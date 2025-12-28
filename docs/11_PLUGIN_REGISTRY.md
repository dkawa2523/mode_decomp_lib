# Plugin Registry（拡張規約）

拡張されやすいポイントは必ずプラグイン化し、coreの変更を最小化する。

---

## 1. 対象
- preprocess: `PreprocessOp`
- vector: `VectorTransform`
- decompose: `Decomposer`
- coeff_post: `CoeffPost`
- model: `Regressor`
- metrics: `MetricFn`（必要なら）

---

## 2. インターフェース（例）
### Decomposer
- `fit(dataset, domain_spec) -> self`
- `transform(field, mask, domain_spec) -> coeff`
- `inverse_transform(coeff, domain_spec) -> field_hat`
- `coeff_meta()` を提供（index対応）

### CoeffPost
- `fit(A_train) -> self`
- `transform(A) -> Z`
- `inverse_transform(Z) -> A_hat`

### Regressor
- `fit(X_cond, Y) -> self`
- `predict(X_cond) -> Y_hat`
- `save/load`（artifact契約に従う）

---

## 3. registry の原則
- registry key は config の `*.method` と一致
- 未登録 key で実行したら即エラー
- 追加時は必ずテスト（少なくとも smoke）を追加

---

## 4. Hydra との統合
- config から `hydra.utils.instantiate` でプラグインを生成できる形にする
- constructor には config の params を渡す（kwargs）
