# TASK 496: Decomposer: PODDecomposer骨格（backend/solver/inner_product切替、state仕様固定）

## 目的
POD系を増やしても分岐地獄にならないよう、**PODDecomposer 1クラス**で
- backend: sklearn | modred
- solver: direct | snapshots | randomized | incremental
- inner_product: euclidean | domain_weights
を切り替え可能にします。

## 作業内容
1. `PODDecomposer` を `data_driven/pod.py` 等に追加
2. 既存decomposerと同じI/F（fit/transform/inverse/meta）に揃える
3. state保存の設計を固定（後続タスクで共通利用）
   - `mean`（field平均）
   - `modes`（空間モード）
   - `eigvals`（固有値）
   - `weights_type` / `inner_product` / `backend` / `solver` / `seed`
4. scalar/vector対応は **channel-wise adapter** で実現（POD内部で分岐しない）
5. v1では `raw_coeff` は単純に `(N,K)` のfloatで良い（Codec不要、ただし将来のためI/Fは維持）

## 受け入れ条件
- `decomposer=pod` として registry から呼べる
- `fit/transform/inverse` が揃い、stateをartifactに保存できる
- vector場でも “chごとにPOD” として動く（モードと係数がch別に管理される）

## 検証
- 最小データ（既存の評価dataset）で POD fit→reconstruct が動く
- `states/` に保存され、再ロードして predict/reconstruct が可能
