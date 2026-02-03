# TASK 498: POD backend=modred（inner_product_weights対応、optional依存）

## 目的
backend=modred を用いて、Weighted POD（inner_product_weights）を現実的に実装します。

## 作業内容
1. modred を optional dependency として導入（requirements/extras）
2. `backend=modred` 実装：
   - solver=snapshots を主に実装（大特徴数でも成立しやすい）
   - `inner_product=domain_weights` のとき、Domain.integration_weights() を `inner_product_weights` として渡す
3. modredが期待するデータ形状（vecs列がスナップショット等）に合わせて reshape/stack を共通関数化
4. modredの truncation（atol等）がある場合、v1はデフォルトに従い、必要なら options へ露出（YAML増殖禁止）
5. stateの保存形式を sklearn backend と共通化（modes/eigvals/mean）

## 受け入れ条件
- modredが無い環境でもプロジェクトが壊れない（optional）
- modredがある環境で `backend=modred, inner_product=domain_weights` のfitが動く
- Weighted POD の導入により、mask/diskでの再構成が安定する

## 検証
- modred有りの環境で、disk/maskのdatasetでfit→reconstructが動く
- modred無しの環境で doctor が “optional missing” として適切に案内する
