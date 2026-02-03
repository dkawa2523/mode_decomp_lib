# TASK 495: Domain: integration_weights/mass_matrix I/F導入（Weighted PODの土台）

## 目的
Weighted PODをスパゲッティ化させずに実装するため、Domainが **重み/質量行列を提供するI/F** を追加します。
POD実装側では domain種別分岐を作らず、Domainから受け取って処理します。

## 作業内容
1. Domain基底クラス（またはprotocol）に以下を追加（無いなら新設）
   - `integration_weights()` → 1D or 2D array（格子点重み、mask込み）
   - `mass_matrix()` → optional（mesh用の質量行列、v1はNone許容）
2. rectangle/disk/arbitrary_mask の Domain 実装で `integration_weights()` を実装
   - maskがある場合は重み=重み×mask
   - diskは面積要素や座標正規化規約に合わせる（既存domainの定義を尊重）
3. mesh domain（存在するなら）では v1として `mass_matrix()` のhookを追加し、未対応の場合は明確に TODO
4. 上記I/Fが無いDomainでも動くよう、POD側は `euclidean` にフォールバック可能にする（警告を出す）

## 受け入れ条件
- Domain側に重みI/Fが追加され、POD側がdomain分岐なしで利用できる
- 既存の解析基底 decomposer には影響しない（インタフェース追加の副作用なし）
- mask domainで `integration_weights()` がmaskを含む

## 検証
- ドメイン単体の簡易テスト（rectangle/disk/maskで weights の shape/値域を確認）
- 既存の最小実行が動く（Task492時点のコマンド）

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/domain/__init__.py`, `src/mode_decomp_ml/plugins/decomposers/data_driven/pod_svd.py`, `configs/decompose/data_driven/pod_svd.yaml`, `tests/test_domain.py`
- 重要な関数/クラス: `DomainSpec.integration_weights()` / `DomainSpec.mass_matrix()`（`src/mode_decomp_ml/domain/__init__.py`）、`PODSVDDecomposer.fit/transform/inverse_transform`（`src/mode_decomp_ml/plugins/decomposers/data_driven/pod_svd.py`）
- 設計判断: Domain側に重みI/Fを追加し、PODは `inner_product` 設定で domain_weights を要求可能にした。Domainが重み未提供なら警告の上で euclidean にフォールバックし、既存挙動を保持する。
- リスク/注意点: `inner_product=domain_weights` では重みが 0 の点は再構成で 0 埋めになる。meshの mass_matrix は v1 未実装（TODO）。
- 検証コマンドと結果:
  - `python3 -m pytest tests/test_domain.py -q`（10 passed）
  - `PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml`（成功）
- 削除一覧: なし
