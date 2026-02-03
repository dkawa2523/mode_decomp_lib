# Addon after Task 492: Data-driven Decomposition (POD suite) — Design & Spec (v1)

このドキュメントは、**Task 492 完了時点のコード**を起点に、以下のデータ駆動分解を「スパゲッティ化させず」に追加するための仕様・設計指針をまとめます。

- Weighted POD（重み付きPOD / domain-aware POD）
- Randomized SVD / Randomized POD（高速SVD）
- Incremental POD（Out-of-core / Online）
- Gappy POD（欠損/部分観測で係数推定）

対象データは以下を不変条件とします：
- 入力: 条件テーブル `cond`（N×D）
- 出力: 時間変化のない定常2D分布 `field`（N×H×W×C）
- domain: rectangle / disk / arbitrary_mask / （将来）mesh
- scalar / vector（C=1 or 2）

NOTE: DomainSpec は annulus / sphere_grid / mesh も実装済み。POD対象範囲に含めるかは運用方針で明記する。

---

## 1. 目的と非目的

### 目的
- POD系の追加で「分解→係数学習→推論→再構成→評価」が一貫して回る
- domain（mask/mesh含む）でも比較可能性を壊さない
- YAML増殖・出力階層肥大を起こさない（run.yaml 1枚運用を維持）
- 既存の特殊関数系 decomposer（FFT/Zernike/Bessel/…）と区別しやすい配置にする
- 将来、ClearMLで step を Task 化できる粒度を意識（今は入れない）

### 非目的（v1ではやらない）
- SPOD/DMDなど時間相関を本質とする分解（定常データでは主戦場ではない）
- AE/VAE 等の非線形分解の大規模導入（別PR）
- mesh上の“完全な”重み付きPOD（v1は hook と最小の質量行列対応まで）

---

## 2. スパゲッティ化を防ぐ「責務分割（不変契約）」

POD系でやりがちな失敗は「PODクラスに全部入れる」ことです。
以下を **必ず分離**します。

1) **Domain**：重み/内積/質量行列（integral weight）を提供する  
2) **PODDecomposer**：基底（modes）と平均（mean）を学習し、transform/inverseを提供  
3) **GappyPODDecomposer**：欠損観測での係数推定だけを担当（PODと分離）  
4) **CoeffCodec**：係数のpack/unpack（PODは基本不要だが、将来Tensor等で必要）  
5) **CoeffPost**：分布正規化（Quantile/Power）やモード重み付け等（全decomposerで共通化）

---

## 3. 追加するファイル配置（既存の特殊関数系と区別）

（実際のパスはリポジトリ現状に合わせて調整）

推奨：
- `src/mode_decomp_ml/plugins/decomposers/*`：FFT/Zernike/Bessel/POD/Gappyなど
- `src/**/coeff_post/*`：Quantile/Power/ModeWeightなど（全decomposer共通）
- `src/**/domain/*`：integration_weights / mass_matrix を提供

NOTE: 現行repoは `src/mode_decomp_ml/plugins/decomposers/*` のフラット構成で、
`analytic/` フォルダは存在しない（config は `configs/decompose/analytic` に集約）。

互換性のため、既存の `pod_svd.py` が存在する場合は：
- 可能なら `data_driven/pod.py` へ移し、旧パスは薄い re-export にする（import互換）
- configの `decompose=pod_svd` が既に使われているなら alias を残す

NOTE(Task493): `pod_svd` は `src/mode_decomp_ml/plugins/decomposers/pod_svd.py` にあり、
config は `configs/decompose/data_driven/pod_svd.yaml`。

---

## 4. PODDecomposerの仕様（“1クラス + 2バックエンド + solver切替”）

### 4.1 PODDecomposer（v1）
- backend: `sklearn` | `modred`
- solver: `direct` | `snapshots` | `randomized` | `incremental`
- inner_product: `euclidean` | `domain_weights`

**重要**：PODDecomposer内で `if domain == ...` の分岐を作らない。  
Domainから `integration_weights()` / `mass_matrix()` を受け取り、共通処理として扱う。

TODO(Task493): 現行 `pod_svd` は NumPy SVDのみ（backend/solver/inner_product/mean_centered なし）。
`mask_policy` でマスク処理を行うため、本specのI/Fと差分がある（Task496以降で収束させる）。

#### Decomposer I/F（既存契約に合わせる）
- `fit(train_fields, domain) -> state`
- `transform(fields, domain) -> coeff`（N×K）
- `inverse(coeff, domain) -> fields_hat`
- `meta()`：K、eigvals、mode_order、weights_typeなどを保存

### 4.2 backend=sklearn（推奨の使い分け）
- solver=direct/snapshots：`sklearn.decomposition.PCA(svd_solver="full")`
- solver=randomized：`PCA(svd_solver="randomized", random_state=seed)`
- solver=incremental：`sklearn.decomposition.IncrementalPCA`

注意：
- inner_product=domain_weights は v1 では「sqrt(weights) で前処理 → PCA → inverseで除算」で対応する。
  - `integration_weights()` が無い場合は警告の上 `euclidean` にフォールバック。
- `IncrementalPCA.fit` と `partial_fit` は内部更新が異なり、**同一結果にならない**可能性がある
- `IncrementalPCA` は batch 学習で基底が得られるが、厳密に全データPCAと一致しない（許容範囲を明示）

### 4.3 backend=modred（重み付きPODの主戦場）
modred PODは `inner_product_weights`（1D/2D）で重み付き内積を与えられる。
- solver=snapshots：スナップショット法で大規模特徴数でも成立しやすい
- inner_product=domain_weights：Domainのintegration weightsを渡す

---

## 5. Weighted POD（domain-aware）の仕様

### 5.1 何を“重み”にするか
- rectangle/disk：積分重み（面積要素）+ mask
- arbitrary_mask：mask内のみ有効（重み=mask）
- mesh：質量行列（mass matrix）を想定（v1は hook と最小対応）

### 5.2 実装ポリシー
- Domainが `integration_weights()` を返せる場合：modredへ `inner_product_weights` として渡す
- 返せない場合：euclideanへフォールバック（ただし警告 or doctorで注意喚起）

UPDATE: DomainSpec は `integration_weights()` / `mass_matrix()` を提供済み。
mesh は頂点面積ベースの `weights` を提供し、`domain.mass_matrix` 指定時に対角質量行列を返す。

---

## 6. Gappy POD（欠損/部分観測）の仕様

Gappy POD は **“学習済みPOD基底を使って”** 欠損観測から係数を推定する段です。
PODと混ぜずに `GappyPODDecomposer` として分離します。

### 6.1 推定式（概念）
観測マスク M の下で、  
`argmin_a || M (x - mean - Phi a) ||^2`  
を最小二乗で解く（必要ならTikhonov正則化）。

### 6.2 v1でのスコープ
- まずは `mask` があるケース（任意マスク・欠損）に対応
- 正則化 λ を小さく入れ、数値不安定を回避できるようにする（設定は options でON/OFF）

UPDATE: Gappy POD は `data_driven/gappy_pod.py` として実装済み。

---

## 7. 周辺技術（全decomposer共通にする）

### 7.1 rank選択の自動化（energy閾値 + CV）
- energy閾値：累積寄与率でKを決める（最短・頑健）
- CV：K候補を試して `field_error` を最小化（高コスト、任意でON）

**重要**：YAML増殖させず、`options.rank_select.enable` のみでON/OFFできる設計にする。

UPDATE: `options.rank_select.*` と `options.mode_weight.*` は POD / POD-SVD で実装済み。
`options.coeff_normalize.*` は未実装のまま（全decomposer共通化の要件が残る）。

### 7.2 係数の分布正規化（Quantile/Power）
- `coeff_post` として実装し、特殊関数系decomposerでも利用できるようにする
- train-only fit を強制（リーク防止）

### 7.3 モード毎の重み付け（eigvalでスケール）
- POD/PCAの eigvals を使って、モード毎にスケール（whitening的）
- eigvalsが無い decomposer では no-op or error（設定により選べる）

---

## 8. 参考（調査メモ）
- modred POD: `inner_product_weights` による重み付き内積（docsに記載）
- Gappy POD: Willcoxのgappy POD（欠損観測から係数推定）の古典
- Randomized SVD: Halko et al. “Finding Structure with Randomness”
- Incremental PCA: scikit-learn IncrementalPCA（out-of-coreの標準実装）

（リンクは docs/ 内の他ドキュメントに集約して良い）

---

## 9. Task492時点のコード配置メモ（現行repoの実体）
- Decomposer registry: `src/mode_decomp_ml/plugins/registry.py`（`register_decomposer` / `build_decomposer`）
- Decomposer実体: `src/mode_decomp_ml/plugins/decomposers/**`（data_driven/..., grid/..., bessel/..., eigen/..., sphere/...）
- Domain実体: `src/mode_decomp_ml/domain/__init__.py`（rectangle/disk/annulus/arbitrary_mask/mask/sphere_grid/mesh）
  - mesh helper: `src/mode_decomp_ml/domain/mesh.py`
- run.yaml 1枚運用入口: `src/mode_decomp_ml/run.py`（`python -m mode_decomp_ml.run --config run.yaml`）
  - Hydra入口: `src/mode_decomp_ml/cli/run.py`
- Artifact契約: `docs/04_ARTIFACTS_AND_VERSIONING.md`（runs/<tag>/<run_id>/ 構造）
  - run dir生成: `src/mode_decomp_ml/pipeline/utils.py`（`RunDirManager`）

---
