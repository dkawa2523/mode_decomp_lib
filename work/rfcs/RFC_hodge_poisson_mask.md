# RFC: Hodge/Poisson decomposition on arbitrary masks (grid domains)

## 背景
`helmholtz` / `helmholtz_poisson` は rectangle 上のベクトル場に対して curl-free / div-free を提供するが、
任意マスク（不規則境界）の領域上で同等の “物理解釈” を提供できていない。

任意マスク上の Poisson/Hodge は、境界条件・離散化・マスクの扱い次第で結果の意味が大きく変わるため、
実装前に method の仕様を固定して比較可能性を担保する必要がある。

## 目的
- grid domains 上で、vector field `(H,W,2)` を curl-free / div-free に分解する method を定義する
- 境界条件・離散化・mask の扱いを明示し、再現性と比較可能性を確保する

## 非目的
- 速度最適化（前処理キャッシュ、multigrid 等）
- FEM など高次の離散化（v1 は FD/sparse）
- mask がサンプルごとに大きく変わるケースの高速化（可能だが高コスト）

## 変更点（提案）
- 新規 decomposer `hodge_poisson_mask` を追加（次タスク）
- `coeff_layout: PHWC`、parts=`["curl_free","div_free"]` を標準化

## 影響範囲
- `src/mode_decomp_ml/plugins/decomposers/`（新規追加）
- `src/mode_decomp_ml/domain/__init__.py`（互換チェック）
- `configs/decompose/`（config coverage）
- `tests/`（再構成・curl/div 検証）
- `docs/`（method catalog / recommendations）

## 仕様（v1 提案: decision complete）
### 対象
- domain: grid domains のうち `arbitrary_mask|mask|disk|annulus|rectangle`
- field: `float` のベクトル場 `(H,W,2)`

### 入力 mask の扱い
- domain mask（`domain_spec.mask`）は **固定**（basis/operator 構築の対象）
- dataset/sample mask は transform 時に許可するが、mask が可変の場合は毎回 sparse solve が必要になり高コスト
  - v1 の推奨運用は「domain mask 固定 + sample mask は軽微」

### 離散化（演算子）
- ノード集合: `domain_mask == True` の画素（flatten index）
- Laplacian: masked 5-point finite difference（近傍が domain 外なら項を落とす）
- divergence/curl:
  - interior: centered difference
  - boundary: one-sided difference（近傍が無い場合は 0 扱い）
  - どの差分を使うかを実装で固定し、docs に明記する

### 境界条件
- v1 default: “自然境界(Neumann)” 相当（境界外フラックス 0）
- optional: Dirichlet（境界のポテンシャル 0）

### 解く方程式
- scalar potentials `phi` / `psi` を求める:
  - `Δ phi = div(field)`
  - `Δ psi = -curl(field)`
- parts:
  - `curl_free = grad(phi)`
  - `div_free = rot_grad(psi)`

### ソルバ
- `cg` + Jacobi precondition を基本
- 小規模は `splu`（direct）を許可（閾値は config）

### 出力
- raw coeff: `parts x H x W x 2`（`coeff_layout: PHWC`, `complex_format: real`）
- inverse: parts を足し合わせる（mask 外は 0）

## 代替案
- graph Laplacian（`graph_fourier` 系）で Hodge を定義する
  - トポロジ依存で強力だが、設計が大きくなり比較可能性が落ちやすい
- FEM/mesh 化して Laplace-Beltrami 上で処理する
  - 精度は上がり得るが v1 の実装コストが高い

## 互換性・移行
- 新規 method 追加のため既存 pipeline 互換は壊さない
- 係数レイアウト `PHWC` は既存 `helmholtz`/`helmholtz_poisson` と揃える

## 受け入れ条件
- 任意マスク domain で `fit/transform/inverse_transform/coeff_meta` が動作する
- `inverse_transform(transform(field))` が interior で小さい誤差（許容誤差を明示）
- parts の interior curl/div RMS が小さい
- `pytest -q` が通る、config coverage が通る

## 実装タスク案
- work/queue_p2.json に `hodge_poisson_mask` を追加（RFC承認後）

