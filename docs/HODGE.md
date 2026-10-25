# 非周期領域の離散 Hodge 分解

`hodge` は固定された2次元 Cartesian 格子上の**辺の接線線積分**を exact・coexact・harmonic に直交分解します。矩形・disk・annulus・arbitrary mask に対応し、穴に沿う循環を harmonic 成分として保持します。周期格子の節点ベクトル射影には既存の `helmholtz` を使います。旧 `helmholtz_poisson` は廃止し、alias は設けません。

## 学習表現と独立した成分解析

Hodgeを学習用の係数表現に選ばなくても、POD/DCT等で再構成・予測した物理fieldを分解できます。同じYAMLへ次を追加します。

```yaml
analysis:
  components:
    enabled: true
    method: hodge
    params: {boundary: natural, tolerance: 1.0e-11}
```

kind:vector、x/y成分と座標の単位、固定非周期geometry等の前提は同じです。surrogateでは物理単位へ戻した予測fieldを再分解します。学習されたcomponent座標のラベルだけでは物理的なexact/coexact条件を保証しません。周期uniform gridにはmethod:helmholtzを明示します。disabledなら追加operatorを準備しません。

## YAML から使う

```powershell
python examples/create_hodge_dataset.py --samples 1
python -m ml_platform_mode_decomp config/tasks/mode_decomp_hodge.yaml
```

例は円環上の勾配・回転・穴周りの循環を含む単一の節点ベクトル観測です。生成する入力は `field.npy` だけで、geometry 等は1つの run YAML に書きます。生成先が既にある場合は生成コマンドを省略します。`--samples` の既定値は1で、24等を指定しても同じ `NHWC` 設定を使えます。全入力を直接分解するタスクなので、`split`・`analysis.kind`・条件列は不要です。profile は環境設定を分けたい場合だけ `--profile PATH` で追加できます。従来の `--task PATH` 形式も利用できます。

自分の単一 NPY を使う場合は、`field.npy` と次の `run.yaml` だけで実行できます。ここでは field が `(25,25,2)`、座標単位が m、成分順が x/y である例を示します。path は起動 cwd 基準で指定します。

```yaml
task: mode_decomp_representation
data:
  format: npy
  local_path: .
  fields:
    path: field.npy
    layout: HWC
    kind: vector
    components: [u_x, u_y]
    units: [m/s, m/s]
  geometry:
    kind: rectangle
    coordinate_units: {x: m, y: m}
    axes:
      x: {start: -1, step: 0.08333333333333333, count: 25}
      y: {start: -1, step: 0.08333333333333333, count: 25}
representation:
  decomposition: {name: hodge, params: {boundary: natural}}
analysis:
  components: {enabled: true}
report: {enabled: true}
```

```powershell
python -m ml_platform_mode_decomp run.yaml
```

`HWC` は単一場、`NHWC` は複数場です。sample table を省略すると ID を自動生成します。入力に合わせて geometry と実座標を指定し、円環等ではその domain を記述します。別の manifest や schema 名を追加して書く必要はありません。

```yaml
representation:
  field_preprocess: {name: none}
  decomposition:
    name: hodge
    params:
      boundary: natural
      tolerance: 1.0e-11
```

`name: hodge` で有効になります。`enabled` は不要です。`boundary` は `natural` だけに対応し、既定値も同じです。`tolerance` は反復最小二乗の相対停止許容値、`maxiter` は任意の反復上限です。未収束を harmonic 成分として採用せず、エラーにします。

例設定は `analysis.components.enabled: true` により、直接分解した全入力の物理成分を明示的に保存します。分解方法の選択と、追加成果物の保存を分ける設定です。未指定時に全 sample の成分・energy を常時保存する処理は行いません。

`model.joblib`、A/Z を含む `coefficients.npz`、`reconstructed_fields.npz`、原 field の `metrics.json` と `analysis_summary.json` を保存します。scope は **`fitted_input_reconstruction`** です。入力自身を fit・再構成した値であり、未知 sample の性能を表しません。未知場の比較には独立 holdout を持つ `mode_decomp_analysis` を使います。

field は `(H,W,2)`、channel 順は **x 方向・y 方向**です。既知の2成分の単位を揃え、座標軸も共通の長さ単位へ変換してください。自動単位変換は行いません。物理ベクトルの分解に channel 別標準化は使いません。domain mask は固定境界であり、sample ごとの未観測値ではありません。domain 内の未観測点は拒否します。domain 外の値は使用しません。

Python配列からは、同じ意味を次のように明示します。`field` は `(H,W,2)`、`x` / `y` は実座標です。

```python
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

data = FieldDataset.from_arrays(
    field,
    layout="HWC",
    x=x,
    y=y,
    field_kind="vector",
    components=["u_x", "u_y"],
    field_units=["m/s", "m/s"],
)
model = FieldRepresentation({"decomposition": {"name": "hodge"}}).fit(data)
```

周期軸を宣言した geometry は拒否します。周期格子を暗黙に切断して自然境界へ変更しません。

## 何を直交分解するか

valid な格子点を頂点、横・縦に隣接する valid 頂点間を辺、4頂点が全て valid な格子セルを面とします。対角接続や周期 wrap は追加しません。形状はこの raster cell complex に離散化され、連続な円の境界を厳密に表すものではありません。

辺の向きは +x または +y、面の向きは反時計回りです。`d0` は頂点 potential の差を辺に割り当て、`d1` は辺の線積分を面周囲で足します。incidence から作るため **`d1 @ d0 = 0`** です。

geometry の正の節点積分面積を `m_v`、辺長を `ell_e` とします。同方向の incident 辺への配分を

```text
gamma(v,e) = ell_e / sum(同方向でvに接続する辺長)
A_e = gamma(p,e)*m_p + gamma(q,e)*m_q
W_e = A_e / ell_e²                 # e=(p,q)
<a,b>_W = sum_e W_e*a_e*b_e
```

と定義します。これが本実装の正の diagonal edge metric です。通常の直交矩形格子では dual length / primal length に一致し、mask 境界では明示した node-area lumping を使います。非等間隔軸でも実座標と実面積を使います。domain 内の面積は有限かつ正である必要があります。

辺1-cochain `a` を次に分けます。

```text
g = d0*phi                              # exact
c = W^-1*d1.T*psi                       # coexact
h = a - g - c                          # harmonic
d1*h = 0,  d0.T*W*h = 0
```

`phi` は `||sqrt(W)*(d0*phi-a)||` を最小化し、coexact は `W^-1/2*d1.T` の像へ同じ座標で射影します。3成分はこの辺内積で直交し、`energy(a) = energy(g) + energy(c) + energy(h)` です。解は solver の許容誤差に従います。potential の定数 gauge は分解成分へ影響しません。

`natural` は全ての頂点 potential と保持面 cochain を使う有限複体の規約です。harmonic の `d0.T*W*h=0` は境界頂点を含む弱い収支条件です。**節点速度の境界法線が点ごとにゼロ、任意の Dirichlet／Neumann PDE 境界値、流入・流出の指定**を実装したという意味ではありません。

数値確認では、一様な矩形通過流のほぼ全energyがexactへ入り、harmonic比は約1.16×10⁻²¹でした。境界での流入出をゼロへ変えていません。一方、coexactの辺上の弱発散が約2.78×10⁻¹⁷でも、節点へliftした境界法線の最大値は約0.00951でした。辺の弱収支から節点ごとの不透過境界を推論できません。

harmonic 次元は選んだ複体の1次 Betti 数です。この有限 planar complex では `E - V + connected_components - F` になります。通常の disk は0、穴が1つの円環は1です。細い部分・点接触・粗い mask は離散 topology を変えるため、連続領域の穴の数と一致するとは限りません。

例えば外半径1・内半径0.9の薄い円環は、25×25では16連結成分・harmonic次元0、33×33では1連結成分・次元1でした。厚い円環の循環は17×17 / 33×33 / 65×65で2πを保持しましたが、native harmonic energyは6.5939 / 7.4893 / 7.4218で、連続場の積分参照7.5648へ単調には収束しません。実領域への適用では穴の数だけでなく、連結成分と境界の解像度依存を確認します。

## native 辺 API

```python
import numpy as np
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.config import load_config
from ml_platform_mode_decomp.decomposition.hodge import HodgeDecomposition

data = load_field_dataset(load_config("config/tasks/mode_decomp_hodge.yaml")["data"])
hodge = HodgeDecomposition(data.geometry)
a = hodge.sample_nodes(data.fields[0])
parts = hodge.decompose_edges(a)
np.testing.assert_allclose(hodge.reconstruct_edges(parts), a)
print({name: hodge.energy(value) for name, value in parts.items()})
print(hodge.harmonic_dimension)
```

`decompose_edges` は `exact`・`coexact`・`harmonic` の辞書を返します。測定した線積分を直接渡す場合は、`edge_vertices` の向きと順序に合わせます。`node_indices` は元配列を flat にした頂点位置、`vertex_coordinates` は座標、`edge_axes` は x=0／y=1、`edge_lengths` は実辺長です。`d0`、`d1`、`edge_weights` も参照できます。geometry から作る operator は sample ごとに再構築しません。

円環の `dtheta` のように、全ての保持面で circulation がゼロでも穴周りの循環が非ゼロなら harmonic 成分を持ちます。対して `d0*log(r)` は大域的な potential の差なので exact に属します。発散・回転の有限差分残差が小さいだけで harmonic と判定する実装ではありません。

## 節点 field との bridge と再構成

`sample_nodes` は各辺で両端の接線成分を平均し、辺長を掛けます。

```text
a_e = ell_e * (u_p[axis] + u_q[axis]) / 2
```

`lift_edges` はこの sampling の right inverse です。同じ辺値を表す節点ベクトルのうち、geometry の節点面積 norm が最小のものを返します。各横列・縦列の連結 path で交互符号の自由度を解析的に固定し、節点へ戻してから同じ sampling を行うと元の辺値になります。

節点 adapter の raw 係数は `(4,H,W,2)` で、次の順です。

1. `exact_edge_lift`
2. `coexact_edge_lift`
3. `harmonic_edge_lift`
4. `sampling_residual = original_node_field - sum(3 lifts)`

4成分の和で元の domain 内節点場を再構成します。交互符号の節点パターン等は endpoint 平均で消えるため、`sampling_residual` に残ります。これは harmonic ではありません。lift 後の節点面積内積で3成分が直交する保証もありません。native 辺 energy と、原 field の再構成 RMSE は別の量です。

これは圧縮表現ではありません。任意に編集した4成分や回帰器が出す packed 座標には、transform が作る exact／coexact／harmonic 条件を自動保証しません。物理成分を調べるときは、予測した原 field を native API で改めて sampling・分解します。

## YAML による物理成分の成果物

`analysis.components.enabled: true` は Hodge を選んだ表現について、入力または予測した原 field を **sampling → native Hodge 分解**し直して保存します。直接分解では全入力 field、holdout 比較の representation では未使用入力 field、surrogate では条件から予測した field が対象です。回帰が出した4個の packed part を物理成分とみなす処理ではありません。

- `components.npz`：`edge_sampled`、`edge_exact`、`edge_coexact`、`edge_harmonic`、3個の `node_*_edge_lift`、`node_sampling_residual`。sample IDs、成分名・単位、固定 mask、節点座標・面積、辺向き・長さ・内積重みも含みます。
- `component_energy.csv`：sample ごとの native な入力・3成分・成分合計の edge energy と、node-area norm による sampling 残差。
- `component_summary.json`：境界・内積・単位・topology と、物理保証が native edge cochain に限られることを記録します。

NPZ の `edge_vertex_indices` は `node_coordinates` 内の index です。`node_flat_indices` から元の H/W 配列位置へ対応させられます。node 配列は元の `(N,H,W,2)`、辺配列は `(N,E)` です。domain 外の node 出力はゼロです。energy は密度や係数1/2を含まない二乗 norm であり、流体の運動エネルギーそのものとは区別します。

## 根拠と検証範囲

離散 incidence、正定値内積、Hodge 分解と cohomology の対応は [Desbrun et al., SIGGRAPH 2006 notes, §7](https://www.geometry.caltech.edu/pubs/GSD06.pdf) を参照してください。対角 Hodge star の幾何学的な比は [DGtal DEC documentation](https://www.dgtal.org/doc/stable/moduleDECIntroduction.html)、反復最小二乗は [SciPy LSMR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html) に基づきます。本実装の mask 境界の metric は上記の配分式で定義し、一般 mesh の Whitney mass や連続 PDE への収束保証とは区別します。

数値テストでは manufactured exact／coexact、穴周りの2π循環、closed／coclosed、直交・energy・射影冪等性、非等間隔座標と重み尺度、sampling nullspace、部分観測の拒否、未収束の拒否を確認します。mesh・球面・3D・周期複体・移動境界は対象外です。
