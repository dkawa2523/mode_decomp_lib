# 散布点を一回の実行で解析する

2次元の散布点から固定Cartesian格子への変換と、その後の分解・学習・解析を、一つのrun YAMLから実行できます。別のimport YAML・schema・profile・変換先の手動指定は不要です。

```powershell
python examples/create_scattered_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_scattered_analysis.yaml
```

生成する原入力はpoints.npy、values.npy、samples.csvの3ファイルです。[同梱analysis YAML](../config/tasks/mode_decomp_scattered_analysis.yaml)が入力仕様・target geometry・比較候補を持ちます。実行開始時に一度だけ正規NPYへ変換し、全候補で同じ入力を使います。

## 元観測点を主評価にする

同梱例は格子を主評価にし、元点を補助評価します。元点への予測精度を候補選択・CV・最終評価の主目的にする場合は、同じYAMLのmetricsを変更します。

```yaml
metrics:
  reference: observations
  names: [observation_rmse, observation_mae]
model:
  selection_metric: observation_rmse
```

点重みはsample内で正規化し、その後channel尺度・重みとsample重みを使います。点数の多いsampleを自動的に重くしません。元点R²は提供しません。参照から固定したsupportを候補間で使い、正重みsampleに採点可能点がなければ事前に拒否します。予測NaNによってcoverageを減らすことはありません。

予測帯を併用しても、その較正対象はgrid固定supportです。元点coverageが十分でも、grid全正measure地点のtruth要件を満たさなければ帯の較正・coverage評価はできません。

## run.yamlのdata

```yaml
data:
  local_path: data/scattered_measurements
  format: scattered
  samples: {path: samples.csv}
  conditions: {format: table, columns: [control_a, control_b]}
  observations:
    format: npy
    coordinates: points.npy
    values: values.npy
  fields:
    layout: NHWC
    kind: scalar
    components: [signal]
  geometry:
    kind: rectangle
    axes:
      x: {start: 0, step: 0.0625, count: 17}
      y: {start: 0, step: 0.0625, count: 17}
analysis:
  kind: surrogate
  scattered: {enabled: true}
```

上は一つのrunの一部です。Taskや候補を含む完全な例は同梱YAMLを使ってください。補間はmethod: linear / outside: maskが既定なので省略できます。明示する場合は `data.interpolation: {method: linear, outside: mask}` とします。未実装のmethodやnearest外挿は拒否します。

線形写像はDelaunay三角形上の重心座標に基づき、凸包外はNaNと無効maskにします。座標の自動rescaleやjitterは行いません。[SciPy Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)、[LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html)の公式仕様に対応します。

## 元観測の形式

| observationsの指定 | shape・意味 |
| --- | --- |
| coordinates | 共有 `(M,2)` またはsample別 `(N,M,2)` のNPY。最後はx,y |
| values | `(N,M,C)` のNPY。fields.components順 |
| mask | 任意の `(M,)` または `(N,M)`。boolean/0–1の元観測mask |
| weights | 任意の `(M,)` または `(N,M)`。有限・非負の点重み |
| simplex_support | 任意の `(S,3)` または `(N,S,3)`。許可するDelaunay三角形の元点index triple |

NPYのsample軸はvaluesのNから対応を決め、samplesを省略すれば位置IDを生成します。独自ID・条件・group/timeがある場合だけ表を追加します。array_index列がなければ表の行順を使い、明示indexがある場合はその対応を保持します。全pathはdata.local_pathの入力root基準です。

sampleごとに点数が違う場合は1 sample 1 CSVを使い、同じrun内のobservationsを次に替えます。

```yaml
observations:
  format: csv
  path_column: point_csv
  coordinates: [position_x, position_y]
  columns: [measured_signal]
  # mask_column: observed
  # weight_column: reliability
```

samples.csvにpoint_csv列を持たせます。単一sampleのCSVはobservations.pathで指定でき、sample表は不要です。成分はcolumnsとfields.componentsの同じ位置で対応し、行順・CSV列順は問いません。既知値は全成分で有限、欠測は明示maskで無効にします。点重みは評価用であり補間重心座標を変更しません。補間にも使わない点はmaskで無効にします。重複点・一直線配置・不安定な三角形はエラーです。

初期対応は固定された2D Cartesian格子へのscalar/vector/multichannel補間です。sphere、3D点群、meshへの直接補間、周期方向の回り込みは含みません。元座標とtarget軸の単位を合わせます。補間値の有効maskは変換後の教師値が定義される位置であり、新たに実測された点という意味ではありません。

NPY単体読込のindex座標既定とは異なり、散布点ではtarget geometryを明示します。元点の座標単位やtargetの物理的な幅を、点群や配列shapeから推測しません。

## 穴や壁は明示した支持範囲で扱う

点群の凸包から穴・壁・領域分離は分かりません。domain maskで出力を切り抜くだけでは、穴を跨ぐsource三角形の妥当性は判断できません。annulus、arbitrary_mask、maskではobservations.simplex_supportを必須にします。

allowed_triangles.npyには実際の境界に基づいて許可した三角形の元点index tripleを保存します。順序は問いません。未指定の三角形は使わず、Delaunayに存在しないtripleは拒否します。全三角形を許可する指定は、穴を考慮した指定にはなりません。

Pythonの `BarycentricMap.build(points, queries).simplices` で候補三角形を確認できます。数値APIはその三角形順のboolean supportも受けますが、ファイルでは列挙順に依存しないvertex tripleを使います。

格子から元点への評価は隣接格子cell内の固定三角形を使い、無効domain頂点を含むcellを跨ぎません。実際に使う逆補間の三角形がsource禁止三角形と正の面積で交わる場合も採点しません。2頂点だけを使う線分は、許可されて全頂点が観測済みのsource三角形の閉unionで全区間が覆われることを要求します。許可境界の線分は保持しますが、欠測で2D supportを失った孤立edgeは保守的に除外します。1頂点だけの評価は既存の有効格子値を保持します。

粗い格子が細い壁・穴を飛び越え、全格子頂点が有効に見える場合にも適用します。支持範囲外を再Delaunayやnearestで迂回しません。採点できない元点はcoverageで示します。

## 出力と三つの誤差

変換後Datasetへの参照は `result.artifacts["input_dataset_manifest"]` です。原入力を変更せず、field.npy・observation_mask.npy・manifestと、元点を保つscattered/index.yamlとNPY群を自動生成します。別のDataset設定を手で二重に作りません。

元点storeはpickle/object配列を使わず、sample_ids、offsets、points、values、observed、weights、simplex_offsets、simplicesを独立NPYとして持ちます。offsetsはsampleごとの点範囲です。対象sampleだけをmmapで読み、fieldとmaskは逐次保存します。元の数値・座標を保持し、raggedな点数も同じ形式で扱います。旧compressed NPZは読込境界で継続対応します。

| 評価 | 比較と意味 |
| --- | --- |
| field | 変換後の教師格子と予測格子。固定geometryによる通常の場指標 |
| observation | 予測格子を元点へ写し、保存した元観測と比較 |
| interpolation | 教師格子を元点へ戻し、保存した元観測と比較。格子化・往復写像の誤差 |

analysis.scattered.enabledをtrueにすると元点評価を追加し、falseなら追加採点の準備を省きます。写像・観測maskに依存するsupport・往復誤差はreferenceごとに共有します。元点と補間baselineは同じ固定採点支持範囲を使い、予測NaNで採点範囲を狭めず、その候補の採点を失敗とします。成分別RMSE/MAE、coverage、known/scored points・samplesを報告します。採点点0はrmse=Noneで示します。人工欠測のimputationとは、隠したtruthと元観測を混同しないため併用しません。

sample内は点平均、weightsがあればその明示点重みです。全体はsampleの二乗誤差を等重みに平均してから平方根を取ります。点間隔から物理面積を推定しません。物理面積のfield評価とは分け、異なる単位の成分も黙って混ぜません。三つのRMSEは足し算の関係ではなく、補間往復RMSEを全モデルの理論的下限とは扱いません。

## 一つのYAMLを使う動作確認

まだ作っていない出力先で次を実行できます。

```powershell
python examples/create_scattered_dataset.py --verify
```

同じanalysis YAMLを一回run_taskへ渡し、scattered_verification.jsonへfield RMSE・元点RMSE・補間往復RMSE・coverageと自動変換先を保存します。既定toyは共有点群で、全sampleの格子supportを共通にしています。sample別の点配置でsupportが変わる実データには対応する欠測手法か明示共通supportが必要で、nearest埋めにより通常PODへ通しません。

toyの元点coverageは1未満で、元点誤差と補間往復誤差も非zeroになります。採点できない境界点があることを隠しません。補間値や欠測を完全な実測truthとみなして予測帯の校正を有効化しないでください。[予測帯ガイド](UNCERTAINTY.md)は固定support上のtruth条件を説明しています。

## Pythonで写像を再利用する

```python
from ml_platform_mode_decomp.data import ScatteredGridMap, point_scores

mapping = ScatteredGridMap.build(points_xy, target_x, target_y, domain_mask=domain_mask)
grid, grid_support = mapping.forward(observed_values, observation_mask=observed_mask)
values_at_points, support = mapping.to_points(predicted_grid, grid_mask=grid_support, observation_mask=observed_mask)
scores = point_scores(
    observed_values, values_at_points, support, observation_mask=observed_mask, point_weights=explicit_weights
)
```

欠測観測がある場合は、forwardとto_pointsへ同じobservation_maskを渡します。粗い格子のmaskだけでは格子点の間にある欠測支持を復元できません。通常のrunで元点評価を有効にした場合は、保存した元観測maskを自動的に使います。写像オブジェクトは前回のforward呼出しの状態を記憶しません。

## 配布後の受入

`scripts/verify_wheels.py`は同梱散布点例でraw取込と解析を実行し、`report.json`の`extensions.scattered`と`scattered_verification.json`へ元点誤差・補間往復誤差・coverageを保存します。現在の実行結果を参照し、同じsampleの空間点を独立sample数として扱わないでください。実観測への適用には観測配置・domain宣言・補間の妥当性を別途確認します。
