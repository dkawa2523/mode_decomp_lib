# 一つのrun YAMLからデータを読む

単体利用では入力仕様をrun.yamlのdataへ直接書きます。別manifest・schema宣言・samples.csvは常に必要なものではありません。起動は `python -m ml_platform_mode_decomp run.yaml` です。[設定ガイド](CONFIGURATION.md)に最小の実行例があります。

## NPYの最小入力とshape

```yaml
data:
  local_path: data/my_fields
  fields: {path: field.npy, layout: HWC}
```

この例はfield.npy一つの単一場です。data.formatは省略するとnpyです。samplesを省略するとsample数に応じてsample_000000...のIDとarray indexを生成します。分解だけならconditionsも不要です。geometryのない格子は配列index座標で、index coordinate measureを使います。PythonのFieldDataset.from_arraysと同じbuilderで、物理単位・物理幅は推定しません。

| layout | 入力shape | 内部shape |
| --- | --- | --- |
| NHWC | `(N,H,W,C)` 格子batch | 同じ |
| HWC | `(H,W,C)` 単一格子場 | `(1,H,W,C)` |
| NVC | `(N,V,C)` mesh batch | 同じ |
| VC | `(V,C)` 単一mesh場 | `(1,V,C)` |

layout省略はNHWCです。単一場HWC・mesh NVC/VCは明示し、`(N,H,W)` をbatchと単一多成分場のどちらかへ推測しません。scalarもC=1の軸を持ちます。NPYのfields.kind省略はC=1ならscalar、C>1ならmultichannelで、vectorとは推測しません。components省略はcomponent_0...を生成します。物理名やunitsは必要に応じて成分順に指定します。meshはgeometryを省略できず、固定vertices/facesが必要です。

## 条件とsampleの対応

条件をNPYで与える場合は同じdataへ追加します。sample表は不要です。

```yaml
conditions: {format: npy, path: conditions.npy, columns: [pressure, speed]}
```

conditions.npyは `(N,P)` の有限な実数で、fieldのsample軸と対応します。単一場でも `(1,P)` にします。カテゴリの自動符号化や欠損補完は行いません。

NPYのcolumnsを省略するとcondition_0...を生成します。条件tableのcolumnsは、metadataを勝手にfeatureへ混ぜないよう明示します。

条件をCSVにまとめる場合や独自ID・group/timeを使う場合はsamplesを指定します。

```yaml
samples: {path: samples.csv, id_column: sample_id, index_column: array_index}
conditions: {format: table, columns: [pressure, speed]}
```

```csv
sample_id,array_index,pressure,speed,experiment_id
run_b,1,110,4,trial_b
run_a,0,100,3,trial_a
```

array_indexがfield・NPY条件・sample別maskの先頭軸を参照します。表の行順を変えても対応を保ち、ID・indexの重複や参照漏れを拒否します。index_columnを指定せずarray_index列もない表では行順を使用します。条件を別tableへ置く場合はconditions.path/id_columnでID結合します。group/timeは明示した条件列に含めない限りfeatureへ混ぜません。

## geometryとmask

明示したgeometryは同じrunの全sampleで共有し、index座標へ置き換えません。axesはstart/step/count、values、または座標NPYへのpathを使います。円・円環はcenter/radius等、sphere_gridはlon/lat・angle_unit・radius、meshはvertices/facesを指定します。小さなmeshならvertices/facesをYAML内へ書け、大きなmeshは `mesh: {path: surface.npz}` を使えます。CSV・散布点のtarget geometryは明示します。

geometryの未知key、fieldとのshape不一致、axesとrangeの矛盾を拒否します。同じ意味の座標・maskをinlineとpath、または別名のkeyで二重指定しません。`FieldDataset.from_arrays()`のx/yとgeometryも択一です。geometry内のmaskと`masks.domain`は共通部分をdomainとします。

非等間隔入力を読めてもFFT等の各手法の条件は別に満たす必要があります。周期軸は重複端点を持たせません。物理面積は実際の座標から求め、点間隔から仮の面積を付けません。

```yaml
masks:
  domain: domain_mask.npy
  observation: observation_mask.npy
```

| mask | 意味 | shape |
| --- | --- | --- |
| domain | 固定された物理領域 | `(H,W)` / `(V,)` |
| observation | sampleで値が観測されている位置 | 共通空間shapeまたは`(N,H,W)` / `(N,V)` |

どちらも必要な場合だけ指定します。有効点はdomain AND observationで、そこだけの有限値を要求します。mask外のNaNや保存上の0を統計・損失へ入れません。共通maskをN枚保存する必要はありません。初期の観測maskは全成分で共有し、成分別欠測は別定義を必要とします。評価maskは観測maskと同じとは限らず、人工欠測では隠した既知正解を採点します。

Pythonの`FieldRepresentation.transform(dataset, masks=...)`へ追加maskを渡す場合も、Datasetの観測maskと固定domainとの共通部分だけを使います。追加maskで未知だった位置を観測済みへ広げません。

## 格子CSVも同じrunから読む

data.format: csvを選ぶと実行開始時に一度正規化し、そのまま分解・学習・解析へ進みます。手動importや別の入力YAMLは不要です。

```yaml
data:
  local_path: data/csv_fields
  format: csv
  samples: {path: samples.csv}
  conditions: {format: table, columns: [pressure]}
  fields:
    path_column: field_csv
    layout: NHWC
    kind: vector
    components: [vx, vy]
    columns: [velocity_x, velocity_y]
    coordinates: {x: position_x, y: position_y}
  geometry:
    kind: rectangle
    axes:
      x: {start: 0, step: 0.5, count: 3}
      y: {values: [-1, 1]}
```

sample表はsample_id,field_csv,pressure等を持ち、1 sample 1 field CSVへ対応付けます。単一CSVはfields.pathで指定し、samplesとarray_indexを省略できます。値の列順はcomponentsとcolumnsの同じ位置で指定します。CSVの行・列順は問いません。全宣言座標を1行ずつ含め、重複・欠落・格子外座標はエラーです。既に格子のCSVを補間しません。

丸めだけを許す場合はfields.coordinate_atolを座標単位の絶対許容差として指定し、最小間隔の半分未満にします。欠測点でも座標行は残し、値nanと明示observation maskを使います。mesh CSVはlayout: NVCと `coordinates: {vertex_index: node}` を選び、0始まりの全頂点を1行ずつ含めます。球面格子CSVではcoordinates.lon/latを指定し、角度単位をgeometryへ揃えます。

## 散布点と旧データ

散布点はdata.format: scatteredを選び、observationsと固定target geometryを同じrunへ書きます。既定はlinear補間・凸包外maskで、method/outsideを毎回書く必要はありません。元の座標と値も保存し、元点誤差と補間往復誤差を別に測れます。[散布点ガイド](SCATTERED.md)を参照してください。

付属の旧NPYは `data: {format: legacy-npy, local_path: data/mode_decomp_eval_dataset_v1/scalar_annulus}` で自動変換できます。field.npy/cond.npyは変えず、座標・maskの意味を付けます。旧manifest.jsonを読み、任意の別manifest名は使いません。旧geometryが不十分ならdata.geometryへ明示します。位置名x1等を物理条件名と推測しません。

CSV/scattered/legacy-npyの変換先は実行側が生成し、`result.artifacts["input_dataset_manifest"]` に保存します。元入力は変更しません。変換だけを別に再利用したい場合には、任意でmode_decomp_importやscripts/import_dataset.pyも使えます。

## 再利用用の外部manifestは任意

```yaml
data:
  local_path: data/reusable_dataset
  source_manifest: field_dataset.yaml
```

外部manifestはdataのfields/geometry/samples/conditions等と同じ内容に、対応するschemaを付けたものです。NPYはfield_dataset_v1、格子CSVはfield_csv_import_v1、散布点はscattered_field_import_v1です。CSV/scatteredではdata.formatも指定します。inline指定との併用は、片方を黙って優先しないよう拒否します。

local_pathは起動cwd基準、その他の参照は入力root基準です。サブフォルダ内のmanifestでも、その所在フォルダへpath基準を変えません。meshとdomain maskの参照も同じ規則です。Datasetのrevisionには実ファイルとinline仕様を含め、任意のDataset IDを付けても内容確認を省略しません。

PythonのDatasetはfield_kind/field_unitsを正規属性として持ち、manifestは来歴です。geometryの同一性は座標・topology・固定mask・measure・数値単位から判定し、pathや表示用metadataは含めません。from_arraysは原則配列を借用し、処理中に入力を書き換えません。呼出し中の外部変更は行わず、任意index抽出やdtype変換でcopyが必要になる場合があります。学習済みgeometryはモデルが独立所有します。

## 変換後入力と外部Stage

raw入力はrun開始後に同じrunの`input/`へ一度だけ変換します。変換失敗もそのrunの部分結果として取得できます。既にcanonicalなNPY入力はcopyせず読みます。

外部Stageのprepareは`canonical_dataset`と`split`を出力します。canonical Datasetはfield・mask・座標・topology・元観測storeの必要参照を含む自己完結したdirectoryで、後続Stageはraw入力を再変換しません。directory artifactのhashがnullであることを内容検証とみなさず、受入時にDataset revisionとsplitの意味を照合します。通常のlocal利用でStage用ファイルを用意する必要はありません。

prepareは一度読み込んだ参照pathとrevisionからsnapshotを作り、コピーのためだけにDatasetを再読込しません。生成snapshotの`stage_input.json`と入力payloadの`dataset/`を分け、元入力のファイル名と衝突しない配置にします。旧root配置のsnapshotも読み込めます。利用者が配置や記録fileを手作業で用意する必要はありません。独立したDataset読込では内容hashを計算し、size・mtimeだけのcacheで入力更新を見落とさないようにしています。

## 条件だけの推論

```yaml
task: mode_decomp_infer
data: {local_path: data/new_conditions.csv}
model: {artifact_path: outputs/trained/model.joblib}
```

モデルが条件列と順序を持ち、CSVをその順へ整列します。field、訓練Dataset、geometryの再指定は不要です。条件NPYはモデルと同じ列順の `(N,P)` とし、必要ならdata.condition_columnsで一致を確認します。独自ID付きNPYや複数ファイルの推論は任意のcondition_batch_v1外部manifestで指定できます。

Pythonは `FieldDataset.from_arrays(fields, conditions, ...)` でファイルを使わず同じ内部表現を作れます。x/yを省略した座標はindexであり、物理単位は推測しません。field_kindの省略時もNPYと同じscalar / multichannelで、物理ベクトルは `field_kind="vector"` と明示します。[設定ガイド](CONFIGURATION.md)の短いAPI例を参照してください。

## 予測の保存と部分読込

保存形式は既存の `output.prediction_name` の拡張子で選びます。最終学習のtest予測と、保存モデルからの条件推論で共通です。

```yaml
output: {prediction_name: predictions.npy}
```

| 出力名 | 保存内容・読込 |
| --- | --- |
| `predictions.npz`（既定） | 従来と同じ予測・ID・成分・単位・任意の帯を持つ圧縮archive。`np.load(path)`で読込 |
| `predictions.npy` | field配列。`np.load(path, mmap_mode="r")`で必要なsampleだけ読込 |

NPYを選ぶと `predictions_sample_ids.npy`、`predictions_metadata.json` と、有効な場合の `predictions_lower.npy` / `predictions_upper.npy` を自動保存します。metadataには相対array path、field shape/kind・成分・単位・較正情報があり、利用者による別manifest作成や入力設定は不要です。artifact keyは `predictions`、`prediction_sample_ids`、`prediction_metadata`、任意の `prediction_lower` / `prediction_upper` です。

```python
import numpy as np

fields = np.load(result.artifacts["predictions"], mmap_mode="r", allow_pickle=False)
sample_ids = np.load(result.artifacts["prediction_sample_ids"], allow_pickle=False)
one_field = fields[10]
```

条件推論は係数・field・帯を小さいbatchで計算して順に書込みます。計算途中に失敗した予測を成果物として登録しません。NPZは互換性を保つため一時NPYを圧縮するので、未圧縮予測・帯と圧縮archiveの両方のディスク領域を一時的に使います。NPYは一時ファイルを最終名へ移すので二重のデータ保存を伴いません。圧縮が不要な大きい出力にはNPYが適します。

Pythonの `predict()` / `predict_interval()` / `predict_coefficients()` は配列を返すため最終配列を全件保持します。部分的に利用する場合は `model.iter_predict(conditions)`、較正済みの帯も使う場合は `model.iter_predict(conditions, intervals=True)` を順に消費します。任意の `batch_size` はこの配列APIでだけ指定でき、実行YAMLに追加指定は不要です。

新しいfield出力は共通の`field_output_v1`です。既存の予測・再構成キーに、実際に宣言されたgeometry座標・topology・重み・単位と、domain/入力観測/calibrationのsupportを付随します。NPZは内部にmetadataを持ち、NPYは相対参照のsidecarを自動保存します。モデルや訓練入力がなくても解釈できます。

書込前にsample IDの一意性・非空、field shapeとgeometryの一致、成分名・単位とchannel数の一致を検査します。読込時も保存された宣言とarrayを照合します。観測されない位置のNaNは保持し、全domainが観測済みであるとは解釈しません。

```python
from ml_platform_mode_decomp import read_field_output

saved = read_field_output(result.artifacts["predictions"])
metadata = saved["metadata"]
fields = saved["arrays"]["predictions"]
```

NPYのreaderは配列をmmapで読みます。NPY出力は関連sidecarも一緒に移動してください。旧出力も配列として読み、保存されていなかったgeometryはunknownとして扱います。架空の物理座標や単位を補いません。

readerはfield用metadataか、宣言と配列のshape・sample ID軸・成分・帯・supportが対応するかを確認します。別用途のJSONや壊れたNPYを物理fieldとして成功扱いしません。field値の全面走査や別のhash契約は追加せず、NPYの部分読込を維持します。旧NPY metadataのgeometry未記録はNoneに統一します。

source場／保存source係数を入力にする場合も、targetの通常Dataset定義と出力形式を維持します。`data.input`のroot、source/target IDの完全一致、係数NPZの`space`とモデルhash、保存bundleだけでの係数推論は[対応学習ガイド](FIELD_MAPPING.md)を参照してください。
