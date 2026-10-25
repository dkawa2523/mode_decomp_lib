# 一つのYAMLで実行する

mode 0.11 / core 0.2.1 の単体利用は **run.yaml一つ** を標準にします。入力の形・座標・手法・解析目的を同じファイルへ書き、次のように実行します。profile、Dataset manifest、シナリオ定義、分割ファイルを別途作る必要はありません。

```powershell
python -m ml_platform_mode_decomp run.yaml
```

## 単一の場を分解する最小例

手元にfield.npyが1枚の `(H,W,C)` 配列としてある場合です。同じ作業フォルダのrun.yamlを次のようにします。

```yaml
task: mode_decomp_representation
data:
  local_path: .
  fields: {path: field.npy, layout: HWC}
representation:
  decomposition: {name: dct2}
```

必要な入力ファイルはfield.npyとrun.yamlだけです。sample IDとcomponent_0等の成分名を生成し、samples.csvや条件は要求しません。geometryを省略した座標は配列indexで、重みもindex coordinate measureです。物理単位・物理的な幅を推定しません。物理的な積分・微分には実際のgeometryを指定します。POD等のデータから学ぶ手法ではsample数・実効rankの条件も満たしてください。

## 目的に合わせてTaskを選ぶ

| task / analysis.kind | 目的 |
| --- | --- |
| mode_decomp_representation | 入力場を分解し、再利用できる表現と係数を保存 |
| mode_decomp_analysis / representation | 訓練場で学んだ表現の、未使用場に対する表現能力を比較 |
| mode_decomp_analysis / surrogate | 条件から場を予測する候補を、共通の未使用sampleで比較 |
| mode_decomp_analysis / imputation | 一部を隠した既知正解場を使い、欠測復元能力を比較 |
| mode_decomp_pipeline | 固定候補を学習、または訓練内で候補選択し、最終test評価・モデル保存 |
| mode_decomp_infer | 保存モデルへ新しい条件・source場・source係数を渡して推論 |

analysisは比較用、pipelineは選択・配備用です。別々のシナリオをすべて実行する必要はありません。通常の分解だけなら最初の例で足ります。基盤向けのmode_decomp_stageや、変換だけを別に保存するmode_decomp_importもありますが、単体利用の準備として必須ではありません。

targetは既存のdata定義を使い、別の場を入力にする場合だけ`data.input: {kind: field, local_path: ...}`とcandidateの`input_representation`を加えます。保存係数は`kind: coefficients`で対応するsourceモデルを指定します。条件入力は従来どおりです。[対応学習ガイド](FIELD_MAPPING.md)を正本として、二つの固定geometry、ID対応、path、評価scopeを説明しています。

## 入力形式も同じdataへ書く

`data.format` はnpy（既定）/csv/scattered/legacy-npyです。NPYは直接読み、他の形式は実行開始時に一度だけ正規NPYへ変換してから、同じ分解・学習・評価へ進みます。手動のimport実行とanalysis実行に分けません。

```yaml
data:
  local_path: data/my_fields
  format: npy
  fields: {path: field.npy, layout: NHWC, kind: scalar, components: [temperature]}
  conditions: {format: npy, path: conditions.npy, columns: [pressure, speed]}
  geometry:
    kind: rectangle
    axes:
      x: {start: 0, step: 0.1, count: 12}
      y: {start: 0, step: 0.1, count: 10}
```

schemaやdata.manifestという追加の入れ子は不要です。samplesを省略すれば配列順にIDを生成します。条件をCSVで与える、独自IDやgroup/timeを保持する場合だけsamplesを指定します。対応するshape・mask・CSVは[データ形式](DATA_FORMAT.md)へまとめています。

散布点の実行例は既存の[一つのanalysis YAML](../config/tasks/mode_decomp_scattered_analysis.yaml)です。

```powershell
python examples/create_scattered_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_scattered_analysis.yaml
```

元点の三つの入力ファイルから自動取込し、元点誤差・補間往復誤差・coverageまで一回で計算します。別のimport YAMLや変換先pathは指定しません。変換後の参照先は `result.artifacts["input_dataset_manifest"]` に残ります。詳細は[散布点ガイド](SCATTERED.md)を参照してください。

## 手法とパラメータ

条件から場を学習・比較する候補は次の形です。

```yaml
model:
  candidates:
    - id: pod_ridge
      representation:
        decomposition: {name: pod_svd, params: {n_modes: 6, inner_product: domain_weights}}
        coeff_post: {name: none}
      regression: {name: ridge, params: {alpha: 0.01, cond_scaler: standardize}}
```

idは候補名、nameは手法、paramsはその手法の数値設定です。CLI・task・Pythonは同じresolverを使い、未知の手法名・parameter・representation keyは入口で拒否します。誤記を捨てて既定手法へ切り替えません。parameterの既定値は各実装のconstructorが正本です。Pythonのconstructorでは明示した`cfg`内の設定がkeyword引数に優先しますが、`seed`と`random_state`の矛盾はエラーになります。別の手法は別candidateとして書きます。不要な係数後処理はname: noneを選びます。場前処理と条件前処理、分解と回帰を同じ設定名へ混ぜません。対応手法と適用条件は[手法一覧](METHODS.md)を参照してください。

既定値を使う場合は設定を省略します。`representation: null / false / []`や`decomposition: null`を省略と同じ意味には扱いません。係数後処理・場前処理の無処理は、そのcomponentに`name: none`を指定します。boolean parameterはYAMLの`true`／`false`を使い、文字列`"false"`、数値、nullをbooleanへ変換しません。run／profile YAMLはmappingが必要です。

設定はそのtaskが実際に利用するものだけを指定します。例えば直接分解・比較解析で`output.prediction_name`を指定しても予測の保存先にはならないため、エラーにします。直接分解の再構成は`reconstructed_fields.npz`、比較解析はCSV／JSON、pipelineとinferの予測名は`output.prediction_name`です。pipeline／inferへmode・component解析設定を置く場合も、無視して成功にはしません。

`analysis.kind: representation / imputation`の候補には回帰器や`input_representation`を置きません。観測mask・乱数による欠測設定はimputationだけへ指定します。同じYAMLを別用途へ変更する場合は、その用途で使わない設定も取り除きます。

## 選択とON/OFFの区別

Task、data.format、analysis.kind、split.method、各component.nameは処理の選択です。独立した追加処理だけenabledで切り替えます。

| 設定 | true | false・省略時 |
| --- | --- | --- |
| tuning.enabled | 候補数によらずpipeline内の共通CV/HPO | 複数候補はinner holdout選択。固定1候補は選択なし |
| uncertainty.enabled | 独立に校正した場の予測帯 | 点予測のみ |
| analysis.scattered.enabled | 保存した元点の誤差・補間誤差・coverage | 変換後の場だけで評価 |
| analysis.modes.enabled | 指定A座標の応答・累積再構成 | absent/falseなら生成しない。indicesも不要 |
| analysis.components.enabled | 指定したHodge/Helmholtzの物理成分 | 既定false |
| report.enabled | analysis全体／直接分解のHTML生成。analysisの既定はtrue、直接分解はfalse | 全体HTML生成を省き、主CSV/JSONを残す |

各enabledは出力を隠すだけでなく、その追加計算と準備も省きます。散布点が無効なら元点評価の読込・写像・往復誤差、モードが無効なら指定座標の応答・曲線、成分が無効なら追加分解・NPZ・energy表、不確実性が無効なら較正分割・較正fit・帯生成を行いません。raw散布点をgridへ変換する入力処理と、主評価に必要な演算は残ります。

`analysis.save_models` は解析モデルの明示保存です。既定falseではサイズ測定のためにもserializeせず、comparisonのmodel_bytesは未測定です。trueなら保存payloadの実byte数を記録します。

report.enabledはanalysisと直接分解の全体HTMLに使います。モード個別HTMLはmodes.enabledが独立して制御します。scattered追加評価は元観測を保持した入力だけが対象で、人工欠測のimputationとの併用は拒否します。Hodgeの物理成分は[Hodgeガイド](HODGE.md)、モードの意味は[モードガイド](MODES.md)を参照してください。

予測帯は学習・surrogate解析で例えば次のように指定します。

```yaml
uncertainty:
  enabled: true
  method: split_conformal
  coverage: 0.9
  scope: simultaneous
  normalization: channel
  calibration_size: 0.2
  calibration_seed: 49
```

calibration_methodは通常outerのrandom/group/timeを継承します。推論は `uncertainty: {enabled: true}` だけで保存モデルの設定を使います。[学習例](../config/tasks/mode_decomp_uncertainty.yaml)、[比較例](../config/tasks/mode_decomp_uncertainty_analysis.yaml)、[推論例](../config/tasks/mode_decomp_uncertainty_infer.yaml)、[予測帯の条件](UNCERTAINTY.md)を参照してください。CVの予算・分割は[探索ガイド](SEARCH.md)にあります。

## 評価対象と実際の観測mask

`metrics.reference`で評価対象を一箇所だけ選びます。pipelineの候補選択・CV・最終testと、analysisの比較で同じ対象を使います。

| reference | 指標 | 用途 |
| --- | --- | --- |
| grid（通常の既定） | field_rmse / field_mae / field_r2 / field_skill | 格子・mesh上の原field |
| observations | observation_rmse / observation_mae / observation_skill | 保存された元観測点上の正解 |
| hidden（imputationの既定） | hidden_rmse / hidden_mae / hidden_r2 / hidden_skill | 入力から隠した、もともと既知の正解 |

```yaml
metrics:
  reference: observations
  names: [observation_rmse, observation_mae]
model:
  selection_metric: observation_rmse
```

referenceだけを明示してnamesやselection_metricを省略すれば、対象に合うRMSE/MAEとRMSE選択を補います。明示した対象と選択指標の矛盾は拒否します。元点には元観測storeが必要です。元点のR²は実装していません。`analysis.scattered.enabled`はgrid主評価への補助出力で、元点を主評価にする場合はreference自体が必要な処理を有効にします。

実sensor配置や連続欠損は、random設定の代わりに一つのNPYを指定できます。

```yaml
analysis:
  kind: imputation
  observation_mask: sensors.npy
metrics:
  reference: hidden
  names: [hidden_rmse, hidden_mae]
```

maskはDataset root基準で、trueを入力観測として使います。shapeは格子のH×WまたはN×H×W、meshのVまたはN×V。sample別maskは元field配列のarray_index順で、samples表と同じ規則で整列します。明示したobserved_fraction / observation_pattern / mask_seedとの併用は拒否します。maskを省略すれば既存のrandom設定を使います。imputationの主指標は`hidden_*`です。旧`field_*`指定は0.11からエラーとし、隠れた正解をgrid指標名で採点しません。

明示maskには乱数設定の既定値を注入しません。解決済み設定や保存したconfigも同じ入口から再利用できます。data.local_pathがcanonical manifestファイルの場合も、準備段階でその親をDataset rootに揃え、mask等の参照を同じ基準で解決します。

物理成分は学習表現と独立に選べます。PODやDCTの出力fieldにも同じ解析を適用します。

```yaml
analysis:
  components:
    enabled: true
    method: hodge
    params: {boundary: natural, tolerance: 1.0e-11}
```

methodの既定はhodge。周期場には明示的にhelmholtzを指定し、paramsにはboundary_condition: periodicとmask_policy: errorまたはzero_fillを使えます。geometryから手法を自動変更しません。物理単位へ戻したCartesian vectorが対象で、imputationへの自動適用はしません。

`runtime.use_clearml`と`output.upload_plots`は無作用だったため廃止しました。明示された旧設定は削除方法が分かるエラーにします。外部公開の設定は基盤adapter側が所有します。disabled sectionの未知keyも検査するので、enabledの誤記で別の動作へ進みません。

## 分割・seed・path

分割はsample単位です。`split.valid_size` はouter final testの割合、`selection_size` は較正を除いたproper train内の候補選択の割合です。較正の `calibration_size` はouter testを除いたouter train内の割合です。analysisはouter train/testと任意calibrationを使います。groupの割合はgroup数基準なので、実際のsample比率とは異なる場合があります。

固定1候補・tuning無効のpipelineは、selection分割を作らずproper trainへ1回fitします。最終summaryは `selection_scope: fixed_configuration`、`selection_score: null` となり、候補順位表はありません。複数候補はinner holdout、tuning有効なら候補数に関わらず共通CVを使います。

outerがgroup/timeならinner selectionもその分離とgapを維持します。outerがfixedで複数候補のholdout選択をする場合だけ `selection_method` を明示します。固定1候補では不要です。group/time/fixedを選ぶ場合は対応metadata列を指定します。splitやfoldファイルは実行側が生成するので、利用者が別ファイルとして用意する必要はありません。

run.seedはseedを受け付ける手法の既定値、split.seedは分割、analysis.split_seeds/model_seedsは分割とモデル乱数の反復です。analysis.split_seedsを明示する場合はsplit.seedを省略し、studyの分割seedを一箇所で指定します。componentに明示したseed/random_stateは優先されます。`pod_svd` など乱数parameterを持たない手法にseedは渡しません。物理的意味を持つ列を分割の都合で自動的に条件featureへ追加しません。

run.yaml内のdata.local_pathとruntime.output_dirは起動cwd基準です。fields.path、samples.path、geometry参照等はdata.local_pathの入力root基準です。YAMLの置き場所からrepoを探索しません。

保存モデルを`load_model(..., expected_context={...})`で照合する場合、None／空文字列以外の全keyを来歴と照合し、不一致・未記録はエラーにします。falseや0も検査対象です。`model_sha256`はartifactのhash、`execution_release_id`は保存された`source_release_id`に対応します。contextを省略した通常の読込へ新しい来歴指定は要求しません。

## Pythonならファイルなしでも使える

```python
from ml_platform_mode_decomp import FieldRepresentation
from ml_platform_mode_decomp.data import FieldDataset

# fieldsは(N,H,W,C)。単一(H,W,C)ならlayout="HWC"を明示。
data = FieldDataset.from_arrays(fields, x=physical_x, y=physical_y)
model = FieldRepresentation({"decomposition": {"name": "dct2"}})
coefficients = model.fit_transform(data)
reconstructed = model.inverse_transform(coefficients)
```

IDと成分名を省略すれば位置名を生成します。x/yを省略した格子はindex座標として扱い、物理単位や物理面積を推測しません。field_kindの既定はC=1ならscalar、それ以外はmultichannelです。Hodgeへ物理ベクトルを渡す場合は `field_kind="vector"` を指定し、成分名・単位をcomponents / field_unitsで与えます。条件付き学習は `FieldDataset.from_arrays(fields, conditions, ...)` をFieldSurrogateへ渡します。直接fitでは渡したデータを全部学ぶので、未使用データの評価は呼出し側で分離するかanalysis/pipelineを使ってください。

設定をPythonで読む場合も一つで足ります。

```python
from ml_platform_mode_decomp.config import load_config
from ml_platform_mode_decomp import run_task

result = run_task(load_config("run.yaml"))
```

## 再利用するときだけ設定を分ける

複数の実験で同じDataset定義を共有する場合は `data: {local_path: ..., source_manifest: field_dataset.yaml}` を使えます。manifestには対応するschemaを持たせます。inline fields/geometry等との併用は矛盾を隠さないため拒否します。

環境を共有する場合だけ `--profile config/profiles/local.yaml` を追加します。profile → run YAML → --setの順で後を優先します。既存の `--task PATH` も同じrun YAMLを選ぶ代替入口として残っています。実際の解決済み設定はrun内のconfig.yamlで確認できます。

YAML内の`profile:`キーでは別ファイルを読み込みません。無作用だったこのキーは受け付けず、profileの選択は`--profile`または`load_config(..., profile_path=...)`で明示します。
