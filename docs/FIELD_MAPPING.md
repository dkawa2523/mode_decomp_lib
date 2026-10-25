# 定常場から別の定常場を予測する

既存の`FieldSurrogate`と`mode_decomp_pipeline / mode_decomp_analysis / mode_decomp_infer / mode_decomp_stage`で、条件だけでなく、別の場または保存した係数を入力にできます。追加するのは`data.input`と、field入力時の`input_representation`です。基盤、分解器、回帰器の入口は共通です。

```text
条件行列 ───────────────────────────────┐
source場 → source表現 → source Z係数 ───┼→ 回帰器 → target Z係数 → target表現の逆変換 → target場
保存済みsource Z係数 ──────────────────┘
```

ここでの場は、各sampleに対応する定常な空間分布です。sourceとtargetは異なる物理量、成分数、単位、固定geometry、解像度を持てます。空間点の対応や、両側の係数数の一致は要求しません。同一caseのsample対応が必要です。入力側の分布が変わった場合の適用精度を自動保証する機能ではありません。

既存のpacked係数空間`A`と混同しないよう、入力をsource、予測対象をtargetと呼びます。学習へ渡す係数は双方とも`Z`です。

## field入力の一つのYAML

既存の`data`はtargetの定義です。`data.input`へsourceを追加し、candidateの`input_representation`でsource表現を選びます。`representation`は従来どおりtarget表現です。

```yaml
task: mode_decomp_pipeline
data:
  local_path: data/paired/target
  fields: {path: field.npy, layout: NHWC}
  input:
    kind: field
    local_path: data/paired/source
    fields: {path: field.npy, layout: NHWC}
model:
  candidates:
    - id: field_mapping
      input_representation:
        decomposition: {name: tucker, params: {ranks: [3, 3, 1]}}
      representation:
        decomposition: {name: quadratic_manifold, params: {n_modes: 2}}
      regression: {name: ridge, params: {alpha: 0.000001}}
```

sourceにも通常のfields、geometry、samples、source_manifest定義を使います。CSV／散布点等は、targetと同じ入力準備でrun内の正規NPYへ変換します。sourceの変換先は`source/input/`で、元入力を変更しません。`load_surrogate_inputs()`を単独で呼ぶ場合は、変換済みのNPY入力を渡します。manifestは必須ではありません。上の例で省略した座標・単位は通常のindex座標規約です。物理的な値を扱う場合は各側へgeometry・成分・単位を指定します。

`data.input.local_path`は通常の`data.local_path`と同じく起動cwd基準です。省略した場合はtargetのrootを使います。入力内のfields等の参照はそのsource root基準です。

同梱例は温度場`6×8×1`から別geometryの応力場`7×5×1`を学びます。

```powershell
python examples/create_field_mapping_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_field_mapping.yaml
```

表示された`artifacts.model`を使って、新しいsource場だけから推論します。

```powershell
python -m ml_platform_mode_decomp config/tasks/mode_decomp_field_mapping_infer.yaml --set "model.artifact_path=outputs/学習run/model.joblib"
```

保存bundleは両側の表現を含むので、targetの訓練入力や元のrunは不要です。出力はtargetのgeometry・成分・単位を持つ、通常の予測NPZまたはNPYです。

## 配列API

```python
from ml_platform_mode_decomp import FieldDataset, FieldSurrogate

# source_fieldsとtarget_fieldsは同じcaseの順序。空間shapeは異なってよい。
source = FieldDataset.from_arrays(source_fields, sample_ids=case_ids)
target = FieldDataset.from_arrays(target_fields, sample_ids=case_ids)
model = FieldSurrogate(
    {
        "input_representation": {"decomposition": {"name": "tucker", "params": {"ranks": [3, 3, 1]}}},
        "representation": {"decomposition": {"name": "quadratic_manifold", "params": {"n_modes": 2}}},
        "regression": {"name": "ridge", "params": {"alpha": 0.000001}},
    }
).fit(target, inputs=source)

predicted_fields = model.predict(new_source)
predicted_target_z = model.predict_coefficients(new_source)
for fields in model.iter_predict(new_source):
    consume(fields)
```

`new_source`も学習時と同じsource geometry・成分・単位を持つ`FieldDataset`です。推論は渡されたsourceのsample順を保ちます。直接`fit()`は渡した全ペアで学習するため、未知caseの評価にはpipelineまたはanalysisを使います。

`data.input`と`input_representation`を使わなければ、従来どおり`model.fit(dataset_with_conditions)`、`model.predict(new_conditions)`です。

## sampleの対応と分割

sourceとtargetのsample ID集合は完全一致させます。順序が違う場合はIDでtargetの順へ整列し、欠落・重複・余剰はエラーにします。暗黙のinner joinでsampleを落としません。samplesを省略すると位置IDを生成するため、両配列の同じ行が同じcaseを表す場合にだけ省略します。

split・group・timeのmetadataはtarget側を正本として、同じindicesでsourceも分けます。field入力ではsource表現、target表現、係数後処理、回帰器を各training partition内で新しくfitします。source表現だけを全sampleで先にfitしてから通常のCVへ渡す処理は行いません。解析summaryの`input_representation_fit_scope: training_partition`がこの経路です。

外部Stageでも同じ`data.input`を使います。`prepare_fields`はtargetとsourceの必要参照を`canonical_dataset`へ保存します。保存係数入力ではNPZと対応する表現モデルも含めます。後続の`train_candidate`／`finalize_surrogate`はこのsnapshotとsplitを受け取り、入力revisionと候補結果を照合するため、元の入力rootを移動した後も再開できます。Stageの種類や設定sectionを追加する必要はありません。

## 保存係数を入力にする

sourceの直接分解が出力した`coefficients.npz`と、その係数に対応する`model.joblib`を指定できます。source側の`input_representation`は指定しません。保存された表現を固定して使います。

```yaml
data:
  local_path: data/paired/target
  source_manifest: field_dataset.yaml
  input:
    kind: coefficients
    local_path: outputs/source_representation_run
    path: coefficients.npz
    model_artifact: model.joblib
model:
  candidates:
    - id: coefficient_mapping
      representation:
        decomposition: {name: pod_svd, params: {n_modes: 8}}
      regression: {name: ridge, params: {alpha: 0.01}}
```

`path`、`model_artifact`、任意の`info_path`はすべて`data.input.local_path`基準です。モデル／infoへの絶対pathも明示できます。NPZはsource root内を参照します。NPZの`coefficients`、`sample_ids`、`space: z`、`model_sha256`を検査し、別モデル・別空間・異なる次元の係数を接続しません。sourceモデルがsurrogateの場合も、その出力側の表現を使います。

保存係数の表現はCVごとに再学習しません。summaryは`input_representation_fit_scope: fixed_input_representation`となり、固定表現を条件とした評価です。source表現が評価sampleも使って作られた場合、その事実を含めて結果を解釈します。較正caseで表現をfit済みの場合も、独立較正の前提を確認します。表現の学習まで含む評価にはfield入力を使います。

学習後のbundleはsource表現とsourceモデルhashを内包します。新しい係数NPZで推論する場合は、保存bundleを指定し、次の入力だけを渡します。元のsourceモデルファイルの再指定は不要です。

```yaml
task: mode_decomp_infer
data:
  input:
    kind: coefficients
    local_path: data/new_coefficients
    path: coefficients.npz
model:
  artifact_path: outputs/学習run/model.joblib
```

Pythonでは`ml_platform_mode_decomp.inputs.load_surrogate_inputs()`で同じ設定から`CoefficientBatch`を読み、`predict()`または`predict_coefficients()`へ渡せます。手動でhashや表現を組み立てる必要はありません。学習時のtargetは通常のfield Datasetです。任意の裸のtarget係数行列だけを学ぶ入口は追加しません。

## 予測係数を次の対応学習へ渡す

pipeline／inferに次の指定を加えると、fieldと同じcase順のtarget Zを保存します。pipelineは最終test分、inferは渡した全query分です。

```yaml
output:
  coefficients_name: target_z.npz
```

`result.artifacts["coefficients"]`のNPZは`coefficients`、`sample_ids`、`space: z`、`model_sha256`、`coefficient_source: surrogate_prediction`を持ちます。ここでhashは、このtarget表現を内包する上流surrogateの保存モデルです。次段の`data.input`にそのNPZと上流の`model.joblib`を指定すれば、既存の係数入力として読み込めます。直接分解の係数は同じ形式で`coefficient_source: field_transform`となり、`packed_a`も保持します。

予測Zは正解fieldを符号化したZとは異なります。連鎖モデルを学習する場合、上流がそのcaseの正解を学習済みかどうかも評価設計に含めます。元fieldが手元にあることを理由に、予測Zを正解Zへ置き換えて評価しません。現在の追加出力は任意で、省略時には計算しません。有効時は係数予測を再実行し、`N×K_Z`行列を保持します。field出力は従来どおりbatchで書き込みます。

## 評価と追加解析

主指標はtargetの原field・support・measureで計算します。`predict_coefficients()`はtargetの`Z`を返すため、条件→field、field→field、係数→係数の予測が同じ回帰器・target表現を通ります。係数値の大小を異なるfitや手法間の共通精度として比較しません。

学習対象はtarget Zです。二乗誤差でZの条件付き平均を学習しても、Quadratic manifoldや非線形係数後処理を通したfieldは一般にfieldの条件付き平均ではありません。例えば同じ入力から正負のmodeが生じる場合、平均係数0を復号したfieldと、各fieldの平均は異なります。入力からtargetがほぼ定まる用途を基本とし、曖昧さが大きい用途では原field誤差とtraining平均baselineへのskillを確認します。多峰分布の推定やfield損失での直接学習は現在の処理に含みません。

較正を使う場合はsourceとtargetの較正ペアをfit・候補選択から除外し、`model.calibrate(target_calibration, inputs=source_calibration, training=target_training)`へ渡します。得られるのはtarget場の予測帯です。通常と同じ交換可能性等の条件が必要です。

既存のtarget mode解析はそのまま利用できます。source第1modeとtarget第1modeが対応するとは限りません。基底の符号・回転・係数尺度が違うため、回帰行列だけから物理的寄与や因果関係を断定しません。非線形表現の応答は基準点に依存します。[modeの意味](MODES.md)を参照してください。

## 現在の範囲

単一sourceと単一target、各側で固定されたgeometry、対応済みsampleを扱います。source表現のqueryが対応していれば、部分観測sourceを符号化できます。欠測学習をサポートしない表現には完全なtraining fieldが必要です。

複数sourceの結合、fieldと補助conditionsの同時入力、sampleごとの可変geometry、未ペア学習、target部分観測との同時推定は追加していません。`data.input`とtarget内のconditionsを同時に入力featureとして結合しません。
