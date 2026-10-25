# Mode Decomposition 0.11

場の分解・表現、条件や別の場からの予測、欠測復元を扱う単体パッケージです。**実行YAML一つ、またはPythonの配列から利用できます。** profile、別の入力manifest、シナリオファイル、手法別設定ファイルは必須ではありません。

`ml-platform-core>=0.2.1,<0.3`へ依存し、ClearML／実行SDKなしで動きます。基盤への統合は単体パッケージの受入後に扱います。

配布先では[配布用の導入手順](docs/DISTRIBUTION.md)を使います。`dist/0.11.2/ml_platform_mode_decomp-0.11.2.zip`に実行用wheel、設定例、入力生成例、利用guideをまとめており、利用先ではGit・uv・基盤checkoutを要求しません。

## 最初の実行

Python 3.11以上とuvで初期構築します。固定core wheelをvendor/wheelsへ同梱しているため、基盤checkoutやcoreの再buildは不要です。

```powershell
uv sync --locked
uv run python examples/create_single_field.py
uv run python -m ml_platform_mode_decomp config/tasks/mode_decomp_representation.yaml
```

入力は `data/single_field/field.npy` 一つです。同じ実行YAMLで配列の配置と手法を選び、座標は既定のindex座標を使います。出力先の `representation.html`、再構成NPZ、係数NPZを確認できます。全入力を使う直接分解なので、train/test分割や条件列は不要です。

配布一式は `uv run --no-project python scripts/build_distribution.py` で生成します。modeのsource archiveからwheelをbuildし、固定core wheel、自己完結した例と利用guide、SHA256SUMSをまとめます。wheel/source archiveだけなら `scripts/build_wheels.py` も使えます。配布先では生成済み2wheelをインストールします。

```powershell
python -m pip install dist/0.11.2/ml_platform_core-0.2.1-py3-none-any.whl dist/0.11.2/ml_platform_mode_decomp-0.11.2-py3-none-any.whl
python -m ml_platform_mode_decomp run.yaml
```

## 入力と設定を一つにする

例えば単一の格子場 `(H,W,C)` を配列index座標でDCT分解する設定です。

```yaml
task: mode_decomp_representation
data:
  local_path: data/single_field
  fields: {path: field.npy, layout: HWC}
representation:
  decomposition: {name: dct2}
report: {enabled: true}
```

`HWC`は単一格子場、既定の`NHWC`は複数sampleの格子場です。meshは`VC`／`NVC`と明示します。省略したsample ID・成分名・NPY条件名は配列順の名前を生成します。省略したgeometryはindex座標で、物理単位は推測しません。条件やgroup/time metadataがある場合はCSVまたはNPYを追加し、物理解析には座標・成分・単位・domain maskを明示します。2成分を自動的に物理ベクトルとは扱いません。

同じデータ定義を複数の実験で共有する場合だけ、外部 `field_dataset.yaml` を `data.source_manifest` から参照できます。inline定義と外部manifestの二重指定は拒否します。実行の入力・出力pathはcwd基準、データ内部の参照は `data.local_path` 基準です。

## Pythonの配列から使う

```python
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

# fields: (N,H,W,C)。単一の(H,W,C)にはlayout="HWC"を指定。
data = FieldDataset.from_arrays(fields)
model = FieldRepresentation({"decomposition": {"name": "dct2"}})
coefficients = model.fit_transform(data)
restored = model.inverse_transform(coefficients)
```

条件からの予測には `FieldDataset.from_arrays(fields, conditions)` と `FieldSurrogate` を使います。省略した格子座標は配列indexで、物理単位を推測しません。物理解析には `x`／`y` または明示した `geometry` を渡します。物理ベクトルは `field_kind="vector"`、成分・単位は `components`／`field_units` で指定します。分割・学習・較正はこのconstructorでは実行しません。

別の場から予測する場合は、同じ`FieldSurrogate`に`input_representation`を加え、`model.fit(target, inputs=source)`とします。両側は異なる固定geometry・物理量で構いません。YAMLは既存taskへ`data.input`を追加するだけです。保存済みsource係数も入力にできます。[場と係数の対応学習](docs/FIELD_MAPPING.md)にAPI、ID整列、fold内の表現fit、保存後推論をまとめています。

大量の新条件は `surrogate.iter_predict(new_conditions)` から順にfieldを受け取れます。較正済みなら `intervals=True` で各batchの予測・下限・上限を受け取ります。通常の `predict` も内部では分割計算し、返すfieldだけを全件保持します。

保存する予測を後から部分読込したい場合は、既存の出力名だけを変えます。

```yaml
output: {prediction_name: predictions.npy}
```

既定は従来の `predictions.npz` です。NPYは `np.load(result.artifacts["predictions"], mmap_mode="r")` で部分読込でき、ID・任意の帯・metadataは同じstemの別ファイルへ自動保存します。利用者が追加manifestを作る必要はありません。詳しくは[予測の保存](docs/DATA_FORMAT.md#予測の保存と部分読込)を参照してください。

YAMLをPythonから実行する場合も入口は一つです。

```python
from ml_platform_mode_decomp import load_config, run_task

result = run_task(load_config("run.yaml"))
print(result.run_dir, result.metrics)
```

## 目的から実行を選ぶ

| 目的 | 設定 | 判断・成果物 |
| --- | --- | --- |
| 手元の場を分解する | `task: mode_decomp_representation` | 全入力の係数・再構成・必要なモード／物理成分。sample数1も可 |
| 未使用場への表現を比べる | `task: mode_decomp_analysis`、`analysis.kind: representation` | 保持次元と再構成誤差・費用 |
| 条件からの予測手法を比べる | 同Task、`analysis.kind: surrogate` | 未使用条件のfield誤差、baseline、表現／回帰の寄与 |
| 欠測復元手法を比べる | 同Task、`analysis.kind: imputation` | 人工的に隠した既知正解への復元精度 |
| 最終予測モデルを学習する | `task: mode_decomp_pipeline` | 内側評価またはCVで選択し、勝者を再学習、独立testで評価・保存 |
| 保存モデルから予測する | `task: mode_decomp_infer` | 新条件・source場・保存係数からfieldと任意の予測帯 |

直接分解の `fitted_input_reconstruction` は入力自身の再構成です。未知sampleの性能は比較解析で判断します。比較解析は固定候補を共通holdoutで採点し、test順位から勝者を自動選択・公開しません。pipelineで候補が一つ・探索無効なら、選択用分割を作らず一度fitして独立testを評価します。

## CSV・散布点も一回で解析する

`data.format: csv`／`scattered`を選び、その入力定義を同じYAMLに書きます。実行時に一度だけ正規Datasetへ変換し、分解・比較・学習へ渡します。変換結果はrun出力へ保持し、元データは変更しません。

```powershell
uv run python examples/create_scattered_dataset.py
uv run python -m ml_platform_mode_decomp config/tasks/mode_decomp_scattered_analysis.yaml
```

取込YAML・中間manifest・別の取込コマンドを手で用意する必要はありません。変換済みDatasetを共有したい場合には、任意で `mode_decomp_import` または `scripts/import_dataset.py` を使えます。

## 必要な追加解析だけを有効にする

| 設定 | 内容 |
| --- | --- |
| `analysis.modes.enabled` | 指定係数の応答・選択再構成。直接分解／比較で共通 |
| `analysis.components.enabled` | 学習表現と独立したHodge／Helmholtz成分。method/paramsで選択 |
| `analysis.scattered.enabled` | 保存した元観測での誤差・補間往復誤差・coverage |
| `uncertainty.enabled` | 独立較正したfield予測帯。surrogate学習／比較／推論 |
| `tuning.enabled` | 最終学習内の共通CVと同数予算のgrid／random探索 |
| `report.enabled` | HTML。比較では既定true、直接分解では既定false |

上記追加処理はHTML以外既定falseです。Hodgeそのものは `decomposition.name: hodge` で選びます。予測帯は標準偏差ではなく、交換可能性を条件とする上下限です。Hodgeの直交・energyはnativeな辺内積に対する性質で、節点sampling残差を別に保持します。

```powershell
uv run python examples/create_hodge_dataset.py --samples 1
uv run python -m ml_platform_mode_decomp config/tasks/mode_decomp_hodge.yaml
uv run python examples/create_analysis_dataset.py
uv run python -m ml_platform_mode_decomp config/tasks/mode_decomp_uncertainty.yaml
```

fixture生成は既存先を上書きしません。既に生成したデータには生成コマンドを省略します。実行環境を共有するときだけ `--profile local.yaml` を追加でき、最終上書きは `--set key=value` です。旧 `--task run.yaml` も同じ入口を使います。

- [設定と目的の選び方](docs/CONFIGURATION.md)
- [入力形式・座標・mask](docs/DATA_FORMAT.md)
- [比較解析と評価定義](docs/ANALYSIS.md)
- [別の場・保存係数からの予測](docs/FIELD_MAPPING.md)
- [モード](docs/MODES.md)・[散布点](docs/SCATTERED.md)・[予測帯](docs/UNCERTAINTY.md)・[Hodge](docs/HODGE.md)
- [CV/HPO](docs/SEARCH.md)・[手法一覧](docs/METHODS.md)
- [定常分布のTucker／Quadratic manifold](docs/METHODS.md#定常分布のtuckerとquadratic-manifold)：共通係数API・条件予測・部分観測queryに対応
- [責務と成果物](docs/ARCHITECTURE.md)・[品質検査](docs/07_TESTING_CI.md)
- [現在地・課題・残件](docs/IMPLEMENTATION_STATUS.md)
