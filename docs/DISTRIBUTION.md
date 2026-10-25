# Mode Decomposition 0.10.4

場の分解、条件からの場の予測、欠測復元を扱うPythonパッケージです。実行YAML一つ、または配列APIから利用できます。Python 3.11以上を使います。利用先にGit、uv、基盤repository、ClearML、実行SDKは不要です。

## インストール

ZIPを展開し、`wheels/`があるフォルダで実行します。既存の分析環境と分ける場合は標準venvを作ります。

Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install wheels/ml_platform_core-0.2.1-py3-none-any.whl wheels/ml_platform_mode_decomp-0.10.4-py3-none-any.whl
```

macOS / Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install wheels/ml_platform_core-0.2.1-py3-none-any.whl wheels/ml_platform_mode_decomp-0.10.4-py3-none-any.whl
```

以降のコマンド例はWindows用です。macOS / Linuxでは`.venv/Scripts/python.exe`を`.venv/bin/python`に置き換えます。venvを有効化している場合は、どちらも`python`だけで実行できます。

coreとmodeのwheelを同梱しています。NumPy・SciPy等の依存パッケージはpipが取得するため、通常の導入にはネットワーク接続が必要です。オフライン導入では対象OS・Pythonに合う依存wheelを別途用意してください。

## 最初の実行

単一の合成fieldを生成し、DCT分解します。

```powershell
.venv/Scripts/python.exe examples/create_single_field.py
.venv/Scripts/python.exe -m ml_platform_mode_decomp config/tasks/mode_decomp_representation.yaml
```

表示された`run_dir`内の`representation.html`、係数、再構成fieldを確認できます。この例は入力自身の再構成であり、未知sampleへの精度評価ではありません。生成scriptは既存入力を上書きしません。

手元の`field.npy`を分解する場合は、同じ作業フォルダの`run.yaml`を次のようにします。

```yaml
task: mode_decomp_representation
data:
  local_path: .
  fields: {path: field.npy, layout: HWC}
representation:
  decomposition: {name: dct2}
```

```powershell
.venv/Scripts/python.exe -m ml_platform_mode_decomp run.yaml
```

`HWC`は単一field、`NHWC`は複数sampleです。座標を省略すると配列indexを使い、物理単位は推測しません。別manifest、samples.csv、条件、train/test分割は単一fieldの直接分解には不要です。

## 条件から学習・推論する

```powershell
.venv/Scripts/python.exe examples/create_tiny_dataset.py
.venv/Scripts/python.exe -m ml_platform_mode_decomp config/tasks/mode_decomp_pipeline.yaml
```

学習結果に表示される`artifacts.model`のpathを、次の`model.artifact_path`へ指定します。引用符内のpathを実際の値へ置き換えてください。新条件は生成済みの`data/tiny_fields/new_conditions.csv`を使います。

```powershell
.venv/Scripts/python.exe -m ml_platform_mode_decomp config/tasks/mode_decomp_infer.yaml --set "model.artifact_path=outputs/学習run/model.joblib"
```

推論には保存bundleと新条件だけを使い、訓練入力は不要です。モデルbundleはjoblib形式なので、信頼できる提供元のものだけを読み込んでください。再利用時は保存したPython minor・数値backendの互換条件も確認します。

## 配列API

```python
import numpy as np
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

fields = np.load("field.npy", allow_pickle=False)  # (H,W,C)
data = FieldDataset.from_arrays(fields, layout="HWC")
model = FieldRepresentation({"decomposition": {"name": "dct2"}})
coefficients = model.fit_transform(data)
restored = model.inverse_transform(coefficients)
```

条件からの学習には`FieldDataset.from_arrays(fields, conditions)`と`FieldSurrogate`を使います。大量の予測は`surrogate.iter_predict()`、部分読込する保存形式は`output.prediction_name: predictions.npy`を選べます。関連sidecarは一緒に保持します。

## 用途と追加機能

| 目的 | Task / 設定 |
| --- | --- |
| 入力fieldを分解する | `mode_decomp_representation` |
| 未使用sampleで手法を比較する | `mode_decomp_analysis`、`analysis.kind: representation / surrogate / imputation` |
| 学習・候補選択・最終test・保存 | `mode_decomp_pipeline` |
| 保存モデルで新条件を予測する | `mode_decomp_infer` |
| 元散布点の誤差とcoverageを追加する | `analysis.scattered.enabled: true` |
| 独立較正したfield予測帯を使う | `uncertainty.enabled: true` |
| モード応答・物理成分を解析する | `analysis.modes.enabled` / `analysis.components.enabled` |

散布点の補間support、元点誤差、格子field誤差は別の量です。予測帯は固定grid supportのsplit conformalで、任意点・条件付き・分布変化へのcoverage保証ではありません。Hodgeの`natural`は非周期の離散cochain境界であり、指定PDE境界値や点ごとの法線速度を課しません。詳しくは各guideを参照してください。

通常のFFT/DCT・POD SVD・Ridge・散布点・Hodgeはbaseインストールで使えます。必要なbackendだけextraを追加します。

```powershell
.venv/Scripts/python.exe -m pip install --find-links wheels "ml-platform-mode-decomp[wavelet]==0.10.4"
```

`wavelet`はPyWavelets、`pod`はmodredを使うPOD方式、`torch`はautoencoder、`mtgp`は多出力GP、`gbdt`はXGBoost・LightGBM・CatBoostです。`pod_svd`のために`pod` extraを導入する必要はありません。

## 同梱物とguide

`wheels/`は実行用の2wheel、`config/tasks/`は用途別の自己完結したYAML、`examples/`は合成入力の生成scriptです。`sources/`はmodeのsource archive、`core.json`は同梱coreの来歴、`SHA256SUMS`は配布ファイルの照合用です。通常の実行にsource buildやhash照合は必須ではありません。

- [設定と一つのYAML](CONFIGURATION.md)
- [入力・出力形式](DATA_FORMAT.md)
- [手法と適用条件](METHODS.md)
- [比較解析](ANALYSIS.md)、[CV・探索](SEARCH.md)
- [散布点](SCATTERED.md)、[較正帯](UNCERTAINTY.md)
- [Hodgeの物理仕様](HODGE.md)、[モードの意味](MODES.md)

同梱例は合成データです。実観測での有効性や必要容量は、実際のsample数・格子/点数・成分数・物理境界に合わせて確認してください。
