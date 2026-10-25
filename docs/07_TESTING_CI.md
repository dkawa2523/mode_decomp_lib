# テストと品質検査（mode 0.11）

検査は通常開発と配布受入の二段です。[CI設定](../.github/workflows/quality.yml)はPython 3.11/Linuxと3.12/Windowsのbase検査、およびPython 3.12/Linux・Windowsのclean wheel受入を定義します。push・pull request・手動実行に対応します。CIのリモート実行結果と、ローカルで実行した検査は[実装状況](IMPLEMENTATION_STATUS.md)で区別します。

## 通常の検査

```powershell
uv sync --locked
uv run python -m pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check --python-interpreter-path .venv/Scripts/python.exe
uv run lint-imports --config pyproject.toml
```

基盤checkoutやcore buildは不要です。optional backendを検証する環境だけ、extraを追加します。

Linuxの型検査ではinterpreterを`.venv/bin/python`へ変更します。Pyrefly実行fileの場所だけでは型検査に使うPython環境は確定しないため、CIもinterpreterを明示します。未導入のoptional backendは該当import行の`missing-import`だけを除外し、他のimportや型エラーは検査します。

```powershell
uv sync --locked --all-packages --extra torch --extra mtgp --extra wavelet --extra pod
```

| 対象 | 残す検証 |
| --- | --- |
| 数値 | FFT独立実数自由度・norm・singleton軸、POD rank/energy、Hodge incidence/edge直交/energy/node残差、較正rank、重み付き損失 |
| 入力 | shape、kind/units、IDとcondition対応、観測finite、正measure、root相対参照、元source不変 |
| workflow | 固定1候補fit1回、共有foldとfresh fit、group/time/gap、test/較正漏洩防止、source/targetを含む外部Stage snapshot、失敗時partial result |
| 解析 | A/Zとcross term、hidden truth、固定採点support、散布点の壁・穴・欠測帯/read-only mask |
| 保存 | 実旧bundle、未知schema/不一致、選択backend、保存後推論と帯、上書き拒否と不完全pair除去 |
| 依存方向 | SDK/別業務package非依存、数値層から実行・保存・描画への逆依存禁止の2規則 |

旧catalog件数、private保存属性、厳密なfile数、local内部JSON全一致、同じCLIの用途別反復を正しさの条件にしません。必要なpublic動作へ置き換えた重複試験を削除します。

## 保存互換

追加の[実0.9.0 fixture](../pkgs/mode_decomp/tests/fixtures/models_v0_9_0/README.md)はschema2のweighted POD/Gappy、非中心SVD、offset+DCT、Hodge、較正済みsurrogateを保持します。新schema4の読込でこれらの係数・field・帯を確認します。旧offsetのfloat32丸めを除去した再encodeには、その元精度に対応する許容差を使い、期待値自体は変更しません。

[実0.7.1 fixture](../pkgs/mode_decomp/tests/fixtures/models_v0_7_1/README.md)は現行版のsaveで再生成しません。対応producer環境では8種類の旧bundleを読み、期待係数・再構成・予測・帯を確認します。Python minorや数値backendが異なる場合は、互換を主張せず理由付きskipとします。現行formatの保存・推論試験はその環境でも実行します。

## 配布受入

```powershell
uv run --no-project python scripts/build_wheels.py
uv venv --python 3.12 work/wheel_env
uv pip install --python work/wheel_env/Scripts/python.exe dist/0.11.2/ml_platform_core-0.2.1-py3-none-any.whl dist/0.11.2/ml_platform_mode_decomp-0.11.2-py3-none-any.whl
work/wheel_env/Scripts/python.exe scripts/verify_wheels.py --work-dir work/wheel_acceptance
```

出力先は未作成のpathを選びます。Linuxのvenv PythonはScripts/python.exeの代わりにbin/pythonです。

release受入では、wheelのPython source集合・bytes、installed CLI、固定候補・小さいCV、3種類の解析、散布点・較正・Hodge出力、訓練入力を移動した後の別process推論を確認します。この検査のYAML・生成例はcheckoutを参照します。大きな探索や数学的詳細をpytestと二重に反復しません。ソースhash検査は配布時だけで、通常のfit・推論へ追加しません。

配布先単独の受入は別にZIPを使います。

```powershell
uv run --no-project python scripts/build_distribution.py
uv run --no-project python scripts/verify_distribution.py --bundle dist/0.11.2/ml_platform_mode_decomp-0.11.2.zip --work-dir work/distribution-acceptance
```

この検査はZIP内のwheel・例・設定だけを参照し、新しい標準venvへ通常pipで導入します。開発package/PYTHONPATHを使わないisolated Pythonで、配列API、単一YAML、学習・保存、訓練入力を隠した後のfresh推論を確認します。通常利用者がこの開発用検査を実行する必要はありません。同じ保存モデルを固定requirementsから導入した二つ目のvenvへcopyし、元のモデルbytesと予測の一致も確認します。sdistからwheelを構築し、tests/fixtures/cacheが配布へ混入しないことも受入で確認します。

CIはOS別に配布物、checksum、受入JSON、pip/consumer logを`distribution-acceptance-<OS>-python-3.12` artifactへ14日間保存します。base検査のJUnit結果も別artifactへ保存します。受入JSONにはbundle SHA256、installed version、訓練入力を外した推論、sourceだけを渡したfield mappingの確認結果が入ります。受入例は合成dataであり、実用途の予測精度の検証とは区別します。

## 計測と適用範囲

性能は同一入力・数値環境・seedで変更前後を測定します。load/fit回数は処理目的に対応させ、内部constructor回数を大量のテストで固定しません。小規模の実行時間改善から大規模のメモリ低減を保証しません。

OMPの線形従属やGPyTorchの既知警告は、別手法への変更や警告抑制で隠しません。未導入backendのskip、旧producer環境が必要なskip、実行して成功した受入を分けて記録します。
