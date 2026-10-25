# モードと係数座標の解析（mode 0.10 / core 0.2.1）

学習済み `FieldRepresentation` は、モードの一覧、指定座標の場への応答、選択・累積再構成を共通 API で扱えます。基底画像は要求した座標だけ生成します。既定の解析空間は、係数後処理に入る前の実数 packed **A** です。

```text
原 field → field 前処理 → 分解 raw → codec → A → 係数後処理 → Z
条件 → 回帰器 → Z → 逆合成 → 原 field
```

A の index は、この fit 済み表現の codec 順です。手法間で同じ index や同じ `n_modes` が同じ形状・自由度を意味するわけではありません。Z は回帰用の座標で、PCA 後の Z を Fourier モードとは呼びません。

独立POD/Gappy/pod_svdの新fitは成分別のKcを保持します。Aは成分順のblockで、component_offsetsが区間を示し、list_modesは成分名と成分内indexを返します。rank0成分にはモードを作らず、学習meanまたはzero baselineとして復元します。全成分rank0なら一覧は空です。旧モデルのCK順序は読み続けます。

## 単一場を1つの YAML で解析する

`field.npy` が `(20,24,1)` の単一場なら、次の `run.yaml` だけで直接分解できます。別の manifest・sample CSV・profile は不要です。例の座標は配列 index の単位なので、物理周波数が必要な場合は実際の step と座標単位を指定します。

```yaml
task: mode_decomp_representation
data:
  format: npy
  local_path: .
  fields: {path: field.npy, layout: HWC, kind: scalar, components: [signal]}
  geometry:
    kind: rectangle
    axes:
      x: {start: 0, step: 1, count: 24, periodic: true}
      y: {start: 0, step: 1, count: 20, periodic: true}
representation:
  decomposition: {name: fft2_lowpass, params: {block_size: 3}}
analysis:
  modes: {enabled: true, indices: [0, 1, 2], cumulative_counts: [1, 2, 3]}
report: {enabled: true}
```

```powershell
python -m ml_platform_mode_decomp run.yaml
```

path は起動 cwd 基準です。環境設定を分けたい場合だけ `--profile PATH` を追加します。`--task PATH` の従来形式も使えます。`HWC` は1場、`NHWC` は複数場で、sample table を省略すると ID を自動生成します。

直接タスクは split せず、全入力から表現を fit して再構成します。`fitted_input_reconstruction` は未知場の評価ではありません。学習済み bundle、Z と `packed_a` を含む係数 NPZ、再構成 field NPZ、指標・追加解析を保存します。複数の未知 sample での比較は、別の目的として `mode_decomp_analysis` を使います。

## Python API

```python
import numpy as np
from ml_platform_mode_decomp import FieldRepresentation
from ml_platform_mode_decomp.data import FieldDataset
from ml_platform_mode_decomp.evaluation import score_fields

field = np.load("field.npy")  # (H,W,C)
reference = FieldDataset.from_arrays(field, layout="HWC", components=["signal"])
representation = FieldRepresentation({"decomposition": {"name": "fft2_lowpass", "params": {"block_size": 3}}}).fit(
    reference
)

order = [0, 1, 2]  # 呼出側が明示する順序。energy 順ではない。
metadata = representation.list_modes(order)
a = representation.transform(reference, space="a")
responses = representation.mode_fields(order)  # (3, *field_shape)
selected = representation.reconstruct_modes(a, order[:2])
for count, fields in enumerate(representation.cumulative_reconstruct(a, order), start=1):
    print(
        count, score_fields(reference, fields, {"reference_scope": "fitted_input_reconstruction"}).metrics["field_rmse"]
    )
```

`from_arrays` はファイル保存や split・fit を行いません。省略した x/y は index 座標であり、物理単位を推測しません。実座標には x/y、mask・面積・単位を含む固定領域には geometry を渡します。

| API | 意味 |
| --- | --- |
| `list_modes(indices=None, space="a")` | 座標の説明を返す。field は生成しない。省略時だけ全座標を列挙 |
| `mode_fields(indices, space="a", reference=None, amplitude=1.0)` | `inverse(reference + amplitude*e_i) - inverse(reference)` を返す |
| `reconstruct_modes(coefficients, indices, space="a", reference=None)` | 指定座標に与えた係数値を使い、残りは reference（既定ゼロ）の値で再構成 |
| `cumulative_reconstruct(coefficients, order, space="a", reference=None)` | 明示した順に座標を追加した再構成を1つずつ返す iterator |

`coefficients` は sample×座標の2次元配列です。`reference` は座標 vector または sample×座標配列です。ただし `mode_fields` は1つの reference vector を使います。`indices` は fit 後の実次元内にある、重複しない index を指定します。

固定された affine な逆合成では、ゼロ座標の再構成 `b = inverse(0)` が基準場です。POD の学習平均や field 標準化の平均がここに含まれることがあります。単位応答は `phi_i = inverse(e_i) - b` なので、選択再構成は `b + Σ A_i*phi_i` です。各モードへ平均を足し直しません。`additive=true` はこの合成の性質を示し、直交性や係数の独立性を保証しません。

## FFT の離散実モード

既定の auto codec は `fft2` / `fft2_lowpass` に `real_fft` を使います。共役 pair の片側の実部・虚部をそれぞれ √2 倍して packed A に保存し、DC／自己共役 Nyquist 点は実数値だけ保持します。全 FFT では channel ごとに H×W 個の独立実座標です。共役対称性は、任意の実 A を復号しても成立します。

一覧には `channel`、`kx` / `ky`、`kind`（`dc` / `nyquist` / `cosine` / `sine`）、`self_conjugate`、`nyquist_axes`、物理周波数等が含まれます。片軸が Nyquist でも、他軸の波数によっては非ゼロの sine 応答があります。

```text
Lx = 元の幅 W × dx       Ly = 元の高さ H × dy
frequency_x = kx / Lx    frequency_y = ky / Ly
phase = 2π [frequency_x (x-x0) + frequency_y (y-y0)]
```

`angular_wavenumber_x/y` は周波数の 2π 倍、`phase_origin` は `(x0,y0)` です。`frequency_unit_x/y` は geometry の座標単位を使い、単位未指定なら `cycles/coordinate_unit` とします。FFT は明示された周期軸とは別に周期延長を仮定するため、`fft_periodic_extension` と `declared_periodic_axes` を分けて記録します。

`norm="ortho"` の正規化で、pair の単位実座標は `sqrt(2/(HW))*cos(phase)`、単位虚座標は **`-sqrt(2/(HW))*sin(phase)`** を合成します。自己共役点は `cos(phase)/sqrt(HW)` です。これらは mask 適用・field 逆前処理前の周期グリッド上の式です。`mode_fields` の返却値は逆前処理後の原 field の応答なので、標準化の scale 等を反映します。単位は `coefficient_unit`、`basis_units_before_field_inverse`、`field_response_unit_per_coefficient_unit` で区別します。

lowpass の `block_size` は保持周波数の幅で、モード数ではありません。部分保持の偶数幅は共役閉包のため1増える場合があります。物理周波数の分母は保持 block ではなく元の H、W です。例の2 channel データでは POD の `n_modes=2` は packed 4 次元、FFT の `block_size=3` は packed 18 次元になります。

FFT の係数 norm 保存は周期グリッド上の性質です。mask 後の領域、非一様な物理面積、field scale を含む評価内積での直交性や物理 energy 保存まで意味しません。

## 潜在座標と有限摂動

`list_modes(..., space="z")` は常に `latent_coordinate` として記録し、Fourier 波数を付けません。PCA 等の affine な逆合成は Z でも加算可能ですが、方向は元の A の方向と異なり得ます。

AE、mag/phase codec 等の非線形合成や、offset＋residual の複合座標は固定した独立モード系として扱いません。metadata の `interpretation`、`additive`、`reference_required` を確認し、応答を求める際は reference を指定します。

```python
# fitted_model は fit 済みの表現。Z の基準点付近で1座標を0.1だけ動かす。
z = fitted_model.transform(reference)
delta = fitted_model.mode_fields([0], space="z", reference=z[0], amplitude=0.1)
```

これは有限摂動であり、一般には微分や reference に依存しない基底ではありません。複合表現では、たとえば offset と残差の DC が冗長になる場合もあります。全手法に共通の「係数二乗＝物理 energy」や、累積座標数に対する再構成誤差の単調減少は仮定しません。

## opt-in の解析レポート

[例設定](../config/tasks/mode_decomp_mode_analysis.yaml) は非線形条件依存を持つ2 channel fixture で、POD2 と FFT3 を同じ単一 split で解析します。

```powershell
python examples/create_analysis_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_mode_analysis.yaml
```

fixture が既にある場合、生成コマンドは省略します。解析を有効にする設定は次だけです。

```yaml
analysis:
  modes:
    enabled: true
    indices: [0, 1, 2]
    cumulative_counts: [1, 2, 3]
```

候補・split・model seed ごとに `modes.html`、`modes.csv`、`reconstruction_curve.csv` を保存し、主 HTML からリンクします。CSV の `selection_order=explicit_index_order` は指定順を表し、energy 順位ではありません。全 fit 次元へ依存する画像・表を常時生成する設定ではありません。

曲線の `selected_a_coordinates` は指定 index の prefix、`all_a_coordinates` は全 A による再構成です。両者とも原 field と主解析と同じ support・面積・sample/channel 重みで採点します。用途ごとの入力と終点は次の通りです。

| 用途 | 曲線へ渡す A | 全 A 終点と主指標の関係 |
| --- | --- | --- |
| representation | held-out field を符号化した A | 係数後処理前の分解再構成。lossy PCA 等があると主 Z 再構成と異なる |
| surrogate | 条件から予測した Z を逆係数後処理した A | 元の surrogate 予測と一致。held-out field の符号化値は使わない |
| imputation | 入力観測だけから符号化した A | 隠した既知正解だけで採点。係数後処理前なので主 Z 再構成と異なる場合がある |

非線形・複合表現の単独表示はゼロ座標を基準とした有限摂動です。解析タスクは候補の選択やモデル公開を行いません。全体の評価定義は [ANALYSIS.md](ANALYSIS.md)、手法の適用条件は [METHODS.md](METHODS.md) を参照してください。
