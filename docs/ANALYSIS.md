# 科学解析ガイド（mode 0.10 / core 0.2.1）

`mode_decomp_analysis` は、事前に決めた手法・ハイパーパラメータを共通の outer train/test で比較するタスクです。原 field 上の損失、表現による誤差、baseline、次元、コストを保存します。test の順位による候補選択やモデル公開は行いません。

手元の一つの場を分解するだけなら `mode_decomp_representation` を使います。全入力で表現をfitし、係数・再構成・任意の物理成分／モードを保存します。scopeは `fitted_input_reconstruction` で、未使用sampleへの性能とは解釈しません。目的が異なるため、この入口にholdoutを強制しません。

## 3つの用途

| `analysis.kind` | train で fit するもの | test 時の入力と正解 |
| --- | --- | --- |
| `representation` | field 前処理・分解・codec・係数後処理 | 未使用 field を変換・再構成し、原 field と比較 |
| `surrogate` | 表現全体と、条件から Z への回帰器 | 条件だけを入力し、未使用の原 field と比較 |
| `imputation` | train field から学ぶ復元器 | 未使用 sample の既知値の一部だけを入力し、人工的に隠した既知値と比較 |

全候補で同じ test sample・geometry・評価 support・重みを使います。Datasetはrun開始時に一度読み、各splitのreference・measure・baselineを候補とmodel seedで共有します。学習済みの前処理・basis・回帰器は共有しません。representation と imputation は条件列がなくても利用できます。imputation はもともと未観測の値を正解として扱いません。隠す前の既知値を評価側に保持し、入力観測maskと評価maskを分けます。隠した値を除去した入力とhidden/full/observedのmeasureはsplitごとに一度準備します。

## CLI と設定

展開先フォルダ（source利用時はrepository root）で実行します。インストールは [README](../README.md) を参照してください。

```powershell
python examples/create_analysis_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_analysis.yaml
python -m ml_platform_mode_decomp config/tasks/mode_decomp_representation_analysis.yaml
python -m ml_platform_mode_decomp config/tasks/mode_decomp_imputation_analysis.yaml
```

fixture は非線形な条件依存と観測ノイズを含む `data/nonlinear_fields` です。生成スクリプトは既存出力を上書きしません。評価の正解はノイズ付き観測 field であり、ノイズなし潜在場ではありません。Local の入力・出力 path は起動 cwd 基準、manifest 内の参照は Dataset root 基準です。

最小の surrogate 比較設定は次の形です。候補の既定値も含めた解決済み設定は protocol に保存されます。

```yaml
task: mode_decomp_analysis
run: {seed: 42}
data: {local_path: data/nonlinear_fields}
split: {method: random, valid_size: 0.25}
analysis:
  kind: surrogate
  split_seeds: [11, 23, 47]
  model_seeds: [42]
  bootstrap_samples: 1000
  confidence_level: 0.95
  allow_failures: false
  save_models: false
model:
  candidates:
    - id: pod6_ridge
      representation:
        decomposition: {name: pod_svd, params: {n_modes: 6, inner_product: domain_weights}}
      regression: {name: ridge, params: {alpha: 0.01}}
metrics:
  names: [field_rmse, field_mae]
```

`metrics.channel_scales`、`metrics.channel_weights`、`metrics.sample_weight_column` で評価の尺度と重みを指定できます。imputation は `analysis.observed_fraction`、`analysis.observation_pattern`（`shared` / `per_sample`）、`analysis.mask_seed` も使います。この固定候補解析は探索を行いません。配備用pipelineの任意CV/HPOは [SEARCH.md](SEARCH.md) に分けています。比較前に範囲と予算を決め、testを見た後の変更は別の探索として記録してください。

## 主評価を選ぶ

通常は`metrics.reference:grid`で、元観測を主評価にする場合は`observations`、人工欠測は`hidden`を使います。指標名もそれぞれfield_* / observation_* / hidden_*に分けます。明示referenceと指標の矛盾は拒否します。旧imputationでreference未指定のfield_*はhidden互換名です。

元点ではsample内の点重みを正規化し、channel尺度・重み、sample重みを適用します。候補間のsupportを参照から固定し、予測NaNで評価点を減らしません。coverageは評価可能な範囲であり精度ではありません。元点R²は提供しません。secondary R²/skillが未定義でも有効RMSE/MAEを保持します。

実sensor配置や連続欠損には`analysis.observation_mask:sensors.npy`を指定できます。trueが入力観測で、元から未知の値はhidden正解にしません。sample別NPYは元fieldのarray_index順です。[設定とshape](CONFIGURATION.md)を参照してください。

通常pipelineの選択・CV・最終採点は予測一度の採点経路です。このanalysisタスクと公開analyze_* APIは、目的に応じた詳細誤差分解を扱います。

## Python API

タスクを実行する場合は core と同じ設定合成・RunResult を使います。

```python
from ml_platform_mode_decomp import load_config, run_task

cfg = load_config("config/tasks/mode_decomp_analysis.yaml")
result = run_task(cfg)
print(result.artifacts["analysis_report"])
print(result.tables["comparison"])
print(result.tables["per_sample"])
```

直接評価する場合は fit と分割を呼出側が管理します。以下は同じ train/reference を3用途で使う例です。

```python
from ml_platform_mode_decomp import FieldRepresentation, FieldSurrogate
from ml_platform_mode_decomp.data import load_field_dataset, split_dataset
from ml_platform_mode_decomp.evaluation import (
    analyze_imputation,
    analyze_representation,
    analyze_surrogate,
    observation_mask,
)

dataset = load_field_dataset("data/nonlinear_fields")
split = split_dataset(dataset, {"method": "random", "valid_size": 0.25}, seed=11, nested=False)
train = dataset.subset(split["train_indices"])
reference = dataset.subset(split["test_indices"])
representation_cfg = {"decomposition": {"name": "pod_svd", "params": {"n_modes": 6, "inner_product": "domain_weights"}}}
options = {"names": ["field_rmse", "field_mae"], "reference_scope": "outer_test"}

representation = FieldRepresentation(representation_cfg, seed=42).fit(train)
representation_report = analyze_representation(representation, train, reference, options)

surrogate = FieldSurrogate(
    {"representation": representation_cfg, "regression": {"name": "ridge", "params": {"alpha": 0.01}}},
    seed=42,
).fit(train)
surrogate_report = analyze_surrogate(surrogate, train, reference, options)

gappy = FieldRepresentation(
    {
        "decomposition": {
            "name": "gappy_pod",
            "params": {"n_modes": 6, "inner_product": "domain_weights", "reg_lambda": 1e-6},
        }
    },
    seed=42,
).fit(train)
observed = observation_mask(reference, fraction=0.3, seed=81, pattern="shared")
imputation_report = analyze_imputation(gappy, train, reference, observed, options)
print(surrogate_report.metrics)
```

`analyze_*` は fit を行いません。戻り値 `EvaluationReport` は `metrics`、sample 別 DataFrame の `per_sample`、JSON 化可能な定義・来歴の `summary`、`predictions`、診断用の `fields` を持ちます。データ形式は [DATA_FORMAT.md](DATA_FORMAT.md) を参照してください。

## 共通の評価 measure

評価重みが0のsampleは表に残し、誤差を未採点の空欄にします。集約・paired比較・previewには使いません。sample/channel/point重みは評価対象の中で安定に正規化し、大きい有限重みの合計overflowを避けます。CVの一部foldだけが全0重みでも、全体で正重みがある場合は全体のsample統計から集約します。

元点の主評価にはsample/channel/pointの重みを使い、格子quadratureを要求しません。格子は逆補間の幾何情報として使います。格子主評価とgrid予測帯では正measureとsupportの既存条件を維持します。

面積重みは geometry の正の `integration_weights`、未指定なら一様です。原 field の有効 mask と用途別評価 mask を交差させ、sample ごとに面積を正規化します。各評価 sample に正の評価面積が必要です。

```text
mse[n] = Σchannel channel_weight[c]
         × Σpoint normalized_area[n,p] × ((prediction-truth)[n,p,c] / channel_scale[c])²
field_rmse = sqrt(Σsample normalized_sample_weight[n] × mse[n])
```

channel weight は非負・総和正、scale は正です。sample weight は既定で一様で、指定した metadata 列から変更できます。集約 RMSE は sample RMSE の平均ではありません。MAE も同じ重みを使います。channel 別 RMSE/MAE は scale で割る前の物理単位です。異なる単位の channel を集約するときは明示した scale が必要です。

`field_r2`は全sample・全空間の変動に対するchannel別R²の重み付き平均です。共通の空間形状だけで高くなる場合があり、条件予測能力を単独では示しません。constant channelでは未定義理由を記録し、有効RMSE/MAEは保持します。R²をprimaryにした場合は選択できないためエラーにします。

baseline は train の観測値だけから地点・channel ごとの平均 profile を作ります。この平均の fit は sample-weighted ではありません。同じ評価 measure で次を計算します。

```text
field_skill = 1 - model_MSE / training_mean_profile_MSE
```

skill は 0 で平均 profile と同等、負なら劣ります。train の観測不足で評価地点の baseline が未定義なら、理由を summary に記録して baseline RMSE と skill を省きます。baseline MSE が 0 のときも skill は省きます。surrogate 解析には train の条件から原 field を推定する線形回帰 baseline も含まれます。

## A / Z と誤差分解

A は decomposition の raw 出力を codec が実数化した係数、Z は係数後処理後の学習 target です。基底の正規化・codec・保持次元が異なるため、異手法の係数誤差を共通の性能指標にはしません。

| 指標 | 比較する field |
| --- | --- |
| `decomposition_rmse` | 原 field と A からの復元（前処理・分解・codec を含む） |
| `coefficient_processing_rmse` | A からの復元と Z からの復元 |
| `representation_rmse` | 原 field と Z からの復元 |
| `regression_rmse` | 正解を Z 経由で復元した field と、条件からの予測 |
| `field_rmse` | 原 field と最終予測 |

surrogate では次の恒等式を保存します。同じ重みで計算した MSE の関係で、RMSE 同士の加算ではありません。

```text
e_rep = represented_truth - original_truth
e_reg = prediction - represented_truth
field_MSE = representation_MSE + regression_MSE + error_cross_term_mse
error_cross_term_mse = 2 × weighted_inner_product(e_rep, e_reg)
```

cross term は負にもなります。誤差の相殺があるため、回帰誤差の改善が最終 field 誤差の改善と一致するとは限りません。sample 別 CSV にも各 MSE と cross term を保存します。

`FieldRepresentation` のモード一覧・単独応答・選択／累積再構成と、任意の `analysis.modes` 出力は [MODES.md](MODES.md) を参照してください。モードHTMLの全A終点は係数後処理前です。representationとimputationでは、lossy PCA等を含む主Z再構成と異なる場合があります。surrogateは条件から予測したZをAへ戻して使うため、全A終点は主予測に一致します。係数の由来と評価supportを混同しません。

## seed、分割、paired 比較

`split_seeds` はデータ分割、`model_seeds` は fit の乱数を分離します。同じ split seed の全候補・model seed は同じ train/test を使います。seedを受け付けるcomponentにだけモデルseedを既定注入します。componentに明示した `seed` / `random_state` は優先されます。乱数を使わない `pod_svd` などに無意味なseedは渡しません。imputation の観測 mask は `mask_seed + split_seed` で固定し、候補や model seed によって変えません。

random / group / time / fixed split に対応します。group の割合は group 数基準なので sample 比率と一致しない場合があります。time は同じ時刻を境界で分離せず過去から未来へ分け、`split.gap_seconds` で評価直前の学習履歴を除外できます。embargoとouterのrolling origin反復は未対応です。time / fixed は seed を変えても同じ分割になる場合があります。繰返し holdout の重複を独立な標本数として数えないでください。gapの定義は[探索ガイド](SEARCH.md)を参照してください。

paired CSV は各 split/model seed 内で同じ sample ID・重み・coverage を照合し、`RMSE(a) - RMSE(b)` を比較します。bootstrap は sample または指定した独立 group を単位として丸ごと再抽出します。pixel・channel・係数を独立標本として扱いません。`analysis.bootstrap_group_column` を指定でき、group split では分割用 group 列を既定で使います。

区間は **学習済みモデルと一つの test split に条件付けた、補正なしの区間** です。再学習による変動や多数候補・多数比較への補正を含みません。独立単位が2つ未満、draws が0の場合は区間を出しません。time split は独立 group を明示しない限り再抽出を行いません。group を独立とみなせるかは実験設計で判断する必要があります。

候補 summary の反復 RMSE 平均・標準偏差は記述統計です。全 sample をまとめた RMSE や独立反復の信頼区間とは異なります。

## 成果物とコスト

| 出力 | 内容 |
| --- | --- |
| `analysis.html` | 指標・baseline・次元／誤差図・誤差の大きい sample・CSV へのリンク |
| `comparison.csv` | 候補×split/model seed の集約指標・次元・コスト |
| `per_sample.csv` | sample ID・重み・support・MSE/MAE・baseline と誤差分解 |
| `paired_comparisons.csv` | paired RMSE 差・条件付き区間・再抽出単位 |
| `protocol.json` | 解決済み候補、seed、分割 sample ID、入力 revision |
| `analysis_summary.json` | 評価定義、候補別記述統計、失敗と不完全性、来歴 |
| `modes/<候補・seed>/`（明示時のみ） | modes.html、座標metadataのmodes.csv、指定順と全A終点のreconstruction_curve.csv |

次元は ambient field、geometry mask 内の入力、packed A、target Z を区別します。`target/input` の比率だけでなく codec の独立実数座標や基底内積も確認してください。座標変換、学習表現、部分観測復元、物理成分分離では次元の意味が異なり、同じ Z 次元でもモデル容量・正則化・費用は等しくありません。HTML の次元図は横軸に log1p を使い、候補を色・凡例で区別します。上部の Field previews リンクから場の図へ移動できます。

`fit_seconds` は fit と、有効化した場合の校正を含む実測時間です。surrogate の `prediction_seconds` は点予測のpredict、imputation は reconstruct の実測です。representation は A への変換と、A→Z→field の計測を合計し、A 復元などの全診断時間は含みません。coverage評価・成分出力等の追加解析時間はprediction_secondsに含めません。wall clock には初回 import/cache や環境差が入り、厳密な速度 benchmark には warmup・反復・資源固定が必要です。baseline の未計測時間は欠損値です。

Pythonの`analyze_representation()`と`analyze_surrogate()`は既定で詳細fieldをreportへ残します。指標と最終予測だけを使う場合は`retain_fields=False`で未使用の診断field保持を省けます。YAMLからの解析はこの経路を使い、preview・モード・物理成分の要求出力を維持します。`mode_fields()`は要求したfield配列だけをまとめて返し、モードHTMLは応答を一つずつ描画します。

`model_bytes` は `analysis.save_models=true` で実際に保存したmodel payloadのbyte数です。falseでは未測定の欠損値にし、計測のためだけのjoblib化を行いません。peak memoryやwheelサイズではありません。preview は最初の反復の大誤差 sample を最大3 channelで縮小表示します。mesh は先頭2座標で表示します。指標は縮小画像でなく全評価 sample・全有効地点を使います。

数値計算に失敗した候補も理由を記録し、可能なCSVと不完全表示付きHTMLを保存します。既定は部分成果物付き `RunError`、`allow_failures=true` は候補失敗による不完全性を保持したまま完了します。`save_models=true` は反復ごとのモデルを保存しますが、解析の `model_publications` は空です。

数値評価後の成分出力や全体HTML生成が失敗した場合、成功した数値行・CSV・metricsは保持し、出力失敗をsummaryへ分けて記録します。要求出力が完了していないため、この場合は `allow_failures` に関わらず `RunError.result` から部分成果物を取得します。モデル保存失敗も成功扱いにはしません。

モード表示は指定座標だけを生成します。表示できない場合は `mode_analysis` に理由を残し、主評価とは区別します。指定順はenergy順位ではなく、非直交・非線形・複合座標では誤差の単調減少や物理energy分解を仮定しません。

## 配備用フローとの関係と制約

`mode_decomp_pipeline` は複数候補ならinner holdout、`tuning.enabled=true` なら候補数によらず共通CV/grid・random探索で選択します。選択した勝者だけproper train全体へrefitしてfinal testを評価します。固定1候補・探索なしではselection分割を作らず、proper trainへ一度fitして最終評価・保存します。同じevaluatorを使いますが、解析タスクはinner selectionを持ちません。`mode_decomp_representation` は単一場を含む全入力を直接分解して保存する用途で、指標のscopeは `fitted_input_reconstruction` です。

通常のlocal pipelineは同じprocess内の評価結果を直接使い、保存候補JSONを読み戻しません。外部Stageで保存評価を受け取る場合だけ入力revision、split、候補／探索、fit IDs、評価設定、scope、明示されたpipeline/release/runtime/workspaceのcontextを照合します。runtimeのpackage辞書全一致と、Stage固有の評価Task IDの一致は要求しません。保存モデルを読むときのfile hash・field schema・backend互換検査は別に維持します。解析タスクにrun間の自動評価cacheはありません。testを見て候補を追加・変更する場合は、探索と最終評価の独立性を別途確保してください。

離散CV/HPOと校正済みfield予測帯は実装済みです。外側も反復するnested CV、連続／適応的探索、条件最適化は未実装です。基盤接続の追加受入はpackageを固めた後に扱います。固定geometryの観測正解に対する比較であり、未知の真値・別geometry・外挿条件への保証は与えません。各手法の前提は [METHODS.md](METHODS.md) を参照してください。

## 任意の追加解析を選ぶ

`uncertainty.enabled=true` はsurrogateの学習と解析だけに使えます。outer trainから校正sampleを先に隔離し、全候補を同じproper trainでfitしてから共通校正sampleで帯を作ります。comparison CSVにinterval_scope、interval_coverage_target、実測のinterval_sample_simultaneous_coverage／interval_pointwise_coverage／幅を保存します。channel別物理幅と保証条件はdetails[].field_uncertaintyにあります。固定候補比較のpaired bootstrapとは異なる量です。[予測帯ガイド](UNCERTAINTY.md)

`analysis.scattered.enabled=true` は保存した元散布点のobservation／interpolation誤差とcoverageをcomparison・per_sample CSVに追加します。成分番号ch0等はmanifestのcomponents順です。集約値・支持範囲・点数はdetails[].scatteredに保存します。原観測点での予測誤差、観測→格子→観測の往復誤差、面積重み付き格子field誤差を区別します。補間往復誤差を予測誤差から差し引いたり、全モデルの誤差下限とみなしたりしません。人工欠測のhidden truthとの混同を避けるためimputationには適用しません。[散布点ガイド](SCATTERED.md)

`analysis.components.enabled=true`は供給fieldの物理成分を保存します。`method:hodge`が既定で、周期場には`method:helmholtz`を明示します。POD/DCT等の学習表現にも同じ物理解析を適用できます。representationは未使用input field、surrogateは条件だけから予測した物理fieldを分解し、回帰component座標のラベルを物理成分とみなしません。kind・単位・geometryが指定methodに適合しない場合は実行前に拒否します。[Hodgeガイド](HODGE.md)

`analysis.scattered`・`analysis.modes`・`analysis.components` と不確実性は明示した場合だけ計算します。散布点が無効なら元点評価storeの読込・逆写像・往復誤差の準備を省きます。raw散布点入力をgridへ変換する処理は入力自体に必要なので残ります。有効時は元観測・support・往復誤差をsplitごとに共有し、共通support上の予測NaNによってcoverageを縮めずエラーにします。

モード無効なら座標応答・再構成曲線・個別HTMLを、成分無効なら再分解・成分NPZ・energy表を省きます。`report.enabled=false` は全体HTMLの生成を省き、主CSV/JSONを残します。モード個別HTMLは `analysis.modes.enabled` が独立して制御します。[YAML設定](CONFIGURATION.md)に一覧があります。
