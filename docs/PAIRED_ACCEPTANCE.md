# Paired fieldの精度・容量と配布受入

2026-10-25に、公開された定常Darcy数値計算benchmarkの保存値で評価しました。ユーザーの実測・実用途データは現在のworkspaceにないため、その用途での精度確認は未実施です。測定完了と、用途ごとの合格判定を区別します。

## データと分割

出典は[NeuralOperator Darcy Flow Dataset v2](https://zenodo.org/records/12784353)です。[公式loader](https://neuraloperator.github.io/dev/_modules/neuralop/data/datasets/darcy.html)が扱う同じファイルの`x[i]`と`y[i]`をペアにしました。入力は保存された二値係数指示場、出力は保存されたDarcy出力場です。

- 32×32×1、両側float32。公式train 5,000件の先頭1,000件と、公式test 1,000件の先頭200件を使用。
- case IDは公式partitionと行番号から作り、sourceとtargetの対応を保持。
- 全公式train/testのsource hashを比較し、partition間の同一場は0件。選択した各partition内の重複も0件。
- 公式archiveのMD5 `24c0033f0c0b75f3df546073111ae154`を照合。原archive、tensorファイル、派生配列のhashを`data/benchmarks/darcy_32/provenance.json`へ保存。
- 800件fit／200件validationで8候補を比較し、選択した候補をtrain 1,000件で再fit。両側の表現と前処理も各fit partitionだけで学習。
- 公式test 200件を候補選択に使いません。物理的なgroup metadataはないため、未知の実験装置・ロットに対する評価ではありません。

archiveの出力には負値・非ゼロ境界があり、物理単位・保存前の逆変換metadataはありません。0/1を3/12等へ推測変換せず、targetも保存値を維持しました。単位名は`stored_benchmark_scale`、座標は配列indexを[0,1]へ正規化した規約です。以下の誤差をPa等の物理誤差や、論文の元物理尺度のrelative L2と同一視しません。

## 測定結果

validationだけで選ばれたのは、中心化したsource POD64 → target POD32、標準化入力を使う多出力GPR（RBF length scale 16、optimizer無効）でした。fieldの標準化とdomain weightsを使用しています。

| 最終testの指標 | 結果 |
| --- | ---: |
| target予測RMSE | 0.101904 |
| 学習平均場baseline RMSE | 0.264729 |
| baselineに対するRMSE低減 | 61.5% |
| target表現残差RMSE | 0.034682 |
| source表現残差RMSE | 0.211636 |
| target予測の相対L2 | 19.32% |
| target表現残差の相対L2 | 6.57% |
| case別予測RMSEの中央値／90 percentile／最大 | 0.09210／0.13436／0.25628 |

RMSEは各sample内で正規化したgeometry重みを使い、sample間は等重みです。相対L2は集約RMSEを同じsupport上の正解RMSで割った値で、case別相対誤差の平均ではありません。最大誤差のcaseは`darcy_32_test_000094`です。

| validationでの比較 | RMSE |
| --- | ---: |
| POD64 → POD32、GPR length scale 16 | 0.103006 |
| POD64 → POD32、GPR length scale 8 | 0.103482 |
| Tucker 8×8×1 → Tucker 8×8×1、Ridge | 0.126912 |
| POD64 → POD32、Ridge α=1 | 0.126993 |
| POD64 → POD32、Ridge α=0.001 | 0.126999 |
| POD64 → Quadratic16、二次α=0.01、Ridge | 0.130243 |
| POD64 → Quadratic16、二次α=0.0001、Ridge | 0.130510 |
| POD128 → POD32、Ridge | 0.130882 |

この構成では係数回帰誤差が全MSEの約88.4%を占め、target表現残差より大きくなりました。改善の優先候補はsourceの情報保持と係数回帰です。TuckerやQuadratic、source次元の増加が常に改善するとは確認できません。次の調整もtrain内で行い、新しい最終評価には追加の未使用caseを使います。

| 容量・速度の実測 | 結果 |
| --- | ---: |
| source＋target配列 | 9.375 MiB |
| 保存モデル | 5.406 MiB |
| 8候補の選択・最終fit・保存を含むpipeline | 4.86秒 |
| 200件の予測 | 0.0153秒 |
| pipeline processの最大resident memory | 260.0 MiB |

測定環境はWindows、Intel Core Ultra 7 265KF、Python 3.12.13、NumPy 2.4.2、SciPy 1.17.0、scikit-learn 1.8.0です。BLAS／OMPは4 threads、GPU不使用。memoryはimport等も含むOSのprocess high-water markで、動作に必要な最小メモリではありません。精度・容量の用途別閾値は未指定です。GPRの学習はsample数に対し概ね時間O(N³)・kernel容量O(N²)で、今回の秒数・容量を大規模入力へ外挿しません。

## 再実行

repository rootで、PyTorchを利用できる環境から一度だけデータを準備します。PyTorchは`.pt`変換用で、変換後の学習に追加依存はありません。

```powershell
python examples/prepare_darcy_benchmark.py
python scripts/verify_paired_data.py config/tasks/mode_decomp_darcy_benchmark.yaml --work-dir work/verification/darcy_acceptance_new --data-kind simulation --provenance "Zenodo 12784353; data/benchmarks/darcy_32/provenance.json; stored Darcy pairs"
```

既存の変換先や評価結果を上書きしません。異なるデータには通常のpaired run YAMLを指定し、`--data-kind measured|simulation|synthetic`と出典を明記します。測定scriptは既存pipelineを使い、入力形式・候補選択・分割を独自実装しません。

今回の詳細結果は`work/verification/darcy_32_acceptance/report.json`、case別誤差は同じdirectoryの`per_sample.csv`です。生成データ・log・JSONはGitへ追加しません。

## Remote CI

[GitHub Actions run 40174035094](https://github.com/dkawa2523/mode_decomp_lib/actions/runs/40174035094)は、実装commit `912497507365f7dbdff98ef713bfb4352ea6853b`で全4 jobs成功しました。

| 環境 | 成功 | skip | 失敗 |
| --- | ---: | ---: | ---: |
| Linux／Python 3.11、品質検査 | 768 | 33 | 0 |
| Windows／Python 3.12、品質検査 | 782 | 19 | 0 |
| Linux／Python 3.12、配布受入 | 成功 | — | 0 |
| Windows／Python 3.12、配布受入 | 成功 | — | 0 |

skipは未導入optional backendと、旧モデルfixtureのproducer Python minor条件です。Pyreflyには検査対象のPythonを明示しています。

両OSでsdist→wheel→ZIP build、fresh pip導入、元学習入力を外した別process推論、sourceだけのfield mapping、target係数・単位、109 source filesの一致を確認しました。artifact・配布物・ZIP内部checksumも照合しています。OS間のZIPバイト列は異なるため、同一バイト列buildとは主張しません。CIの精度受入は同梱合成例による動作確認で、上記Darcy評価はローカルの別実験です。

CI証跡は`work/verification/remote_ci_40174035094/summary.json`にも保存しています。最新の配布受入は検証branchのActions結果とartifactを参照してください。

ユーザー実用途の確認にはsource／targetの保存先、case対応、物理座標・単位、独立に分けるgroup情報が必要です。今回の公開benchmarkで、その用途での精度合格を代用しません。
