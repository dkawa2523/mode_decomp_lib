# CVと離散パラメータ探索（mode 0.11 / core 0.2.1）

配備用 `mode_decomp_pipeline` で `tuning.enabled: true` を指定すると、outer train内の共通CVで候補のパラメータを選びます。outer final testは探索へ渡しません。各foldでfield前処理・分解・codec・係数後処理・回帰を新しくfitし、原fieldのOOF予測を評価します。

各候補の最良設定を固定し、候補間のCV指標で選んだ勝者だけをouter train全体へrefitします。`uncertainty.enabled: true` の場合は、outer trainから最初にcalibrationを隔離し、残るproper train内だけでCV・候補選択・refitします。その後に予測帯を較正し、一度だけfinal testを評価してモデルを公開対象にします。

`tuning.enabled: false` は、複数候補ならinner holdoutで選択します。固定1候補なら選択を行わずproper trainへ1回fitし、任意較正→final test→保存へ進みます。selection分割・selection score・leaderboardは作りません。outerがfixedでも、この経路にはselection_methodが不要です。

これは独立outer testを持つinner CV/HPOです。固定候補を反復outer holdoutで比較する `mode_decomp_analysis` や、外側もK-foldで繰り返すnested CVとは別の用途です。

## CLIとPython

インストール後、展開先フォルダ（source利用時はrepository root）で実行します。fixtureが既にあれば生成コマンドは省きます。

```powershell
python examples/create_tiny_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_search.yaml
```

同じ一つの設定をPythonから実行できます。profileは実行環境を共有するときだけ追加します。

```python
from ml_platform_mode_decomp import load_config, load_model, run_task

cfg = load_config("config/tasks/mode_decomp_search.yaml")
result = run_task(cfg)
print(result.tables["leaderboard"])
print(result.metrics)  # final_test
print(load_model(result.artifacts["model"]).candidate)  # 固定した勝者の設定
```

Localの入力・出力は起動cwd基準、manifest内参照はDataset root基準です。入力形式は [DATA_FORMAT.md](DATA_FORMAT.md) を参照してください。

## 探索範囲と予算

[実行例](../config/tasks/mode_decomp_search.yaml)はPODの保持数とRidge alpha、DCTのRidge alphaを探索します。`tuning.search_spaces` は候補IDごとに、componentの `params` を指すドット区切りpathと有限の選択肢リストを指定します。手法名の切替は候補を分け、例えば `representation.decomposition.name` を探索pathにしません。

全候補の試行数は共通の `tuning.n_trials` です。どれかの候補で重複しない組合せが足りなければ設定エラーにします。設定の探索なしでCVだけを行う場合は `n_trials: 1` と `search_spaces: {}` を使えます。

- `method: grid` はparameter pathを文字列順に並べ、選択肢の記載順による直積を先頭から評価します。右端のpathが最も速く変わります。`n_trials` が全組合せより小さい場合は先頭部分だけです。
- `method: random` は同じ離散直積から、`tuning.seed` を使って重複なしで抽出します。連続分布や適応的探索・Optuna連携はありません。
- 同数試行は同じ計算時間・モデル容量・探索の難しさを意味しません。範囲と予算はtestを見る前に決めます。

`run.seed` はseedを受け付ける手法の既定値、`tuning.seed` は試行抽出、`tuning.cv.seed` はCV分割です。componentに明示したseedは既定値より優先されます。全候補・全試行で同じCV分割を使います。分割、validationのreference、評価measure、fit集合だけから求めたbaselineを一度準備して共有します。各foldの学習済み変換・回帰器は毎試行で作り直します。ユーザーが指定した物理channel scaleは共有できますが、学習から求めるmean・scale・basisは各foldのfit集合だけから推定します。

## 分割と指標

`metrics.reference`と`model.selection_metric`で対象を指定し、候補・全fold・最終testへ共通適用します。gridはfield_*、元観測はobservation_*、人工欠測はhidden_*です。通常のpipelineはgridまたはobservationsを使います。元点には保存された元観測storeが必要です。

CVはfoldのRMSEを単純平均せず、sample別損失をpoolして元のsample重みで再集約します。採点のための全OOF fieldやA/Z詳細再構成は保持しません。任意のsecondary指標が未定義でもprimaryが有効なら比較を続けます。

| CV方法 | 動作と条件 |
| --- | --- |
| random | sampleをshuffleしてK-fold。各sampleを1回validationへ使う |
| group | 同じgroupをfit/validationに分けない。group数でfoldを分けるためsample数は不均等になり得る |
| time | 同時刻を分断せず、過去だけのfitと後続blockのvalidationを繰り返す。最初の学習blockはOOF採点に含まない |

outerがgroup/timeなら、CVも同じ方法・同じmetadata列を使います。tuning無効の複数候補選択でも、inner holdoutはouterのgroup/time分離と必要gapを維持します。outerがfixedなら `tuning.cv.method` を明示し、group/timeを選ぶ場合はCV用の列も指定します。fold数以上のsample/group、timeではfold数＋1以上の異なる時刻が必要です。

時間的に隣接した学習データを除外する場合は `split.gap_seconds` を指定します。有限・非負の秒数で、outerとinnerのtime holdoutに同じ規則を使います。各境界で `validation開始時刻 - 学習時刻 >= gap_seconds` を満たす過去sampleだけを残し、同時刻のsampleを分断しません。境界と同じ秒差のsampleは保持し、validation側は動かしません。ISO日時はUTCの経過秒へ換算し、timezoneのない日時はUTCとして扱います。

```yaml
split:
  method: time
  time_column: measured_at
  gap_seconds: 86400  # 最低24時間離す。sample数・暦日数ではない。
tuning:
  enabled: true
  cv: {n_splits: 3}  # method/column/gapをouterから継承
```

time CVの `tuning.cv.gap_seconds` はouterの指定を継承し、明示する場合はouter以上にします。fixed等のouterからtime CVだけを行う場合は、CV側へ秒数を指定できます。purge後に学習履歴がなくなればエラーです。`split.json` のouter/inner purged IDsと `cv_folds.json` のfold別purged IDsで除外を確認できます。gapは既定0で、random/group/fixedへの指定、row-count gap、embargo、groupとtimeの複合分割は未対応として拒否します。学習は常に過去だけなので、未来側のembargoは追加していません。

gapだけで統計的独立性が保証されるわけではありません。特徴量の集約windowや正解の予測horizon、系列の依存時間を踏まえて秒数を決めます。

CVの主指標は全validation sampleのOOF予測をまとめ、共通の面積・channel scale/weight・sample weightで計算します。fold RMSEの単純平均ではありません。評価定義とfield_skill、誤差分解は [ANALYSIS.md](ANALYSIS.md) と共通です。CVで最良値を選ぶため、その値を独立な最終精度と解釈せず、`final_test` を別に確認します。

## 保存結果と失敗

local pipelineは一つのrun内でDatasetとsplitを一度準備し、JSONや外部Stageを経由せず続行します。候補の証跡だけを `candidates/<id>/` に保存します。

| 出力 | 内容 |
| --- | --- |
| run直下の `split.json` | outer train/testと任意calibrationの固定ID。探索時はinner holdoutを作らない |
| `candidates/<id>/cv_folds.json` | fit/validationのsample IDと親Datasetのindex。`index_space: dataset` |
| `candidates/<id>/search_trials.csv` | 試行設定、成功/失敗、指標、実測経過時間 |
| 候補内の `candidate_result.json` / `analysis_summary.json` / `per_sample.csv` | 最良試行の解決済み設定、`inner_cv` 指標、OOF sample損失・fold |
| run直下の `leaderboard.csv` | `inner_cv` の候補順位 |
| run直下のbundle・summary・予測 | 勝者のproper train refitモデルと `final_test` 評価。モデルはここだけ公開対象 |

foldを作るときのproper train内の位置と、保存する親Datasetのindexは区別します。保存indexへの変換は準備時に一度行い、sample IDでも対象を確認できます。

不適合rank、逆変換の定義域、backend等による失敗試行は理由を表へ記録し、残りの試行を続けます。失敗も試行予算に含みます。候補の全試行が失敗すると、その表とfoldを部分成果物として残して実行を停止し、モデルを公開しません。localでは候補内の表とfoldを残し、`RunError.result` から取得できます。

外部実行のために `mode_decomp_stage` を個別起動する場合は、各Stageのrunへ結果を保存します。この境界だけで保存split・候補設定・fit IDs・評価設定・scope・明示されたpipeline/release/runtime/workspace contextを照合します。candidateとfinalizeで異なる評価Task ID、およびruntimeのpackage辞書全一致は要求しません。保存モデルのbackend互換検査は別に行います。旧SDK資材は歴史的な接続例で、現行packageの基盤運用受入とは分けています。

検証済みDatasetの選択は再利用しますが、dense fitに必要な配列のgatherは残ります。CVやincremental solverの利用だけで全工程のout-of-core処理を保証しません。
