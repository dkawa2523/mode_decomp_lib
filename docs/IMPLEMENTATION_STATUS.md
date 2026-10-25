# 現在の実装と次の工程

現在のsourceはmode 0.11.2、共通依存はcore 0.2.1です。定常場の分解・条件や別の場からの予測・欠測復元を、実行YAML一つまたは配列APIで扱います。SDK／ClearMLは単体利用の前提ではありません。

## 到達点

| 計画 | 現在の動作 |
| --- | --- |
| 共通表現 | field → 前処理 → raw → 実数A → 学習座標Z。逆変換は保存したfieldの成分・単位へ戻す |
| 定常分布の手法 | POD等に加えTuckerとQuadratic manifold。同じcodec・mode・評価APIを利用 |
| 条件／場／係数入力 | 同じFieldSurrogateでtarget Zを学習。sourceとtargetは独立した固定geometry |
| case対応 | sourceとtargetは同じID集合。IDで整列し、欠落・余分・重複caseを拒否 |
| 学習・評価 | 両表現をfold内fit。独立test・任意較正。保存source Zは固定表現を条件とした評価 |
| 入出力の明確化 | 名前付き条件表の列名整列、有限なRidge学習状態、Gappyの観測rank、保存fieldの有効重みを検証 |
| 再利用 | 両表現を含むbundle、runtime固定requirements、任意の予測target Z出力 |
| CLI | 成否・run・metrics・artifacts・tables・plotsのJSON。失敗時も途中成果物と失敗段階を保持 |
| 外部Stage | source場／係数／表現を含む必要参照のsnapshotで再開可能 |

利用仕様は[設定](CONFIGURATION.md)、[入力と出力](DATA_FORMAT.md)、[手法](METHODS.md)、[場・係数の対応学習](FIELD_MAPPING.md)へ集約しています。責務は[アーキテクチャ](ARCHITECTURE.md)、配布は[導入ガイド](DISTRIBUTION.md)、検証は[品質検査](07_TESTING_CI.md)を参照してください。

## 今回の整理

条件DataFrameの列順による予測意味の入替え、重複caseを生成するbatch選択、非有限のRidgeを学習済みとする状態を修正しました。回帰器9種の再学習開始時にも状態を共通に無効化し、失敗後に新しいscalerと古いモデルが混在した予測を返さないようにしました。正則化なしGappyで係数を識別できない観測配置は明示的な失敗とします。正則化を勝手に追加したり、指定solverを別手法に変更したりしません。

CLIのCSV出力漏れと失敗時のrun参照漏れを解消し、予測target Zを既存の保存係数入力へ接続できるようにしました。係数NPZ書込を共通化し、重複処理・tolの重複検証・古い仕様に依存する試験ケースを削除しました。package READMEの重複した内部説明や版ごとの履歴を短くし、SDK説明の矛盾も解消しました。追加調査で判明した無作用の条件ID指定・空manifest指定・不正な保存空間重みも共通境界で拒否します。

数値の正しさ、sample分離、旧bundleの読込、実際の配布受入を確認する検証は維持します。不要な診断framework・schema更新・core変更は追加していません。旧producerの実bundle fixtureとcoreの再構築patchには互換性確認・来歴の役割があり、維持します。未使用SDK資材や旧RFCは既に削除済みで、生成data・過去の配布物は現行実行の依存にしません。

## 検証と配布

現行build先は`dist/0.11.2/`です。旧versionの配布物はその時点のsnapshotです。新規buildはsdistからwheel、consumer ZIPを作り、fresh pip環境で利用を確認します。同じ保存モデルをrequirementsから構築した別venvへ移し、再学習なしで同じ予測になることも配布受入に含めます。外部registry公開は別操作です。

公開Darcy定常計算benchmarkの出典・保存尺度・独立testを保った評価は[評価記録](PAIRED_ACCEPTANCE.md)にあります。この記録は0.11.0時点の測定で、0.11.1でも同じ分割と精度を再現済みです。ユーザーの実測／実用途dataでの受入に読み替えません。0.11.2のlocal検証は860件成功・3件skip、Ruff・型検査・依存方向検査が成功しました。installed wheelでは109 source fileの一致、固定候補・CV・3種類の解析・fresh推論・散布点・較正・Hodge出力を確認済みです。Darcyも同じcase分割・勝者を保持し、RMSE 0.10190444、target表現残差RMSE 0.03468200を再現しました。remote CIはこのbranchの最新Actionsと保存artifactを参照します。

## 改良中に明確になった制約

- 非線形decoderで平均Zを復号したfieldは、一般にfieldの条件付き平均ではありません。原field誤差と平均baselineへのskillで判断します。
- 欠測復元はモデルによる全field近似です。観測位置の値も変わり得ます。観測値を保持する合成は利用側で明示します。
- 予測Zと正解fieldを符号化したZは別です。連鎖学習では上流学習へのcase混入も評価設計に含めます。
- 係数追加出力はN×K行列を保持し、係数予測をもう一度計算します。通常のfield出力はbatchで書き込みます。
- bundleの移送には記録されたPython minorと数値backendが必要です。requirementsはOS／GPU driverを含む完全な環境imageではありません。
- 保存の公開処理にはhard link対応の出力filesystemが必要です。

## 次に進める工程

1. **実用途の受入**：ユーザーpaired data、物理尺度、未知group/caseの分割、許容誤差と容量閾値を使って評価する。公開benchmarkだけではこの工程は完了しません。
2. **配布先の確定**：確認済みwheel／ZIPを実際の利用先へ渡し、必要な配布channelへの公開を判断する。
3. **必要な統合だけ追加**：SDK adapterと実ClearML環境の受入は別工程。複数source、補助条件結合、可変geometry、未ペア学習、target部分観測との同時推定は具体的な用途と評価方法が決まってから扱う。

現在は「共通基盤と入出力の意味をそろえる」工程を実装・検証し、配布受入を行う位置です。次の主工程は実用途dataでの受入です。
