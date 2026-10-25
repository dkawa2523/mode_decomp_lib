# 現在の実装と次の工程

現在の開発sourceはmode 0.11.0です。単一YAML／配列APIによる場の分解、条件・別の場・保存係数からの予測、欠測復元を扱います。共通依存はcore 0.2.1で、SDKやClearMLなしで動作します。

## 到達点

| 項目 | 現在の動作 |
| --- | --- |
| 入力 | inline定義または任意manifest。NPY・CSV・散布点を既存taskで扱い、元入力を変更しない |
| 表現 | FFT/DCT・POD等に加え、定常場用TuckerとQuadratic manifold。共通A/Z codecとmode APIを使用 |
| 条件予測 | 従来の`FieldSurrogate.fit(dataset_with_conditions)`を維持 |
| 場の対応学習 | `fit(target, inputs=source)`。sourceとtargetは独立した固定geometry、完全なsample ID対応 |
| 保存係数入力 | Z空間・sourceモデルhash・次元を照合し、既学習のsource表現を固定して利用 |
| 分割と評価 | 共通holdout/CV、field入力表現のfold内fit、独立test、任意の独立較正。原target fieldで評価 |
| 設定と入出力 | 未知・無作用指定、曖昧な値の変換を拒否。grid／hidden／元観測の指標名と採点supportを一致させる |
| 保存と推論 | bundleに両表現と入力契約を保存。新しいsource場／係数だけでtarget係数・fieldを予測 |
| 外部Stage | targetとsourceの必要参照をsnapshotし、元入力を移動した後も候補評価・finalizeを再開 |
| 出力 | target座標・成分・単位・supportを持つNPZ／部分読込NPY、CSV・JSON、任意HTML |

利用仕様は[設定](CONFIGURATION.md)、[入力形式](DATA_FORMAT.md)、[手法](METHODS.md)、[場と係数の対応学習](FIELD_MAPPING.md)へ集約しています。内部の責務は[アーキテクチャ](ARCHITECTURE.md)、検査方法は[品質検査](07_TESTING_CI.md)を参照してください。

TuckerとQuadratic manifoldは追加依存なしで使えます。学習時は固定domain上の完全観測が必要で、学習後の部分観測queryは対応するsolverで扱います。Hodgeのnativeな辺内積、非線形modeの基準点依存、較正帯の固定supportなど、既存の数値規約を維持します。

保存schemaは4です。入力kindとoptionalなsource表現を検証し、旧schema1〜3のモデルも元のschema／runtime条件で読み込みます。旧producerの実bundle fixtureは互換性検査のため維持します。

## 配布済みsnapshotとの区別

`dist/`のmode 0.10.4 wheel・source archive・ZIPは、これらのsource追加前に作成したrelease snapshotです。現行sourceの到達点をそのまま表す配布物として扱いません。既存の配布物やユーザー入力dataは保持しています。開発環境のeditable mode packageは0.11.0へ更新し、他の依存packageは変更していません。

現行sourceのbuild出力は`dist/0.11.0/`です。`scripts/build_distribution.py`はpackage version別の出力先を既定とし、cleanなsdist→wheel→consumer ZIPを作成します。旧root配布物とSHA256SUMSは保持します。新規配布物は配布先だけのvenv／pip環境で受け入れ、外部公開は別操作です。[配布ガイド](DISTRIBUTION.md)を参照してください。

remote CIはLinux／Windowsの品質検査と配布受入が成功しました。公開Darcy定常計算benchmarkでも、公式testを保持して未知case誤差・表現残差・容量を測定しています。[評価結果と再実行](PAIRED_ACCEPTANCE.md)に出典・保存尺度・分割・CI証跡をまとめています。ユーザー実用途での精度確認、外部registry公開、現行版のSDK統合、実ClearML環境での受入は別工程です。core wheelの来歴と再構築手順は[integration](../integration/README.md)に保持します。

## 現行の制約

- sourceは一つ、targetも一つ。両側のgeometryはそれぞれsample間で固定です。
- 保存係数入力は固定表現を条件とした評価です。表現のfitを含むCVにはfield入力を使います。
- 複数source、補助conditionsとの結合、可変geometry、未ペア学習、target部分観測との同時推定は未対応です。
- 全件arrayを返すAPI、dense SVD・GP、厳密pointwise較正等には問題規模に応じた容量が必要です。

## 次の工程

1. 公開benchmarkの保存値での評価を、ユーザー実用途のpaired data・物理尺度・group分割へ置き換え、用途別の精度・容量閾値を確認する。
2. 成功したremote CIのartifactを確認し、配布channelへの公開を判断する。
3. 実際に必要になった入力の拡張だけを追加する。複数sourceや可変geometryは用途と評価方法が明確になった時点で検討する。

旧計画IDの一覧、生成済み監査JSON／log、旧SDK patchと検討途中のRFCは現行実行の依存ではないため整理しました。現在の仕様と受入手順を上記guideへ一本化します。
