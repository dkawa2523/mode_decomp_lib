# mode_decomp_eval_dataset_v1

主な8subset（scalar_rect/disk/mask/annulus/sphere、vector_rect/disk/mask）の条件・場・maskと旧manifestを原データとして保持しています。全て200sampleで、scalarの8条件・vectorの16条件は固定空間パターンの生成係数です。実測非線形条件の一般化性能を証明するデータではありません。

現行packageは[入力設定](../../docs/DATA_FORMAT.md)の`data.format: legacy-npy`で実行時に一度だけ変換できます。変換済Datasetを共有する場合だけ`python scripts/import_dataset.py legacy-npy SOURCE DESTINATION`を使います。field/condの値を保ってID・条件名・maskの役割を明示します。旧Hydra例と旧CLIは廃止しました。

`offset_noise_30`は別の追加生成データで、各subsetに旧manifest.jsonがありません。通常loaderで自動推測せず、利用時にgeometry・条件・maskを確認した正規manifestを用意してください。現行形式に対応しない旧generatorは除去し、元配列は保持しています。現行のサンプル生成は`examples/create_tiny_dataset.py`と`examples/create_analysis_dataset.py`を使います。
