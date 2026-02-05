# 設定ファイルの説明

## run.yaml（トップレベル）
| 変数名 | 型 | 内容 | 既定 | Tips |
| --- | --- | --- | --- | --- |
| task | str | 実行タスク | pipeline | decomposition/preprocessing/train/inference 等 |
| dataset | mapping | 入力データ設定 | なし | CSV または NPY を指定 |
| pipeline | mapping | decomposer/coeff_post/model の指定 | なし | pipeline 実行時のみ使用 |
| output.root | str | 出力ルート | runs | 任意ディレクトリに変更可 |
| output.name | str | プロジェクト名 | default | runs/<name>/<process> に出力 |
| seed | int | 乱数 | 123 | 再現性確保 |
| params.decompose | mapping | 分解パラメータ | なし | decomposer に依存 |
| params.model | mapping | 学習パラメータ | なし | model に依存 |

## dataset（CSV 形式）
| 変数名 | 型 | 内容 | 既定 | Tips |
| --- | --- | --- | --- | --- |
| conditions_csv | str | 条件CSV | 必須 | id 列必須 |
| fields_dir | str | 分布CSV格納 dir | 必須 | `<id>.csv` or `<id>_fx.csv/_fy.csv` |
| id_column | str | ID列名 | id | 任意名可 |
| field_components | list[str] | ベクトル成分 | なし | [fx, fy] でベクトル |
| grid.H / grid.W | int | 格子サイズ | 必須 | データと一致必須 |
| grid.x_range / y_range | [float,float] | 範囲 | 必須 | ドメインと整合 |
| mask_file | str | マスク | 任意 | 2D or 3D mask.npy |

## inference（基本）
| 変数名 | 型 | 内容 | 既定 | Tips |
| --- | --- | --- | --- | --- |
| mode | str | 推論モード | single | single/batch/optimize |
| sampler | str | Optuna sampler | tpe | tpe/cmaes/random/grid |
| n_trials | int | 試行数 | 50 | yaml で設定 |
| objective.name | str | 目的関数 | field_std | 単目的最適化 |
