# scalar / sphere_grid サブセット

このフォルダには sphere_grid ドメイン用のスカラー場データが入っています。

## 含まれるファイル
- `cond.npy` : cond: shape [N,8]
- `field.npy`: field: shape [N,H,W,1]
- `mask.npy` : shape [H,W]（全て 1）
- `conditions.csv` : 条件テーブル（id + x1..x8）
- `fields/` : 各条件の `x,y,f` CSV（1条件1ファイル、x=lon, y=lat）

## 生成方法（概要）
- 緯度経度の固定 8 パターンの線形結合。

## CSV 利用例
`dataset=csv_fields` と `dataset.conditions_csv` / `dataset.fields_dir` を指定してください。
