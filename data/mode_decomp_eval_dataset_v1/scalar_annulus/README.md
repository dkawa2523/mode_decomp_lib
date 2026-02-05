# scalar / annulus サブセット

このフォルダには annulus ドメイン用のスカラー場データが入っています。

## 含まれるファイル
- `cond.npy` : cond: shape [N,8]
- `field.npy`: field: shape [N,H,W,1]
- `mask.npy` : shape [H,W]（0/1, annulus 内のみ 1）
- `conditions.csv` : 条件テーブル（id + x1..x8）
- `fields/` : 各条件の `x,y,f` CSV（1条件1ファイル）

## 生成方法（概要）
- annulus 内部の固定 8 パターンの線形結合。
- 外側は 0 に固定。

## CSV 利用例
`dataset=csv_fields` と `dataset.conditions_csv` / `dataset.fields_dir` を指定してください。
