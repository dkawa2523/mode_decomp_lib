# scalar / disk サブセット

このフォルダには `scalar` 場（スカラー場）の synthetic データが入っています。

## 含まれるファイル
- `cond.npy` : cond: shape [N,8] （8個の基底パターン係数）
- `field.npy`: field: shape [N,H,W,1] （スカラー場）
- `mask.npy` : shape [N,H,W]（0/1, 1が有効領域）
- `conditions.csv` : 条件テーブル（id + x1..x8）
- `fields/` : 各条件の `x,y,f` CSV（1条件1ファイル）

## 生成方法（概要）
- 各サンプルは **固定の8個の空間パターン**の線形結合で作られています。
- `cond` はその線形結合係数（ベクトル場は vx,vy 別々の係数）です。
- `mask` 外（0領域）の field 値は 0 にしてあります。

## 目的
- 分布モード分解の再構成精度比較（FFT/Zernike/Bessel/POD/Graph...）
- `cond -> mode係数` 回帰モデル（Ridge/GPR/...）の学習・推論評価

## CSV 利用例
`dataset=csv_fields` と `dataset.conditions_csv` / `dataset.fields_dir` を指定してください。
