# vector / rect サブセット

このフォルダには `vector` 場（2Dベクトル場）の synthetic データが入っています。

## 含まれるファイル
- `cond.npy` : cond: shape [N,16] （vx用8係数 + vy用8係数）
- `field.npy`: field: shape [N,H,W,2] （ベクトル場：vx,vy）
- `mask.npy` : shape [N,H,W]（0/1, 1が有効領域）
- `conditions.csv` : 条件テーブル（id + x1..x16）
- `fields/` : 各条件の `x,y,f` CSV（1条件2ファイル: `_fx`, `_fy`）

## 生成方法（概要）
- 各サンプルは **固定の8個の空間パターン**の線形結合で作られています。
- `cond` はその線形結合係数（ベクトル場は vx,vy 別々の係数）です。
- `mask` 外（0領域）の field 値は 0 にしてあります。

## 目的
- 分布モード分解の再構成精度比較（FFT/Zernike/Bessel/POD/Graph...）
- `cond -> mode係数` 回帰モデル（Ridge/GPR/...）の学習・推論評価

## CSV 利用例（vector）
`dataset=csv_fields` で `field_components: [fx, fy]` を指定してください。
