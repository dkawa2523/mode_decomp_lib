# データセット仕様

## 取得方法
- `data/mode_decomp_eval_dataset_v1` にテスト用データセットを格納
- 新規生成は `tools/generate_offset_noise_testsets.py` を使用

## CSV 形式（最重要）
### 条件CSV
- 必須列: `id`
- それ以外は任意列名
- 目的変数列は yaml 側で指定

### 分布CSV
- 1条件1ファイル
- ファイル名: `<id>.csv`
- 列名: `x,y,f`

### ベクトル場
- 1条件につき `*_fx.csv` / `*_fy.csv`
- dataset.field_components: `[fx, fy]`

## 欠損・不等間隔
- 混在を許可
- 既存コードの補完/マスク処理を使用
