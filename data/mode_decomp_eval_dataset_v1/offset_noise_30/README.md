# offset_noise_30 (per-domain testsets)

各ドメイン向けに、以下の条件を満たすテストデータを生成しています。

- 条件数: 約30
- オフセット成分: 各サンプルで一定値
- ふきゅう(ゆらぎ)成分: オフセットの約10%
- ノイズ: オフセットに対して空間位置ごとに約1%

## サブセット
- scalar_rect
- scalar_disk
- scalar_annulus
- scalar_mask
- scalar_sphere
- vector_rect
- vector_disk
- vector_annulus
- vector_mask
- vector_sphere

各サブセットは以下のファイルを含みます:
- `cond.npy` / `field.npy` / `mask.npy`
- `conditions.csv`（scalar_*: id + x1..x4, vector_*: id + x1..x8）
- `fields/`（1条件1ファイル、x,y,f）
  - vector_* は `*_fx.csv` / `*_fy.csv` の2ファイル構成

生成スクリプト: `tools/generate_offset_noise_testsets.py`
