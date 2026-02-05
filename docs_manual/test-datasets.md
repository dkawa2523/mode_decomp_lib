# ドメイン別テストデータ

## データセット一覧
| サブセット | ドメイン | 種別 | 概要 |
| --- | --- | --- | --- |
| scalar_rect | rectangle | スカラー | 矩形格子の基準データ |
| scalar_disk | disk | スカラー | 円盤領域 |
| scalar_annulus | annulus | スカラー | 環状領域 |
| scalar_mask | arbitrary_mask | スカラー | 任意マスク |
| scalar_sphere | sphere_grid | スカラー | 球面格子 |
| vector_rect | rectangle | ベクトル | fx/fy |
| vector_disk | disk | ベクトル | fx/fy |
| vector_annulus | annulus | ベクトル | fx/fy |
| vector_mask | arbitrary_mask | ベクトル | fx/fy |
| vector_sphere | sphere_grid | ベクトル | fx/fy |
| offset_noise_30/* | all | スカラー/ベクトル | オフセット+ゆらぎ+ノイズ | 

## 生成コマンド
```bash
.venv/bin/python tools/generate_offset_noise_testsets.py
```

## テストケース（例）
| ケース | 想定問題 | 解決方法 |
| --- | --- | --- |
| disk + zernike | 円盤分解 | Zernike 基底 |
| annulus + annular_zernike | 環状分解 | Annular Zernike |
| sphere + spherical_harmonics | 球面分解 | 球面調和関数 |
| mask + pod_svd | 欠損領域 | POD + マスク処理 |
