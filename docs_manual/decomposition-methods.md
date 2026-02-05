# モード分解手法

## 分解前の空間補完
- 欠損・不等間隔・マスク領域は、ドメインに応じて補完・無効化を行います。
- disk/annulus/sphere などは専用の基底・座標変換を用います。

## スカラー/ベクトルの扱い
- スカラー場: `f` のみ
- ベクトル場: `fx`, `fy` をチャンネルとして扱い、分解器によりチャネル単位で処理

## 解析関数系（代表）
| 手法 | 用途 | ドメイン | 強み |
| --- | --- | --- | --- |
| Zernike | 円盤 | disk | 円対称の分解に強い |
| Annular Zernike | 環状 | annulus | 環状境界に適合 |
| Spherical Harmonics | 球面 | sphere_grid | 球面基底で表現 |
| Slepian | 球面 | sphere_grid | 局所領域に強い |
| Fourier-Bessel / PSWF | 円盤系 | disk | 周波数制御が容易 |

## データ駆動系（代表）
| 手法 | 用途 | ドメイン | 強み |
| --- | --- | --- | --- |
| POD / POD-SVD | 汎用 | all | 安定・再構成誤差が低い |
| Gappy POD | 欠損対応 | arbitrary_mask | 欠損に強い |
| Dictionary Learning | 非線形 | all | スパース表現 |
| Autoencoder | 非線形 | all | 非線形表現 |

## ドメイン別の推奨
| ドメイン | 代表分解 | 想定用途 |
| --- | --- | --- |
| rectangle | FFT/DCT/POD | 一般的な格子分布 |
| disk | Zernike / Fourier-Bessel | 円盤領域 |
| annulus | Annular Zernike | 環状領域 |
| arbitrary_mask | POD / Gappy POD | 欠損・複雑形状 |
| sphere_grid | Spherical Harmonics / Slepian | 球面分布 |
