# ドメインごとのテストデータ

## 目的

- domain×手法で、破綻しやすい点（mask、境界、重み、複素係数）を早期に検出する。
- ベンチで method 間比較ができる形にする。

## 生成コマンド（ベース）

```bash
PYTHONPATH=src python3 tools/bench/generate_benchmark_datasets_v1.py --help
```

## ケース一覧（テンプレ）

ベンチ（v1）は「offset 優勢 + 低周波揺らぎ + 小ノイズ」の合成データで、domain ごとの差を観察しやすいように設計しています。

- データ生成: `tools/bench/generate_benchmark_datasets_v1.py`
- ベンチ実行/集計: `tools/bench/run_benchmark_v1.py`

代表ケース（v1）:

| case | domain | scalar/vector | 想定問題設定 | 比較する手法 |
|---|---|---|---|---|
| rectangle_scalar | rectangle | scalar | offset+低周波+ノイズ | fft2/dct2/pod/... |
| rectangle_vector | rectangle | vector | offset(u,v)+低周波+ノイズ | helmholtz/pod_joint/... |
| disk_scalar | disk | scalar | disk mask + polar patterns | zernike/fourier_bessel/... |
| disk_vector | disk | vector | disk mask + vector patterns | graph_fourier/polar_fft/... |
| annulus_scalar | annulus | scalar | annulus mask + polar patterns | annular_zernike/polar_fft/... |
| annulus_vector | annulus | vector | annulus mask + vector patterns | graph_fourier/gappy_*... |
| arbitrary_mask_scalar | arbitrary_mask | scalar | 不規則 mask + 欠損 | gappy_graph_fourier/rbf/pod_em |
| arbitrary_mask_vector | arbitrary_mask | vector | 不規則 mask + 欠損 | gappy_graph_fourier/rbf/pod_joint_em |
| sphere_grid_scalar | sphere_grid | scalar | lon/lat 低次パターン | spherical_harmonics/slepian |
| sphere_grid_vector | sphere_grid | vector | lon/lat 低次ベクトル | spherical_harmonics/slepian |

注意:

- disk/annulus/arbitrary_mask は **mask 外を評価に含めない**（`field_r2` の水増し防止）。
- “手法の前提” があるもの（Slepian 系など）は、前提に合うテストケースを別途追加する価値があります（ROI/帯域制限、境界テーパ等）。
