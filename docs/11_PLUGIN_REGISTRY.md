# Plugin Registry（拡張規約）

拡張されやすいポイントは必ずプラグイン化し、coreの変更を最小化する。

---

## 1. 対象
- preprocess: `PreprocessOp`
- vector: `VectorTransform`
- decompose: `Decomposer`
- codec: `CoeffCodec`
- coeff_post: `CoeffPost`
- model: `Regressor`
- metrics: `MetricFn`（必要なら）

---

## 1.1 配置（plugins 配下）
プラグインは `src/mode_decomp_ml/plugins/` に集約する。

```
src/mode_decomp_ml/plugins/
  registry.py
  decomposers/
    base.py
    fft_dct.py
    wavelet2d.py
    pswf2d_tensor.py
    zernike_decomposer.py  # zernike/ パッケージとの衝突回避のため名称変更
    annular_zernike.py
    zernike_shared.py
    fourier_bessel.py
    spherical_harmonics.py
    spherical_slepian.py
    graph_fourier.py
    laplace_beltrami.py
    helmholtz.py
    autoencoder.py
    dict_learning.py
    pod.py
    pod_svd.py
    gappy_pod.py
  codecs/
    basic.py
    fft_complex.py
    zernike_pack.py
    wavelet_pack.py
    sh_pack.py
  coeff_post/basic.py
  models/base.py
  models/gbdt.py
  models/mtgp.py
  models/sklearn.py
```

  - 新規実装・参照は `mode_decomp_ml.plugins.*` を使う。

---

## 2. 現在のプラグイン一覧（domain/コスト/用途）
### 2.1 Domain compatibility（Decomposer）
| decomposer | rectangle | disk | annulus | mask/arbitrary_mask | sphere_grid | mesh | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fft2 | ok | ok* | - | - | - | - | disk_policy=mask_zero_fill で disk 可 |
| dct2 | ok | ok* | - | - | - | - | disk_policy=mask_zero_fill で disk 可 |
| wavelet2d | ok | ok* | - | ok* | - | - | mask_policy=zero_fill, requires pywt |
| zernike | - | ok | - | - | - | - | disk only |
| annular_zernike | - | - | ok | - | - | - | annulus only |
| fourier_bessel | - | ok | - | - | - | - | disk only |
| graph_fourier | ok | ok | ok | ok | - | - | grid graph, 固定mask推奨 |
| pod | ok | ok | ok | ok | - | - | mask_policy=ignore_masked_points で固定mask |
| pod_svd | ok | ok | ok | ok | - | - | mask_policy=ignore_masked_points で固定mask |
| dict_learning | ok | ok | ok | ok | - | - | 固定mask推奨, iterative |
| autoencoder | ok | ok | ok | ok | - | - | requires torch, mask_policy=zero_fill |
| helmholtz | ok | - | - | ok | - | - | vector field only |
| spherical_harmonics | - | - | - | - | ok | - | requires pyshtools |
| spherical_slepian | - | - | - | - | ok | - | region mask/cap, pyshtools optional |
| laplace_beltrami | - | - | - | - | - | ok | mesh only |

Notes:
- `ok*` はポリシー設定（`disk_policy` / `mask_policy`）が必須。
- `configs/decompose/` は各 decomposer の最小設定を用意済み。新規追加時は同様に config を追加すること。
- Optional dependencies の一覧は `docs/14_OPTIONAL_DEPENDENCIES.md` を参照。
- 運用上の推奨パラメータは `docs/24_DECOMPOSER_RECOMMENDATIONS.md` を参照。

### 2.2 Cost & recommended use（Decomposer）
| decomposer | cost | recommended use |
| --- | --- | --- |
| fft2 | low | 周期性/全体構造のベースライン |
| dct2 | low | 端点で滑らかな場、圧縮ベース |
| wavelet2d | medium | 局所パターン/マルチスケール |
| zernike | medium | disk 上の滑らかな放射状構造 |
| annular_zernike | medium | ring/annulus 上の構造 |
| fourier_bessel | medium-high | disk 上の放射振動 |
| graph_fourier | high | 不規則mask/トポロジ依存 |
| pod | medium-high | 低ランク・固定maskのデータ |
| pod_svd | medium-high | 低ランク・固定maskのデータ |
| dict_learning | high | 疎表現・局所反復パターン |
| autoencoder | high | 非線形パターン, GPU推奨 |
| helmholtz | medium | ベクトル場のdiv/curl分離 |
| spherical_harmonics | high | 球面データの全球基底 |
| spherical_slepian | high | ROI集中・局所欠陥の低次表現 |
| laplace_beltrami | high | 曲面/mesh の固有基底 |

### 2.3 CoeffCodec catalog
- `none`: lossless flatten（real-valued coeff 用）
- `fft_complex_codec_v1`: FFT 複素係数（real/imag or mag/phase or logmag/phase）
- `zernike_pack_v1`: Zernike/Annular Zernike 係数の lossless pack
- `wavelet_pack_v1`: wavedec2 係数の lossless pack（wavelet2d 用）
- `sh_pack_v1`: spherical harmonics 係数の lossless pack
- `slepian_pack_v1`: Slepian 係数の lossless pack（集中度メタ付き）

### 2.4 CoeffPost catalog
- `none`: 何もしない
- `standardize`: 平均0/分散1（実装済み、config未整備）
- `quantile`: 外れ値に強い（逆変換は端で近似）
- `power_yeojohnson`: 歪度補正（逆変換可）
- `pca`: 次元削減（直交基底）
- `dict_learning`: 疎辞書（実装済み、config未整備）

### 2.5 Model catalog
- `ridge`: baseline（multi-output）
- `elasticnet`: 疎性+L2（実装済み、config未整備）
- `multitask_elasticnet`: 共有疎性（実装済み、config未整備）
- `multitask_lasso`: 共有疎性（configあり）
- `gpr`: Gaussian Process（実装済み、config未整備）
- `mtgp`: Multi-task GP（gpytorch optional, 出力相関を学習・高コスト）
- `xgb` / `lgbm` / `catboost`: GBDT（optional dependency）

---

## 3. インターフェース（例）
### Decomposer
- `fit(dataset, domain_spec) -> self`
- `transform(field, mask, domain_spec) -> coeff`
- `inverse_transform(coeff, domain_spec) -> field_hat`
- `coeff_meta()` を提供（index対応）

### CoeffCodec
- `encode(raw_coeff, raw_meta) -> vector_coeff`
- `decode(vector_coeff, raw_meta) -> raw_coeff`
- `coeff_meta(raw_meta, vector_coeff)` を提供

### CoeffPost
- `fit(A_train) -> self`
- `transform(A) -> Z`
- `inverse_transform(Z) -> A_hat`

### Regressor
- `fit(X_cond, Y) -> self`
- `predict(X_cond) -> Y_hat`
- `save/load`（artifact契約に従う）

---

## 4. registry の原則
- registry key は config の `*.name`（互換で `*.method` も許容）と一致
- 未登録 key で実行したら即エラー
- 追加時は必ずテスト（少なくとも smoke）を追加

---

## 5. 追加手順（テンプレ）
最短で以下を揃える（README 追記は必要最低限）。
1) 実装: `src/mode_decomp_ml/plugins/<type>/...` に追加し `@register_*("<name>")` を付与
2) config: `configs/<type>/<name>.yaml` を追加（`name: <name>` + 必須パラメータ）
3) tests: `tests/` に smoke を追加（encode/decode or transform/inverse）
4) sweep: 必要なら `scripts/bench/matrix.yaml` に登録（optional は `optional: true` + `requires`）

Template（Decomposer + config）:
```python
from mode_decomp_ml.plugins.registry import register_decomposer

@register_decomposer("my_method")
class MyDecomposer(BaseDecomposer):
    def fit(self, dataset=None, domain_spec=None):
        return self
    def transform(self, field, mask, domain_spec):
        ...
    def inverse_transform(self, coeff, domain_spec):
        ...
```
```yaml
# configs/decompose/my_method.yaml
name: my_method
# required params...
```

---

## 6. Hydra との統合
- config から `hydra.utils.instantiate` でプラグインを生成できる形にする
- constructor には config の params を渡す（kwargs）
