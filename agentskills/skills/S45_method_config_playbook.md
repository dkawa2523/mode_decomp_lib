# S45: 手法設定プレイブック（decompose / codec / coeff_post / model）

## 目的
- ドメインとデータ特性に合う `decompose` を選び、`codec`/`coeff_post`/`model` を破綻しない組み合わせで設定する。
- 「動く」だけでなく、比較可能な設定に寄せる。

## このスキルを使う場面
- `decompose` を何にするか決めたい。
- `decompose` は選んだが `codec` や `disk_policy` で失敗する。
- 別環境で Copilot に手法設定を任せると、組み合わせ不整合が起きる。

## 手順
1. domain で候補を絞る（必須）。
- `rectangle`: `dct2`, `fft2`, `fft2_lowpass`, `pswf2d_tensor`, `helmholtz`, `helmholtz_poisson`, `pod*`, `graph_fourier`, `rbf_expansion`, `wavelet2d`
- `disk`: `zernike`, `pseudo_zernike`, `fourier_bessel`, `fourier_jacobi`, `polar_fft`, `disk_slepian`, `pod*`, `graph_fourier`, `rbf_expansion`
- `annulus`: `annular_zernike`, `polar_fft`, `pod*`, `graph_fourier`, `rbf_expansion`
- `arbitrary_mask`: `pod*`, `gappy_graph_fourier`, `graph_fourier`, `rbf_expansion`, `wavelet2d`, `helmholtz`
- `sphere_grid`: `spherical_harmonics`, `spherical_slepian`
- `mesh`: `laplace_beltrami`

2. ハード制約を先に消す。
- `fft2` / `dct2` / `fft2_lowpass` を `disk` に使うなら `decompose.disk_policy=mask_zero_fill` が必須（`error` は失敗）。
- `pswf2d_tensor` は `rectangle` 専用。
- `annular_zernike` は `annulus` 専用。
- `laplace_beltrami` は `mesh` 専用。

3. `codec` を決める（迷ったら `auto`）。
- 推奨デフォルト: `codec.name=auto_codec_v1`。
- 複素係数（FFT系）や構造化係数（wavelet/offset_residual）を自動判定できる。
- `offset_residual` を使う場合は `auto_codec_v1` か `offset_residual_pack_v1` を使う。
- `codec.name=none` は「実数の単純配列係数のみ」に限定して使う。

4. `coeff_post` を決める。
- ベースラインは `coeff_post.name=none`。
- 次点は `pca`（高次元係数で有効）。
- 注意: `preprocessing` で一部手法に対する `pca` は自動で `none` にスキップされる実装がある。

5. `model` を決める。
- ベースライン: `model.name=ridge`。
- 非線形を試すなら `xgb/lgbm/catboost`（追加依存が必要）。
- 不確実性まで見るなら `gpr` / `mtgp`（依存と計算コストに注意）。

6. offset 優勢データでは `offset_split` を使う。
- 推奨開始値: `enabled=auto`, `f_offset=5.0`, `max_samples=128`。
- 係数圧縮比較 (`n_components_required`) が安定しやすい。

## 実運用の初手（ベンチ実績ベース）
- `rectangle_scalar`: `fft2` または `dct2`。圧縮重視なら `pod_svd`。
- `disk_scalar`: `pseudo_zernike` を第一候補、次に `zernike`/`fourier_jacobi`。
- `annulus_scalar`: `annular_zernike` を第一候補、次に `polar_fft`。
- `arbitrary_mask_scalar`: `gappy_graph_fourier_bench` または `rbf_expansion_k64`。
- `rectangle_vector`: `helmholtz`（比較として `pod_joint_em`）。
- `disk_vector`/`annulus_vector`/`arbitrary_mask_vector`: `pod_joint_em`, `gappy_graph_fourier_bench`, `rbf_expansion_k64`。
- `sphere_grid_*`: `spherical_harmonics_scipy_bench` または `spherical_slepian_scipy`。
- `mesh_scalar`: `laplace_beltrami`。

## 最小テンプレ（Hydra）
```bash
PYTHONPATH=src python -m mode_decomp_ml.cli.run \
  task=decomposition \
  dataset=npy_dir dataset.root=data/my_dataset dataset.mask_policy=allow_none \
  decompose=pseudo_zernike \
  codec=auto \
  coeff_post=none \
  model=ridge \
  output.name=trial_disk_scalar
```

## 事故りやすい点
- `domain` と `decompose` の不整合（例: `annular_zernike` を `disk` で実行）。
- `disk` で `fft2` 系を使うのに `disk_policy` を入れない。
- `codec=none` のまま複素/構造化係数を流して失敗。
- optional 依存（torch, pyshtools, xgboost など）未導入で実行。

## 実装の根拠（読む場所）
- `src/mode_decomp_ml/domain/__init__.py` (`validate_decomposer_compatibility`)
- `src/mode_decomp_ml/plugins/decomposers/*.py`
- `src/mode_decomp_ml/plugins/codecs/auto.py`
- `src/processes/preprocessing.py` (`_select_coeff_post`)
- `summary_benchmark.md`（2026-02-11 生成）
