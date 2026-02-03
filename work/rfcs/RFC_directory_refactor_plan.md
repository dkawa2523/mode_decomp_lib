# RFC: ディレクトリ再編（段階移行計画）

## 背景
- `src/mode_decomp_ml/plugins/` を中心に階層が深く、追加/修正の導線が分散している。
- 既存の alias / config / import を壊さずに整理するには段階的な移行が必要。

## 提案
段階的に “読みやすい入口” を追加し、既存パスは互換aliasで維持する。

### 目標（最終形の方向性）
- `src/mode_decomp_ml/plugins/decomposers/` を **フラット構成** に寄せ、深い階層を削減
- 既存の import パスは shim で維持（互換性維持）
- 開発者が触るべき起点を `docs/27_CODE_MAP.md` に明記
- 既存の config group 名や plugin 名は **変更しない**

### 段階移行の概要
1) **Phase 0: 影響調査**
   - 既存 import / config group / registry 参照の棚卸し
   - `docs/27_CODE_MAP.md` に現状導線を明記
2) **Phase 1: 集約入口の追加**
   - `plugins/decomposers/__init__.py` に整理された “一覧” を追加
   - 既存のサブディレクトリは残し、alias維持
3) **Phase 2: パスの整理**
   - 深い階層の主要ファイルを浅い階層へ移動（最終構成案は後述）
   - 移動前の場所には “import だけ残す” shim を配置
4) **Phase 3: ドキュメント/テスト更新**
   - `docs/27_CODE_MAP.md` と `docs/11_PLUGIN_REGISTRY.md` を更新
   - import alias の動作をテストで担保
5) **Phase 4: 段階的削除**
   - 移行完了後に shim を削除（RFCで削除期限を確定）

## 代替案
- 既存構造のまま docs だけ整備する  
  → 変更リスクは低いが、長期的な散逸は解決しない。

## 互換性/移行
- 既存の plugin 名、config group 名は **維持** する。
- 移動後も import alias を残し、破壊的変更を回避する。
- 影響範囲が広いため、Phaseごとにテスト実行を必須化する。

## Phase 2 最終構成案（具体）
対象: `src/mode_decomp_ml/plugins/decomposers/` 配下

### 変更後（案）
```
plugins/decomposers/
  __init__.py
  base.py
  autoencoder.py
  dict_learning.py
  gappy_pod.py
  pod.py
  pod_svd.py
  fft_dct.py
  wavelet2d.py
  pswf2d_tensor.py
  graph_fourier.py
  laplace_beltrami.py
  fourier_bessel.py
  zernike_decomposer.py
  annular_zernike.py
  zernike_shared.py
  helmholtz.py
  spherical_harmonics.py
  spherical_slepian.py
  legacy_shims/
    analytic/__init__.py
    grid/__init__.py
    eigen/__init__.py
    data_driven/__init__.py
    bessel/__init__.py
    sphere/__init__.py
    vector/__init__.py
    zernike/__init__.py
```

### 移動/alias マッピング
- `plugins/decomposers/data_driven/autoencoder.py` → `plugins/decomposers/autoencoder.py`
- `plugins/decomposers/data_driven/dict_learning.py` → `plugins/decomposers/dict_learning.py`
- `plugins/decomposers/data_driven/gappy_pod.py` → `plugins/decomposers/gappy_pod.py`
- `plugins/decomposers/data_driven/pod.py` → `plugins/decomposers/pod.py`
- `plugins/decomposers/data_driven/pod_svd.py` → `plugins/decomposers/pod_svd.py`
- `plugins/decomposers/grid/fft_dct.py` → `plugins/decomposers/fft_dct.py`
- `plugins/decomposers/grid/wavelet2d.py` → `plugins/decomposers/wavelet2d.py`
- `plugins/decomposers/grid/pswf2d_tensor.py` → `plugins/decomposers/pswf2d_tensor.py`
- `plugins/decomposers/eigen/graph_fourier.py` → `plugins/decomposers/graph_fourier.py`
- `plugins/decomposers/eigen/laplace_beltrami.py` → `plugins/decomposers/laplace_beltrami.py`
- `plugins/decomposers/bessel/fourier_bessel.py` → `plugins/decomposers/fourier_bessel.py`
- `plugins/decomposers/zernike/zernike.py` → `plugins/decomposers/zernike_decomposer.py`
- `plugins/decomposers/zernike/annular_zernike.py` → `plugins/decomposers/annular_zernike.py`
- `plugins/decomposers/zernike/shared.py` → `plugins/decomposers/zernike_shared.py`
- `plugins/decomposers/vector/helmholtz.py` → `plugins/decomposers/helmholtz.py`
- `plugins/decomposers/sphere/spherical_harmonics.py` → `plugins/decomposers/spherical_harmonics.py`
- `plugins/decomposers/sphere/spherical_slepian.py` → `plugins/decomposers/spherical_slepian.py`

### Shim 方針
- 旧パスの `__init__.py` は新パスを re-export するだけに変更。
- 旧モジュール名（例: `plugins.decomposers.grid.fft_dct`）は shim によって維持。

## リスク
- パスの移動に伴う循環importや暗黙の依存が顕在化する可能性
- 互換aliasの維持期間が長引くと管理が複雑化

## 決定したいこと
### 合意済み（実装反映）
- Phase 2での最終的なディレクトリ構成案（下記「Phase 2 最終構成案」）
- shim維持の前提（旧パス互換は維持、import shim を残す）

### TODO（合意が必要）
### 合意済み

- shimの即時削除を実施（期限到来済みとして扱う）
  - 外部依存が無いことを確認済みとして、旧shimを削除
- shim維持期間と削除期限
  - Phase 2完了後 **最低2リリース** または **60日** のいずれか長い方まで維持
  - 期限到来前に削除タスクを起票し、shim削除は別タスクで実施
- テスト・ドキュメント更新の必須範囲
  - `tests/test_plugin_shims.py` + 主要 decomposer の代表テスト
  - `docs/11`/`docs/27` を更新
