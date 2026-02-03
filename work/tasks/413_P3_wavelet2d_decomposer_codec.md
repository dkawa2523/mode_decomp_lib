# Task: 413 Add: Decomposer Wavelet2D + Codec（PyWavelets, lossless pack/unpack）

- Priority: P1
- Status: done
- Depends on: 410
- Unblocks: 440, 490

## Intent
2D Wavelet 分解（局所×スケール）を追加し、局所欠陥・多スケール構造に強い基底を比較基盤に入れる。

## Context / Constraints
- wavelet係数は階層構造なので codec が必須（wavelet_pack_v1）
- domainは当面 rectangle を正式対応とし、disk/maskは 0埋め＋mask評価で許容
- optional dependency: PyWavelets（未導入時は明確エラー or registryから外す）

## Plan
- [x] decomposer `wavelet2d` を追加（transform=inverseが可能）
- [x] codec `wavelet_pack_v1` を追加（flatten + metaにshape）
- [x] run.yaml 例を追加（scalar_rectでwavelet+ridge）
- [x] tests: lossless roundtrip（field->coeff->field）
- [x] docs/23 に wavelet の params と制約を追記

## Acceptance Criteria
- [x] wavelet2d が registry から選択できる
- [x] roundtrip テストが通る（PyWaveletsあり環境）
- [x] bench matrix に wavelet を追加できる（rectangleケース）

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_wavelet_ridge.yaml`
- Expected:
  - metrics.json が生成され、reconstruct が動く

## Review Map
- **変更ファイル一覧**: `src/mode_decomp_ml/plugins/decomposers/grid/wavelet2d.py`, `src/mode_decomp_ml/plugins/codecs/wavelet_pack.py`, `src/mode_decomp_ml/plugins/decomposers/__init__.py`, `src/mode_decomp_ml/plugins/codecs/__init__.py`, `src/mode_decomp_ml/domain/__init__.py`, `configs/decompose/wavelet2d.yaml`, `configs/codec/wavelet_pack_v1.yaml`, `examples/run_scalar_rect_wavelet_ridge.yaml`, `docs/23_METHODS_ADDON_SPEC.md`, `tests/test_decompose_wavelet2d.py`
- **重要な関数/クラス**: `src/mode_decomp_ml/plugins/decomposers/grid/wavelet2d.py:Wavelet2DDecomposer`, `src/mode_decomp_ml/plugins/codecs/wavelet_pack.py:WaveletPackCodecV1`
- **設計判断**: wavedec2 の階層係数を保持したまま codec 側で flatten/unflatten する構成にし、decomposer は raw_coeff 構造を meta に保存して比較可能性を確保。mask は明示的に `mask_policy` で zero_fill を許可する場合のみ 0 埋め。
- **リスク/注意点**: PyWavelets 未導入環境では wavelet2d 生成時に ImportError。mask を使う場合は `mask_policy=zero_fill` が必須。wavelet再構成が入力より大きい場合は grid shape にトリム。
- **検証コマンドと結果**: `python -m pytest tests/test_decompose_wavelet2d.py -q`（PyWavelets 未導入の場合は skip になる想定）
- **削除一覧**: なし
