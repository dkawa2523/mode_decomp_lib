# Task: 416 Add: Decomposer Slepian（sphere_grid, state保存, ROI集中基底）

- Priority: P2
- Status: done
- Depends on: 410, 415
- Unblocks: 440, 490

## Intent
Slepian（領域集中基底）を sphere_grid 向けに追加し、ROI集中・局所反応/欠陥に対して
少数成分で表現できる可能性を比較基盤に入れる。

## Context / Constraints
- Slepian は region に依存するため `fit()` が必要（state保存が必須）
- optional dependency: pyshtools を推奨
- 固有基底系として EigenBasisDecomposerBase に乗せる（コピペ禁止）

## Plan
- [ ] decomposer: `spherical_slepian` を追加（l_max, region_mask/cap, k）
- [ ] state: eigenvalues/eigenvectors/region_spec を `states/` に保存
- [ ] codec: `slepian_pack_v1`（k次元、集中度もmetaへ）
- [ ] tests: basis集中度の妥当性（eigenvalues）、roundtrip
- [ ] docs更新: region指定方法（dataset mask or config）

## Acceptance Criteria
- [ ] spherical_slepian が registry で選べる（依存あり環境）
- [ ] state が保存/ロードでき、再実行で同じbasisを使える
- [ ] roundtrip テストが通る（許容誤差内）
- [ ] 依存が無い環境では明確にガイドする

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_sphere_slepian_ridge.yaml`
- Expected:
  - `states/slepian_basis.npz` 等が出力される

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/plugins/decomposers/sphere/spherical_slepian.py`, `src/mode_decomp_ml/plugins/decomposers/sphere/__init__.py`, `src/mode_decomp_ml/plugins/decomposers/__init__.py`, `src/mode_decomp_ml/plugins/codecs/slepian_pack.py`, `src/mode_decomp_ml/plugins/codecs/__init__.py`, `src/mode_decomp_ml/domain/__init__.py`, `configs/decompose/spherical_slepian.yaml`, `configs/codec/slepian_pack_v1.yaml`, `examples/run_sphere_slepian_ridge.yaml`, `tests/test_decompose_spherical_slepian.py`, `tests/test_codec_roundtrip.py`, `tests/test_domain.py`, `docs/11_PLUGIN_REGISTRY.md`, `docs/20_METHOD_CATALOG.md`
- 重要な関数/クラス: `src/mode_decomp_ml/plugins/decomposers/sphere/spherical_slepian.py:SphericalSlepianDecomposer`（fit/transform/inverse/state保存）, `src/mode_decomp_ml/plugins/codecs/slepian_pack.py:SlepianPackCodecV1`
- 設計判断: sphere_grid 上で Slepian 基底を SciPy の球面調和＋重み付き直交化で構築し、region は cap または dataset/domain mask で指定。mask が無い場合は重み付き射影、mask 有りは最小二乗で係数推定。
- リスク/注意点: region mask が小さすぎる／grid が粗いと Gram が不安定になる可能性。backend=pyshtools は未導入環境で明確にエラーを返す。
- 検証コマンドと結果: `python3 -m pytest tests/test_codec_roundtrip.py tests/test_domain.py tests/test_decompose_spherical_slepian.py`（pass）。`python3 -m mode_decomp_ml.run --config examples/run_sphere_slepian_ridge.yaml` は `processes.train` モジュール未検出で失敗（既存CLI構成に依存）。
- 削除一覧: なし
