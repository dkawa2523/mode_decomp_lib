# Task: 417 Add: Decomposer PSWF（pswf2d_tensor, research track, optional dependency）

- Priority: P2
- Status: done
- Depends on: 410, 398
- Unblocks: 440, 490

## Intent
PSWF（Prolate Spheroidal, スフェロイダル）を “研究枠” として導入する。
ただし実装負荷が高いので、まずは rectangle 向けの `pswf2d_tensor` として最小導入し、
スパゲッティ化を避ける。

## Context / Constraints
- optional dependency（SciPy special 等）に依存
- 近似や数値安定性が課題になりやすいので、roundtripの許容誤差・限界を明示する
- eigenbasis系へ拡張する道（将来）を docs に残す

## Plan
- [x] decomposer: `pswf2d_tensor`（1D PSWFの外積）を実装
- [x] codec: `tensor_pack_v1`
- [x] tests: toyのroundtrip（誤差閾値を明示）
- [x] docs/23 に「研究枠」「制約」「推奨用途」を明記
- [x] benchmarkには “任意” 追加（heavyなら off by default）

## Acceptance Criteria
- [x] registry に `pswf2d_tensor` が追加される（依存あり環境）
- [x] 実行例（dry-run含む）が docs にある
- [x] toy test で動作確認できる（誤差許容明記）

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_pswf_ridge.yaml --dry-run`
- Expected:
  - 依存未導入でもエラーメッセージが明確（導入済なら実行可）

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/plugins/decomposers/grid/pswf2d_tensor.py`, `src/mode_decomp_ml/plugins/codecs/tensor_pack.py`, `src/mode_decomp_ml/plugins/decomposers/__init__.py`, `src/mode_decomp_ml/plugins/codecs/__init__.py`, `src/mode_decomp_ml/domain/__init__.py`, `configs/decompose/pswf2d_tensor.yaml`, `configs/codec/tensor_pack_v1.yaml`, `examples/run_scalar_rect_pswf_ridge.yaml`, `docs/23_METHODS_ADDON_SPEC.md`, `docs/20_METHOD_CATALOG.md`, `scripts/bench/matrix.yaml`, `tests/test_decompose_pswf2d_tensor.py`, `tests/test_codec_roundtrip.py`, `tests/test_domain.py`
- 重要な関数/クラス: `PSWF2DTensorDecomposer.transform/inverse_transform/_get_basis`（tensor DPSS 基底）, `TensorPackCodecV1.encode/decode`
- 設計判断: 連続PSWFの代替としてDPSS外積を採用し、`c_x/c_y` を time-bandwidth として扱う; maskは明示的 `mask_policy` のみ許可; 係数順は `(y, x)` の row-major で meta に固定
- リスク/注意点: 研究枠の近似（フル基底以外は再構成誤差が出る）; rectangle以外は非対応; SciPy未導入時は明確エラー
- 検証コマンドと結果: `pytest tests/test_codec_roundtrip.py tests/test_decompose_pswf2d_tensor.py tests/test_domain.py`（pass）, `python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_pswf_ridge.yaml --dry-run`（dry-run 出力確認）
- 削除一覧: なし
