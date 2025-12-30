# Decomposer: FFT2 + DCT2（transform/inverse/係数meta）

**ID:** 040  
**Priority:** P0  
**Status:** done  
**Depends on:** 030  
**Unblocks:** 060  

⚠️ **DO NOT CONTINUE**: このタスクは Acceptance Criteria / Verification を全て満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない（必要なら TODO と RFC にまとめて停止）。  

## Compactness & Review（必須）
- 最小差分（無関係な整形・過剰抽象化禁止）: docs/14
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15
- 不要になったコード/ファイルは削除: docs/16

---
## Description
矩形領域向けの分解として FFT2 と DCT2 を実装する。
目的は「最小で動く・比較可能な baseline」を作ること。

- transform: field → a（実数ベクトル）
  - FFT2は complex を real/imag に分解
- inverse: a → field_hat（未使用係数は0埋め）
- coeff_meta: flatten順、keepルール、complex扱い、shape を保持

## Acceptance Criteria
- [x] FFT2 が transform/inverse を持つ（合成波で復元できる）
- [x] DCT2 が transform/inverse を持つ
- [x] coeff_meta が artifact に保存される

## Verification
- [x] 合成正弦波入力で FFT2→inverse の再構成誤差が小さい

## Review Map（必須）
- 変更ファイル一覧（追加/変更/削除）: `src/mode_decomp_ml/decompose/__init__.py`（更新）, `tests/test_decompose_fft_dct.py`（追加）
- 重要な関数/クラス: `FFT2Decomposer`, `DCT2Decomposer`, `BaseDecomposer.save_coeff_meta`, `build_decomposer`
- 設計判断: FFTは`norm="ortho"`でreal/imag分解し、DCTは`dctn/idctn`を使用。maskは`disk_policy=mask_zero_fill`時のみ0埋めし、それ以外は明示エラーにして比較可能性を守る。
- リスク/注意点: 係数のflatten順は`coeff_meta`に固定（CHW/CHWRI + row-major）。`inverse_transform`は直前の`transform`で得たshapeに依存するため、別プロセスで復元する場合は`coeff_meta`の保存/復元が前提。
- 検証コマンドと結果: `pytest -q tests/test_decompose_fft_dct.py`（pass）
- 削除一覧: なし
