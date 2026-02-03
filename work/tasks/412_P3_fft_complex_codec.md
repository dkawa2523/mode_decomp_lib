# Task: 412 Add: FFT複素係数Codec（real/imag・mag/phase・logmag/phase）

- Priority: P1
- Status: todo
- Depends on: 410, 411
- Unblocks: 440

## Intent
FFT系の複素係数表現を codec として統一し、学習・保存・可視化が一貫するようにする。
（real/imag / mag/phase / logmag/phase を選択可能）

## Context / Constraints
- FFT decomposer は raw_coeff として複素ndarrayを返す（codecで表現変換）
- lossless mode（real/imag）は必ず roundtrip を保証
- mag/phase系は位相の扱い（unwrap/clip）をmetaに明示する

## Plan
- [ ] `fft_complex_codec_v1` を実装（plugins/codecs）
- [ ] mode切替: `real_imag`, `mag_phase`, `logmag_phase`
- [ ] 可視化: スペクトル診断（magnitude）を standard に含める（必要なら）
- [ ] unit test: lossless mode の `decode(encode(x))≈x`
- [ ] config例: run.yaml で codec mode を指定できる

## Acceptance Criteria
- [ ] FFT + codec(real_imag) で reconstruct が従来と同等に動く
- [ ] lossless roundtrip テストが通る
- [ ] mag/phase系は meta に表現ルールが保存される

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_rect_fft_ridge.yaml`
- Expected:
  - `states/coeff_meta.json` に codec mode が記録される

## Review Map
- 変更ファイル一覧: `src/mode_decomp_ml/plugins/codecs/fft_complex.py`, `src/mode_decomp_ml/plugins/codecs/__init__.py`, `src/mode_decomp_ml/plugins/decomposers/grid/fft_dct.py`, `src/mode_decomp_ml/evaluate/__init__.py`, `src/mode_decomp_ml/viz/__init__.py`, `src/processes/viz.py`, `configs/codec/fft_complex_codec_v1.yaml`, `examples/run_scalar_rect_fft_ridge.yaml`, `tests/test_codec_roundtrip.py`, `tests/test_decompose_fft_dct.py`
- 重要な関数/クラス: `FFTComplexCodecV1.encode/decode/coeff_meta`（`src/mode_decomp_ml/plugins/codecs/fft_complex.py`）, `FFT2Decomposer._forward_fft/_inverse_fft`（`src/mode_decomp_ml/plugins/decomposers/grid/fft_dct.py`）, `_coeff_energy_vector`（`src/mode_decomp_ml/evaluate/__init__.py`）, `coeff_energy_spectrum`（`src/mode_decomp_ml/viz/__init__.py`）, `_normalize_k_list`（`src/processes/viz.py`）
- 設計判断: FFT2はcomplex生係数を返し、表現変換はcodecに集約。codec metaにraw_metaを保持しつつvector側のcoeff_shape/complex_formatを更新し、mag/phase系の位相ルールを明示。スペクトル/energyはmag成分で評価するよう補正。
- リスク/注意点: FFT2+codec=noneは不可になるためrun.yamlでcodec指定が必要。mag/phase/logmag_phaseはlossyで、phase_unwrapは最後の2軸のみ適用。
- 検証コマンドと結果: `pytest tests/test_codec_roundtrip.py tests/test_decompose_fft_dct.py` ✅（5 tests passed）
- 削除一覧: なし
