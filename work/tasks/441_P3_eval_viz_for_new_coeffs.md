# Task: 441 Enhance: 評価・可視化（新係数表現の診断、図を最小拡張）

- Priority: P2
- Status: done
- Depends on: 412, 413, 415
- Unblocks: 490

## Intent
新しい係数表現（FFT mag/phase、Wavelet、SH/Slepian等）でも比較が成立するように
評価・可視化を最小限拡張する（“何が違うか” を現場が理解できるようにする）。

## Context / Constraints
- 既存の評価プロトコルを壊さない（docs/09）
- 追加は “本当に必要な最小” に留め、図の数を増やしすぎない
- 出力は `figures/` `tables/` に固定

## Plan
- [ ] coeff診断の共通関数を追加（分布、スペクトル、上位モード寄与）
- [ ] FFTの場合は magnitude スペクトルを標準で出せるようにする（codecに依存せず）
- [ ] Waveletは level別エネルギー（band energy）を出す（簡単に）
- [ ] SH/Slepianは l別エネルギー/集中度を図示できるようにする
- [ ] docs: 可視化の読み方を短く追加（非DS向け）

## Acceptance Criteria
- [x] FFT/Wavelet/SH の少なくとも1つで新しい図が生成される
- [x] 図が多すぎない（1 run あたり 5〜10枚程度目安）
- [x] 既存の評価（field/coeff誤差）は維持される

## Verification
- 任意のrunで `figures/` を確認し、診断図が増えている

## Review Map（必須）
- 変更ファイル一覧: `src/mode_decomp_ml/viz/__init__.py`, `src/processes/viz.py`, `configs/viz/basic.yaml`, `docs/09_EVALUATION_PROTOCOL.md`, `tests/test_processes_e2e.py`
- 重要な関数/クラス: `src/processes/viz.py:main`（diagnostics出力）, `src/mode_decomp_ml/viz/__init__.py:coeff_energy_vector`, `src/mode_decomp_ml/viz/__init__.py:fft_magnitude_spectrum`, `src/mode_decomp_ml/viz/__init__.py:wavelet_band_energy`, `src/mode_decomp_ml/viz/__init__.py:spherical_l_energy`, `src/mode_decomp_ml/viz/__init__.py:slepian_concentration`
- 設計判断: 既存の`coeff_spectrum`に加え「分布/上位寄与/方式別」の最小セットを追加し、出力は`figures/`のみ維持。方式固有は`coeff_meta`由来の判定で自動出力。
- リスク/注意点: 係数メタの`coeff_shape`/`complex_format`が不整合だと診断がスキップ/失敗するため、事前のメタ保存が前提。FFT/Wavelet/SH/Slepian以外では方式別図は出力されない。
- 検証コマンドと結果: `python3 -m pytest tests/test_processes_e2e.py`（pass）。FFTスモーク: `PYTHONPATH=src python3 - <<'PY' ... PY` で `fft_magnitude_spectrum.png` の生成を確認。
- 削除一覧: なし
