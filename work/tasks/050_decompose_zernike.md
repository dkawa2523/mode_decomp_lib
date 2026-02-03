# Decomposer: Zernike（legacy移植 or 最小実装）

**ID:** 050  
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
disk 領域向けの主力分解として Zernike を実装する。

優先:
1) `legacy/` に旧Zernike実装がある場合は **必要最小限だけ** `src/mode_decomp_ml/decompose/zernike.py` へ移植する
2) 旧実装が無い場合は、SciPy/既存依存のみで成立する最小実装を作る（高次最適化はしない）

必須:
- transform: field→a
- inverse: a→field_hat
- coeff_meta: (n,m) など次数情報、正規化、mask扱い

## Acceptance Criteria
- [x] disk domain で Zernike transform/inverse が動作する
- [x] 係数次元が設定で決められる（n_max 等）
- [x] coeff_meta に (n,m) の対応が残る

## Verification
- [x] 低次モードのみでの逐次再構成が可視化できる（k=1,2,4...）

## Review Map（必須）
- 変更ファイル一覧（追加/変更/削除）: `src/mode_decomp_ml/decompose/zernike.py`（追加）, `src/mode_decomp_ml/decompose/__init__.py`（更新）, `configs/decompose/zernike.yaml`（更新）, `tests/test_decompose_zernike.py`（追加）
- 重要な関数/クラス: `ZernikeDecomposer.transform`, `ZernikeDecomposer.inverse_transform`, `_build_nm_list`, `_zernike_mode`
- 設計判断: disk専用のZernike基底を `n_max` で固定し、`n_then_m` 順で係数化。離散格子の非直交性を考慮し、mask+weightsを使った最小二乗で係数推定する。正規化は `orthonormal` とし、`boundary_condition` を coeff_meta に記録。
- リスク/注意点: `n_max` が大きい場合は最小二乗が重く、valid点数が少ないとrank不足でエラー。coeffの並びは `nm_list` に依存するため比較時は必ず参照する。
- 検証コマンドと結果: `python -m pytest tests/test_decompose_zernike.py`（失敗: pytest未導入）, `python - <<'PY' ...`（tests内の関数を直接実行: ok）
