# Task: 415 Add: Domain sphere_grid + Decomposer Spherical Harmonics + Codec

- Priority: P2
- Status: done
- Depends on: 410, 398
- Unblocks: 416, 440

## Intent
球面調和関数（Spherical Harmonics）を追加し、球面（または球面近似）の曲面データに対して
標準的・高速な基底展開を可能にする。

## Context / Constraints
- mesh+LB に比べ、sphere_grid は “球面格子” に限定して簡素に導入する
- optional dependency（pyshtools 等）。未導入時は registry から除外 or 明確エラー
- codec `sh_pack_v1` で (l,m) 順序を固定

## Plan
- [x] domain: `sphere_grid`（n_lat,n_lon,radius,座標）を追加（manifest対応）
- [x] decomposer: `spherical_harmonics` を追加（l_max, norm, real_form）
- [x] codec: `sh_pack_v1` を追加
- [x] tests: 低次Y_lmの再現、roundtrip
- [x] docs/23 更新 + optional dependency の導入方法を docs に追記

## Acceptance Criteria
- [x] sphere_grid domain が manifest で指定できる
- [x] spherical_harmonics が選択でき、reconstruct が通る
- [x] roundtrip テスト（許容誤差内）が通る
- [x] optional dependency が無い場合は “分かりやすいエラー” を出す

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_sphere_sh_ridge.yaml --dry-run`
- Expected:
  - dependency未導入でも原因が明確（導入済なら実行可）
