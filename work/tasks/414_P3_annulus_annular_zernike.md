# Task: 414 Add: Domain annulus + Decomposer Annular Zernike（Zernikeコード共有）

- Priority: P1
- Status: done
- Depends on: 410, 399
- Unblocks: 440, 490

## Intent
Annular Zernike（環状ゼルニケ）を追加し、穴あき円板/リング状観測に対して解釈性の高い展開を可能にする。

## Context / Constraints
- 既存Zernikeとコードを共有し、radial部分だけ差分にする（コピペ禁止）
- 新domain `annulus` を追加（center, r_inner, r_outer）
- codec は既存 zernike_pack_v1 を再利用（metaで annular を区別）

## Plan
- [x] domain: `annulus` の mask/座標/重みを実装（diskの派生）
- [x] decomposer: `annular_zernike` を追加（ZernikeFamilyBase推奨）
- [x] meta: (n,m,kind)順序を固定し保存
- [x] tests: annulus roundtrip + r_inner=0 のとき disk zernike と整合する範囲を確認
- [x] examples: run.yaml（scalar_disk_annulus）を追加

## Acceptance Criteria
- [x] annulus domain が dataset manifest で指定できる
- [x] annular_zernike が registry で選択でき、reconstruct が通る
- [x] roundtrip テストが通る（許容誤差内）
- [x] docs/23 が更新される

## Verification
- Command:
  - `python -m mode_decomp_ml.run --config examples/run_scalar_annulus_annular_zernike_ridge.yaml`
- Expected:
  - metrics.json が生成され、figures/ に係数スペクトル等が出る
