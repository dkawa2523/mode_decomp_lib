# Task: 411 Refactor: 既存DecomposerをCodec前提に統一（flatten/複素/順序の共通化）

- Priority: P1
- Status: todo
- Depends on: 410
- Unblocks: 412, 413, 414, 415, 440, 490

## Intent
既存 decomposer（FFT2/DCT2/Zernike/Fourier–Bessel/POD/SVD/Graph Laplacian/LB）を
新しい Codec 前提のI/Fに揃え、係数の表現（flatten/complex扱い）を codec 側に移して共通化する。

## Context / Constraints
- 既存の評価・可視化・ベンチが壊れないよう、段階的に移行する
- 互換性のため、旧I/F を一時的に Adapter で支えるのは可（ただし最終的に削除）
- raw_coeff_meta を必ず返す（順序/周波数/モードidなど）

## Plan
- [ ] 各decomposerが返す raw_coeff の型と meta を定義（docs/23参照）
- [ ] “flattenして返す”系のロジックを削除し codec へ移す
- [ ] Vector場は ChannelwiseAdapter を導入し、スカラー実装を再利用
- [ ] 既存の roundtrip テストを更新（codec込みで成立）
- [ ] docs更新: plugin catalog と coeff_meta の保存仕様

## Acceptance Criteria
- [ ] 主要decomposerが `raw_coeff + meta` を返し、codecで学習できる
- [ ] 既存の baseline（FFT2+Ridge, Zernike+Ridge, POD+Ridge, Graph+Ridge）が回る
- [ ] テストが通り、bench quick が最低1回成功する

## Verification
- Command:
  - `bash scripts/bench/run_p0p1_p2ready.sh`（存在する場合）
- Expected:
  - 失敗せずに数ケースが完走し、leaderboard/metricsが生成される
