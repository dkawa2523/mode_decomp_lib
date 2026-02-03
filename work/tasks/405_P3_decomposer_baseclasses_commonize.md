# Task: 405 Core: Decomposer baseclasses + ChannelwiseAdapter（共通化で増殖停止）

- Priority: P1
- Status: todo
- Depends on: 404
- Unblocks: 411, 414, 416

## Intent
共通化（Grid/Zernike/Eigenbasis + ChannelwiseAdapter）を実装し、
手法追加のたびにコピペが増える問題を根本から止める。

## Context / Constraints
- docs/22 の方針に従う
- 既存 decomposer はこのベースクラスに乗せる（最低限FFT/Zernike/Graphは対応）
- 符号不定性（eigenbasis）は “符号固定ルール” を1箇所に集約する

## Plan
- [ ] `GridDecomposerBase`, `ZernikeFamilyBase`, `EigenBasisDecomposerBase` を追加
- [ ] `ChannelwiseAdapter` を追加（vector場を scalar decomposer で再利用）
- [ ] 既存の代表手法をベースクラスへ移行（例: FFT2, Zernike, Graph Laplacian）
- [ ] 共通の meta/state 保存/ロードをベース側に寄せる
- [ ] unit tests: ベースクラス経由でも roundtrip が成立すること

## Acceptance Criteria
- [ ] 既存 decomposer の重複コードが減っている（レビューで確認可能）
- [ ] 新手法が “小さな差分” で追加できる土台になっている
- [ ] vector場の処理が ChannelwiseAdapter で簡潔に書ける

## Verification
- `pytest` が通る
- baselineのrunが壊れていない
