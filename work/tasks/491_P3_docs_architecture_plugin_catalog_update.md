# Task: 491 Docs: アーキテクチャ/プラグインカタログ更新（第三者理解最優先）

- Priority: P1
- Status: done
- Depends on: 490
- Unblocks: (none)

## Intent
第三者（別アーキテクト/別開発者）が短時間で理解できるように、
アーキテクチャ概要・プラグイン一覧・追加手法の位置づけを docs として確定する。

## Context / Constraints
- “読む順” を最優先（docs/README）
- 追加した手法は「どのdomainで使えるか」「コスト」「推奨用途」を明示
- 仕様と実装がズレていたら、ズレを TODO として起票する（捏造禁止）

## Plan
- [x] plugin catalog を更新（decomposer/codec/coeff_post/model）
- [x] domain compatibility 表を追加（rectangle/disk/annulus/mask/sphere_grid/mesh）
- [x] run.yaml の最小例と、bench（quick/full）の使い方をまとめる
- [x] 今後の計画（Helmholtz、AE/VAE、ClearML）の接続点を明記

## Acceptance Criteria
- [x] 第三者が docs を読めば “どの手法をいつ使うか” 判断できる
- [x] plugin追加手順が短く明確（テンプレあり）
- [x] docs とコードの対応が取れている（リンク/パスが正しい）

## Verification
- docs を別人が読んで、最低1つの run.yaml を実行できる（レビュー観点）
