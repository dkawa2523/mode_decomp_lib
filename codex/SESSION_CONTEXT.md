# Codex Session Context（このプロジェクト用）

## 目的
- 2Dスカラー/ベクトル場の **モード分解（Zernike/FFT/Bessel/RBF…）** と
  **係数後処理（PCA/ICA…）** と **回帰モデル学習** を、
  反復改善しても比較可能性が壊れない形で実装する。

## 最重要契約
- docs/00_INVARIANTS.md を破らない
  - 設定が真実（Hydra）
  - Process単位で単独実行
  - train/serve skew禁止（PCA等はfit/transform分離）
  - 係数の並びは coeff_meta.json として必ず保存

## 進め方
- work/queue.json の P0 から順に work/tasks/*.md を実行する
- タスクの Acceptance Criteria と Verification を満たしたら done
- 進められない場合は質問せず：
  - status=blocked
  - blocker解消の小タスクを起票（unblocks/depends_on）

## 実装スタイル
- plugin registry を必ず使う（decompose / coeff_post / model）
- 既存Zernike実装がある場合はまずラップして動かし、後でリファクタする
- maskの扱いは “明示的なポリシー” を config に置く（silent fill禁止）

## 出力（artifact）
- outputs/<process>/... 配下に config/meta/model/metrics/preds/viz を保存する（docs/04）


- docs/13_TASK_FLOW.md（タスク運用ルール）も必ず読む。

## 追加の不変ルール（コード量・レビュー・削除）
- コード量を増やしすぎない（docs/14）
  - “将来のため”の過剰抽象化は禁止。今必要な最小限で実装する。
- レビュー容易性を最優先（docs/15）
  - タスク完了時に task md へ **Review Map**（重要ファイル/入口/設計判断/検証結果）を追記する。
- 不要物を残さない（docs/16）
  - タスク中に不要になったコード/ファイルは削除する。
  - 削除が大きい場合はRFC/ADRか、削除期限タスクを起票して放置しない。

## 拡張時の起票ルール（将来）
- 追加機能は「Task化 → depends_on/unblocks → doctorで整合 → 実装」の順（docs/17）。
- 追加手法は “プラグイン + config + test + docs更新” をセットで行う。
