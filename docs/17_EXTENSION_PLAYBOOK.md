# 拡張プレイブック（Codexが追加Task/ファイルを作りやすい指針）

このドキュメントは、今後ユーザーから指示される拡張（特徴量化、モデル、I/O、可視化、ClearML、アーキテクトなリファクタ等）を、
**仕様をぶらさず**に追加するための手順書です。

## 拡張の基本方針
- 既存の I/O 契約（docs/02, 10）を壊さない
- 既存の比較可能性（docs/09）を壊さない
- 追加は “プラグイン” と “設定” と “テスト” の3点セットで行う（docs/11）

## 新規手法を追加する手順（共通）
1. **Method定義**
   - 何を入力として、何を出力するか（shape/型/単位）
   - どの domain/boundary で有効か（rectangle/disk/arbitrary_mask/mesh）
2. **実装**
   - `src/` の該当プラグインカテゴリに追加
     - decomposer / coeff_post / model / dataset / viz / process
3. **Hydra設定**
   - `configs/<group>/<name>.yaml` を追加し、切替可能にする（docs/03）
4. **テスト**
   - unit（小配列）+ e2e（smoke）を最低限追加（docs/07）
5. **比較基盤**
   - metrics/leaderboard/pipeline に載ること（docs/09, 04）
6. **ドキュメント更新**
   - docs/20_METHOD_CATALOG.md などに “追加した手法” を追記

## 新しいTaskを起票する手順（Codex向け）
- `work/templates/TASK.md` をベースに `work/tasks/<id>_<slug>.md` を作成
- `work/queue.json` の `tasks[]` に追記（depends_on/unblocks を明示）
- doctor で整合性確認：`python tools/codex_prompt.py doctor`

### 推奨：タスク自動生成
`tools/taskgen.py` を使うと、タスクmd作成とqueue追記を自動化できます（work/TASK_AUTHORING_GUIDE.md 参照）。

## よくある拡張例の“差し込み位置”
- 特徴量化（分解後）:
  - coeff_post に PCA/ICA/NMF/whitening など（train-only fit）
- モデル拡張:
  - sklearn（Ridge/ElasticNet/GPR）
  - torch（MLP/CNN）
- 異なるデータ入出力:
  - dataset adapters を追加し、下流I/Oは不変に
  - sphere_grid を扱うデータ生成では `mode_decomp_ml.domain.sphere_grid` のユーティリティを使用する（range の再実装禁止）
  - dataset テンプレは `docs/addons/35_DATASET_TEMPLATE_SAMPLES.md` を参照
- 可視化の拡充:
  - decomposition/inference の plots 出力で追加（artifactに残す）
- ClearML対応:
  - まず `tracking hooks` の抽象点を用意（docs/12 を参照し、必要ならタスク化）
- アーキテクトなリファクタ:
  - RFC/ADR を起票し、段階的に移行（work/templates）

## Doneの定義（拡張でも不変）
- 動く
- 比較できる（同じ指標/同じ手順）
- 不要物を残さない（docs/16）
- レビューしやすい（docs/15）
