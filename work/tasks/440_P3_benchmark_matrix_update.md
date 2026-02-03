# Task: 440 Update: Benchmark sweep（domain整合・quick/full・optional依存スキップ）

- Priority: P1
- Status: done
- Depends on: 401, 410
- Unblocks: 490

## Intent
P0/P1/P2 の decomposer・coeff_post・model を **domain整合を保ったまま網羅**する benchmark sweep 雛形を更新する。
追加手法（Wavelet/Annular/SH/Slepian/PSWF、Quantile/Power、GBDT/MTLasso/MTGP）も
“壊れない範囲で” matrix に組み込み、quick/full を分けて運用できるようにする。

## Context / Constraints
- domain compatibility（rectangle/disk/annulus/mask/sphere_grid/mesh）を守り、無意味な組合せを作らない
- optional dependency の手法は “存在すれば回す/無ければスキップ” を標準とする
- 出力は `runs/` のフラット構造に揃える（タスク401）

## Plan
- [ ] `scripts/bench/matrix.yaml` を再設計（quick / full）
- [ ] domainごとに decomposer候補セットを定義（互換表）
- [ ] modelごとに推奨 coeff_post を定義（例: gpr/mtgp は pca前提）
- [ ] スキップ理由をログに出す（依存なし/非対応domain等）
- [ ] leaderboard 集計が新出力レイアウトで動くよう更新

## Acceptance Criteria
- [ ] `run_p0p1_p2ready.sh` が完走する（少数ケース）
- [ ] `run_full.sh` は optional を含みつつ、未導入でも止まらない（skipされる）
- [ ] 出力が `runs/<tag>/<run_id>/` に揃う
- [ ] matrix.yaml が第三者に理解できる（コメント/見出し）

## Verification
- Command:
  - `bash scripts/bench/run_p0p1_p2ready.sh`
- Expected:
  - 最低でも rectangle/disk/mask の baseline が完走し、tables/ に集計が出る

## Review Map
- **変更ファイル一覧**: `scripts/bench/matrix.yaml`, `scripts/bench/run_matrix.py`, `scripts/bench/run_p0p1_p2ready.sh`, `scripts/bench/run_full.sh`, `src/processes/benchmark.py`
- **重要な関数/クラス**: `scripts/bench/run_matrix.py:main`, `scripts/bench/run_matrix.py:_prepare_models`, `scripts/bench/run_matrix.py:_run_leaderboard`
- **設計判断**: benchmark は codec を固定するため、matrix は domain→decomposer(+codec) で分解し 1 decomposer/1 codec 単位で benchmark を回す構成にした。optional 依存（pywt/pyshtools 等）は matrix 側で明示し、存在しない場合は skip で継続する方針に統一。
- **リスク/注意点**: optional 手法は config 未導入だと skip されるため、追加時は `configs/*` の投入が必須。mask domain は synthetic の mask_mode を disk に固定している。
- **検証コマンドと結果**: `bash scripts/bench/run_p0p1_p2ready.sh`（4 runs + leaderboard 生成を確認）
- **削除一覧**: なし
