# Task 250 (P1): Uncertainty（係数→場）伝播と可視化（GPR）

**ID:** 250  
**Priority:** P1  
**Status:** done  
**Depends on:** 240  
**Unblocks:** 260, 270  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Context
あなたはすでに GPR（Task230）と sweep benchmark（Task240）を完了しています。
残る課題は「モデルがどれくらい信じられるか」を判断できる情報を artifact と可視化で残すことです。

## Goal（最小）
- GPRの `predict_std`（係数の標準偏差）を artifact として保存する
- 代表サンプルで “場の不確かさ” を可視化する（近似でOK）

## Recommended minimal approach（汎用：Monte Carlo）
- 係数の予測分布を **独立正規** と仮定して係数サンプルを生成し、復元画像の分散から std map を作る
  - `a ~ Normal(mu, diag(std^2))`
  - サンプル数 `S` は 16〜64 程度で十分（設定で変更可能）
- この方法なら decomposer が線形/非線形でも **inverse があれば動く**

## Implementation notes
- config追加:
  - `uncertainty.enabled: true/false`
  - `uncertainty.num_mc_samples: 32`
  - `uncertainty.num_cases: 3`（可視化するサンプル数）
- 保存（artifact）:
  - `preds/coeff_mean.npy`
  - `preds/coeff_std.npy`（GPRの場合）
  - `preds/field_std.npy`（MC結果：可視化対象だけでも可）
  - `viz/uncertainty_map_<id>.png`
- docs追記:
  - 仮定（独立・正規・MC近似）と “分かる/分からない” を短く書く

## Acceptance Criteria
- [ ] GPR実行時に `coeff_std` が保存される
- [ ] 指定サンプルで uncertainty map（std）がpngで出力される
- [ ] 仮定と計算方法がdocsに明記される（追記先は docs/20_METHOD_CATALOG.md or docs/addons）

## Verification
- [ ] 既存のGPR run か toy run で `uncertainty.enabled=true` をONにして実行
- [ ] outputs配下に preds/viz が生成されることを確認

## Cleanup Checklist
- [ ] 一時的なデバッグコード/printを削除
- [ ] 不要な大規模依存を追加しない（最小）

---
## Review Map
- **変更ファイル一覧**: `configs/config.yaml`, `configs/uncertainty/gpr_mc.yaml`, `src/mode_decomp_ml/models/__init__.py`, `src/processes/predict.py`, `src/processes/reconstruct.py`, `src/processes/viz.py`, `src/mode_decomp_ml/viz/__init__.py`, `docs/20_METHOD_CATALOG.md`, `tests/test_models_gpr.py`, `work/tasks_p1_tail/250_uncertainty_viz.md`, `work/queue.json`
- **重要な関数/クラス**: `src/mode_decomp_ml/models/__init__.py` の `BaseRegressor.predict_with_std` / `GPRRegressor.predict_with_std`, `src/processes/predict.py:main`, `src/processes/reconstruct.py:_mc_field_std` / `src/processes/reconstruct.py:main`, `src/processes/viz.py:main`, `src/mode_decomp_ml/viz/__init__.py:plot_uncertainty_map`
- **設計判断**: GPRのpredict_stdは常に保存し、MC不確かさ伝播は `uncertainty.enabled` のみで実行。field_stdは代表サンプルのみ（等間隔index）で保存し、vizは保存済みメタに従う。
- **リスク/注意点**: 独立正規（diag std）の仮定で係数相関は無視。`uncertainty.enabled=true` で coeff_std が無いモデルはエラー。case indexは dataset順に対応。
- **検証コマンドと結果**: `pytest -q tests/test_models_gpr.py` -> pass。`python - <<'PY' ...` で train/predict/reconstruct/viz を実行し `outputs/_uncertainty_smoke/.../preds` と `viz/uncertainty_map_*.png` を確認（Matplotlib cache warningあり）。
