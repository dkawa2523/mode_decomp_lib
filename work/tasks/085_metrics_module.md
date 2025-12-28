# Task 085 (P0): 評価指標モジュール（field/coeff）と共通レポート形式

## 目的
分解手法や学習モデルが変わっても同じ指標で比較できるよう、
評価指標を共通モジュール化し、出力形式（metrics.json）を固定する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/080_process_e2e.md

## スコープ
### In
- `metrics/field_metrics.py`（RMSE, relL2, max_err, SSIM optional）
- `metrics/coeff_metrics.py`（RMSE, energy capture 等）
- mask内評価を徹底（mask外は無視）
- metrics.json のスキーマを固定（docs/09 に合わせる）

### Out
- 複雑な物理指標（応力不変量等）はP2

## 実装方針（Codex向け）
### 1) API
- `compute_field_metrics(field_true, field_pred, mask) -> dict`
- `compute_coeff_metrics(A_true, A_pred, coeff_meta) -> dict`
- `coeff_meta` が一致しない場合は coeff metric を出さない（または `available=false` を返す）

### 2) SSIM
- 画像としての近さが重要なら追加
- ただし外部依存が増えるので optional にする（configでON）

### 3) energy capture
- A_true のエネルギー（L2）に対し、上位K成分が占める割合など

## ライブラリ候補
- numpy
- scikit-learn（rmse等）
- scikit-image（SSIM optional）

## Acceptance Criteria（完了条件）
- [ ] metrics が共通関数で計算され、metrics.json が固定スキーマで出る
- [ ] maskが反映されている
- [ ] coeff_meta不一致時の挙動が明確

## Verification（検証手順）
- [ ] 簡単な入力で metrics が期待通りになる unit test を追加
- [ ] tinyのevalで metrics.json が生成される
