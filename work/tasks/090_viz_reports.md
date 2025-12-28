# Task 090 (P1): 可視化・比較レポート（recon/error/coeff spectrum）

## 目的
非データサイエンティストの開発エンジニアでも差分が理解できるよう、
再構成画像・誤差マップ・係数スペクトルなどの可視化をProcessとして整備する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/085_metrics_module.md

## スコープ
### In
- `processes/viz.py` を実装し、run dir から図を生成する
- field_true/field_pred/error_map を保存（PNG）
- 係数のスペクトル（Zernikeなら(n,m)ヒートマップ、FFTなら周波数平面）を保存
- 比較表（method, metric, params）をCSV/Markdownで出す

### Out
- GUI（Streamlit等）はP3

## 実装方針（Codex向け）
### 1) 入力はrun dir
- viz process は “再計算しない” を基本（preds/metrics を読む）
- 例外：必要なら reconstruct を内部で呼ぶが、その場合は依存runを明記

### 2) 図の種類（P1）
- recon: true/pred の並置
- error_map: abs error（mask内）
- coeff_spectrum:
  - Zernike: |a_{n,m}| を (n,m) 格子に配置
  - FFT: |A[kx,ky]| を log 表示
  - DCT: C[u,v] を log 表示

### 3) 出力
- `viz/` に保存し、ファイル名規約を固定（比較しやすさ）
- `viz/summary.md` を自動生成（主要指標と図へのリンク）

## ライブラリ候補
- matplotlib
- numpy
- pandas

## Acceptance Criteria（完了条件）
- [ ] viz process が1コマンドで図を生成できる
- [ ] Zernike/FFT/DCT それぞれでスペクトル図が出る
- [ ] summary table が出力される

## Verification（検証手順）
- [ ] tiny run の出力run dirに対して viz が実行できる
- [ ] 出力画像が空でない（ファイルサイズ>0）
