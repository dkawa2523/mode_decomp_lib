# Task 045 (P1): ベクトル場の表現変換（component/div-curl）

## 目的
ベクトル場を分解する際の選択肢（成分別、div/curlなど）をプラグイン化し、
分解手法と独立に切り替えられるようにする。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/040_preprocess_pipeline.md

## スコープ
### In
- `VectorTransform` registry を実装
- P0相当: componentwise（vx,vyを別チャネルとして扱う）
- P1: div_curl（2D格子上のdiv/curlを計算し、2枚のスカラー場に変換）
- vector transform の情報を artifact に保存

### Out
- Hodge分解（mesh含む）はP2

## 実装方針（Codex向け）
### 1) interface
- `transform(sample: FieldSample) -> FieldSample`（fieldチャネルが変わる）
- `inverse_transform(sample_like) -> FieldSample` は必須ではない（div/curlは不可逆）
  - ただし、不可逆なら `is_invertible=false` を明示し、後続で扱いを変える

### 2) div/curl 実装メモ（rect grid）
- `div = d(vx)/dx + d(vy)/dy`
- `curl_z = d(vy)/dx - d(vx)/dy`
- 差分スキーム（中央差分/前進差分）を config で選べるようにする
- mask境界の扱いは要注意：mask外はNaNとして差分を工夫 or 内側だけ評価

### 3) 後続との接続
- ベクトル分解を「スカラー2枚の分解」として扱えるため、既存decomposerを再利用しやすい

## ライブラリ候補
- numpy
- scipy.ndimage（差分フィルタ）

## Acceptance Criteria（完了条件）
- [ ] vector transform を config で切替できる
- [ ] componentwise でベクトル場がそのまま下流に流れる
- [ ] div_curl の出力shapeが正しく、maskを尊重する

## Verification（検証手順）
- [ ] 合成ベクトル場でdiv/curlが理論通りになる簡易テストを追加
- [ ] tinyデータで componentwise -> decompose が通る
