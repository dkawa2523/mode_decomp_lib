# Artifacts and Versioning（保存契約）

## 原則
- すべての Process は run dir に出力する
- 出力は「次の Process が読める形式」で保存する（再実行可能）

---

## 代表的な run dir 構造（例）
```
outputs/<process>/<date>/<time>_<tag>/
  hydra/                    # Hydraが保存する設定
  meta.json                 # 実行環境、git hash、dataset hash等
  artifacts/
    dataset_meta.json
    decomposer/
      state.pkl             # basis/indices 等（必要な場合）
      coeff_meta.json       # 係数のindex対応
    coeff_post/
      state.pkl             # PCA等
    model/
      model.pkl or model.pth
  metrics/
    metrics.json
  preds/
    coeff.npy               # a_hat もしくは z_hat
    field.npy               # field_hat
  viz/
    recon.png
    error_map.png
    coeff_spectrum.png
```

---

## 必須artifact（最低限）
- `hydra/config.yaml`（自動）
- `meta.json`（seed、git、lib versions、dataset hash）
- `metrics/metrics.json`（評価指標）
- `preds/`（必要な場合のみ：predict/reconstruct）
- `artifacts/decomposer`（transformの再現に必要なstateがある場合）
- `artifacts/coeff_post`（PCA等のstateは必須）
- `artifacts/model`（学習モデル）

---

## dataset versioning
- 生データのハッシュ（ファイルhashまたはサンプルID一覧hash）を `dataset_meta.json` に保存
- split方式とseedも保存（比較可能性の根拠）

---

## 係数の互換性
- 係数 `a` の次元・インデックス対応は `coeff_meta.json` に保存する
  - 例: Zernikeの(n,m)順序、FFTの周波数並び、Besselの( m, n ) など
- 係数互換性が壊れる変更（並び順変更等）は破壊的変更なのでRFC必須
