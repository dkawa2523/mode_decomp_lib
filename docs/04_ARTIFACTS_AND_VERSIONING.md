# Artifacts and Versioning（保存契約）

## 原則
- すべての Process は run dir に出力する
- 出力は「次の Process が読める形式」で保存する（再実行可能）

---

## 代表的な run dir 構造（例）
```
runs/<name>/<process>/
  configuration/
    run.yaml
    resolved.yaml
    inputs/
  logs/
    run.log
    steps.json
  outputs/
    manifest_run.json
    metrics.json
    preds.npz
    coeffs.npz
    optuna_trials.csv          # inference optimize
    optuna_best.json           # inference optimize
    states/
      decomposer/
        state.pkl           # basis/indices 等（必要な場合）
        coeff_meta.json     # 係数のindex対応
      coeff_post/
        state.pkl           # PCA等
      preprocess/
        state.pkl
    tables/
  model/
    model.pkl or model.pth
  plots/
    recon.png
    error_map.png
    coeff_spectrum.png
```

---

## 必須artifact（最低限）
- `configuration/run.yaml`（入力設定の保存）
- `outputs/manifest_run.json`（seed、git、dataset hash、dataset_meta、upstream_artifacts 等）
- `outputs/metrics.json`（評価指標）
- `outputs/preds.npz`（inference の予測）
- `outputs/coeffs.npz`（decomposition/preprocessing の係数）
- `outputs/states/decomposer`（transform再現に必要なstateがある場合）
- `outputs/states/coeff_post`（PCA等のstateは必須）
- `model/`（学習モデル）

---

## dataset versioning
- 生データのハッシュ（ファイルhashまたはサンプルID一覧hash）を `manifest_run.json` 内 `dataset_meta` に保存
- split方式とseedも保存（比較可能性の根拠）

---

## 係数の互換性
- 係数 `a` の次元・インデックス対応は `coeff_meta.json` に保存する
  - 例: Zernikeの(n,m)順序、FFTの周波数並び、Besselの( m, n ) など
- 係数互換性が壊れる変更（並び順変更等）は破壊的変更なのでRFC必須
