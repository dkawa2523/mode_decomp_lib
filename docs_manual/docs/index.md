# 概要

この MkDocs マニュアルは、本リポジトリの「モード分解 → 学習 → 推論 → 評価/可視化 → ベンチマーク」の全体像を、第三者が短時間で把握できる形に整理します。

!!! note "Mermaid 図が文字列に見える場合"
    Mermaid 図（`<div class="mermaid">...</div>`）は **MkDocs サイトとして表示したときだけ** SVG にレンダリングされます。  
    `.md` を IDE/リポジトリビューアで直接開くと、Mermaid のソース文字列がそのまま見えます。  
    ローカルで確認する場合は `docs_manual/` で `mkdocs serve -f mkdocs.yml` を使ってください。

## 最初に読む順番

1. `overview.md`
2. `execution.md`
3. `code-structure.md`
4. `decomposition-methods.md` / `training.md` / `inference.md`
5. `plots.md` / `test-datasets.md`

## 用語（固定）

| 用語 | 意味 |
|---|---|
| field | 空間上の場（スカラー: C=1 / ベクトル: C=2） |
| mask | 有効領域（domain mask + dataset mask の合成） |
| モード分解 | field → coeff への変換（基底による展開、またはデータ駆動による次元削減） |
| coeff(a) | codec によりベクトル化された係数（学習モデルの標準ターゲット） |
| coeff(z) | coeff_post による後処理後の係数（例: PCA） |
| artifact | `run_dir/` 以下に生成される成果物（tables/plots/states 等） |

## 実行入口（2本に統一）

Hydra（推奨。sweep/benchmark向き）:
```bash
PYTHONPATH=src python3 -m mode_decomp_ml.cli.run task=pipeline
```

run.yaml（非Hydra。1-runを確実に回す用途）:
```bash
PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml --dry-run
PYTHONPATH=src python3 -m mode_decomp_ml.run --config examples/run_scalar_rect_fft2_ridge.yaml
```

## 追加で考慮すべき項目（navには含めない）

- Glossary（用語集）
- Troubleshooting（よくある失敗: mask/codec/coeff_meta/optional deps）
- Extension checklist（decomposer/codec/model/coeff_post/preprocess 追加チェックリスト）
- Optional dependencies matrix（torch/gpytorch/pywt/pyshtools 等）

## 参照（canonical docs）

- `docs/01_ARCHITECTURE.md`
- `docs/10_PROCESS_CATALOG.md`
- `docs/11_PLUGIN_REGISTRY.md`
- `docs/21_CODEC_LAYER_SPEC.md`
- `docs/28_COEFF_META_CONTRACT.md`
- `docs/30_BENCHMARKING.md`
- `docs/31_CODE_TOUR.md`
