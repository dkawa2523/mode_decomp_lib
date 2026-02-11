# Benchmark Test (v1): 全ドメイン網羅・再生成可能

このリポジトリの decomposer / domain を **同一条件の問題セット**で横並び評価するための手順です。

## 1. データセット仕様（v1）
- サンプル数: `N=36`（30〜40）
- 生成場:
  - `field = offset + fluct + noise`
  - `fluct`: mask内RMSが `0.07 * offset`（デフォルト、5〜10%相当）
  - `noise`: mask内RMSが `0.01 * offset`（約1%）
- ケース（10）:
  - `rectangle_scalar`, `rectangle_vector`
  - `disk_scalar`, `disk_vector`
  - `annulus_scalar`, `annulus_vector`
  - `arbitrary_mask_scalar`, `arbitrary_mask_vector`
  - `sphere_grid_scalar`, `sphere_grid_vector`

生成先（デフォルト）:
- `data/benchmarks/v1/offset_noise_36/<case>/`
- 各caseは `cond.npy`, `field.npy`, `manifest.json`（必要なら `domain_mask.npy`）を持ちます。

## 2. データ生成
```bash
python tools/bench/generate_benchmark_datasets_v1.py \
  --out-root data/benchmarks/v1/offset_noise_36 \
  --n-samples 36 \
  --fluct-ratio 0.07 \
  --noise-ratio 0.01 \
  --seed 123
```

## 3. ベンチ実行（全ケース）
```bash
python tools/bench/run_benchmark_v1.py \
  --dataset-root data/benchmarks/v1/offset_noise_36 \
  --runs-root runs/benchmarks/v1 \
  --seed 123
```

## 3.1 未評価手法の追加ベンチ（v1_missing_methods）
v1本体では未評価になりがちな data-driven/特殊系（`pod`, `pod_joint`, `dict_learning`, `autoencoder`, `laplace_beltrami`）を
最小ケース数で追加実行します。

meshケース（`laplace_beltrami` 用）生成:
```bash
python tools/bench/generate_benchmark_datasets_v1_mesh.py \
  --out-root data/benchmarks/v1/offset_noise_36 \
  --n-samples 36 \
  --seed 123
```

missing methods ベンチ実行:
```bash
python tools/bench/run_benchmark_v1_missing_methods.py \
  --dataset-root data/benchmarks/v1/offset_noise_36 \
  --runs-root runs/benchmarks/v1_missing_methods \
  --seed 123
```

出力:
- `runs/benchmarks/v1_missing_methods/summary/benchmark_summary_decomposition.csv`
- `runs/benchmarks/v1_missing_methods/summary/benchmark_summary_train.csv`
- `runs/benchmarks/v1_missing_methods/summary/benchmark_axes_summary.md`

## 3.2 gappy_pod 専用評価（fit=mask無し / transform=観測mask）
`gappy_pod` は fit と transform のマスク契約が pipeline と一致しないため、専用スクリプトで評価します。

```bash
python tools/bench/eval_gappy_pod_v1.py \
  --dataset-root data/benchmarks/v1/offset_noise_36/rectangle_scalar \
  --out-dir runs/benchmarks/v1_missing_methods/gappy_pod_rectangle_scalar \
  --obs-frac 0.7 \
  --seed 123
```

出力:
- `runs/benchmarks/v1_missing_methods/gappy_pod_rectangle_scalar/metrics.json`
- `runs/benchmarks/v1_missing_methods/gappy_pod_rectangle_scalar/plots/`

## 3.3 Coverage確認（decomposer未評価が残っていないか）
```bash
python tools/bench/check_benchmark_coverage_v1.py
```

途中で失敗する手法があっても止めない（推奨）:
- `run_benchmark_v1.py` はデフォルトで continue-on-error 相当で動作します。
- `src/processes/pipeline.py` も `task.continue_on_error: true` をサポートします。

## 4. 出力と比較方法
### 4.1 caseごとの出力
各caseは次に出力されます:
- `runs/benchmarks/v1/<case>/pipeline/outputs/tables/leaderboard_decomposition.csv`
- `runs/benchmarks/v1/<case>/pipeline/outputs/tables/leaderboard_train.csv`

加えて、分解/学習の plot は各run配下の `plots/` に出ます（`viz.validity` を含む）。

### 4.2 全ケース集計
`run_benchmark_v1.py` が最後に集計します:
- `runs/benchmarks/v1/summary/benchmark_summary_decomposition.csv`
- `runs/benchmarks/v1/summary/benchmark_summary_train.csv`

## 5. 注意（重要）
- 係数の型が混在するため、ベンチでは codec に `auto_codec_v1` を使います（complex/wavelet/real を自動dispatch）。
- `graph_fourier` は dense 解を避けるため `graph_fourier_bench`（eigsh固定）を使います。
- `sphere_grid` は optional dependency を避けるため `spherical_*_scipy` を使います。
