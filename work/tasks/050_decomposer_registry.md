# Task 050 (P0): 一次モード分解の共通インターフェースとregistry

## 目的
Zernike/FFT/DCT/Bessel/RBFなど複数の分解手法を同じ枠で扱うため、
Decomposer の共通インターフェースと registry を実装する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/020_hydra_mvp.md
- depends_on: work/tasks/030_data_domain_io.md

## スコープ
### In
- `Decomposer` base class（fit/transform/inverse_transform/coeff_meta）を実装
- registry（method key -> class）を実装し、Hydraから instantiate できるようにする
- decompose.transform Process を実装して係数 a と coeff_meta を保存する
- エラー時の検証（domain不一致、mask不正、次数不正）を実装

### Out
- 各分解手法の実装（次タスクで追加）

## 実装方針（Codex向け）
### 1) 係数の表現を統一する
- 係数 `a` は `np.ndarray` を基本（必要なら complex）
- 追加で `CoeffMeta`（dict）を返し、indexの意味を保存する
  - 例: `type: zernike`, `nm_list: [(n,m),...]`, `ordering: ...`
- 係数の dtype/shape を明確にし、必ず `coeff_meta.json` に書き出す

### 2) fitの扱い
- 固定基底（Zernike/FFT/DCT/Bessel）は fit=no-op でもOK
- データ駆動辞書（POD/NMF/DictionaryLearning等）は fitが必要
- どちらも同じ interface を守る

### 3) mask/domain
- `transform(field, mask, domain_spec)` を引数に持ち、domainを必須とする
- disk系は `domain_spec.type == disk` を要求し、それ以外は明示エラー

### 4) Process実装
- `processes/decompose_transform.py` を作り、
  - dataset -> coeff.npy (N, K) と coeff_meta.json を保存する
- 係数は sample_id 順の対応が必要（indexファイルを保存）

## ライブラリ候補
- numpy
- pydantic/dataclasses（coeff meta）
- joblib/pickle（state保存）

## Acceptance Criteria（完了条件）
- [ ] decompose=... を config で切り替えられる（少なくともzernike/fft/dctは後続タスクで）
- [ ] coeff_meta.json が必ず保存され、係数の並びが説明可能
- [ ] domain不一致などは明確にエラーになる

## Verification（検証手順）
- [ ] tinyデータで `python -m processes.decompose_transform decompose=<method>` が走り、coeff.npyができる
- [ ] coeff_meta.json を開いて意味が追える
