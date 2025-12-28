# Task 057 (P3): 任意境界：Laplace固有関数（FEM）分解（準備タスク）

## 目的
任意形状境界に対して物理解釈性の高い基底（Laplacian eigenfunctions）を得るため、
FEMベースの固有問題を解く分解手法を導入する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md

## スコープ
### In
- FEMライブラリ候補を選定し、最小実装（矩形以外の簡単な多角形）で固有関数を計算
- boundary_condition（Dirichlet/Neumann）を config 化
- 固有関数 φ_k を basis として係数 a_k を投影で計算
- state にメッシュ・固有値・固有ベクトルを保存

### Out
- 高精度メッシュ生成・高速化（P4）

## 実装方針（Codex向け）
### 0) 先に決める（リスク）
- 本タスクは依存が重い。FEniCS/mesh生成が環境で詰まりやすい
- まずは scikit-fem で簡単な領域の固有問題が解けるか検証する

### 1) メッシュ
- mask-domain を等値線抽出して多角形化 → メッシュ化（P3の課題）
- 代替：事前にメッシュを用意して読み込む（mesh-domain）

### 2) 固有問題
- -Δφ = λφ を解き、最小K個の固有関数を得る
- 係数: a_k = <f, φ_k>（内積は領域積分）

### 3) 保存
- メッシュと固有関数は必ず保存（再現性）
- 同じdomainでもメッシュが違うと基底が変わるため比較に注意

## ライブラリ候補
- scikit-fem（候補）
- numpy/scipy（固有分解）
- mesh生成：pygalmesh / meshio（候補）

## Acceptance Criteria（完了条件）
- [ ] 選定したFEMライブラリで最小例が動く
- [ ] 固有関数basisで transform->inverse が可能
- [ ] boundary_conditionが保存される

## Verification（検証手順）
- [ ] 簡単な領域（例：円/矩形）で固有値が理論と近いことを確認
- [ ] Kを増やすと再構成誤差が下がることを確認
