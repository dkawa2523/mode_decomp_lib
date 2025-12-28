# Task 058 (P3): 曲面/メッシュ：Laplace–Beltrami固有展開（Mesh Decomposer）

## 目的
将来的に曲面メッシュ（2次元多様体）データを扱う可能性に備え、
Laplace–Beltrami 固有関数（Manifold Harmonics）による分解の枠を用意する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/030_data_domain_io.md
- depends_on: work/tasks/050_decomposer_registry.md

## スコープ
### In
- mesh-domain（vertices, faces）を DomainSpec に追加/拡張
- cotangent Laplacian を計算し、最小K固有ベクトルを求める
- 係数 = 固有ベクトルへの投影で計算
- state に mesh と eigenpairs を保存

### Out
- メッシュ品質改善・リメッシュなど（P4）

## 実装方針（Codex向け）
### 1) 実装方針
- まず mesh-domain のI/O（meshio等）を整備
- cotan Laplacian L と mass matrix M を作り、一般化固有問題 L φ = λ M φ を解く
- 係数: a = Φ^T M f
- inverse: f_hat = Φ a

### 2) 注意
- メッシュが違うと基底が変わり、比較が難しい
- 符号反転の不定性（固有ベクトルは±が等価）→ coeff_metaに注意

### 3) 優先度
- 今すぐ必要ではないが、後から追加すると設計が壊れやすいので “枠” だけ先に決めるのが狙い

## ライブラリ候補
- potpourri3d（cotan laplacian/eigsの候補）
- trimesh（mesh util）
- scipy.sparse.linalg.eigsh

## Acceptance Criteria（完了条件）
- [ ] mesh-domainの最小例が読み込める
- [ ] Laplace–Beltrami eigenbasis で transform->inverse が動く
- [ ] state/coeff_meta が保存される

## Verification（検証手順）
- [ ] 小さな球メッシュ等で低周波成分が滑らかになることを確認
- [ ] Kを増やすと再構成誤差が下がることを確認
