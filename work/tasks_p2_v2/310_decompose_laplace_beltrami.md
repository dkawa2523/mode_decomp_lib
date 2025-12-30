# Task 310 (P2): Domain+Decomposer: Mesh + Laplace–Beltrami（曲面）

**ID:** 310  
**Priority:** P2  
**Status:** todo  
**Depends on:** 300  
**Unblocks:** 320  

⚠️ **DO NOT CONTINUE**: Acceptance Criteria / Verification を満たすまで `done` にしない。  
⚠️ **NO TASK CREEP**: 実行中に `work/queue.json` に新規タスクIDを追加しない。必要ならRFCに落として停止。  

## Compactness & Review（必須）
- 最小差分 / 過剰抽象化禁止: docs/14_COMPACT_CODE_POLICY.md
- 重要箇所は `# CONTRACT:` / `# REVIEW:` で目印: docs/15_REVIEWER_GUIDE.md
- 不要になったコード/ファイルは削除: docs/16_DELETION_AND_PRUNING.md

---
## Context
あなたは Graph Laplacian（300）を完了しています。
次は “曲面/メッシュ” を扱う Laplace–Beltrami 展開です。ここはブレやすいので、
**最小実装（SciPyのみ）** をまず成立させてください。

## Scope（最小）
- mesh domain（固定メッシュ）を扱う
- per-vertex scalar/vector field を展開できる（C=1/2）
- cotan Laplacian + mass matrix による一般化固有値問題:
  - L φ = λ M φ
- transform/inverse が成立する
- eigenvalues/eigenvectors を decomposer_state として保存

## Recommended file plan（目安）
- add: `src/mode_decomp_ml/domain/mesh.py`
- add: `src/mode_decomp_ml/decompose/laplace_beltrami.py`
- add: `src/mode_decomp_ml/mesh_ops/cotan_laplacian.py`（小さく）
- add: `configs/domain/mesh.yaml`
- add: `configs/decompose/laplace_beltrami.yaml`
- update: plugin registry（domain/decompose）

## Implementation steps
1) **Toy mesh generator** を用意（コードで生成）
   - 平面の正方形を格子→三角化（nV~1k以下）
   - テストに使う（外部データ無しで動かすため）

2) cotan Laplacian と mass matrix（barycentricでOK）を実装
   - sparse CSR を返す
   - 数値安定のため、対称性や対角の扱いを確認

3) 固有分解（eigsh）
   - k=16〜64
   - 最小固有値（定数モード）を扱う方針を決める（除外推奨）
   - λ と φ を保存（npy）

4) transform/inverse
   - a_i = φ_i^T M f（離散内積）
   - f_hat = Σ a_i φ_i
   - vector field はチャネル別に同じ処理（C次元）

5) artifact 保存
   - `outputs/.../model/decomposer_state/` に eigenvalues/eigenvectors
   - `meta` に mesh signature（nV,nF,k,hash）を残す

## Acceptance Criteria
- [ ] toy mesh（平面）で k>=16 が計算できる
- [ ] transform→inverse で再構成誤差が十分小さい（kを増やすと改善）
- [ ] scalar と vector の両方で動く（同じコードパス）
- [ ] eigenvalues/eigenvectors が artifacts に保存される（docs/addons/42）

## Verification
- [ ] toy mesh の 1サンプルで decompose→inverse のスモークテスト
- [ ] `task=reconstruct`（既存入口）で field_hat が生成される（可能な範囲で）

## Cleanup Checklist
- [ ] mesh ops の試行コードを消し、最小の関数にまとめる
- [ ] 未使用の依存を追加しない
