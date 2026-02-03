# TASK 497: POD backend=sklearn（direct/snapshots/randomizedの基礎実装）

## 目的
backend=sklearn の PODを実装し、direct/snapshots/randomized の基礎を提供します。

## 作業内容
1. sklearn PCA を用いたPOD（PCA=主成分）を `PODDecomposer` 内で実装
   - solver=direct/snapshots: svd_solver="full"（またはautoで良いが再現性を重視）
   - solver=randomized: svd_solver="randomized", random_state=seed
2. center（mean）を明示的に扱う（sklearnに任せる場合も mean_ を state に保存）
3. inverseは `components_` を使って再構成（shape戻しは共通関数へ）
4. inner_product=domain_weights の場合、sklearn backend では v1として
   - まずは `euclidean` のみサポート（警告+フォールバック）か
   - あるいは “重み付け前処理（sqrt(w)を掛ける）” を入れる（ただし慎重に）
   どちらにするかを docs に明記（捏造禁止）

## 受け入れ条件
- `backend=sklearn` で fit/transform/inverse が動く
- randomized が seed により再現可能
- stateに mean/modes/eigvals が保存される

## 検証
- 小さなdatasetで fit→reconstruct のRMSEが K増加で単調に下がる

---

## Review Map
- **変更ファイル一覧**: `src/mode_decomp_ml/plugins/decomposers/data_driven/pod.py`（sklearn PCA実装/solver追加）, `tests/test_decompose_pod.py`（randomized再現性+RMSE単調性テスト追加）, `docs/addons/30_POD_SUITE_SPEC.md`（sklearnのdomain_weights前処理方針明記）
- **重要な関数/クラス**: `_PODScalarDecomposer.fit`（sklearn PCA + solver切替）, `_PODScalarDecomposer._validate_backend`（randomized許可）, `test_pod_sklearn_randomized_seed_reproducible`, `test_pod_sklearn_rmse_decreases_with_k`
- **設計判断**: sklearn PCAに統一し、direct/snapshotsは`svd_solver="full"`、randomizedは`svd_solver="randomized"`でseedを反映。meanは既存の前処理（center後weight）を維持し、components_をmodesとして保存。
- **リスク/注意点**: inner_product=domain_weightsはsqrt(weights)前処理の近似で、integration_weights未定義時は警告フォールバック。randomizedはseed無しだと非再現になり得る。
- **検証コマンドと結果**: `pytest -q tests/test_decompose_pod.py` → 4 passed
