# Task 054 (P1): 任意領域：RBF基底展開（RBF expansion）Decomposer

## 目的
円/矩形以外の任意形状マスクでも使える汎用的な分解として、
RBF（Radial Basis Function）基底展開を Decomposer として導入する。

## 背景 / 根拠
- docs/00_INVARIANTS.md
- docs/01_ARCHITECTURE.md
- docs/10_PROCESS_CATALOG.md
- docs/04_ARTIFACTS_AND_VERSIONING.md

## 依存関係
- depends_on: work/tasks/050_decomposer_registry.md
- depends_on: work/tasks/030_data_domain_io.md

## スコープ
### In
- `rbf_expansion` decomposer を実装（mask-domain/points-domainを想定）
- 中心（centers）の選び方を config 化（grid/subsample/kmeans）
- 係数 = RBF重み w を返す
- regularization（ridge）を導入し、欠損やノイズに強くする
- coeff_meta に center座標と kernel params を保存

### Out
- 大規模N（>1e5）での高速化（近似法）はP2

## 実装方針（Codex向け）
### 1) 分解の定義
- 有効点座標 `X`（N,2）と値 `y`（N,）を取り出す
- center `C`（M,2）を決める
- Φ_{i,j} = φ(||X_i - C_j||; ε, kernel)
- w = argmin ||Φ w - y||^2 + λ||w||^2 （ridge）
- 係数 a = w（M次元）

### 2) center の選択
- `grid`: 有効点から等間隔サブサンプル
- `subsample`: ランダムにM点（seed固定）
- `kmeans`: 有効点をkmeansでMクラスタにして中心（P1/optional）

### 3) kernel
- gaussian: exp(-(r/ε)^2)
- multiquadric, inverse_multiquadric など（拡張）
- ε と λ は config で制御

### 4) inverse_transform
- 任意の評価点 X_eval（grid点）で `y_hat = Φ_eval w`
- mask外はNaN/0にする規約を config で

### 5) 実装上の注意
- Φは巨大になり得る → まずは tiny/中規模向け実装でOK
- 数値安定のために標準化（座標スケール）を導入（domain側の正規化）
- wのorderingはcenterの並び順なので、必ず center一覧を state に保存

## ライブラリ候補
- numpy
- scipy.linalg（ridge解法）
- scikit-learn（kmeans など optional）
- scipy.interpolate.RBFInterpolator（代替案：内部実装を使う場合）

## Acceptance Criteria（完了条件）
- [ ] rbf_expansion decomposer が registry に登録される
- [ ] center座標とkernel paramsが state/coeff_meta に保存される
- [ ] mask-domainで transform->inverse が動作する

## Verification（検証手順）
- [ ] 合成RBF場（既知w）で transform->inverse の誤差が小さい
- [ ] center数Mを変えると再構成精度が変わることを確認
