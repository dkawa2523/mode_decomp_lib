# Testing / CI（テスト方針）

## 1. 目的
- 分解・逆変換・係数後処理・学習が **比較可能** であることを保証する
- “silent failure” を早期に検知する

## 2. テスト種別
### Unit
- `Decomposer`:
  - transform -> inverse_transform の再構成誤差が閾値以下
  - 係数shape/orderingが `coeff_meta` と一致
- `CoeffPost`:
  - fit/transform/inverse_transform の round-trip が成立（PCA等）
- `Preprocess`:
  - mask が壊れない
  - 期待するshapeが維持される

### Integration
- `decomposition -> preprocessing -> train -> inference` の一連が動く（smoke）
- pipeline で method を切り替えても動く

## 3. 数値系の注意
- 浮動小数は `np.allclose` を使う（tolをconfig化）
- FFT/波形は位相・符号の規約があるので `coeff_meta` を常に参照

## 4. CI（将来）
- GitHub Actions等で `pytest -q` を回す前提で設計する
- 大きなデータに依存しない “tiny dataset fixture” を用意する
