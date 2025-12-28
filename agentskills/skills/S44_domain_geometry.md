# S44: DomainSpec（rect/disk/mask/points/mesh）と座標/重み

## 目的
- 境界形状が変わっても、分解が壊れないように domain を明示的に扱う。

## やること手順
1. **domain type を明確に**
   - rect: 均一格子（x,y）
   - disk: r<=1 に正規化した座標 + disk mask
   - mask: 任意mask（rect座標を流用）
   - points: (N,2)
   - mesh: vertices/faces

2. **座標正規化**
   - disk: 中心/半径で正規化して r<=1 を作る
   - points: スケール統一（RBF等で重要）

3. **内積重み**
   - disk: 面積要素 `r` を重みに含む（極座標）
   - rect: dx*dy（均一なら定数）

4. **validate**
   - mask shape, field shape, domain consistency をチェックして早期に落とす

## 事故りやすい点
- diskなのに中心がずれている（r計算が狂う）
- dx,dy を無視して内積が歪む
- points の順序が変わり係数の意味が変わる

## DoD
- domainごとに `to_coords()` 等があり、decomposerがそれを使う
- validate があり、例外がわかりやすい
