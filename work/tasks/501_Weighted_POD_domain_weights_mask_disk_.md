# TASK 501: Weighted POD（domain_weights運用、mask/diskでの安定化）

## 目的
Weighted POD（domain-aware）を “Domainが重みを供給する” 方式で完成させます。
POD側にdomain分岐を増やさない。

## 作業内容
1. `inner_product=domain_weights` のときに Domain.integration_weights() を取得
2. backendごとに扱いを統一：
   - modred backend: `inner_product_weights` にそのまま渡す
   - sklearn backend: v1では “サポートしない→フォールバック” か “sqrt(w)で前処理” のどちらかで統一（ドキュメントも更新）
3. mask domain のとき、重みがmask込みであることを確認
4. 評価指標に、mask内/外の誤差分解（可能なら）を追加

## 受け入れ条件
- disk/maskで Weighted POD が動く
- Euclidean POD と比較して、特にmask/disk境界近傍の誤差が安定する（定性的でOK）
- 実装が Domain→POD の単方向依存になっている

## 検証
- disk/maskのケースで recon_error を出し比較（figures/ へ保存）
