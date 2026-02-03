# TASK 503: 周辺技術: rank自動選択（energy閾値 + 任意CV、optionsでON/OFF）

## 目的
rank選択の自動化を追加し、K（モード数）を人手で迷わず決められるようにします。
- energy閾値（必須）
- CV（任意・高コスト）

## 作業内容
1. energy法：
   - eigvals から累積寄与率を算出
   - `options.rank_select.energy` で最小Kを決定（max_modes上限あり）
2. CV法（任意）：
   - K候補を少数（例：[8,16,32,64]）に絞って評価
   - metricは field誤差（reconstruct or predict後の誤差）から選べる
3. 追加設定は `options.rank_select.*` に閉じ込める（新しいyamlファイルを増やさない）
4. 選択されたKを state/meta/metrics に記録

## 受け入れ条件
- `options.rank_select.enable=true` でKが自動決定される
- energy法で `energy_cum.png` が生成され、K選定根拠が残る
- CV法は有効/無効を明確に切り替えられる（デフォルトOFF）

## 検証
- 同じdatasetで energy=0.9 と 0.99 のKが変わることを確認
- CV有効時に実行時間が増えるが、結果が保存される
