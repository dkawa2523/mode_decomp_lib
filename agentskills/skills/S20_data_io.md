# S20: Data I/O / Split / Audit

## 目的
- データの読み込み・正規化・split・監査を安定させる

## 手順
- データ契約（入力列/型/主キー/目的変数）を docs/02 と docs/10 に明記
- splitは保存し、seed固定
- auditで重複/漏洩/欠損/外れ値を可視化

## DoD
- audit_report が生成できる
- split再現できる
