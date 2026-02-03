# Security & Secrets

## 原則
- APIキー等の秘密情報を repo にコミットしない
- 自動化（autopilot）では sandbox を維持する
- ネットワークアクセスは必要最小限

## Codex自動化の安全設定（推奨）
- sandbox: workspace-write（workspace外を書けない）
- approval: never は止まりにくいが危険。feature branch運用/ログ/レビューが前提。
- approval: on-failure は安全寄り（止まる可能性あり）

## ログと機密
- `work/.autopilot/` にプロンプト/出力が保存される
- 機密が混じる場合はログ保存を制限する（autopilotの設定でOFF可）

## TODO（対象repo用）
- 社内/顧客データの取り扱いルール
- CI環境における秘密管理
