# Autopilot Troubleshooting（詰まった時）

## 1) zipが壊れる/空
- 生成したディレクトリに実ファイルがあるか確認
- zipは `zipfile` で再生成

## 2) 係数shapeが合わない
- `coeff_meta.json` の index/shape と実装を照合
- FFTは複素→実数変換の規約を固定（mag/phase or real/imag）

## 3) 再構成誤差が大きい
- domainの座標系/重みが一致しているか
- mask内の内積定義（重み）が正しいか
- 分解の次数/周波数上限が低すぎないか

## 4) 学習が不安定
- coeff_post の標準化（per_mode）を入れる
- 目的変数が z か a かを明確にし、lossを整合させる

## `Not inside a trusted directory and --skip-git-repo-check was not specified.`

**症状**: `codex_err_*.txt` に上記が出て、Autopilotが停止する。

**原因**: Codex CLI が「gitリポジトリ内（または trusted dir）」での実行を要求している。
GitHubからZIPで取得した作業フォルダ等で `.git/` が無い場合に起きやすい。

**対処（推奨）**:
- このoverlay v7 以降では、Codexが対応していれば自動で `--skip-git-repo-check` を付与するため、通常は再実行で解消します。
- それでも解消しない場合は、以下のいずれかを実施してください：
  1) 作業ディレクトリを git repo 化する（例: `git init && git add -A && git commit -m "init"`）
  2) `CODEX_GLOBAL_FLAGS` を手動指定する（例: `CODEX_GLOBAL_FLAGS="--ask-for-approval never --skip-git-repo-check"`）

**確認**:
- `work/.autopilot/<timestamp>/codex_help.txt` に `--skip-git-repo-check` が出ているか
- `RESOLVED_CODEX_GLOBAL_FLAGS` に `--skip-git-repo-check` が含まれているか
