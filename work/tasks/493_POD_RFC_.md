# TASK 493: POD拡張RFC（スコープ固定・既存構造確認・命名確定）

## 目的
Task492完了状態から、POD系拡張（Weighted/Gappy/Randomized/Incremental）を **最小の設計変更**で追加するための合意を固定し、以降の実装がブレないようにします。

## 背景（このタスクで必ず確認）
- 既存の decomposer（解析基底）と POD系（データ駆動）を **コード配置で区別**する必要がある
- YAML増殖・出力階層肥大は絶対に起こさない（run.yaml 1枚運用を維持）
- POD実装でありがちな「PODクラスに分岐を詰め込む」を避ける（Domain/Decomposer/Gappy/CoeffPostに責務分割）

## 作業内容
1. `docs/addons/30_POD_SUITE_SPEC.md` と `31_POD_CONFIG_MINIMAL.md` を読み、**現状コード**（Task492時点）と不整合がないか洗い出す
2. 不整合がある場合は、ドキュメント側に TODO と理由を追記（捏造禁止）
3. 既存の以下を特定し、メモに残す（docsに追記でOK）
   - Decomposer registry の実体（どこで名前→クラス解決しているか）
   - Domainの実体（rectangle/disk/mask/meshのどこがあるか）
   - run.yaml 1枚運用の入口（CLIやProcess）
   - 出力のartifact契約（runs配下の構造）
4. 追加するPOD系の“名前”を確定（後で変えない）
   - `decomposer=pod`（PODDecomposer）
   - `decomposer=gappy_pod`（GappyPODDecomposer）
   - `options.rank_select.*`, `options.coeff_normalize.*`, `options.mode_weight.*`

## 受け入れ条件（Acceptance Criteria）
- docs/addons/30,31,32,33 に TODO/追記が反映され、Task492時点との不整合が“見える化”されている
- POD系の plugin 名称・config key 名称が固定されている（以降タスクで流用）

## 検証（Verification）
- `python -m compileall src` が通る（ドキュメント編集のみなら不要だが、誤ってコード壊してないこと確認）
- `python tools/codex_prompt.py list` が壊れていない（queue JSONに触らない）

## Review Map
- 変更ファイル一覧: `docs/addons/30_POD_SUITE_SPEC.md`, `docs/addons/31_POD_CONFIG_MINIMAL.md`, `docs/addons/32_POD_VISUALIZATION_STANDARD.md`, `docs/addons/33_CLEARML_READY_NOTES.md`, `docs/10_PROCESS_CATALOG.md`, `work/queue.json`
- 重要な関数/クラス: `src/mode_decomp_ml/plugins/registry.py`（decomposer registry）, `src/mode_decomp_ml/domain/__init__.py`（DomainSpec/build_domain_spec）, `src/mode_decomp_ml/run.py`（run.yaml entrypoint）, `src/mode_decomp_ml/pipeline/utils.py`（RunDirManager）
- 設計判断: 現状コードとの差分はTODOで明示し、命名ロック（pod/gappy_pod + options.*）はdocsに固定して後続タスクへ委譲
- リスク/注意点: `pod`/`gappy_pod` aliasや `options.*` は未実装のため、run.yaml例は現時点で直接は動かない
- 検証コマンドと結果: `python3 -m compileall src`（OK）, `python3 tools/codex_prompt.py list`（OK）
- 削除一覧: `src/**/._*`（AppleDouble由来の不要ファイルを削除）
