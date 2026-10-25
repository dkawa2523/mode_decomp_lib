# アーキテクチャレビュー（2026-09-25）

- 対象: `master`（a101c25c）の `src/`（`mode_decomp_ml` と `processes`）および `tests/`
- 手段: 今回導入した Ruff / pyrefly / radon / import-linter の結果、コード読解、pytest 実行
- 観点: (1) 計算単位の独立性 (2) 連携計算（合成）の有用性 (3) 責務の明確さ (4) 契約・診断・テストによる複雑化

## 0. 結論

| 観点 | 評価 | 要旨 |
|---|---|---|
| 計算単位の独立性 | 概ね良い | decomposer 28 モジュール・codec・model は基底クラス経由でほぼ独立。違反は private 関数の横借り 3 本と、保存用 import 1 行の逆流のみ |
| 連携計算の有用性 | 弱い | 「条件 → 係数 → 場」の合成が部品として存在せず、processes の中に少なくとも 4 系統コピーされている |
| 責務の明確さ | 計算層は明確、orchestration と診断が肥大 | 循環的複雑度 100 超・500 行級の関数が 4 つ。processes に数値アルゴリズムと指標定義が混入 |
| 契約・診断・テスト | 契約と診断が複雑さの主因 | 契約が型でなく「文字列キーの dict ＋ 実行時検査 ＋ 文書」で表現され、検査とテストが膨らむ。診断は計算と結合し、握りつぶしと巻き込み失敗の両方が起きている |

計算単位そのものは健全です。一方で「単位の契約（インターフェース・メタデータ・保存）」と「単位の合成」が型としてコード化されておらず、その不足を processes 側の分岐・検査・診断コードが埋めています。複雑さの大半はここから来ています。

## 1. 導入したツールと現状値

### 1.1 変更ファイル

- `pyproject.toml`: dev 依存グループ、各ツールの設定、import-linter 契約 7 本、`[tool.uv.extra-build-dependencies]`（2.5 節参照）
- `uv.lock`: 追加パッケージのみ（既存パッケージのバージョンは不変）
- `docs/07_TESTING_CI.md`: 実行方法を追記

### 1.2 実行結果（ベースライン）

| ツール | 結果 | 主な内訳 |
|---|---|---|
| Ruff（設定したルール） | 339 件（うち src 236 件） | src: blind except 71、未使用 import 42、C901 29、分岐過多 27、try/except/pass 23、文過多 19、未定義名 1 |
| pyrefly | 237 エラー | `BaseDecomposer` に `transform` / `inverse_transform` が無い 32、optional 依存を `None` で置く形に起因 約 55、文字列引数と Literal 型・overload の不一致 49、未導入の optional 依存 13 |
| radon cc | 937 ブロック、平均 B（6.27） | F 10、E 5、D 22、C 82 |
| radon mi | C 評価 9 ファイル | MI 0: `data/datasets.py`, `plugins/decomposers/pod.py`, `processes/decomposition.py`, `processes/_decomposition_plots.py` |
| import-linter | 契約 7 本中 4 維持・3 違反 | 1.3 節 |
| pytest（参考） | 142 成功・3 失敗・10 スキップ | スキップは torch / pywt / xgboost などの optional 依存が未導入のため |

### 1.3 import-linter 契約

| 契約 | 結果 |
|---|---|
| `processes` → `mode_decomp_ml` の一方向 | 維持（ただし B6 の動的 import は検出対象外） |
| ライブラリの層構造（entrypoint > tracking > pipeline > evaluate/viz > plugins > registry > data > domain > kernel） | 違反 1 本: codecs/basic → pipeline |
| decomposer の相互独立 | 違反 3 本: gappy_pod → pod、gappy_graph_fourier → graph_fourier、helmholtz_poisson → helmholtz |
| codec の層（auto > offset_residual_pack > 各 pack > basic） | 維持 |
| model 実装の相互独立 | 維持 |
| process 段階の相互独立（合成は pipeline のみ） | 維持 |
| 計算層は matplotlib / Hydra / OmegaConf / Optuna / YAML に依存しない | 違反: codecs/basic → pipeline 経由で OmegaConf と YAML を推移的に取り込む |

## 2. 所見

### 2.1 計算単位の独立性

良い点

- registry とデコレータ登録により、decomposer / codec / model / coeff_post は他の実装を知らずに追加できる（[registry.py](../../src/mode_decomp_ml/plugins/registry.py)）。
- 合成も registry 経由でできている。offset_residual は内側の分解器を `build_decomposer` で組み立て、直接 import しない（[offset_residual.py:209](../../src/mode_decomp_ml/plugins/decomposers/offset_residual.py#L209)）。
- codec は「dispatcher > 合成 pack > 個別 pack > basic」の層がきれいに守られ、model 実装同士も独立している。
- 計算層は描画・設定・チューニングのフレームワークを直接 import していない。basis キャッシュもインスタンス単位で、グローバル状態が無い。

課題

- **A1 保存用 import 1 行による逆流。** [codecs/basic.py:11](../../src/mode_decomp_ml/plugins/codecs/basic.py#L11) が `write_json` のためだけに `mode_decomp_ml.pipeline` を import している。pipeline の `__init__` 経由で OmegaConf・YAML・artifact 層まで計算層に入り込む。import-linter の 2 契約違反はこの 1 行が原因。
- **A2 兄弟モジュールの private 関数の横借り。** [gappy_pod.py:13](../../src/mode_decomp_ml/plugins/decomposers/gappy_pod.py#L13) は `_PODScalarDecomposer` を、[gappy_graph_fourier.py:13](../../src/mode_decomp_ml/plugins/decomposers/gappy_graph_fourier.py#L13) は `_build_laplacian` など 6 個を、[helmholtz_poisson.py:13](../../src/mode_decomp_ml/plugins/decomposers/helmholtz_poisson.py#L13) は `_fft_helmholtz` と `_grid_spacing` を直接 import している。本体側の内部変更が派生手法を壊す経路になる。既に `pod_utils` / `zernike_shared` という共有モジュールの型があるので、そこへ寄せれば契約は通る。
- **A3 decomposer だけインターフェースが宣言されていない。** model・coeff_post・codec の基底は `fit/predict`、`fit/transform/inverse_transform`、`encode/decode` を宣言している。しかし [BaseDecomposer](../../src/mode_decomp_ml/plugins/decomposers/base.py#L76) には `transform` / `inverse_transform` が無い。pyrefly の 32 エラーはこれで、最重要の計算単位の契約が「実装の慣習」に留まっている。
- **A4 時間的結合（transform が状態を書き換える）。** `transform` が `_coeff_shape`・`_coeff_meta`・`_orig_hw` などを更新し、`inverse_transform` はそれに依存する。その結果「transform must be called before inverse_transform」のガードが 29 か所に複製されている。逆変換の出力形も「最後に transform した入力」が 2D か 3D かで変わる。fit で確定する情報は fit で保持し、入力に依存する情報は戻り値で返せば、ガードごと消せる。
- **A5 domain 互換表の集中管理。** [validate_decomposer_compatibility](../../src/mode_decomp_ml/domain/__init__.py#L368) が全 decomposer 名・対応 domain・エラーメッセージの表を持つ。手法を追加するたびに domain 層の編集が必要で、開放閉鎖原則に反する。各 decomposer クラスに `supported_domains` を持たせ、検証は基底で行うのが自然。
- **A6 保存レイアウトを計算単位が知っている。** 各基底の `save_state` がパスを直書きし、しかも不揃い（model だけ `run_dir/model/`、他は `outputs/states/<name>/`）。同じ知識が [loaders.py:24](../../src/mode_decomp_ml/pipeline/loaders.py#L24) に旧レイアウトへのフォールバック付きで重複している。計算単位は状態を返す・受け取るだけにし、パスは pipeline 層で一元管理すると層構造と一致する。

### 2.2 連携計算（合成）の有用性

- **B1 合成された計算が部品として存在しない。** 「cond → model → coeff_post の逆変換 → codec の復号 → decomposer の逆変換 → preprocess の逆変換 → field」は [inference.py:475](../../src/processes/inference.py#L475) のクロージャとしてしか存在しない。同等の連鎖が decomposition・train・診断プロットにも書き直されており、processes 内の逆変換・復号・予測の呼び出しは 84 か所ある。ノートブックや最適化ループから `predict_fields(cond)` のように再利用する手段が無い。5 つの状態（preprocess / decomposer / codec / coeff_post / model）をまとめて持つ合成オブジェクト（例: `encode` / `decode` / `predict_fields` / `save` / `load` を持つクラス）を `mode_decomp_ml` 側に置けば、processes はそれを呼ぶだけになる。
- **B2 processes に数値アルゴリズムが入っている。** docs/01 の「processes は組み立てのみ」に反して、次のコードがある。
  - 係数を K 個に切り詰めた再構成 R²（R²@K）。[decomposition.py](../../src/processes/decomposition.py) の 2 ブロック（計約 540 行）、[_decomposition_plots.py](../../src/processes/_decomposition_plots.py) の r2_vs_k と dashboard、[train.py](../../src/processes/train.py) の k_list 評価の少なくとも 4 か所にある。係数レイアウト（CK / CHW / K）× offset_residual の有無の分岐ごとコピーされている。
  - 分割・K-fold・RMSE / MAE / R² の自前実装（[train.py:95](../../src/processes/train.py#L95) 以降）。`pipeline.utils.split_indices` や scikit-learn と重複する。
- **B3 R² の定義が 4 つある。** `evaluate.field_r2` はサンプルごとの R² の平均、`viz.masked_weighted_r2` は全点をプールした（重み付き）R²、`train._r2` は出力次元の分散重み付き R²、さらに散布図用の `_r2_1d` がある。同じ予測でも metrics.json と図の R² が一致しない可能性があり、docs/00 の比較可能性（指標定義の固定）に反する。
- **B4 processes が特定のプラグイン名を知っている。** codec 名（[decomposition.py:173](../../src/processes/decomposition.py#L173) の `auto_codec_v1` など）や手法名（`fft2` / `fft2_lowpass`、R²@K の分岐 5 か所）で分岐している。手法を追加すると processes の修正が必要になる。係数の並び順と切り詰め方は decomposer（または coeff_meta の並び順情報）が提供すべき情報。
- **B5 設定合成が 3 系統ある。** Hydra による合成（[cli/run.py](../../src/mode_decomp_ml/cli/run.py)）、Hydra 不在時の自前合成フォールバック（同ファイル。hydra-core は必須依存なので通常は通らない経路）、pipeline sweep 用の `defaults` 解決の再実装（[processes/pipeline.py:69](../../src/processes/pipeline.py#L69)）の 3 つ。加えて run.yaml からの変換（[run.py](../../src/mode_decomp_ml/run.py)）もある。同じ config でも経路によって解決結果が変わりうる。sweep は Hydra の compose API に寄せれば 1 系統にできる。
- **B6 ライブラリが `processes` を文字列で動的 import する**（[cli/run.py:53](../../src/mode_decomp_ml/cli/run.py#L53)）。依存の向きが逆転しているうえ、静的解析には見えない。import-linter の一方向契約が「維持」と出るのはこのため。`processes` はトップレベル名として汎用的すぎ、site-packages に同名で入る。
- 良い点: 段階間は artifact（run_dir）で受け渡され、各段階を単独で実行できる（docs/00 不変条件 2）。段階同士も pipeline 以外は互いを import していない（契約維持）。

### 2.3 責務の明確さ

**C1 巨大関数（radon）**

| 関数 | 循環的複雑度 | 行数 |
|---|---|---|
| `processes/decomposition.py` `main` | 183 | 766 |
| `processes/_decomposition_plots.py` `_plot_validity_diagnostics` | 131 | 487 |
| `processes/_decomposition_plots.py` `_plot_key_decomp_dashboard` | 128 | 485 |
| `processes/train.py` `main` | 122 | 589 |
| `data/datasets.py` `CsvFieldsDataset.__init__` | 64 | 156 |
| `domain/__init__.py` `build_domain_spec` | 59 | 258 |
| `evaluate/__init__.py` `coeff_energy_cumsum` | 55 | 124 |
| `plugins/decomposers/pod.py` `_PODScalarDecomposer.fit` / `_fit_incremental` | 54 / 51 | 175 / 207 |
| `processes/inference.py` `main` | 43 | 310 |

上位 4 つは processes 層で、いずれも「計算・指標・描画・保存」を 1 関数で行っている。計算層で F 評価なのは POD 系、CSV データセット、domain 生成で、いずれも種類ごとの if 分岐を 1 関数に並べた形（domain は mesh / sphere_grid / rectangle / disk / annulus / mask を 1 関数で扱う）。

- **C2 `pipeline/utils.py` が雑多。** 設定アクセス、パス解決、run_dir 管理、JSON I/O、dataset から配列への変換、マスク合成、分割、git 情報、meta 生成が同居している。`PROJECT_ROOT = parents[3]`（[utils.py:25](../../src/mode_decomp_ml/pipeline/utils.py#L25)）はリポジトリ配置を前提にしており、非 editable インストールでは `configs/` を見失う。
- **C3 重複ヘルパー。** `_load_yaml` 7 か所、`_require_task_config` 5、`_to_container` 4、`_deep_update` 3、basis キャッシュの実装 9 decomposer、`_resolve_scaler` / `_resolve_target_space` 3 model、旧レイアウト候補の `_CONFIG_CANDIDATES` 2。
- **C4 リポジトリ直下の `mode_decomp_ml/`（shim）が `src/mode_decomp_ml` を覆い隠す。** import-linter はこのため `src/` から実行する必要があった。`uv sync` で editable インストールされる現在は不要なはず。
- **C5 非推奨 shim**（`mode_decomp_ml/coeff_post`、`mode_decomp_ml/models`）が残っている。

### 2.4 契約・診断・テストによる複雑化

契約

- **D1 契約が型ではなく dict と実行時検査で表現されている。** `coeff_meta` や各プラグインの設定は自由形式の dict で、正しさは `raise` と文書と契約テストで担保されている。decomposer の `raise` は 744 件（設定パース約 200、変換時の形状・状態検査約 200、fit 約 120）あり、おおよそ 11 行に 1 回の割合。`CoeffMeta` を dataclass（最低でも TypedDict）にし、各プラグインの設定を dataclass で 1 回だけ検証すれば、検査コードと契約テストの大半を型に吸収できる。
- **D2 形式だけの契約。** `coeff_post.fit(A, split="train")` と `preprocess.fit(..., split="train")` は文字列 "train" を照合するだけで（[coeff_post/basic.py:22](../../src/mode_decomp_ml/plugins/coeff_post/basic.py#L22)）、データが本当に学習分割かは検証できない。呼び出し側は常にリテラル "train" を渡している。train-only fit は分割を持つ orchestration 側の責務であり、計算単位の引数から外せる。
- **D3 文書の契約とコードの乖離。** docs/00 不変条件 1「コード内に暗黙のデフォルトを増やさない」に対し、`cfg_get` 487 件中 328 件がコード内デフォルトを持つ（processes 150、models 68、decomposers 44）。30 本超の docs で宣言された契約が機械的に検査されていないため、守られていない。契約は型・import-linter・lint ルールに落とせるものから落とすべき。

診断

- **D4 診断コードの比重が大きい。** 描画系（viz と `_decomposition_plots.py`）だけで SLOC の 15.5%（3,175 行）を占め、orchestration（processes、14.2%）より大きい。train / inference / preprocessing / pipeline にも独自の `_plot_*` がある。
- **D5 握りつぶしが実バグを隠している。** src に blind except が 71、try/except/pass が 23 ある。例として [viz/coeff_plots.py:496](../../src/mode_decomp_ml/viz/coeff_plots.py#L496) の `plot_coeff_hist_by_mode` は、別モジュールにしかない `_colormap_with_bad` を呼ぶため必ず NameError になる。呼び出し側（[_decomposition_plots.py:820](../../src/processes/_decomposition_plots.py#L820)）が `except Exception: pass` で握りつぶすので、図が出ないことに誰も気づかない（Ruff の F821 と pyrefly の unknown-name で検出）。decomposition の R²@K 計算も丸ごと `except Exception: pass` の下にあり、失敗すると指標が黙って欠ける。
- **D6 逆に、ガードの無い診断が計算段階を落とす。** [_decomposition_plots.py:776](../../src/processes/_decomposition_plots.py#L776) の係数ヒストグラムはビン数 40 固定で、分散がほぼゼロの係数次元では numpy 2.x が ValueError を出す。分解と指標の計算は成功しているのに decomposition 段階が failed になり、pipeline では後続の preprocessing と train がスキップされる。`test_pipeline_continue_on_error_preprocessing_failure_writes_leaderboard` の失敗はこれが原因で、テストは正しく実バグを捉えている。
- 方針: 診断は「計算結果を受け取って描くだけ」の独立した段階にする。失敗は握りつぶさず warning ログと steps の meta に記録し、段階の成否には影響させない。これで D5 と D6 の両方が解消する。

テスト

- **E1 設計は概ね良い。** decomposer ごとの小さな往復（transform → inverse_transform）テストが中心で、計算単位の独立性とよく対応している。全体の実行も約 14 秒と軽い。
- **E2 失敗 3 件の内訳。** 2 件は scipy 1.17 で削除された `special.sph_harm` をテストが直接呼んでいるため（[test_decompose_spherical_harmonics.py:73](../../tests/test_decompose_spherical_harmonics.py#L73)、[test_decompose_spherical_slepian.py:45](../../tests/test_decompose_spherical_slepian.py#L45)）。本体は新 API の `sph_harm_y` に対応済み。残り 1 件は D6 の実バグ。
- **E3 R²@K のテストが process 全体を通す形になっている。** ロジックが processes にあるためで、B2 のとおり evaluate に移せば単体テストにできる。
- **E4 重複と環境依存。** `_rectangle_domain` が 16 ファイル、`_disk_domain` が 9 ファイルに重複している（conftest の fixture にすれば解消）。一部のテストは sys.path を書き換えて `processes` を import している。pytest が依存に宣言されていない。
- **E5 coeff_meta の契約テストは 16 本のほぼ同形のテストで、キーの有無を見るだけ。** D1 の型化で 1 本の parametrize テストに縮む。

### 2.5 その他の検出事項

- [scripts/bench/run_matrix.py:270](../../scripts/bench/run_matrix.py#L270) が構文エラー（f-string の引用符がバックスラッシュでエスケープされて混入）。これを呼ぶ `run_full.sh` と `run_p0p1_p2ready.sh` も動かない。
- optional 依存 modred は sdist のみで、最新の setuptools から削除された `pkg_resources` を必要とする。このため依存を 1 つでも変えると `uv lock` の再解決が失敗する状態だった。今回、modred のビルド時だけ setuptools を 81 未満に固定した（`[tool.uv.extra-build-dependencies]`）。
- `datetime.utcnow()` の非推奨警告（`pipeline/steps.py`、`pipeline/utils.py`）。
- optional 依存をモジュール先頭で `try: import torch` / `except: torch = None` とする形が、pyrefly エラー約 55 件の原因。関数内での遅延 import か `TYPE_CHECKING` の併用で、型と実行を両立できる。

## 3. 推奨対応（優先順）

P0: 小さく直せて効果が大きいもの

1. 実バグの修正: `_colormap_with_bad` の import、ヒストグラムのビン数（ゼロ幅データへのガード）、`run_matrix.py` の構文、scipy 1.17 対応のテスト（`importorskip` か `sph_harm_y` での比較）。
2. `write_json` などの I/O を pipeline より下の小モジュールへ移し、codecs から pipeline への import を消す。import-linter の 2 契約が通る。
3. decomposer 間の private 共有 3 本を `pod_utils`、新設の `graph_utils`、`helmholtz_shared` へ移す。decomposer 独立契約が通る。
4. 以上で import-linter を CI の必須チェックにできる。新たな横依存や逆流を防げる。

P1: 構造

5. `BaseDecomposer` に `transform` / `inverse_transform` を抽象メソッドとして宣言する。逆変換に必要な情報は fit 時の状態か coeff_meta から取る形にし、29 か所の順序ガードを不要にする。
6. 合成オブジェクト（B1）を `mode_decomp_ml` 側に作り、状態の保存・読込もそこで一元化する（A6）。processes はそれを呼ぶだけにする。
7. R²@K を「decomposer が切り詰め方（並び順）を提供し、evaluate が指標を計算する」形に移す（B2・B4）。processes から数百行が消える。
8. 指標定義を evaluate に一本化し、viz は値を受け取って描くだけにする（B3）。
9. `supported_domains` をクラス属性にし、domain の中央表を削除する（A5）。
10. sweep の設定合成を Hydra の compose API に寄せ、CLI の自前フォールバックを削除する（B5）。
11. 診断段階を分離する（D4〜D6）。decomposition / train の main を「計算」「指標」「診断」「保存」の関数に分ける。

P2: 契約とテストの整理

12. プラグイン設定と `CoeffMeta` を型にする（D1）。デフォルト値は型定義に集約し、resolved config に残す（D3）。
13. `split="train"` 引数を廃止する（D2）。
14. blind except を狭い例外とログに置き換え、Ruff の BLE / S110 をゲートにする。
15. テスト fixture の共通化、pytest の dev 依存への追加、sys.path 操作の削除（E4）。
16. ルート shim と非推奨 shim を削除する（C4・C5）。

## 4. ツールの運用提案

- import-linter: P0 の 2・3 を終えると全契約が維持になる。そこから CI 必須にする。
- Ruff: まず `E4 / E7 / E9 / F / B` をゲートにし、BLE・S110 は P2-14 の後でゲートに加える。複雑度系（C90・PLR09）は当面レポート用途にし、件数の推移を追う。
- pyrefly: P1-5 で 32 件、optional 依存の import 形の見直しで約 55 件が消える見込み。残りは numpy / scikit-learn の Literal 型に文字列を渡している箇所が中心で、設定の型化（P2-12）と同時に解消できる。
- radon: F / E ランクの関数数（現在 15）を指標として追う。

## 付録: 再現コマンド

```sh
uv sync
uv run ruff check .
uv run pyrefly check
uv run radon cc src
uv run radon mi src
cd src && uv run lint-imports --config ../pyproject.toml
uv run --with pytest pytest -q
```
