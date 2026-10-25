# 校正済みの場の予測帯

`uncertainty.method: split_conformal` は、学習済みsurrogateの原field残差を独立した校正sampleで校正し、条件からの点予測に上下限を付けます。Ridge等を含む全回帰器に使え、分解・係数後処理・回帰の偏りを含めた総誤差が対象です。点予測を置き換えず、標準偏差・Gaussian分布・Bayesian posteriorとは呼びません。

## YAMLとデータの分離

```yaml
uncertainty:
  enabled: true
  method: split_conformal
  coverage: 0.9
  scope: simultaneous
  normalization: channel
  calibration_size: 0.2
```

`normalization` は `none` / `channel` です。channelではproper fitting trainの観測truthだけからchannel標準偏差を学び、残差を規格化します。定数channelのscaleは1にします。校正truthからscaleを選びません。

outer trainから校正用sampleを**候補選択・CV/HPOの前に**分けます。候補選択と全変換のfitにはproper fitting trainだけを使い、勝者の最終fit後に未使用の校正sampleで帯を校正します。その後に再fitしません。final testはcoverageと幅の独立評価に使います。`calibration_seed` と `calibration_method` を指定でき、分割方法・group/time・時間gapの扱いは実行設定と保存splitで確認します。

pipelineとsurrogate解析が対象で、representation/imputationへの有効化は拒否します。推論時は `uncertainty.enabled: true` で校正済みモデルの帯を要求します。falseなら従来どおり点予測だけです。校正を持たないモデルへ帯を要求するとエラーになります。

## スコアと有限標本の分位点

calibration sample数をn、指定coverageをc、学習時に固定したchannel scaleをsとします。

```text
simultaneous: score[i] = max over valid space/channel |truth[i] - prediction[i]| / s
pointwise:    score[i,p,c] = |truth[i,p,c] - prediction[i,p,c]| / s[c]
k = ceil((n+1) * coverage)
q = calibration scores の k 番目に小さい値
lower, upper = unchanged point prediction ± q * s
```

simultaneousはsampleごとの最大残差を1つのスコアにし、固定された有効場全体が帯へ入る事象を扱います。pointwiseは地点・channelごとに別の分位点を使い、それぞれの座標のcoverageを扱います。pointwiseの90%を「場全体が90%で入る」と読み替えません。

線形補間した分位点や通常の90 percentileは使いません。`k > n` の場合は有限の帯を作れないためエラーです。例えば90%には少なくとも9校正sampleが必要です。不足時に最小／最大へclipしたり、帯を無限大で成功扱いにしたりしません。

有限標本補正と、固定モデルに対するexchangeability下のmarginal coverageは [Angelopoulos & Bates, §1.1 / Appendix D](https://arxiv.org/abs/2107.07511) に基づきます。本実装は場の最大標準化残差をスコアにする適用です。独立同分布sampleはその十分条件ですが、group内相関・時系列依存・分布変化について自動の保証はありません。時間gapやgroup分割だけで保証が成立したとは扱いません。

## 数値APIとsupport

```python
from ml_platform_mode_decomp.uncertainty import FieldCalibrator

# fitted_surrogate / proper_train / calibration / test は独立分割済み。
calibrator = FieldCalibrator(coverage=0.9, scope="simultaneous", normalization="channel")
calibrator.fit_batches(
    calibration,
    fitted_surrogate.iter_predict(calibration.conditions),
    training=proper_train,
)
points = fitted_surrogate.predict(test.conditions)
lower, upper = calibrator.intervals(points)
coverage_and_width = calibrator.evaluate(test, points)
metadata = calibrator.summary()
```

`fit_batches()`はcalibrationと同じsample順の予測batchを一度だけ消費します。既に予測配列がある場合は`fit(calibration, predictions, training=proper_train)`を使えます。通常の`surrogate.calibrate()`とYAMLからの較正もbatch経路を使い、全較正予測を保持しません。追加のYAML設定は不要です。

`surrogate.calibrate(..., training=proper_train)`へ渡すtraining ID集合は、そのモデルのfit ID集合と一致させます。別caseや評価caseをchannel規格化に混ぜません。較正・coverage参照はgeometry・field kind・成分・単位が一致する場に限定します。保存済み旧モデルの較正stateは、モデルが持つtarget kindを引き継いで読み込みます。

simultaneousはsampleごとの最大残差だけを保持します。pointwiseは正確な順位kを得るため、各地点・成分につき`min(k, n-k+1)`個のscoreと作業batchを保持します。90%指定ならおおむねnの10%、中間のcoverageではおおむね半数のscoreが必要です。近似分位点・再推論・一時diskへ変更せず、固定supportと較正数に比例する容量は残ります。coverage評価は全件の上下限を保持せず、保存時に各batchの帯を生成します。

形状は元のNHWCまたはmesh NVCで、帯は元fieldの単位です。supportはgeometry mask内かつ面積重みが正の固定地点です。両scopeとも校正・coverage評価にその全地点の既知truthを要求し、未観測地点の残差を0として埋めません。support外の上下限はNaNです。geometry・成分・単位が異なる場へ校正を流用しません。

`summary()` は設定・校正数・分位点rank・thresholdの範囲・固定scale・保証の条件を返します。pointwiseの座標別threshold配列は校正オブジェクトが保持します。`evaluate()` は等sampleの全場coverage、任意のsample/面積/channel重みによる点coverage、channel別の物理幅を区別します。異単位のchannelを混ぜた物理幅平均は返さず、channel別幅と規格化幅を確認します。これらは観測したtest上の実測であり、そのtestや各条件値へのcoverage保証ではありません。

帯の幅は校正後に固定され、条件ごとの難しさに適応するheteroscedasticモデルや、epistemic/aleatoricの分解は含みません。比較用途で帯幅・coverageを見て候補や設定を変更した場合は、その校正を再利用せず独立にやり直してください。

元点を主評価にする`metrics.reference:observations`でも、帯の対象はgrid固定supportのままです。calibration/coverage参照にその全正measure地点の既知truthがなければ拒否します。帯を黙って省略したり、元点の較正帯へ変換したりしません。元点採点だけの利用は可能です。

per_sample CSVには全場を覆ったか、sample内の面積・channel加重coverage、channel別coverageを保存します。既存の等sample同時被覆率と、明示sample重みによる値を区別します。これらは一回の帯評価から得る結果で、追加の予測/較正を行いません。

## 配布後の受入

`scripts/verify_wheels.py`は同梱例で学習・独立較正・別process推論を実行し、`report.json`の`extensions.uncertainty`へ実測coverage・幅・sample数を保存します。指定coverageと有限testの実測値を区別して確認してください。用途の異なる実測dataへの保証にはなりません。
