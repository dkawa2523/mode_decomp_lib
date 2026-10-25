# 数値手法と適用条件

分解手法は明示catalogから選び、使用する実装をimportします。parameterと既定値は各実装の明示constructorが正本です。YAML・Pythonは同じresolverを使い、未知parameterを拒否します。

| 区分 | 手法 |
| --- | --- |
| 矩形・学習表現 | fft2、fft2_lowpass、dct2、wavelet2d、pswf2d_tensor、pod、pod_svd、tucker、dict_learning、autoencoder |
| 定常場の非線形学習表現 | quadratic_manifold（grid／meshの全成分を結合） |
| 円・円環 | zernike、pseudo_zernike、annular_zernike、polar_fft、fourier_bessel、fourier_jacobi、disk_slepian |
| graph・mesh・球面 | graph_fourier、gappy_graph_fourier、rbf_expansion、laplace_beltrami、spherical_harmonics、spherical_slepian |
| 欠測・成分分離等 | pod_joint、pod_em、pod_joint_em、gappy_pod、helmholtz、hodge、offset_residual |

実数配列の係数は共通の`none` codecでshapeとflatten順を保って復元します。`dtype_policy`はfloat32/float64です。`auto`は実FFT、複素係数、wavelet構造、offset残差に応じてcodecを選びます。

係数後処理はnone/standardize/quantile/power_yeojohnson/pca/dict_learning、回帰はridge/elasticnet/multitask_elasticnet/multitask_lasso/gpr/mtgp/xgb/lgbm/catboostです。回帰器は実数2次元targetを受け、A/Zの意味は合成層が所有します。fitするsampleとfoldの分離は合成処理が管理します。

完全なFFT/DCT等の可逆変換、modeを切り詰める圧縮表現、Helmholtz等の成分分離、gappy/EM等の欠測処理は用途が異なります。同じ`n_modes`でもchannel別・全channel結合・周波数blockでは自由度が異なるため、実際のpacked/target次元、実効rank、原fieldの評価誤差を併記します。成分分離の再構成誤差が小さくても、圧縮や物理的な分離の妥当性を証明したことにはなりません。

共通の `list_modes`、`mode_fields`、`reconstruct_modes`、`cumulative_reconstruct` はfit済み表現のAまたはZ座標を扱います。要求した座標だけ応答を生成し、`analysis.modes` でHTML/CSVへ出せます。固定affine合成の応答は `inverse(e_i)-inverse(0)` で、学習平均等の基準場をモードごとに重複加算しません。非線形・複合座標では基準点付き有限摂動です。[モードガイド](MODES.md)にFFTの符号・波数・単位と解釈をまとめています。

## 数値規約

- FFT lowpassは保持集合を共役対称にします。偶数幅の指定が元軸より小さい場合、保持幅が1増えることがあり、要求幅と実保持幅をmetadataへ残します。既定のauto codecはCartesian `fft2` / `fft2_lowpass`に`real_fft`を使い、自己共役点の実数値と共役pairの独立な実部・虚部だけを保持します。pairの係数には√2を掛け、係数空間の二乗normを保ちます。任意の復号vectorでも実数場の共役対称性を満たします。offset内のFFT残差もこのcodecを使います。
- 円環Zernikeはn/mの偶奇を満たすmodeだけ列挙します。
- SHの現実装はSciPy/orthoです。schmidt/4piや実行されないpyshtools指定は拒否します。
- 独立POD/Gappy/pod_svdは成分別のrankと保持数Kcを使い、共通の最小rankへ切り詰めません。新fitのraw係数は`component_blocks_v1`の成分順の実数vectorで、component_offsetsが各区間を示します。n_modes:nullの自動選択とenergy閾値は成分ごと、整数は非定数成分ごとの明示countです。非定数のrank不足はエラーにします。中心化POD/Gappyのsample間不変profileはKc=0でmeanを保持し、非中心pod_svdはzero snapshotだけKc=0、非zero不変profileはrank1です。joint/EMの数学は変更しません。零energyの保持率は未定義として扱い、偽modeを追加しません。energyの分母は切詰め前の全training energyです。randomized/incrementalは`spectrum_complete`で完全なスペクトルと区別し、incrementalの保持量は再投影で測ります。
- 全成分がrank0ならA/ZはN×0です。coeff_post:noneだけを許容し、回帰器をfitせずbaseline fieldを返します。条件のschemaと有限値、独立較正、保存後予測は通常どおり検査します。dummy targetや定数専用のYAMLは不要です。
- EMはmask外NaNを演算前に除き、中心化指定と計算を一致させます。gappy PODの`mask=None`は全観測と同じ意味です。重み付きPOD基底は物理座標へ戻してから部分観測LS・再構成に使います。
- Gappyのbasis学習は固定domain内の完全観測を要求し、domain外の値やNaNは要求しません。学習したdomain部分のmean/basisを物理格子へ一度配置し、domain外は0として復元します。同じ観測supportのbatchは成分ごとに複数右辺を共有してsolveし、supportが異なるsampleは個別に解きます。solver・正則化・mode weightの数学は変更しません。
- Cartesian円の物理面積に追加のrを掛けません。球面はradius²を掛け、周期経度の最後から先頭へ戻る区間も積分します。meshは保持facesと整合する頂点massを使います。inclusiveな矩形端点には台形重みを使います。FFT/DCT・Helmholtz・DPSSは必要な等間隔軸を検証し、対応しない非等間隔入力を拒否します。Polar補間は実際のCartesian座標を用い、非等間隔軸にも対応します。補間往復を一般に無損失とは扱いません。
- 周期Helmholtzは実数の離散一次微分として、偶数格子のNyquist微分をゼロとする規約を使います。その微分のnullspaceはdiv-free側へ一度だけ割り当てます。この規約のEuclidean grid内積で冪等性・直交性・energy保存を検証しています。
- `hodge`は固定maskのCartesian cell complex上で、`boundary="natural"`の離散cochain分解を行います。nativeな辺1-formはexact/coexact/harmonicの3成分、節点HWCベクトル場はそのliftと`sampling_residual`の4成分です。直交性とenergy分解はgeometry由来の正定値の辺内積に対する保証で、lift後の節点内積や境界法線の点ごとの条件は保証しません。`sampling_residual`をharmonic成分とは呼びません。詳細は[Hodgeガイド](HODGE.md)を参照してください。
- Spherical Slepianは重み付きSH行列のSVDから正の離散部分空間だけを白色化し、実効rankを超えるkを拒否します。零固有値を正値へclipしてmodeを存続させません。基底をgeometry・重み・maskへ束縛し、別のgeometryで再利用しません。
- Quantile/Powerは入力を破壊せず、Power逆変換の定義域外は明示エラーにします。PCA energy=1.0は全成分保持へ変換します。
- offsetは定数場とeps=0を扱います。

## 定常分布のTuckerとQuadratic manifold

`tucker`と`quadratic_manifold`は、既存の`FieldRepresentation`、条件から予測する`FieldSurrogate`、
直接分解・比較・pipeline・保存後推論で利用できます。追加packageは不要で、NumPy／SciPyで計算します。
どちらも**基底／decoderの学習には固定domain内の完全観測を要求**し、学習後は部分観測のqueryを受け付けます。
学習sample自体の変動欠測には、既存のEM系等を選びます。

### Tucker

sample軸を残し、NHWCのH/W/C軸だけへ共通因子を学習します。raw係数は各sampleの
`(rank_y, rank_x, rank_component)` coreで、AはそのC順flattenです。packed次元は3つのrankの積であり、
PODの`n_modes`と同じ値では比較しません。成分因子は元の物理channelを混合するため、coreの成分indexを
元channel名へ読み替えません。

```yaml
representation:
  decomposition:
    name: tucker
    params:
      ranks: [4, 4, 2]  # y, x, component。各軸の実際の長さ以下にする
      center: true
      inner_product: domain_weights
      n_iter: 20
      tol: 0.00000001
      ridge_alpha: 0.000001
```

`ranks: null`は各軸`min(8, 軸長)`です。`n_iter: 0`はHOSVD、正数はHOSVDを初期値にしたHOOIです。
`center: true`はtraining mean fieldを保持します。対応geometryは完全なrectilinear rectangleで、
不等間隔軸も利用できます。meshや穴のあるdomain maskは受け付けません。
`inner_product: domain_weights`は正で有限な積分重みがy/xに分離できる場合に対応し、重み付き座標で因子を学習します。
`euclidean`は等重みです。近い別のmeasureへ自動置換しません。

完全観測queryは因子への直交射影、部分観測queryは観測位置の因子積に対する重み付きLSでcoreを推定します。
`ridge_alpha`は部分観測のcoreへ適用するL2正則化で、完全観測の射影は縮小しません。
無正則化で観測からcoreを識別できない場合はエラーです。meanを除いた合成はaffineなので、
`mode_fields()`は基準場を除いた加算可能な因子積応答を返します。

### Quadratic manifold

全成分を結合したPOD座標`a`と、そのPOD空間に直交する二次補正を学習します。

```text
u(a) = mean + V a + H (q(a) - q_mean)
q(a) = [a_i * a_j for i <= j]
```

`center: true`ではmeanとq_meanをtraining sampleから求めます。これにより、対称な二次変動も平均からの補正として
学習できます。`inverse(0)`は二次補正の切片を含むため、training meanと異なる場合があります。
`center: false`では両meanを0にします。`domain_weights`では同じ物理重みでPOD、補正fit、観測残差を扱います。

```yaml
representation:
  decomposition:
    name: quadratic_manifold
    params:
      n_modes: 4
      center: true
      inner_product: domain_weights
      quadratic_alpha: 0.000001
      ridge_alpha: 0.000001
      max_nfev: 200
      tol: 0.00000001
```

`n_modes: null`はtrainingの数値rank以下で最大16、整数指定は数値rank以下です。全て定数なら座標はN×0となり、
既存の定数場経路を使います。`quadratic_alpha`はtrainingの二次補正fitに対するL2正則化です。
完全観測のencodeは固定の線形POD射影で、decoderの右逆になります。一般の任意fieldに対する非線形最短距離射影ではありません。
部分観測のencodeは線形LS初期値からの局所非線形LSで、`ridge_alpha`はこのquery座標だけを正則化します。
未収束や、無正則化での観測Jacobianのrank不足はエラーです。局所rankが十分でも大域的な解の一意性は保証しません。

矩形・disk・annulus・mask・sphere_grid・meshの固定geometryを扱えます。重み付きの場合はdomain内に正で有限な積分重みが必要です。
domain外の値は学習に使わず、復元は0です。非線形decoderなので、Aでも`list_modes()`は`latent_coordinate`、
`additive: false`です。`mode_fields()`にはreference座標を渡し、指定点での有限摂動として解釈します。
線形PODのenergy順位や誤差曲線の単調性を二次応答へ引き継ぎません。

```python
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation

data = FieldDataset.from_arrays(fields)  # NHWC。meshにはgeometryとlayout="NVC"を指定
representation = FieldRepresentation({"decomposition": {"name": "quadratic_manifold", "params": {"n_modes": 4}}}).fit(
    data
)
coordinates = representation.transform(observed_fields, masks=observed_mask)
restored = representation.inverse_transform(coordinates)
```

条件予測では同じ`representation`設定を`model.candidates[].representation`へ置き、既存回帰器を選びます。
欠測比較では同じ設定を`analysis.kind: imputation`へ渡します。隠したtruthやtest sampleで因子・decoderを再fitしません。

両手法のfitはdenseなtraining tensor／matrixを使用します。Tuckerは3因子を保持し、部分観測では
観測数×成分数×core次元のLS行列が必要です。Quadratic manifoldは有効field次元Fに対して
`F × K(K+1)/2`の補正行列と、補正fitの一時行列を必要とします。大きいrankは実測して選びます。

既存の入力生成例と次の設定で直接実行できます。入力生成先が既にある場合は生成を省略します。

```powershell
python examples/create_analysis_dataset.py
python -m ml_platform_mode_decomp config/tasks/mode_decomp_tucker.yaml
python -m ml_platform_mode_decomp config/tasks/mode_decomp_quadratic_manifold.yaml
```

## 残る適用制約

`real_fft`の独立実座標とnorm保存はFFT係数の性質です。masked domain上での自由度や物理面積のnorm、offsetと残差を合わせた全体の非冗長性まで保証するものではありません。明示的に選ぶ旧`fft_complex_codec_v1`やmag/phase表現には、任意の回帰出力の共役保証がありません。mag/phaseは位相の周期性も考慮する必要があります。

円環Zernikeは半径写像基底＋LSで、円環面積に対する正規直交系と同じではありません。Fourier–Bessel/Jacobi等の列norm調整も、離散的な面積内積でGram行列が単位行列になる保証ではありません。非直交基底の物理energyには基底のGram行列が必要で、係数二乗和をそのまま使いません。PSWF2DはDPSSのテンソル積です。単位近傍Graphの周波数を物理PDEの周波数と同一視しません。

gappyは完全場等で学んだ表現へ部分観測を投影します。観測数がmode数以上でも配置によって識別できません。変動欠測での学習はEM等の対応手法だけで扱います。

PODの中心化後energy、中心化しないSVDの二乗norm、Slepianの領域集中率、係数PCAの説明分散は異なる量です。係数PCAの`energy_threshold`はpacked係数Aの共分散に対する指定であり、原fieldのenergy保持率ではありません。基底・codec・係数後処理によって異なるA/Zの誤差で手法間の優劣を決めず、同じ原field・support・重みで比較します。

モード再構成曲線の全A終点は、representation/imputationの主Z終点と異なる場合があります。ZのPCA方向を元のFourierモードへ読み替えず、非直交基底や有限摂動に共通のenergy順位・誤差単調減少を仮定しません。

Quantileは学習範囲端で飽和します。非線形inverse後の点予測は分布平均ではありません。係数の周辺stdからfield_stdを作るMC伝播は公開していません。0.6では別方式の[独立校正したfield予測帯](UNCERTAINTY.md)を実装し、総field残差の上下限として扱います。標準偏差やGaussian posteriorとは解釈しません。

基準指標は原fieldのRMSE/MAEです。各sampleの評価可能な空間重みを1へ正規化し、既定ではsampleを等重み平均します。`sample_weight_column`で明示した重みも利用できます。同単位channelのMSEを既定で等重み集約して単一field_rmseを作り、channel別実単位値も残します。異単位をfields.unitsに宣言した場合は明示channel_scalesが必要で、channel_weightsは任意です。scale指定後の集約値は規格化値であり、channel別値は実単位です。単位を宣言しない入力の物理的整合は利用側で確かめます。

`field_r2`はsampleと空間を合わせたchannel別分散を基準とします。constantなchannelで未定義でも有効RMSE/MAEは残し、理由を記録します。R²を主選択指標にした場合は順位を付けられないためエラーです。空間的な定常profileによる高R²を条件に対する予測能力と同一視しません。学習だけから作った平均profileに対するskillも同じ対象と重みで評価します。`metrics.reference`のgrid/observations/hiddenで対象を選び、元点のR²は提供しません。元点ではsample内point重みを正規化し、channel尺度とsample重みを適用します。補間往復誤差は予測誤差から差し引きません。

通常の候補選択と最終採点は予測一度で行い、詳細A/Z誤差分解は明示解析だけで実行します。`decomposition_rmse`はfield前処理・分解・codecまでの誤差で、係数後処理前の再構成を使います。元点の詳細誤差分解は中間fieldも同じ逆補間へ写してから計算し、格子損失と元点損失を足しません。

表現誤差と回帰誤差の二乗を単純に足せるのは、同じ内積での直交射影等の条件が成立する場合です。analysisは原field空間の交差項を測り、`total_mse = representation_mse + regression_mse + error_cross_term_mse`として確認します。非直交基底でも厳密な重み付きLSが直交射影を作る場合は成立しますが、正則化や非線形inverseでは一般に保証しません。

配備用の任意CV/HPOは各foldで前処理から回帰まで新しくfitし、原fieldのOOF損失を使います。異なるfoldの係数やモードを直接比較して順位付けしません。探索範囲と同数試行の予算、group/time制約は [SEARCH.md](SEARCH.md) を参照してください。

Torch・GPyTorch・PyWavelets・modredはCPUの数値テストを実施しています。GBDT・GPU・実データでの精度比較は対応環境で追加します。catalogにあることを、全geometry/全backendの検証完了とは扱いません。
