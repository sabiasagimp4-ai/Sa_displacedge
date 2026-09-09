using System.ComponentModel.DataAnnotations;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Controls;
using YukkuriMovieMaker.Exo;
using YukkuriMovieMaker.Player.Video;
using YukkuriMovieMaker.Plugin.Effects;

namespace SaDisplacedgeYmm;

[VideoEffect("Sa_displacedge", ["フィルタ"], ["輪郭", "変位", "液状化", "色収差"], IsAviUtlSupported = false)]
public sealed class SaDisplacedgeEffect : VideoEffectBase
{
    public override string Label => "Sa_displacedge";

    [Display(Name = "検出サイズ", Description = "輪郭を検出するサンプル間隔。太い輪郭ほど大きく", Order = 0)]
    [AnimationSlider("F1", "px", 0.25, 32)]
    public Animation DetectionScale { get; } = new(1.5, 0.25, 32);

    [Display(Name = "検出しきい値", Description = "大きいほど強い輪郭だけを液状化させる", Order = 1)]
    [AnimationSlider("F0", "", 0, 255)]
    public Animation Threshold { get; } = new(40, 0, 255);

    [Display(Name = "コントラスト", Description = "1未満で反応域を絞り、1より大きいと弱い輪郭も持ち上げます", Order = 2)]
    [AnimationSlider("F2", "", 0.01, 16)]
    public Animation Contrast { get; } = new(1, 0.01, 16);

    [Display(Name = "変位範囲", Description = "輪郭からどこまで変位が染み出すか。0で輪郭上だけ", Order = 3)]
    [AnimationSlider("F1", "px", 0, 512)]
    public Animation Radius { get; } = new(48, 0, 512);

    [Display(Name = "変位強度", Description = "液状化で画素をずらす最大距離", Order = 4)]
    [AnimationSlider("F1", "px", 0, 400)]
    public Animation Strength { get; } = new(46, 0, 400);

    [Display(Name = "乱流", Description = "0%で輪郭の外向きに沿って流れ、100%で渦状の流体ノイズに従います", Order = 5)]
    [AnimationSlider("F1", "%", 0, 100)]
    public Animation Turbulence { get; } = new(60, 0, 100);

    [Display(Name = "乱流の複雑さ", Description = "渦ノイズの重ね合わせ回数。大きいほど計算量が増え、細部が乱れます", Order = 6)]
    [AnimationSlider("F0", "", 1, 8)]
    public Animation TurbulenceDetail { get; } = new(4, 1, 8);

    [Display(Name = "渦の大きさ", Description = "小さいほど細かい渦、大きいほど大きくうねる流れになります", Order = 7)]
    [AnimationSlider("F0", "px", 8, 4000)]
    public Animation SwirlSize { get; } = new(220, 8, 4000);

    [Display(Name = "流速", Description = "渦アニメーションの速さ。0%で静止", Order = 8)]
    [AnimationSlider("F1", "%", 0, 1000)]
    public Animation FlowSpeed { get; } = new(100, 0, 1000);

    [Display(Name = "色収差", Description = "変位方向に沿ってRGBをずらし、プリズムのような縁取りを作ります", Order = 9)]
    [AnimationSlider("F1", "%", 0, 100)]
    public Animation Dispersion { get; } = new(35, 0, 100);

    [Display(Name = "色収差ステップ数", Description = "色収差のサンプリング回数。3で従来通りのRGB分離、大きいほど滑らかな虹のグラデーションになりますが計算量が増えます", Order = 10)]
    [AnimationSlider("F0", "", 3, 128)]
    public Animation DispersionSteps { get; } = new(16, 3, 128);

    [Display(Name = "虹色の強さ", Description = "変位の頂点に薄膜干渉風の虹色ハイライトを重ねます", Order = 11)]
    [AnimationSlider("F1", "%", 0, 100)]
    public Animation Iridescence { get; } = new(55, 0, 100);

    [Display(Name = "光の角度", Description = "虹色ハイライトが最も強くなる向き", Order = 12)]
    [AnimationSlider("F1", "°", -180, 180)]
    public Animation LightAngle { get; } = new(55, -180, 180);

    [Display(Name = "シード", Description = "渦模様のパターンを変える乱数の種", Order = 13)]
    [AnimationSlider("F0", "", -4096, 4096)]
    public Animation Seed { get; } = new(0, -4096, 4096);

    [Display(Name = "位相", Description = "渦アニメーションの開始位置をずらします。流速とは独立で、タイムラインを動かさずに模様を変えられます", Order = 14)]
    [AnimationSlider("F0", "", -10000, 10000)]
    public Animation Phase { get; } = new(0, -10000, 10000);

    [Display(Name = "出力", Description = "変位フィールドの向きや強さマスクを直接確認できます", Order = 15)]
    [EnumComboBox]
    public SaDisplacedgeOutputMode OutputMode { get => _outputMode; set => Set(ref _outputMode, value); }
    private SaDisplacedgeOutputMode _outputMode = SaDisplacedgeOutputMode.Composite;

    [Display(Name = "色収差の品質", Description = "軽量は最大8サンプルに集約。細かい模様や強い色収差では差が出ます", Order = 16)]
    [EnumComboBox]
    public SaDisplacedgeSamplingQuality SamplingQuality { get => _samplingQuality; set => Set(ref _samplingQuality, value); }
    private SaDisplacedgeSamplingQuality _samplingQuality = SaDisplacedgeSamplingQuality.Standard;

    public override IEnumerable<string> CreateExoVideoFilters(int keyFrameIndex, ExoOutputDescription exoOutputDescription) => [];

    public override IVideoEffectProcessor CreateVideoEffect(IGraphicsDevicesAndContext devices) => new SaDisplacedgeProcessor(devices, this);

    protected override IEnumerable<IAnimatable> GetAnimatables() =>
    [
        DetectionScale, Threshold, Contrast, Radius, Strength, Turbulence, TurbulenceDetail,
        SwirlSize, FlowSpeed, Dispersion, DispersionSteps, Iridescence, LightAngle, Seed, Phase,
    ];
}

public enum SaDisplacedgeOutputMode
{
    [Display(Name = "合成")] Composite = 0,
    [Display(Name = "流れの向き")] Flow = 1,
    [Display(Name = "強さマスク")] Mask = 2,
}

public enum SaDisplacedgeSamplingQuality
{
    [Display(Name = "標準")] Standard = 0,
    [Display(Name = "軽量（最大8サンプル）")] Fast = 1,
}
