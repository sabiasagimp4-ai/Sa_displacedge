using Vortice.Direct2D1;
using Vortice.Direct2D1.Effects;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Player.Video;

namespace SaDisplacedgeYmm;

internal sealed class SaDisplacedgeProcessor : IVideoEffectProcessor
{
    private readonly SaDisplacedgeEffect _item;
    private readonly EdgeGradientEffect? _edge;
    private readonly GaussianBlur? _blur;
    private readonly FlowDisplaceEffect? _flow;
    private readonly ID2D1Image? _output;
    private ID2D1Image? _input;
    private bool _blurBypassed;

    public SaDisplacedgeProcessor(IGraphicsDevicesAndContext devices, SaDisplacedgeEffect item)
    {
        _item = item;
        EdgeGradientEffect? edge = null;
        GaussianBlur? blur = null;
        FlowDisplaceEffect? flow = null;
        ID2D1Image? output = null;
        try
        {
            edge = new EdgeGradientEffect(devices);
            blur = new GaussianBlur(devices.DeviceContext);
            blur.Optimization = GaussianBlurOptimization.Quality;
            blur.BorderMode = BorderMode.Soft;
            flow = new FlowDisplaceEffect(devices);
            if (!edge.IsEnabled || !flow.IsEnabled)
                return;
            using (var edgeOutput = edge.Output)
                blur.SetInput(0, edgeOutput, true);
            using (var blurredOutput = blur.Output)
                flow.SetInput(1, blurredOutput, true);
            output = flow.Output;
            _edge = edge; _blur = blur; _flow = flow; _output = output;
            edge = null; blur = null; flow = null; output = null;
        }
        finally
        {
            output?.Dispose();
            flow?.Dispose();
            blur?.Dispose();
            edge?.Dispose();
        }
    }

    public ID2D1Image Output => _output ?? _input ?? throw new InvalidOperationException("入力が未設定です。");

    public void SetInput(ID2D1Image? input)
    {
        _input = input;
        _edge?.SetInput(0, input, true);
        _flow?.SetInput(0, input, true);
    }

    public void ClearInput()
    {
        _input = null;
        _edge?.SetInput(0, null, true);
        _flow?.SetInput(0, null, true);
    }

    public DrawDescription Update(EffectDescription effectDescription)
    {
        if (_edge is null || _blur is null || _flow is null)
            return effectDescription.DrawDescription;
        var frame = effectDescription.ItemPosition.Frame;
        var length = effectDescription.ItemDuration.Frame;
        var fps = effectDescription.FPS;

        _edge.DetectionScale = (float)_item.DetectionScale.GetValue(frame, length, fps);

        float radius = Math.Clamp((float)_item.Radius.GetValue(frame, length, fps), 0f, 256f);
        bool bypass = radius <= 0f;
        if (bypass != _blurBypassed)
        {
            using var edgeField = bypass ? _edge.Output : _blur.Output;
            _flow.SetInput(1, edgeField, true);
            _blurBypassed = bypass;
        }
        if (!bypass) _blur.StandardDeviation = radius;

        _flow.Strength = (float)_item.Strength.GetValue(frame, length, fps);
        _flow.Turbulence = (float)(_item.Turbulence.GetValue(frame, length, fps) / 100.0);
        _flow.TurbulenceDetail = (float)_item.TurbulenceDetail.GetValue(frame, length, fps);
        _flow.NoiseScale = (float)_item.SwirlSize.GetValue(frame, length, fps);
        _flow.FlowSpeed = (float)(_item.FlowSpeed.GetValue(frame, length, fps) / 100.0);
        _flow.Time = (float)frame;
        _flow.Dispersion = (float)(_item.Dispersion.GetValue(frame, length, fps) / 100.0);
        _flow.DispersionSteps = (float)_item.DispersionSteps.GetValue(frame, length, fps);
        _flow.Iridescence = (float)(_item.Iridescence.GetValue(frame, length, fps) / 100.0);
        _flow.LightAngle = (float)_item.LightAngle.GetValue(frame, length, fps);
        _flow.Seed = (float)_item.Seed.GetValue(frame, length, fps);
        _flow.PhaseOffset = (float)_item.Phase.GetValue(frame, length, fps);
        _flow.Threshold = (float)_item.Threshold.GetValue(frame, length, fps);
        _flow.Contrast = (float)_item.Contrast.GetValue(frame, length, fps);
        _flow.OutputMode = (float)_item.OutputMode;

        return effectDescription.DrawDescription;
    }

    public void Dispose()
    {
        ClearInput();
        _blur?.SetInput(0, null, true);
        _flow?.SetInput(1, null, true);
        _output?.Dispose();
        _flow?.Dispose();
        _blur?.Dispose();
        _edge?.Dispose();
    }
}
