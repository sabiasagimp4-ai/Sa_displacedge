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
    private float _appliedBlurRadius = float.NaN;
    private float _lastDetectionScale = float.NaN;
    private float _lastStrength = float.NaN;
    private float _lastTurbulence = float.NaN;
    private float _lastTurbulenceDetail = float.NaN;
    private float _lastNoiseScale = float.NaN;
    private float _lastFlowSpeed = float.NaN;
    private float _lastTime = float.NaN;
    private float _lastDispersion = float.NaN;
    private float _lastDispersionSteps = float.NaN;
    private float _lastIridescence = float.NaN;
    private float _lastLightAngle = float.NaN;
    private float _lastSeed = float.NaN;
    private float _lastPhaseOffset = float.NaN;
    private float _lastThreshold = float.NaN;
    private float _lastContrast = float.NaN;
    private float _lastOutputMode = float.NaN;

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
            // Balanced uses D2D's pre-scaling path at large radii while
            // retaining trilinear filtering; Quality needlessly disables
            // those optimizations for this deliberately soft field.
            blur.Optimization = GaussianBlurOptimization.Balanced;
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

        float detectionScale = (float)_item.DetectionScale.GetValue(frame, length, fps);
        if (detectionScale != _lastDetectionScale)
        {
            _edge.DetectionScale = detectionScale;
            _lastDetectionScale = detectionScale;
        }

        float radius = Math.Clamp((float)_item.Radius.GetValue(frame, length, fps), 0f, 512f);
        bool bypass = radius <= 0f;
        if (bypass != _blurBypassed)
        {
            using var edgeField = bypass ? _edge.Output : _blur.Output;
            _flow.SetInput(1, edgeField, true);
            _blurBypassed = bypass;
        }
        if (!bypass && radius != _appliedBlurRadius)
        {
            _blur.StandardDeviation = radius;
            _appliedBlurRadius = radius;
        }

        float strength = (float)_item.Strength.GetValue(frame, length, fps);
        if (strength != _lastStrength) { _flow.Strength = strength; _lastStrength = strength; }

        float turbulence = (float)(_item.Turbulence.GetValue(frame, length, fps) / 100.0);
        if (turbulence != _lastTurbulence) { _flow.Turbulence = turbulence; _lastTurbulence = turbulence; }

        float turbulenceDetail = (float)_item.TurbulenceDetail.GetValue(frame, length, fps);
        if (turbulenceDetail != _lastTurbulenceDetail) { _flow.TurbulenceDetail = turbulenceDetail; _lastTurbulenceDetail = turbulenceDetail; }

        float noiseScale = (float)_item.SwirlSize.GetValue(frame, length, fps);
        if (noiseScale != _lastNoiseScale) { _flow.NoiseScale = noiseScale; _lastNoiseScale = noiseScale; }

        float flowSpeed = (float)(_item.FlowSpeed.GetValue(frame, length, fps) / 100.0);
        if (flowSpeed != _lastFlowSpeed) { _flow.FlowSpeed = flowSpeed; _lastFlowSpeed = flowSpeed; }

        float time = (float)frame;
        if (time != _lastTime) { _flow.Time = time; _lastTime = time; }

        float dispersion = (float)(_item.Dispersion.GetValue(frame, length, fps) / 100.0);
        if (dispersion != _lastDispersion) { _flow.Dispersion = dispersion; _lastDispersion = dispersion; }

        float dispersionSteps = (float)_item.DispersionSteps.GetValue(frame, length, fps);
        if (dispersionSteps != _lastDispersionSteps) { _flow.DispersionSteps = dispersionSteps; _lastDispersionSteps = dispersionSteps; }

        float iridescence = (float)(_item.Iridescence.GetValue(frame, length, fps) / 100.0);
        if (iridescence != _lastIridescence) { _flow.Iridescence = iridescence; _lastIridescence = iridescence; }

        float lightAngle = (float)_item.LightAngle.GetValue(frame, length, fps);
        if (lightAngle != _lastLightAngle) { _flow.LightAngle = lightAngle; _lastLightAngle = lightAngle; }

        float seed = (float)_item.Seed.GetValue(frame, length, fps);
        if (seed != _lastSeed) { _flow.Seed = seed; _lastSeed = seed; }

        float phaseOffset = (float)_item.Phase.GetValue(frame, length, fps);
        if (phaseOffset != _lastPhaseOffset) { _flow.PhaseOffset = phaseOffset; _lastPhaseOffset = phaseOffset; }

        float threshold = (float)_item.Threshold.GetValue(frame, length, fps);
        if (threshold != _lastThreshold) { _flow.Threshold = threshold; _lastThreshold = threshold; }

        float contrast = (float)_item.Contrast.GetValue(frame, length, fps);
        if (contrast != _lastContrast) { _flow.Contrast = contrast; _lastContrast = contrast; }

        float outputMode = (float)_item.OutputMode;
        if (outputMode != _lastOutputMode) { _flow.OutputMode = outputMode; _lastOutputMode = outputMode; }

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
