using System.Runtime.InteropServices;
using Vortice;
using Vortice.Direct2D1;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Player.Video;

namespace SaDisplacedgeYmm;

// The creative core: blends the edge-normal direction (Input1, blurred)
// with a divergence-free curl-noise field into a flow vector, displaces
// Input0 (the original source) along it, sweeping DispersionSteps taps
// across the displacement for a spectral prism fringe, and adds an
// animated thin-film-style iridescent glint.
// See Shaders/FlowDisplace.hlsl for the full pixel shader.
internal sealed class FlowDisplaceEffect(IGraphicsDevicesAndContext devices)
    : D2D1CustomShaderEffectBase(Create<FlowDisplaceEffect.Impl>(devices))
{
    public float Strength { set => SetValue(0, value); }
    public float Turbulence { set => SetValue(1, value); }
    public float TurbulenceDetail { set => SetValue(2, value); }
    public float NoiseScale { set => SetValue(3, value); }
    public float FlowSpeed { set => SetValue(4, value); }
    public float Time { set => SetValue(5, value); }
    public float Dispersion { set => SetValue(6, value); }
    public float Iridescence { set => SetValue(7, value); }
    public float LightAngle { set => SetValue(8, value); }
    public float Seed { set => SetValue(9, value); }
    public float Threshold { set => SetValue(10, value); }
    public float Contrast { set => SetValue(11, value); }
    public float OutputMode { set => SetValue(12, value); }
    public float DispersionSteps { set => SetValue(13, value); }
    public float PhaseOffset { set => SetValue(14, value); }

    public float FastSampling { set => SetValue(15, value); }

    // S_DistortChroma-compatible controls. Values are normalized ratios in
    // the shader (the YMM UI exposes them as percentages/degrees).
    public float WarpRed { set => SetValue(16, value); }
    public float WarpBlue { set => SetValue(17, value); }
    public float WarpRotation { set => SetValue(18, value); }

    [CustomEffect(2)]
    private sealed class Impl : D2D1CustomShaderEffectImplBase<Impl>
    {
        private Constants _constants = new()
        {
            Strength = 46f,
            Turbulence = .6f,
            TurbulenceDetail = 4f,
            NoiseScale = 220f,
            FlowSpeed = 1f,
            Dispersion = .35f,
            Iridescence = .55f,
            LightAngle = 55f,
            Contrast = 1f,
            DispersionSteps = 16f,
            WarpRed = .5f,
            WarpBlue = 1f,
        };

        [CustomEffectProperty(PropertyType.Float, 0)] public float Strength { get => _constants.Strength; set { _constants.Strength = Math.Clamp(value, 0f, 400f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 1)] public float Turbulence { get => _constants.Turbulence; set { _constants.Turbulence = Math.Clamp(value, 0f, 1f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 2)] public float TurbulenceDetail { get => _constants.TurbulenceDetail; set { _constants.TurbulenceDetail = Math.Clamp(MathF.Round(value), 1f, 8f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 3)] public float NoiseScale { get => _constants.NoiseScale; set { _constants.NoiseScale = Math.Clamp(value, 8f, 4000f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 4)] public float FlowSpeed { get => _constants.FlowSpeed; set { _constants.FlowSpeed = Math.Clamp(value, 0f, 10f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 5)] public float Time { get => _constants.Time; set { _constants.Time = value; UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 6)] public float Dispersion { get => _constants.Dispersion; set { _constants.Dispersion = Math.Clamp(value, 0f, 8f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 7)] public float Iridescence { get => _constants.Iridescence; set { _constants.Iridescence = Math.Clamp(value, 0f, 1f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 8)] public float LightAngle { get => _constants.LightAngle; set { _constants.LightAngle = Math.Clamp(value, -180f, 180f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 9)] public float Seed { get => _constants.Seed; set { _constants.Seed = Math.Clamp(value, -4096f, 4096f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 10)] public float Threshold { get => _constants.Threshold; set { _constants.Threshold = Math.Clamp(value, 0f, 255f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 11)] public float Contrast { get => _constants.Contrast; set { _constants.Contrast = Math.Clamp(value, .01f, 16f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 12)] public float OutputMode { get => _constants.OutputMode; set { _constants.OutputMode = Math.Clamp(MathF.Round(value), 0f, 2f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 13)] public float DispersionSteps { get => _constants.DispersionSteps; set { _constants.DispersionSteps = Math.Clamp(MathF.Round(value), 3f, 128f); UpdateConstants(); } }
        // Not rounded, like Seed: a continuous frame-equivalent offset, so
        // keyframing it animates the swirl's starting point smoothly
        // instead of stepping in whole-frame jumps.
        [CustomEffectProperty(PropertyType.Float, 14)] public float PhaseOffset { get => _constants.PhaseOffset; set { _constants.PhaseOffset = Math.Clamp(value, -10000f, 10000f); UpdateConstants(); } }

        private float _fastSampling;
        private float _warpRotation;
        private int _tableSteps = -1;
        private bool _tableFast;
        [CustomEffectProperty(PropertyType.Float, 15)]
        public float FastSampling { get => _fastSampling; set { _fastSampling = value; UpdateConstants(); } }

        [CustomEffectProperty(PropertyType.Float, 16)]
        public float WarpRed { get => _constants.WarpRed; set { _constants.WarpRed = Math.Clamp(value, -8f, 8f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 17)]
        public float WarpBlue { get => _constants.WarpBlue; set { _constants.WarpBlue = Math.Clamp(value, -8f, 8f); UpdateConstants(); } }
        [CustomEffectProperty(PropertyType.Float, 18)]
        public float WarpRotation { get => _warpRotation; set { _warpRotation = Math.Clamp(value, -180f, 180f); UpdateConstants(); } }

        public Impl() : base(ShaderResourceLoader.Get("FlowDisplace")) { }

        protected override void UpdateConstants()
        {
            int steps = (int)_constants.DispersionSteps;
            bool fast = _fastSampling > .5f;
            if (steps != _tableSteps || fast != _tableFast)
            {
                _constants.Taps = SpectralTable.Build(steps, fast, out int count);
                _constants.TapCount = count;
                _tableSteps = steps;
                _tableFast = fast;
            }
            float angle = _constants.LightAngle * (MathF.PI / 180f);
            _constants.LightX = MathF.Cos(angle);
            _constants.LightY = MathF.Sin(angle);
            float chromaAngle = _warpRotation * (MathF.PI / 180f);
            _constants.WarpCos = MathF.Cos(chromaAngle);
            _constants.WarpSin = MathF.Sin(chromaAngle);
            _constants.Cutoff = _constants.Threshold / 255f * .12f;
            _constants.InvContrast = 1f / MathF.Max(.01f, _constants.Contrast);
            drawInformation?.SetOutputBuffer(BufferPrecision.PerChannel32Float, ChannelDepth.Four);
            drawInformation?.SetPixelShaderConstantBuffer(_constants);
        }

        public override void MapInputRectsToOutputRect(RawRect[] inputRects, RawRect[] inputOpaqueSubRects, out RawRect outputRect, out RawRect outputOpaqueSubRect)
        {
            outputRect = inputRects[0];
            outputOpaqueSubRect = default;
            _constants.Left = outputRect.Left; _constants.Top = outputRect.Top;
            _constants.Right = outputRect.Right; _constants.Bottom = outputRect.Bottom;
            UpdateConstants();
        }

        public override void MapOutputRectToInputRects(RawRect outputRect, RawRect[] inputRects)
        {
            // Worst-case displacement magnitude: the base flow displacement
            // plus the larger S_DistortChroma-style red/blue warp.
            bool displaces = _constants.OutputMode < .5f && _constants.Dispersion > 1e-5f && _constants.Strength > 0f;
            float chromaWarp = MathF.Max(MathF.Abs(_constants.WarpRed), MathF.Abs(_constants.WarpBlue));
            int halo = displaces ? (int)MathF.Ceiling(_constants.Strength * (1f + _constants.Dispersion * chromaWarp)) + 1 : 0;
            inputRects[0] = new(outputRect.Left - halo, outputRect.Top - halo, outputRect.Right + halo, outputRect.Bottom + halo);
            inputRects[1] = outputRect;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Constants
        {
            public float Strength, Turbulence, TurbulenceDetail, NoiseScale;
            public float FlowSpeed, Time, Dispersion, Iridescence;
            public float LightAngle, Seed, Threshold, Contrast;
            public float OutputMode, DispersionSteps, PhaseOffset, TapCount;
            public float Left, Top, Right, Bottom;
            public float LightX, LightY, Cutoff, InvContrast;
            public float WarpRed, WarpBlue, WarpCos, WarpSin;
            public SpectralTable.Buffer Taps;
        }
    }
}
