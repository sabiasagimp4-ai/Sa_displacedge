using System.Runtime.InteropServices;
using Vortice;
using Vortice.Direct2D1;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Player.Video;

namespace SaDisplacedgeYmm;

// Scharr gradient of OKLab L, weight-carried by alpha as (gx*a, gy*a, mag*a,
// a). See Shaders/EdgeGradient.hlsl for why: the following Gaussian blur
// then averages direction and strength correctly under partial
// transparency, the same numerator/denominator convention Sa_aohue's
// ChromaField pass uses.
internal sealed class EdgeGradientEffect(IGraphicsDevicesAndContext devices)
    : D2D1CustomShaderEffectBase(Create<EdgeGradientEffect.Impl>(devices))
{
    public float DetectionScale { set => SetValue(0, value); }

    [CustomEffect(1)]
    private sealed class Impl : D2D1CustomShaderEffectImplBase<Impl>
    {
        private Constants _constants = new() { DetectionScale = 1.5f };

        [CustomEffectProperty(PropertyType.Float, 0)]
        public float DetectionScale { get => _constants.DetectionScale; set { _constants.DetectionScale = Math.Clamp(value, .25f, 32f); UpdateConstants(); } }

        public Impl() : base(ShaderResourceLoader.Get("EdgeGradient")) { }

        protected override void UpdateConstants()
        {
            // Raw gradient/magnitude data, not colour: keep full float
            // precision so the later blur doesn't quantise small slopes.
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
            int halo = (int)MathF.Ceiling(_constants.DetectionScale);
            inputRects[0] = new(outputRect.Left - halo, outputRect.Top - halo, outputRect.Right + halo, outputRect.Bottom + halo);
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct Constants
        {
            public float DetectionScale, Padding0, Padding1, Padding2;
            public float Left, Top, Right, Bottom;
        }
    }
}
