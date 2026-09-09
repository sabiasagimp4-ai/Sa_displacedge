using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SaDisplacedgeYmm;

// Each float4 is normalized RGB weight + signed sweep position. The layout
// exactly matches float4 spectralTaps[128] in the shader constant buffer.
internal static class SpectralTable
{
    [InlineArray(128)]
    [StructLayout(LayoutKind.Sequential)]
    internal struct Buffer
    {
        private Vector4 _element0;
    }

    internal static Buffer Build(int steps, bool fast, out int count)
    {
        steps = Math.Clamp(steps, 3, 128);
        count = fast ? Math.Min(steps, 8) : steps;
        Buffer result = default;
        Vector3 sum = default;
        for (int s = 0; s < steps; s++)
        {
            float t = (float)s / (steps - 1);
            Vector3 w = new(Gauss(t, .85f), Gauss(t, .5f), Gauss(t, .15f));
            sum += w;
            // Redistribute the original weights to two neighboring nodes.
            // This preserves both the total and first moment per channel:
            // constant colours and linear colour ramps remain unchanged.
            float node = t * (count - 1);
            int lo = Math.Min((int)node, count - 1);
            int hi = Math.Min(lo + 1, count - 1);
            float f = node - lo;
            if (!fast) { lo = hi = s; f = 0; }
            result[lo] += new Vector4(w * (1 - f), 0);
            result[hi] += new Vector4(w * f, 0);
        }
        for (int i = 0; i < count; i++)
        {
            Vector4 w = result[i];
            result[i] = new(w.X / sum.X, w.Y / sum.Y, w.Z / sum.Z,
                2f * i / (count - 1) - 1f);
        }
        return result;
    }

    private static float Gauss(float t, float center)
    {
        float d = (t - center) / .35f;
        return MathF.Exp(-.5f * d * d);
    }
}
