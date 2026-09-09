namespace SaDisplacedgeYmm.Tests;

// Portable (non-Windows, no D2D/YMM4 reference) port of the value-noise /
// curl-noise math in Shaders/Common.hlsli, kept only so KernelTests can
// verify the construction's divergence-free property numerically without a
// GPU. This is not linked into the shipping plugin; the runtime effect
// always evaluates the HLSL version on the GPU. Keep the formulas in sync
// with Common.hlsli and prototype/preview.py.
internal static class ReferenceMath
{
    public static double Hash21(double x, double y)
    {
        double v = Math.Sin(x * 127.1 + y * 311.7) * 43758.5453123;
        return v - Math.Floor(v);
    }

    public static double ValueNoise(double x, double y)
    {
        double ix = Math.Floor(x), iy = Math.Floor(y);
        double fx = x - ix, fy = y - iy;
        double a = Hash21(ix, iy);
        double b = Hash21(ix + 1, iy);
        double c = Hash21(ix, iy + 1);
        double d = Hash21(ix + 1, iy + 1);
        double ux = fx * fx * (3 - 2 * fx);
        double uy = fy * fy * (3 - 2 * fy);
        return Lerp(Lerp(a, b, ux), Lerp(c, d, ux), uy);
    }

    private static double Lerp(double a, double b, double t) => a + (b - a) * t;

    private static readonly double[,] OctaveRotation = { { .8, .6 }, { -.6, .8 } };

    public static double Fbm(double x, double y, int octaves)
    {
        double value = 0, amp = .5, freq = 1;
        double px = x, py = y;
        for (int i = 0; i < octaves; i++)
        {
            value += amp * ValueNoise(px * freq, py * freq);
            double rx = OctaveRotation[0, 0] * px + OctaveRotation[0, 1] * py;
            double ry = OctaveRotation[1, 0] * px + OctaveRotation[1, 1] * py;
            px = rx; py = ry;
            freq *= 2.03;
            amp *= .5;
        }
        return value;
    }

    // v = (d(psi)/dy, -d(psi)/dx). See Common.hlsli for why this pairing
    // (not the same-axis pairing) is the one that is divergence-free.
    public static (double X, double Y) CurlNoise(double x, double y, int octaves, double eps = .5)
    {
        double dPsiDy = (Fbm(x, y + eps, octaves) - Fbm(x, y - eps, octaves)) / (2 * eps);
        double dPsiDx = (Fbm(x + eps, y, octaves) - Fbm(x - eps, y, octaves)) / (2 * eps);
        return (dPsiDy, -dPsiDx);
    }

    public static double Gauss(double x, double center, double sigma)
    {
        double d = (x - center) / sigma;
        return Math.Exp(-.5 * d * d);
    }

    // Approximate per-channel spectral response for a dispersion sweep
    // position t in [0, 1]. See Common.hlsli's SpectrumWeight for the
    // rationale (overlapping bumps -> white centre, coloured extremes).
    public static (double R, double G, double B) SpectrumWeight(double t) =>
        (Gauss(t, .85, .35), Gauss(t, .5, .35), Gauss(t, .15, .35));
}
