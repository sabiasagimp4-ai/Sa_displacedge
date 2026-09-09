using SaDisplacedgeYmm.Tests;

var rng = new Random(7);

for (int i = 0; i < 2000; i++)
{
    double x = rng.NextDouble() * 200 - 100;
    double y = rng.NextDouble() * 200 - 100;
    double h = ReferenceMath.Hash21(x, y);
    if (h < 0 || h >= 1)
        throw new Exception($"Hash21 out of range: {h}");
    if (ReferenceMath.Hash21(x, y) != h)
        throw new Exception("Hash21 is not deterministic");
}
Console.WriteLine("PASS: Hash21 range and determinism (2000 samples)");

for (int i = 0; i < 500; i++)
{
    double x = rng.NextDouble() * 50 - 25;
    double y = rng.NextDouble() * 50 - 25;
    double n = ReferenceMath.ValueNoise(x, y);
    if (double.IsNaN(n) || n < -1e-9 || n > 1 + 1e-9)
        throw new Exception($"ValueNoise out of range: {n}");
}
Console.WriteLine("PASS: ValueNoise range (500 samples)");

// Continuity: ValueNoise is smoothstep-interpolated (C1), so a small step
// must not produce a large jump.
for (int i = 0; i < 200; i++)
{
    double x = rng.NextDouble() * 50 - 25;
    double y = rng.NextDouble() * 50 - 25;
    double step = 1e-4;
    double n0 = ReferenceMath.ValueNoise(x, y);
    double n1 = ReferenceMath.ValueNoise(x + step, y);
    if (Math.Abs(n1 - n0) > 0.01)
        throw new Exception($"ValueNoise discontinuity: {n0} -> {n1} over step {step}");
}
Console.WriteLine("PASS: ValueNoise local continuity (200 samples)");

// The optimized curl construction uses the analytic gradient of the scalar
// fBm potential. Verify the gradient against an independent finite difference
// and then verify the 90-degree rotation used by CurlNoise.
const double Eps = 1e-5;
const int Octaves = 4;
for (int i = 0; i < 200; i++)
{
    double x = rng.NextDouble() * 100 - 50;
    double y = rng.NextDouble() * 100 - 50;

    var gradient = ReferenceMath.FbmGradient(x, y, Octaves);
    double finiteX = (ReferenceMath.Fbm(x + Eps, y, Octaves) - ReferenceMath.Fbm(x - Eps, y, Octaves)) / (2 * Eps);
    double finiteY = (ReferenceMath.Fbm(x, y + Eps, Octaves) - ReferenceMath.Fbm(x, y - Eps, Octaves)) / (2 * Eps);
    if (Math.Abs(gradient.X - finiteX) > 1e-6 || Math.Abs(gradient.Y - finiteY) > 1e-6)
        throw new Exception($"FbmGradient mismatch at ({x},{y})");

    var curl = ReferenceMath.CurlNoise(x, y, Octaves);
    if (Math.Abs(curl.X - gradient.Y) > 1e-12 || Math.Abs(curl.Y + gradient.X) > 1e-12)
        throw new Exception($"CurlNoise rotation mismatch at ({x},{y})");
}
Console.WriteLine("PASS: analytic fBm gradient and curl rotation (200 samples)");

// Octave count must materially change the field: more octaves adds
// higher-frequency detail, so a 1-octave and a 6-octave field should not be
// nearly identical almost everywhere.
{
    int differing = 0, total = 0;
    for (double x = -20; x <= 20; x += 1.3)
    {
        for (double y = -20; y <= 20; y += 1.3)
        {
            var (a, _) = ReferenceMath.CurlNoise(x, y, 1);
            var (b, _) = ReferenceMath.CurlNoise(x, y, 6);
            total++;
            if (Math.Abs(a - b) > 0.05) differing++;
        }
    }
    if (differing < total / 10)
        throw new Exception($"TurbulenceDetail has little effect: only {differing}/{total} samples differ");
}
Console.WriteLine("PASS: turbulence detail (octave count) changes the field");

// SpectrumWeight: bounded to [0,1], and peaks at the sweep position that
// matches the classic R (outer, t=1) / G (middle, t=0.5) / B (inner, t=0)
// tap ordering used when DispersionSteps == 3.
for (int i = 0; i <= 20; i++)
{
    double t = i / 20.0;
    var (r, g, b) = ReferenceMath.SpectrumWeight(t);
    foreach (double w in new[] { r, g, b })
        if (w < -1e-9 || w > 1 + 1e-9)
            throw new Exception($"SpectrumWeight out of range at t={t}: {w}");
}
{
    var atOuter = ReferenceMath.SpectrumWeight(1.0);
    var atMiddle = ReferenceMath.SpectrumWeight(0.5);
    var atInner = ReferenceMath.SpectrumWeight(0.0);
    if (!(atOuter.R >= atOuter.G && atOuter.R >= atOuter.B))
        throw new Exception("Red should peak at the outer sweep position (t=1)");
    if (!(atMiddle.G >= atMiddle.R && atMiddle.G >= atMiddle.B))
        throw new Exception("Green should peak at the middle sweep position (t=0.5)");
    if (!(atInner.B >= atInner.R && atInner.B >= atInner.G))
        throw new Exception("Blue should peak at the inner sweep position (t=0)");
}
Console.WriteLine("PASS: SpectrumWeight range and R/G/B peak ordering");
