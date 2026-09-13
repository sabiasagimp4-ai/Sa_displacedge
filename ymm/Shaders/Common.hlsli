// Shared helpers for the Sa_displacedge shader passes: sRGB/OKLab colour,
// and a divergence-free curl-noise field built from a rotated-octave
// value-noise fBm. The curl field uses the analytic gradient of the smooth
// value-noise potential, so one fBm pass replaces the old four finite-
// difference fBm passes per pixel. The matching reference is
// prototype/preview.py; keep the two in sync if either changes.

float3 DecodeSrgb(float3 c)
{
    return float3(
        c.r <= .04045 ? c.r / 12.92 : pow((c.r + .055) / 1.055, 2.4),
        c.g <= .04045 ? c.g / 12.92 : pow((c.g + .055) / 1.055, 2.4),
        c.b <= .04045 ? c.b / 12.92 : pow((c.b + .055) / 1.055, 2.4));
}

// Only OKLab L is needed for edge detection (Sa_aohue uses the same L-only
// shortcut for its line detector).
float OkLabL(float3 c)
{
    float l = pow(max(dot(c, float3(.4122214708, .5363325363, .0514459929)), 0), 1.0 / 3.0);
    float m = pow(max(dot(c, float3(.2119034982, .6806995451, .1073969566)), 0), 1.0 / 3.0);
    float s = pow(max(dot(c, float3(.0883024619, .2817188376, .6299787005)), 0), 1.0 / 3.0);
    return .2104542553 * l + .793617785 * m - .0040720468 * s;
}

float Hash21(float2 p)
{
    float x = sin(dot(p, float2(127.1, 311.7))) * 43758.5453123;
    return frac(x);
}

// Bilinear value noise, smoothstep-interpolated: C1 continuous, in [0, 1).
float ValueNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float a = Hash21(i);
    float b = Hash21(i + float2(1, 0));
    float c = Hash21(i + float2(0, 1));
    float d = Hash21(i + float2(1, 1));
    float2 u = f * f * (3 - 2 * f);
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

static const float2x2 OctaveRotation = { .8, .6, -.6, .8 };

// Analytic gradient of ValueNoise with respect to p. Smoothstep makes the
// value field C1 across cell borders, so rotating this gradient by 90 degrees
// gives a continuous, divergence-free curl field (up to floating-point error).
float2 ValueNoiseGradient(float2 p)
{
    float2 i = floor(p);
    float2 f = frac(p);
    float a = Hash21(i);
    float b = Hash21(i + float2(1, 0));
    float c = Hash21(i + float2(0, 1));
    float d = Hash21(i + float2(1, 1));
    float2 u = f * f * (3 - 2 * f);
    float2 du = 6 * f * (1 - f);
    float x0 = lerp(a, b, u.x);
    float x1 = lerp(c, d, u.x);
    float dValueDx = lerp(b - a, d - c, u.y) * du.x;
    float dValueDy = (x1 - x0) * du.y;
    return float2(dValueDx, dValueDy);
}

// Fractal Brownian motion: each octave is rotated before doubling frequency
// so the summed field has no dominant grid axis (a plain power-of-two
// pyramid without rotation shows visible horizontal/vertical streaking).
float Fbm(float2 p, int octaves)
{
    float value = 0, amp = .5, freq = 1;
    [loop]
    for (int i = 0; i < octaves; ++i)
    {
        value += amp * ValueNoise(p * freq);
        p = mul(OctaveRotation, p);
        freq *= 2.03;
        amp *= .5;
    }
    return value;
}

// Gradient of the rotated-octave fBm with respect to the original p. Each
// octave rotates its local coordinates; axisX/axisY carry the chain rule back
// to the original coordinate system.
float2 FbmGradient(float2 p, int octaves)
{
    float2 gradient = 0;
    float2 axisX = float2(1, 0);
    float2 axisY = float2(0, 1);
    float amp = .5;
    float freq = 1;
    [loop]
    for (int i = 0; i < octaves; ++i)
    {
        float2 gradQ = ValueNoiseGradient(p * freq) * freq;
        gradient.x += amp * dot(gradQ, axisX);
        gradient.y += amp * dot(gradQ, axisY);
        p = mul(OctaveRotation, p);
        axisX = mul(OctaveRotation, axisX);
        axisY = mul(OctaveRotation, axisY);
        freq *= 2.03;
        amp *= .5;
    }
    return gradient;
}

// Curl of a scalar fBm potential psi: v = (d(psi)/dy, -d(psi)/dx). The
// mixed partials cancel in the divergence, which is what makes this field
// swirl instead of pooling or draining.
float2 CurlNoise(float2 p, int octaves)
{
    float2 gradient = FbmGradient(p, octaves);
    return float2(gradient.y, -gradient.x);
}

// Inigo Quilez-style cosine palette: a cheap analytic stand-in for
// thin-film interference colour, used for the iridescent edge glint.
float3 IridescentPalette(float t)
{
    return .5 + .5 * cos(6.2831853 * (t + float3(0, .33, .67)));
}

float Gauss(float x, float center, float sigma)
{
    float d = (x - center) / sigma;
    return exp(-.5 * d * d);
}

// Approximate per-channel spectral response for a dispersion tap at sweep
// position t in [0, 1] (0 = innermost sample, 1 = outermost). Three
// overlapping bumps keep the middle of the sweep close to white while the
// extremes read as colour, like a real lens' chromatic fringing rather than
// a hard three-tap RGB split. Used by FlowDisplace.hlsl to blend an
// arbitrary number of dispersion taps into a smooth spectral gradient
// instead of a fixed R/G/B triple.
float3 SpectrumWeight(float t)
{
    return float3(Gauss(t, .85, .35), Gauss(t, .5, .35), Gauss(t, .15, .35));
}

// S_DistortChroma defaults to reflected borders. Reflecting the high-gain
// chroma taps keeps a large warp from collapsing into a flat edge smear.
float2 ReflectSamplePosition(float2 samplePosition, float4 bounds)
{
    float2 extent = max(bounds.zw - bounds.xy - 1.0, 1.0);
    float2 period = extent * 2.0;
    float2 q = samplePosition - bounds.xy;
    q -= period * floor(q / period);
    q = min(q, period - q);
    return bounds.xy + clamp(q, 0.0, extent);
}
