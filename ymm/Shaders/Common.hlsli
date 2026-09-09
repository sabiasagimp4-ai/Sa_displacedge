// Shared helpers for the Sa_displacedge shader passes: sRGB/OKLab colour,
// and a divergence-free curl-noise field built from a rotated-octave
// value-noise fBm. Ported 1:1 (including the exact derivative pairing) from
// prototype/preview.py, which is verified by tests/test_prototype.py --
// keep the two in sync if either changes.

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

// Curl of a scalar fBm potential psi: v = (d(psi)/dy, -d(psi)/dx). The
// *mixed* partials cancel in the divergence (d2(psi)/dydx == d2(psi)/dxdy),
// which is what makes this field swirl -- dye stirred into water -- instead
// of pooling or draining anywhere. Pairing same-axis derivatives instead
// (an easy mistake -- prototype/preview.py caught exactly this once) gives
// the Laplacian difference, not a divergence-free field. Central-difference
// operators commute exactly regardless of step size, so any eps is
// admissible here; downstream code must not re-differentiate this result
// with a *different* step, which would probe unrelated sub-eps structure.
float2 CurlNoise(float2 p, int octaves)
{
    const float e = .5;
    float dPsiDy = (Fbm(p + float2(0, e), octaves) - Fbm(p - float2(0, e), octaves)) / (2 * e);
    float dPsiDx = (Fbm(p + float2(e, 0), octaves) - Fbm(p - float2(e, 0), octaves)) / (2 * e);
    return float2(dPsiDy, -dPsiDx);
}

// Inigo Quilez-style cosine palette: a cheap analytic stand-in for
// thin-film interference colour, used for the iridescent edge glint.
float3 IridescentPalette(float t)
{
    return .5 + .5 * cos(6.2831853 * (t + float3(0, .33, .67)));
}
