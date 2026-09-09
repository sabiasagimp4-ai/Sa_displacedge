// Pass 2/2: the creative core. Input0 is the original source (swept along
// the displacement at dispersionSteps offsets for a spectral prism fringe);
// Input1 is the blurred edge field from EdgeGradient.hlsl (recovered here
// as an average gradient direction + strength, i.e. how close/aligned this
// pixel is to a nearby contour). The two are blended into a
// divergence-free curl-noise flow, scaled by that closeness, to displace
// the image like the contours are melting into liquid glass; an animated
// rim glint colourised through a thin-film-style cosine palette rides the
// crest of the displacement.
#define D2D_ENTRY main
#include <d2d1effecthelpers.hlsli>
#include "Common.hlsli"

float strength;
float turbulence;
float turbulenceDetail;
float noiseScale;
float flowSpeed;
float time;
float dispersion;
float iridescence;
float lightAngle;
float seed;
float threshold;
float contrast;
float outputMode;
float dispersionSteps;
float phaseOffset;
float padding2;
float4 inputBounds;

D2D_PS_ENTRY(main)
{
    float2 p = D2DGetScenePosition().xy;

    // Recover the alpha-weighted average gradient/magnitude the blur left
    // packed as (gx*a, gy*a, mag*a, a).
    float4 field = D2DSampleInputAtPosition(1, p);
    float weight = field.a;
    float2 gradient = weight > 1e-5 ? field.rg / weight : 0;
    float magnitude = weight > 1e-5 ? field.b / weight : 0;

    float cutoff = saturate(threshold / 255.0) * .12;
    float mask = saturate((magnitude - cutoff) / .05);
    mask = pow(mask, 1.0 / max(.01, contrast));

    // The mask visualization does not need the flow field at all.
    if (outputMode > 1.5)
        return float4(mask, mask, mask, 1);

    float4 source = D2DSampleInputAtPosition(0, p);
    if (outputMode < .5 && (mask <= 0 || source.a <= 0))
        return source;

    float invMag = magnitude > 1e-5 ? 1.0 / magnitude : 0;
    float2 edgeNormal = gradient * invMag;

    // phaseOffset is added in the same frame-equivalent units as time,
    // *before* the .015/.05 rate scaling below, so it shifts the swirl's
    // starting point by a fixed amount regardless of flowSpeed -- unlike
    // seed (a spatial coordinate offset, a different pattern entirely),
    // this only moves *where in its cycle* the same pattern currently is.
    float animTime = time * flowSpeed + phaseOffset;
    float2 coord = p / max(1.0, noiseScale) + seed * 17.0;
    coord.x += animTime * .015;
    int octaves = (int) clamp(round(turbulenceDetail), 1, 8);

    float turb = saturate(turbulence);
    float irid = saturate(iridescence);
    float2 curl = 0;
    float2 curlDir = float2(1, 0);
    if (turb > 1e-5 || irid > 1e-5)
    {
        curl = CurlNoise(coord, octaves);
        float curlLen = length(curl);
        curlDir = curlLen > 1e-5 ? curl / curlLen : float2(1, 0);
    }

    float2 flow = lerp(edgeNormal, curlDir, turb);
    float flowLen = length(flow);
    flow = flowLen > 1e-5 ? flow / flowLen : 0;

    float2 disp = flow * (strength * mask);

    if (outputMode > .5)
        return float4(.5 + .5 * flow.x * mask, .5 + .5 * flow.y * mask, mask, 1);

    // Sweep dispersionSteps taps from scale (1-dispersion) to (1+dispersion)
    // along the displacement vector, each weighted by an approximate
    // spectral response (SpectrumWeight), and blend them into a smoothly
    // graded prism fringe. 3 steps reduces to the classic R/G/B split;
    // more steps trade GPU time for a smoother, more continuous spectrum.
    float d = saturate(dispersion);
    float3 rgb;
    if (d <= 1e-5)
    {
        rgb = source.a > 0 ? source.rgb / source.a : 0;
    }
    else
    {
        float4 uv = D2DGetInputCoordinate(0);
        int steps = (int) clamp(round(dispersionSteps), 3, 128);
        float3 colorSum = 0, weightSum = 0;
        [loop]
        for (int s = 0; s < steps; ++s)
        {
            float t = (float) s / (float) (steps - 1);
            float scale = 1 + d * (2 * t - 1);
            float2 samplePosition = clamp(p + disp * scale, inputBounds.xy, inputBounds.zw - 1);
            float4 c = InputTexture0.SampleLevel(InputSampler0, uv.xy + uv.zw * (samplePosition - p), 0);
            float a = saturate(c.a);
            float3 tap = a > 0 ? c.rgb / a : 0;
            float3 w = SpectrumWeight(t);
            colorSum += tap * w;
            weightSum += w;
        }
        rgb = colorSum / max(weightSum, 1e-5);
    }

    float3 glint = 0;
    if (irid > 1e-5)
    {
        float2 lightDir = float2(cos(radians(lightAngle)), sin(radians(lightAngle)));
        float ndotl = saturate(dot(edgeNormal, lightDir) * .5 + .5);
        float spec = pow(ndotl, 8);
        float phase = magnitude * 10 + animTime * .05 + curl.x * .6;
        glint = IridescentPalette(phase) * (spec * mask * irid);
    }

    float3 result = saturate(rgb + glint);
    float outAlpha = saturate(source.a);
    return float4(result * outAlpha, outAlpha);
}
