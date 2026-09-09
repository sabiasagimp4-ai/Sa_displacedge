// Pass 2/2: the creative core. Input0 is the original source (sampled at
// three dispersed offsets for a prism-like fringe); Input1 is the blurred
// edge field from EdgeGradient.hlsl (recovered here as an average gradient
// direction + strength, i.e. how close/aligned this pixel is to a nearby
// contour). The two are blended into a divergence-free curl-noise flow,
// scaled by that closeness, to displace the image like the contours are
// melting into liquid glass; an animated rim glint colourised through a
// thin-film-style cosine palette rides the crest of the displacement.
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
float padding0;
float padding1;
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
    mask = pow(mask, 1.0 / max(.1, contrast));

    float4 source = D2DSampleInputAtPosition(0, p);
    if (outputMode < .5 && (mask <= 0 || source.a <= 0))
        return source;

    float invMag = magnitude > 1e-5 ? 1.0 / magnitude : 0;
    float2 edgeNormal = gradient * invMag;

    float2 coord = p / max(1.0, noiseScale) + seed * 17.0;
    coord.x += time * flowSpeed * .015;
    int octaves = (int) clamp(round(turbulenceDetail), 1, 6);
    float2 curl = CurlNoise(coord, octaves);
    float curlLen = length(curl);
    float2 curlDir = curlLen > 1e-5 ? curl / curlLen : float2(1, 0);

    float2 flow = lerp(edgeNormal, curlDir, saturate(turbulence));
    float flowLen = length(flow);
    flow = flowLen > 1e-5 ? flow / flowLen : 0;

    float2 disp = flow * (strength * mask);

    if (outputMode > 1.5)
        return float4(mask, mask, mask, 1);
    if (outputMode > .5)
        return float4(.5 + .5 * flow.x * mask, .5 + .5 * flow.y * mask, mask, 1);

    float4 uv = D2DGetInputCoordinate(0);
    float2 spread = disp * saturate(dispersion);
    float2 rp = clamp(p + disp + spread, inputBounds.xy, inputBounds.zw - 1);
    float2 gp = clamp(p + disp, inputBounds.xy, inputBounds.zw - 1);
    float2 bp = clamp(p + disp - spread, inputBounds.xy, inputBounds.zw - 1);
    float4 rc = InputTexture0.SampleLevel(InputSampler0, uv.xy + uv.zw * (rp - p), 0);
    float4 gc = InputTexture0.SampleLevel(InputSampler0, uv.xy + uv.zw * (gp - p), 0);
    float4 bc = InputTexture0.SampleLevel(InputSampler0, uv.xy + uv.zw * (bp - p), 0);
    float ra = saturate(rc.a), ga = saturate(gc.a), ba = saturate(bc.a);
    float3 rgb = float3(
        ra > 0 ? rc.r / ra : 0,
        ga > 0 ? gc.g / ga : 0,
        ba > 0 ? bc.b / ba : 0);

    float2 lightDir = float2(cos(radians(lightAngle)), sin(radians(lightAngle)));
    float ndotl = saturate(dot(edgeNormal, lightDir) * .5 + .5);
    float spec = pow(ndotl, 8);
    float phase = magnitude * 10 + time * flowSpeed * .05 + curl.x * .6;
    float3 glint = IridescentPalette(phase) * (spec * mask * saturate(iridescence));

    float3 result = saturate(rgb + glint);
    float outAlpha = saturate(source.a);
    return float4(result * outAlpha, outAlpha);
}
