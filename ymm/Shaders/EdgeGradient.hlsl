#define D2D_REQUIRES_SCENE_POSITION
// Pass 1/2: Scharr gradient of OKLab L at an adjustable tap spacing
// (detectionScale), weight-carried by alpha as (gx*a, gy*a, mag*a, a).
// D2D's built-in GaussianBlur (applied to this pass's output by
// SaDisplacedgeProcessor) treats the buffer as premultiplied colour, so
// carrying the weight this way makes the blur an alpha-correct weighted
// average of direction and magnitude -- the same numerator/denominator
// trick Sa_aohue's ChromaField/GaussianHorizontal passes use.
#define D2D_ENTRY main
#include <d2d1effecthelpers.hlsli>
#include "Common.hlsli"

float detectionScale;
float padding0;
float padding1;
float padding2;
float4 inputBounds;

static const float3x3 ScharrX = {
    -3, 0, 3,
    -10, 0, 10,
    -3, 0, 3
};
static const float3x3 ScharrY = {
    -3, -10, -3,
    0, 0, 0,
    3, 10, 3
};

D2D_PS_ENTRY(main)
{
    float2 p = D2DGetScenePosition().xy;
    float step = max(.25, detectionScale);
    float4 uv = D2DGetInputCoordinate(0);

    // The centre tap is needed only for alpha because the Scharr centre
    // coefficients are zero. Reusing it removes one texture fetch per pixel.
    float4 center = InputTexture0.SampleLevel(InputSampler0, uv.xy, 0);

    float gx = 0, gy = 0;
    [unroll]
    for (int j = -1; j <= 1; ++j)
    {
        [unroll]
        for (int i = -1; i <= 1; ++i)
        {
            if (i != 0 || j != 0)
            {
                float2 samplePosition = clamp(p + float2(i, j) * step, inputBounds.xy, inputBounds.zw - 1);
                float4 c = InputTexture0.SampleLevel(InputSampler0, uv.xy + uv.zw * (samplePosition - p), 0);
                float a = saturate(c.a);
                float l = a > 0 ? OkLabL(DecodeSrgb(saturate(c.rgb / a))) : 0;
                gx += ScharrX[j + 1][i + 1] * l;
                gy += ScharrY[j + 1][i + 1] * l;
            }
        }
    }
    gx /= 16.0;
    gy /= 16.0;
    float mag = length(float2(gx, gy));

    float alpha = saturate(center.a);
    return float4(gx, gy, mag, 1) * alpha;
}
