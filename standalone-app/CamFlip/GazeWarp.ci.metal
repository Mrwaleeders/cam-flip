#include <CoreImage/CoreImage.h>
using namespace metal;

// Eye-contact warp, run by Core Image on the GPU.
//
// For each output pixel it returns where to read from the camera image. Inside each eye
// region the content is pulled along `shift` (towards the top of the face), with full
// strength near the iris and a smooth falloff to zero at the region's edge. Pixels outside
// both eye regions are untouched, so the rest of the image stays bit-for-bit sharp.
//
// eye   = (centerX, centerY, radiusX, radiusY) in image pixels (Core Image, origin bottom-left)
// shift = displacement in pixels (already oriented to the face's "up" direction)
[[stitchable]] float2 gazeWarp(float4 eyeA, float2 shiftA, float4 eyeB, float2 shiftB,
                               coreimage::destination dest)
{
    float2 p = dest.coord();

    float2 dA = (p - eyeA.xy) / max(eyeA.zw, float2(1.0));
    float wA = 1.0 - smoothstep(0.35, 1.0, length(dA));

    float2 dB = (p - eyeB.xy) / max(eyeB.zw, float2(1.0));
    float wB = 1.0 - smoothstep(0.35, 1.0, length(dB));

    return p - shiftA * wA - shiftB * wB;
}
