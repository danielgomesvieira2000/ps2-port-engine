// SMAA 1x as three compute passes (ANTI-ALIASING: SMAA): each pass's pixel
// computes the offsets SMAA's vertex shader would hand its pixel shader,
// then runs that pixel shader (gpu/third_party/smaa/SMAA.glsl, its HIGH
// preset). The picture's size in the push constant: 1/w, 1/h, w, h.

layout(local_size_x = 8, local_size_y = 8) in;
layout(push_constant) uniform Metrics { vec4 rt; } metrics;

#define SMAA_GLSL_4 1
#define SMAA_RT_METRICS metrics.rt
#define SMAA_PRESET_HIGH 1
#define SMAA_DISCARD            // a pixel with no edge: its edges stay 0
#include "SMAA.glsl"

ivec2 Pixel() { return ivec2(gl_GlobalInvocationID.xy); }
bool Outside(ivec2 p) { return p.x >= int(metrics.rt.z) || p.y >= int(metrics.rt.w); }
vec2 TexCoord(ivec2 p) { return (vec2(p) + 0.5) * metrics.rt.xy; }
