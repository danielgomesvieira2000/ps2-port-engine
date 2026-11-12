#version 450
#extension GL_GOOGLE_include_directive : require
// A GS vertex: position in pixels, colour 0-255 (normalised by the attribute
// format), texture coordinate S, T, Q. The GS samples a pixel at its whole
// coordinates and Vulkan at its centre, so positions move half a pixel.

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 colour;
layout(location = 2) in vec3 stq;
layout(location = 3) in uint z;     // the GS's Z, unsigned; larger is nearer
layout(location = 4) in float fog;  // F, 0-1
// A sprite on a target drawn larger: the S, T the GS reads at its first and
// last pixels (s0, t0, s1, t1) - the texture coordinates kept inside them.
layout(location = 5) in vec4 range;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
    // Drawn larger (the internal resolution): how far a vertex on the
    // target's top or left edge (0) reaches out (0 at the PS2's own). Its
    // pixels sample at their centres, (i + 0.5) / scale - 0.5 in the GS's
    // pixels: the first row and column of the larger picture before 0 -
    // reached by nothing that starts at 0 (a picture's whole-screen
    // overlay), they kept what was there before: a line of other pixels
    // along the top and the left.
    float edge;
} target;

#include "gs_depth.glsl"

layout(location = 0) out vec4 shade;
// The GS interpolates S, T and Q linearly across the screen and divides per
// pixel: no perspective correction of its own.
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

void main() {
    vec2 at = position;
    if (target.edge > 0.0) {
        if (at.x <= 0.0) at.x -= target.edge;
        if (at.y <= 0.0) at.y -= target.edge;
    }
    vec2 ndc = (at + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = colour;
    coordinate = stq;
    fog_out = fog;
    range_out = range;
}
