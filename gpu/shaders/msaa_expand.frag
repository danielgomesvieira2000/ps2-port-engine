#version 450
// ...each sample of a pixel the single-sample image's pixel: its colour, its
// depth (the pipeline's write mask and depth write say which of them).

layout(set = 0, binding = 0) uniform sampler2D colour_in;
layout(set = 0, binding = 1) uniform sampler2D depth_in;

layout(location = 0) out vec4 colour;

void main() {
    const ivec2 at = ivec2(gl_FragCoord.xy);
    colour = texelFetch(colour_in, at, 0);
    gl_FragDepth = texelFetch(depth_in, at, 0).r;
}
