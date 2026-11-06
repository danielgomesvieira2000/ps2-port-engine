#version 450
#extension GL_GOOGLE_include_directive : require
// A native microprogram's vertices made on the GPU (docs/NATIVE_RENDERER.md,
// the GPU's way): a mesh of the game's own vertices kept in a buffer, and per
// draw the header the microprogram would have been given - the GS's values
// made here, as the microcode makes them, then on as gs.vert takes them from
// the bridge. Not used yet: the first such program, Outbreak's room (VU1
// C43D662E; its reference on the CPU FloatOf in games/re_outbreak2/port/
// vu_native_room.cpp, the same arithmetic - plain float, which the microcode's
// matches word for word in every scripted run).
//
// A vertex, four quadwords: its place, normal, colour and texture coordinates.
// The header: rows 0-23 of VU1's data memory, and the program's constant
// registers vf11-vf17; the context's XYOFFSET (sixteenths) beside them.
// Which triangles: the strip's restarts (the vertices marked in their data)
// are the index buffer's; a triangle whose three vertices are all out past
// one plane of the guard band (rows 7-10: CLIP, |x| > |w| and so on - the
// microcode's ADC) is culled by the six cull distances, the same rule. The
// places stay the GS's (w 1, as gs.vert's): a triangle across a plane - the
// microcode's clipper - is not for this shader (its batch the microcode's,
// for now; later homogeneous places and the GPU's clipping).

layout(set = 3, binding = 0, std430) readonly buffer Mesh {
    vec4 quad[];
} mesh;

layout(set = 3, binding = 1, std430) readonly buffer Header {
    vec4 row[24];       // VU1 data memory 0-23
    vec4 k[7];          // vf11-vf17
    vec4 offset;        // XYOFFSET's OFX, OFY (sixteenths of a pixel)
} header;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
} target;

#include "gs_depth.glsl"

out gl_PerVertex {
    vec4 gl_Position;
    float gl_CullDistance[6];
};

layout(location = 0) out vec4 shade;
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

// FTOI: toward zero, saturated.
int Ftoi(float y) {
    if (y >= 2147483648.0) return 2147483647;
    if (y < -2147483648.0) return -2147483647 - 1;
    return int(y);
}

void main() {
    const uint at = uint(gl_VertexIndex) * 4u;
    const vec4 p = mesh.quad[at];
    const vec4 nrm = mesh.quad[at + 1u];
    const vec4 col = mesh.quad[at + 2u];
    const vec4 tx = mesh.quad[at + 3u];
    // (`precise`: no product and sum fused - the microcode rounds each)
    precise vec4 cp = header.row[3] * p.x + header.row[4] * p.y + header.row[5] * p.z + header.row[6];
    // the guard band's six planes, as CLIP judges (|x| > |w| out)
    precise vec4 g = header.row[7] * p.x + header.row[8] * p.y + header.row[9] * p.z + header.row[10];
    const float gw = abs(g.w);
    gl_CullDistance[0] = gw - g.x;
    gl_CullDistance[1] = gw + g.x;
    gl_CullDistance[2] = gw - g.y;
    gl_CullDistance[3] = gw + g.y;
    gl_CullDistance[4] = gw - g.z;
    gl_CullDistance[5] = gw + g.z;
    precise float q = 1.0 / cp.w;
    // the texture coordinates through rows 15-17 (Q: the vertex's own z)
    precise vec2 st = header.row[15].xy * tx.x + header.row[16].xy * tx.y + header.row[17].xy;
    precise vec3 stq = vec3(st * q, tx.z * q);
    // a point light: vf15-vf17's x its place, vf12's x its strength over the distance
    precise vec3 l = vec3(header.k[4].x, header.k[5].x, header.k[6].x) - p.xyz;
    precise float rs = 1.0 / sqrt(dot(l, l));
    precise float atten = min(header.k[1].x * rs, 1.0);
    precise float lambert = max((l.x * rs) * nrm.x + (l.y * rs) * nrm.y + (l.z * rs) * nrm.z, 0.0);
    precise vec3 lit = (header.k[2].xyz * lambert + header.k[3].xyz * col.xyz) * atten + header.row[0].xyz * col.xyz;
    precise vec4 c = min(vec4(lit, col.w), vec4(128.0)) * header.row[1];
    const uvec4 rgba = uvec4(ivec4(c)) & 0xFFu;
    // the fog from w through vf11
    precise float fog = clamp((header.k[0].x - cp.w) * header.k[0].y * 255.0, 0.0, 255.0);
    const uint f = (uint(Ftoi(fog * 16.0)) >> 4) & 0xFFu;
    // X, Y, Z as XYZF2 takes them (12.4 fixed X and Y, 24-bit Z), X and Y less XYOFFSET
    const uint x = uint(Ftoi(cp.x * q * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(cp.y * q * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(cp.z * q * 16.0)) >> 4) & 0xFFFFFFu;
    const vec2 position = (vec2(float(x), float(y)) - header.offset.xy) / 16.0;
    // ...on as gs.vert
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(rgba) / 255.0;
    coordinate = stq;
    fog_out = float(f) / 255.0;
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}
