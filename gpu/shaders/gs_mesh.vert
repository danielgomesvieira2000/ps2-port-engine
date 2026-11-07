#version 450
#extension GL_GOOGLE_include_directive : require
// A native microprogram's vertices made on the GPU (docs/NATIVE_RENDERER.md,
// the GPU's way): a mesh of the game's own vertices kept in a buffer, and per
// draw the header the microprogram would have been given - the GS's values
// made here, as the microcode makes them, then on as gs.vert takes them from
// the bridge. The first such program, Outbreak's room (VU1 C43D662E; its
// reference on the CPU FloatOf in games/re_outbreak2/port/vu_native_room.cpp,
// the same arithmetic - float rounded toward zero, the emulated VU's, which
// matches the microcode word for word in every scripted run).
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
    vec4 offset;        // XYOFFSET's OFX, OFY (sixteenths of a pixel); z not 0: the mesh's first two vertices the kept
    vec4 kept[6];       // the two vertices the strip goes on from, as the GS takes them (STQ, RGBAQ, XYZF2 each)
    vec4 ext[12];       // the enhanced draws' (gs_mesh_enhanced.glsl)
} header;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
} target;

#include "gs_depth.glsl"

#ifdef ENHANCED
// (the guard band's planes clip: a triangle across one drawn in part, by the GPU)
out gl_PerVertex {
    vec4 gl_Position;
    float gl_ClipDistance[6];
};
#define GUARD gl_ClipDistance
#else
out gl_PerVertex {
    vec4 gl_Position;
    float gl_CullDistance[6];
};
#define GUARD gl_CullDistance
#endif

layout(location = 0) out vec4 shade;
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

#include "gs_mesh_common.glsl"
#ifdef ENHANCED
#include "gs_mesh_enhanced.glsl"
#include "gs_shadow.glsl"
uint Enhancements() { return floatBitsToUint(header.ext[0].y); }
vec3 ViewOf(vec4 clip) {
    const vec4 v = shadow.unproject * clip;
    return v.xyz / (abs(v.w) > 1e-20 ? v.w : 1.0);
}
#endif

// A row's dot with the place, as MULA, MADDA, MADDA, MADD (w: 1) make it.
float Dot(int r, vec4 p, int lane) {
    return Add(Add(Add(Mul(header.row[r][lane], p.x), Mul(header.row[r + 1][lane], p.y)),
                   Mul(header.row[r + 2][lane], p.z)), header.row[r + 3][lane]);
}

void main() {
    if (header.offset.z != 0.0 && gl_VertexIndex < 2) {
        // A kept vertex: as its packet has it (the batch before's, under its own header); never culled by itself.
        const int i = gl_VertexIndex * 3;
        const uvec4 rgbaq = floatBitsToUint(header.kept[i + 1]);
        const uvec4 xyzf = floatBitsToUint(header.kept[i + 2]);
        const vec2 position = (vec2(float(xyzf.x & 0xFFFFu), float(xyzf.y & 0xFFFFu)) - header.offset.xy) / 16.0;
        const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
        gl_Position = vec4(ndc, DepthOfZ((xyzf.z >> 4) & 0xFFFFFFu), 1.0);
        for (int d = 0; d < 6; ++d) GUARD[d] = 1.0;
        shade = vec4(Unorm8(rgbaq.x), Unorm8(rgbaq.y), Unorm8(rgbaq.z), Unorm8(rgbaq.w));
        coordinate = header.kept[i].xyz;
        fog_out = Unorm8((xyzf.w >> 4) & 0xFFu);
        range_out = vec4(-1e9, -1e9, 1e9, 1e9);
#ifdef ENHANCED
        // known (the batch before's, as the game kept it): placed, clipped and lit as the batch's own
        lit_on = 0.0;
        lit_program = 0u;
        lit_place = vec4(0.0);
        lit_normal = vec3(0.0);
        lit_colour = vec4(0.0);
        lit_eye = EyeOf(header.row[3], header.row[4], header.row[5], header.row[6]);
        if (((floatBitsToUint(header.ext[0].x) >> gl_VertexIndex) & 1u) != 0u) {
            const int e = 1 + 5 * gl_VertexIndex;
            const vec4 g = header.ext[e];
            gl_Position = ClipPlace(header.ext[e + 1], header.offset.xy, target.size, position,
                                    (xyzf.z >> 4) & 0xFFFFFFu, (Enhancements() & kPrecise) == 0u, InGuard(g));
            const float gw = abs(g.w);
            GUARD[0] = gw - g.x, GUARD[1] = gw + g.x, GUARD[2] = gw - g.y;
            GUARD[3] = gw + g.y, GUARD[4] = gw - g.z, GUARD[5] = gw + g.z;
            lit_place = vec4(header.ext[e + 2].xyz, length(header.ext[e + 3].xyz));
            lit_normal = header.ext[e + 3].xyz;
            lit_colour = header.ext[e + 4];
            lit_on = (Enhancements() & kLitPixels) != 0u ? 1.0 : 0.0;
            lit_view = ViewOf(header.ext[e + 1]);
        } else {
            lit_view = vec3(0.0);
        }
#endif
        return;
    }
    const uint at = uint(gl_VertexIndex) * 4u;
    const vec4 p = mesh.quad[at];
    const vec4 nrm = mesh.quad[at + 1u];
    const vec4 col = mesh.quad[at + 2u];
    const vec4 tx = mesh.quad[at + 3u];
    // the place through rows 3-6 (the view) and 7-10 (the guard band)
    vec4 cp, g;
    for (int l = 0; l < 4; ++l) {
        cp[l] = Dot(3, p, l);
        g[l] = Dot(7, p, l);
    }
    // the guard band's six planes, as CLIP judges (|x| > |w| out)
    const float gw = abs(g.w);
    GUARD[0] = gw - g.x;
    GUARD[1] = gw + g.x;
    GUARD[2] = gw - g.y;
    GUARD[3] = gw + g.y;
    GUARD[4] = gw - g.z;
    GUARD[5] = gw + g.z;
    const float q = Recip(cp.w);
    // the texture coordinates through rows 15-17 (Q: the vertex's own z)
    const float s0 = Add(Add(Mul(header.row[15].x, tx.x), Mul(header.row[16].x, tx.y)), header.row[17].x);
    const float t0 = Add(Add(Mul(header.row[15].y, tx.x), Mul(header.row[16].y, tx.y)), header.row[17].y);
    const vec3 stq = vec3(Mul(s0, q), Mul(t0, q), Mul(tx.z, q));
    // a point light: vf15-vf17's x its place, vf12's x its strength over the distance
    const float lx = Sub(header.k[4].x, p.x), ly = Sub(header.k[5].x, p.y), lz = Sub(header.k[6].x, p.z);
    const float rs = Rsqrt(Add(Add(Mul(lx, lx), Mul(ly, ly)), Mul(lz, lz)));
    const float atten = min(Mul(header.k[1].x, rs), 1.0);
    const float nx = Mul(lx, rs), ny = Mul(ly, rs), nz = Mul(lz, rs);
    const float lambert = max(Add(Add(Mul(nx, nrm.x), Mul(ny, nrm.y)), Mul(nz, nrm.z)), 0.0);
    vec4 c;
    for (int l = 0; l < 3; ++l) {
        const float lit = Add(Mul(header.k[2][l], lambert), Mul(header.k[3][l], col[l]));
        c[l] = Add(Mul(lit, atten), Mul(header.row[0][l], col[l]));
    }
    c.w = col.w;
    for (int l = 0; l < 4; ++l) c[l] = Mul(min(c[l], 128.0), header.row[1][l]);
    const uvec4 rgba = uvec4(ivec4(c)) & 0xFFu;
    // the fog from w through vf11
    const float fog = max(min(Mul(Mul(Sub(header.k[0].x, cp.w), header.k[0].y), 255.0), 255.0), 0.0);
    const uint f = (uint(Ftoi(fog * 16.0)) >> 4) & 0xFFu;
    // X, Y, Z as XYZF2 takes them (12.4 fixed X and Y, 24-bit Z; FTOI4's 16 exact), X and Y less XYOFFSET
    const uint x = uint(Ftoi(Mul(cp.x, q) * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(Mul(cp.y, q) * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(Mul(cp.z, q) * 16.0)) >> 4) & 0xFFFFFFu;
    const vec2 position = (vec2(float(x), float(y)) - header.offset.xy) / 16.0;
    // ...on as gs.vert
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(Unorm8(rgba.r), Unorm8(rgba.g), Unorm8(rgba.b), Unorm8(rgba.a));
    coordinate = stq;
    fog_out = Unorm8(f);
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
#ifdef ENHANCED
    gl_Position = ClipPlace(cp, header.offset.xy, target.size, position, z, (Enhancements() & kPrecise) == 0u,
                            InGuard(g));
    lit_place = vec4(p.xyz, length(nrm.xyz));
    lit_normal = nrm.xyz;
    lit_colour = col;
    lit_on = (Enhancements() & kLitPixels) != 0u ? 1.0 : 0.0;
    lit_program = 0u;
    lit_eye = EyeOf(header.row[3], header.row[4], header.row[5], header.row[6]);
    lit_view = ViewOf(cp);
#endif
}
