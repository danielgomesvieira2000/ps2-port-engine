#version 450
#extension GL_GOOGLE_include_directive : require
// Outbreak's characters drawn on the GPU (docs/NATIVE_RENDERER.md, the GPU's
// way): the characters' microprogram's rigid batch (VU1 EDE38E48, its code at
// 0x0230) as a vertex shader, its arithmetic the emulated VU's (gs_mesh_common.glsl:
// rounded toward zero) in the microcode's order.
//
// A vertex, five quadwords: its place, normal, texture coordinates, the
// addresses of its two matrices (x: the place's, four rows; y: the normal's,
// three), and one unused. The header: the registers vf00-vf21 the batch reads
// (vf01-vf04 the view, vf05-vf08 the guard band, vf09 the colour's scale, vf10
// the fog, vf11-vf14 the lights, vf15-vf21 the thing's colours), rows 11-13
// of data memory (the texture coordinates' matrix), XYOFFSET, the alpha (vi15),
// the two vertices the strip goes on from as the GS takes them (their packet's
// STQ, RGBAQ and XYZF2: the batch before's, under its own matrices - the
// mesh's first two vertices are these) and the data memory rows the matrices
// are in, from `first`. Which triangles:
// the draw's index buffer (the microcode's ADC: any of a triangle's vertices
// out of the guard band, or marked).

layout(set = 3, binding = 0, std430) readonly buffer Mesh {
    vec4 quad[];
} mesh;

layout(set = 3, binding = 1, std430) readonly buffer Header {
    vec4 vf[22];        // vf00-vf21
    vec4 tex[3];        // data memory rows 11-13
    vec4 misc;          // x, y: XYOFFSET's OFX, OFY (sixteenths); z: the first row below's address; w: vi15 (bits)
    vec4 kept[6];       // the two kept vertices' STQ, RGBAQ, XYZF2, as the packet has them
    vec4 row[];         // data memory from misc.z: the matrices
} header;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
} target;

#include "gs_depth.glsl"
#include "gs_mesh_common.glsl"

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec4 shade;
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

vec4 Row(uint address) { return header.row[address - floatBitsToUint(header.misc.z)]; }

void main() {
    if (gl_VertexIndex < 2) {
        // A kept vertex: as its packet has it.
        const int i = gl_VertexIndex * 3;
        const uvec4 rgbaq = floatBitsToUint(header.kept[i + 1]);
        const uvec4 xyzf = floatBitsToUint(header.kept[i + 2]);
        const vec2 position = (vec2(float(xyzf.x & 0xFFFFu), float(xyzf.y & 0xFFFFu)) - header.misc.xy) / 16.0;
        const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
        gl_Position = vec4(ndc, DepthOfZ((xyzf.z >> 4) & 0xFFFFFFu), 1.0);
        shade = vec4(Unorm8(rgbaq.x), Unorm8(rgbaq.y), Unorm8(rgbaq.z), Unorm8(rgbaq.w));
        coordinate = header.kept[i].xyz;
        fog_out = Unorm8((xyzf.w >> 4) & 0xFFu);
        range_out = vec4(-1e9, -1e9, 1e9, 1e9);
        return;
    }
    const uint at = uint(gl_VertexIndex) * 5u;
    const vec4 pos = mesh.quad[at];
    const vec4 nrm = mesh.quad[at + 1u];
    const vec4 tx = mesh.quad[at + 2u];
    const uvec4 matrices = floatBitsToUint(mesh.quad[at + 3u]);
    const uint ma = matrices.x & 0xFFFFu, mb = matrices.y & 0xFFFFu;
    const vec4 a0 = Row(ma), a1 = Row(ma + 1u), a2 = Row(ma + 2u), a3 = Row(ma + 3u);
    const vec4 b0 = Row(mb), b1 = Row(mb + 1u), b2 = Row(mb + 2u);
    // the place through its matrix (MULAX, MADDAY, MADDAZ, MADDW), the normal through its own (three rows)
    vec4 p;
    vec3 n;
    for (int l = 0; l < 4; ++l) p[l] = Add(Add(Add(Mul(a0[l], pos.x), Mul(a1[l], pos.y)), Mul(a2[l], pos.z)), a3[l]);
    for (int l = 0; l < 3; ++l) n[l] = Add(Add(Mul(b0[l], nrm.x), Mul(b1[l], nrm.y)), Mul(b2[l], nrm.z));
    // the view (vf01-vf04)
    vec4 cp;
    for (int l = 0; l < 4; ++l)
        cp[l] = Add(Add(Add(Mul(header.vf[1][l], p.x), Mul(header.vf[2][l], p.y)), Mul(header.vf[3][l], p.z)),
                    header.vf[4][l]);
    const float q = Recip(cp.w);
    // the texture coordinates through rows 11-13 (Q: the vertex's own z)
    const float s0 = Add(Add(Mul(header.tex[0].x, tx.x), Mul(header.tex[1].x, tx.y)), header.tex[2].x);
    const float t0 = Add(Add(Mul(header.tex[0].y, tx.x), Mul(header.tex[1].y, tx.y)), header.tex[2].y);
    const vec3 stq = vec3(Mul(s0, q), Mul(t0, q), Mul(tx.z, q));
    // the fog from w through vf10
    const float fog = max(min(Mul(Mul(Sub(header.vf[10].x, cp.w), header.vf[10].y), 255.0), 255.0), 0.0);
    const uint f = (uint(Ftoi(fog * 16.0)) >> 4) & 0xFFu;
    // The lights: x a direction (vf11-vf13's x as they are), y and z points (vf11-vf13's y and z its place),
    // each point over its distance (y: ERSQRT, z: RSQRT), its strength vf14's, at most 1.
    vec3 lx, ly, lz;
    lx.x = header.vf[11].x, ly.x = header.vf[12].x, lz.x = header.vf[13].x;
    for (int k = 1; k <= 2; ++k) {
        lx[k] = Sub(header.vf[11][k], p.x);
        ly[k] = Sub(header.vf[12][k], p.y);
        lz[k] = Sub(header.vf[13][k], p.z);
    }
    vec3 atten = vec3(1.0);
    for (int k = 1; k <= 2; ++k) {
        const float r = Rsqrt(Add(Add(Mul(lx[k], lx[k]), Mul(ly[k], ly[k])), Mul(lz[k], lz[k])));
        lx[k] = Mul(lx[k], r), ly[k] = Mul(ly[k], r), lz[k] = Mul(lz[k], r);
        atten[k] = min(Mul(header.vf[14][k], r), 1.0);
    }
    vec3 dot3;
    for (int k = 0; k < 3; ++k) {
        dot3[k] = max(Add(Add(Mul(lx[k], n.x), Mul(ly[k], n.y)), Mul(lz[k], n.z)), 0.0);
        if (k > 0) dot3[k] = Mul(dot3[k], atten[k]);
    }
    // the colour: vf15-vf17 by the lights, vf18, vf19 and vf20 by the points' strengths, vf21; at most 128, by vf09
    vec4 c;
    for (int l = 0; l < 3; ++l) {
        float acc = Mul(header.vf[15][l], dot3.x);
        acc = Add(acc, Mul(header.vf[16][l], dot3.y));
        acc = Add(acc, Mul(header.vf[17][l], dot3.z));
        acc = Add(acc, header.vf[18][l]);
        acc = Add(acc, Mul(header.vf[19][l], atten.y));
        acc = Add(acc, Mul(header.vf[20][l], atten.z));
        acc = Add(acc, header.vf[21][l]);
        c[l] = Mul(min(acc, 128.0), header.vf[9][l]);
    }
    const uvec3 rgb = uvec3(ivec3(c.xyz)) & 0xFFu;
    const uint alpha = floatBitsToUint(header.misc.w) & 0xFFu;
    // X, Y, Z as XYZF2 takes them, X and Y less XYOFFSET
    const uint x = uint(Ftoi(Mul(cp.x, q) * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(Mul(cp.y, q) * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(Mul(cp.z, q) * 16.0)) >> 4) & 0xFFFFFFu;
    const vec2 position = (vec2(float(x), float(y)) - header.misc.xy) / 16.0;
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(Unorm8(rgb.r), Unorm8(rgb.g), Unorm8(rgb.b), Unorm8(alpha));
    coordinate = stq;
    fog_out = Unorm8(f);
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}
