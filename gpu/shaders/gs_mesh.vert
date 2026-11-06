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

// The VU rounds toward zero (the emulated VU's arithmetic, under VuRounding:
// what the recorded pictures hold), the GPU to the nearest: each operation
// made toward zero from the nearest and its exact error (a product's by fma,
// a sum's by TwoSum) - one step toward zero where the nearest went past.
// Exact for normal values (the native programs decline the rest).
const float kMax = uintBitsToFloat(0x7F7FFFFFu);
float TowardZero(float x) { return uintBitsToFloat(floatBitsToUint(x) - 1u); }     // (x not 0: its magnitude one less)
float AwayFromZero(float x) { return uintBitsToFloat(floatBitsToUint(x) + 1u); }
float Fix(float r, float e) {
    // r the nearest, e the exact value's excess over it: past it (the other sign) - one toward zero
    return (e != 0.0 && r != 0.0 && ((e < 0.0) != (r < 0.0))) ? TowardZero(r) : r;
}
float Mul(float a, float b) {
    precise float r = a * b;
    precise float e = fma(a, b, -r);
    return Fix(r, e);
}
float Add(float a, float b) {
    precise float r = a + b;
    precise float bb = r - a;
    precise float e = (a - (r - bb)) + (b - bb);
    return Fix(r, e);
}
float Sub(float a, float b) { return Add(a, -b); }
// 1/x toward zero (the VU's DIV): the largest magnitude whose product with x is not past 1.
float Recip(float x) {
    const float ax = abs(x);
    if (ax == 0.0) return x < 0.0 || floatBitsToUint(x) == 0x80000000u ? -kMax : kMax;   // (the VU's: the largest)
    precise float q = 1.0 / ax;
    if (isinf(q) || q == 0.0) return x < 0.0 ? -q : q;
    precise float e0 = fma(-ax, q, 1.0);
    precise float q1 = fma(q, e0, q);
    q = q1;
    for (int i = 0; i < 2; ++i) {
        precise float r = fma(-q, ax, 1.0);
        if (r < 0.0) q = TowardZero(q);
    }
    precise float up = AwayFromZero(q);
    precise float ru = fma(-up, ax, 1.0);
    if (ru >= 0.0) q = up;
    return x < 0.0 ? -q : q;
}
// 1/sqrt(x) toward zero (the VU's RSQRT: the double's 1/sqrt truncated): the largest y with y*y*x not past 1.
bool Past(float y, float x) {
    // y*y*x > 1, near enough exactly: y*y = p + pe (exact), then p*x - 1 and pe*x
    precise float p = y * y;
    precise float pe = fma(y, y, -p);
    precise float t = fma(p, x, -1.0);
    precise float total = fma(pe, x, t);
    return total > 0.0;
}
float Rsqrt(float x) {
    if (x == 0.0) return kMax;      // (the VU's: the largest)
    precise float y = inversesqrt(x);
    if (isinf(y) || y == 0.0) return y;
    // (a Newton step: the GPU's is approximate)
    precise float h = 0.5 * x * y;
    precise float n = fma(-h, y, 1.5);
    y = y * n;
    for (int i = 0; i < 2; ++i)
        if (Past(y, x)) y = TowardZero(y);
    if (!Past(AwayFromZero(y), x)) y = AwayFromZero(y);
    return y;
}
// A row's dot with the place, as MULA, MADDA, MADDA, MADD (w: 1) make it.
float Dot(int r, vec4 p, int lane) {
    return Add(Add(Add(Mul(header.row[r][lane], p.x), Mul(header.row[r + 1][lane], p.y)),
                   Mul(header.row[r + 2][lane], p.z)), header.row[r + 3][lane]);
}

void main() {
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
    gl_CullDistance[0] = gw - g.x;
    gl_CullDistance[1] = gw + g.x;
    gl_CullDistance[2] = gw - g.y;
    gl_CullDistance[3] = gw + g.y;
    gl_CullDistance[4] = gw - g.z;
    gl_CullDistance[5] = gw + g.z;
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
    shade = vec4(rgba) / 255.0;
    coordinate = stq;
    fog_out = float(f) / 255.0;
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}
