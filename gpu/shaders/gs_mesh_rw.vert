#version 450
#extension GL_GOOGLE_include_directive : require
// RenderWare's VU1 programs drawn on the GPU (docs/NATIVE_RENDERER.md, the GPU's
// way): Burnout 3's car program (VU1 image 7E86891B) as a vertex shader, its
// arithmetic the emulated VU's (gs_mesh_common.glsl: rounded toward zero) in the
// microcode's order. host/include/ps2host/rw_native.h is its twin on the CPU and
// documents the vertex and the header; the draw's indices are the strip's
// triangles the microcode leaves without ADC.

layout(set = 3, binding = 0, std430) readonly buffer Mesh {
    uvec4 quad[];
} mesh;

layout(set = 3, binding = 1, std430) readonly buffer Header {
    vec4 q[];
} header;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
} target;

#include "gs_depth.glsl"
#include "gs_mesh_common.glsl"
#include "gs_mesh_rw.glsl"

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec4 shade;
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

// (rw_native.h's kRw*)
const int kMisc = 0, kC = 1, kO = 5, kM = 6, kL = 10, kR = 11, kC0 = 14, kC1 = 15, kEnv = 16, kGuard = 17, kG = 19;
const uint kEnvStream = 1u, kGuardPlace = 2u;

float Madd(float acc, float a, float b) { return Add(acc, Mul(a, b)); }
float Msub(float acc, float a, float b) { return Sub(acc, Mul(a, b)); }

void main() {
    const uint mode = floatBitsToUint(header.q[kMisc].z);
    const uint j = uint(gl_VertexIndex);
    const uvec4 q0 = mesh.quad[3u * j];
    const uvec4 q2 = mesh.quad[3u * j + 2u];
    const vec3 p = uintBitsToFloat(q0.xyz);
    const uint g = j / 4u, lane = j % 4u;
    const uvec4 q1 = mesh.quad[3u * (4u * g + ((lane & 1u) != 0u ? 2u : 0u)) + 1u];
    const uvec2 st = (lane & 2u) != 0u ? q1.zw : q1.xy;
    const vec4 L = header.q[kL];
    const vec3 n = vec3(Sub(uintBitsToFloat(q2.x), L.w), Sub(uintBitsToFloat(q2.y), L.w), Sub(uintBitsToFloat(q2.z), L.w));
    // the place, and Q
    vec4 clip;
    float q;
    if ((mode & kGuardPlace) != 0u) {
        vec4 gg;
        for (int l = 0; l < 4; ++l) {
            float acc = Mul(header.q[kG + 3][l], 1.0);
            acc = Madd(acc, header.q[kG][l], p.x);
            acc = Madd(acc, header.q[kG + 1][l], p.y);
            gg[l] = Madd(acc, header.q[kG + 2][l], p.z);
        }
        q = Recip(gg.w);
        clip.x = Mul(gg.x, header.q[kGuard].x);
        clip.y = Mul(gg.y, Add(0.0, header.q[kGuard + 1].x));
        clip.z = header.q[kC + 3].z;
        clip.w = Mul(gg.w, 1.0);
    } else {
        for (int l = 0; l < 4; ++l) {
            if (l == 2) continue;
            float acc = Mul(header.q[kC + 3][l], 1.0);
            acc = Madd(acc, header.q[kC][l], p.x);
            acc = Madd(acc, header.q[kC + 1][l], p.y);
            clip[l] = Madd(acc, header.q[kC + 2][l], p.z);
        }
        clip.z = Madd(Mul(header.q[kC + 3].z, 1.0), header.q[kC + 2].z, p.z);
        q = Recip(clip.w);
    }
    const vec4 O = header.q[kO];
    const uint x = uint(Ftoi(Madd(Mul(O.x, 1.0), clip.x, q) * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(Madd(Mul(O.y, 1.0), clip.y, q) * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(Madd(Mul(O.z, 1.0), clip.z, q) * 16.0)) >> 4) & 0xFFFFFFu;
    vec3 stq;
    uvec4 rgba;
    if ((mode & kEnvStream) != 0u) {
        // the reflection about the normal, through the sphere map's rows
        const vec4 R0 = header.q[kR], R1 = header.q[kR + 1], R2 = header.q[kR + 2];
        const float dx = Sub(p.x, R0.w), dy = Sub(p.y, R1.w), dz = Sub(p.z, R2.w);
        const float l2 = g == 0u ? Madd(Madd(Mul(dy, dy), dx, dx), dz, dz) : Madd(Madd(Mul(dx, dx), dz, dz), dy, dy);
        const float dn = Madd(Madd(Mul(dx, n.x), dy, n.y), dz, n.z);
        const float k = Add(dn, dn);
        const float rx = Sub(dx, Mul(k, n.x)), ry = Sub(dy, Mul(k, n.y)), rz = Sub(dz, Mul(k, n.z));
        const float ex = Madd(Madd(Mul(rx, R0.x), ry, R1.x), rz, R2.x);
        const float ey = Madd(Madd(Mul(rx, R0.y), ry, R1.y), rz, R2.y);
        const float ez = Madd(Madd(Madd(Add(SqrtTZ(l2), 0.0), rx, R0.z), ry, R1.z), rz, R2.z);
        const float e2 = Madd(Madd(Mul(ex, ex), ey, ey), ez, ez);
        const float s = Mul(Rsqrt(e2), 0.5);
        const float u = Madd(0.5, ex, s), v = Madd(0.5, ey, s);
        stq = vec3(Mul(Add(0.0, u), q), Mul(Add(0.0, v), q), Mul(1.0, q));
        rgba = floatBitsToUint(header.q[kEnv]);
    } else {
        stq = vec3(Mul(Itof12(st.x), q), Mul(Itof12(st.y), q), Mul(1.0, q));
        // the lighting: a quadratic form of the normal; alpha 2|d| - d^2 of the Fresnel direction's
        const vec4 M0 = header.q[kM], M1 = header.q[kM + 1], M2 = header.q[kM + 2], M3 = header.q[kM + 3];
        vec3 r;
        for (int l = 0; l < 3; ++l) r[l] = Madd(Madd(Madd(Add(0.0, M3[l]), n.x, M0[l]), n.z, M2[l]), n.y, M1[l]);
        const float d = Madd(Madd(Mul(n.x, L.x), n.y, L.y), n.z, L.z);
        const float a = abs(d);
        const float f = Msub(Add(a, a), a, a);
        float e = Add(0.0, M3.w);
        e = Madd(e, n.x, M0.w);
        e = Madd(e, n.z, M2.w);
        e = Madd(e, n.y, M1.w);
        e = Madd(e, r.x, n.x);
        e = Madd(e, r.y, n.y);
        e = Madd(e, r.z, n.z);
        const vec4 C0 = header.q[kC0], C1 = header.q[kC1];
        for (int l = 0; l < 3; ++l) rgba[l] = uint(Ftoi(Madd(Add(C0[l], 0.0), Sub(C1[l], C0[l]), e)));
        rgba.w = uint(Ftoi(Msub(Add(VuMax(0.0, C1.w), 0.0), Sub(C1.w, C0.w), f)));
    }
    const vec2 position = (vec2(float(x), float(y)) - header.q[kMisc].xy) / 16.0;
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(Unorm8(rgba.x), Unorm8(rgba.y), Unorm8(rgba.z), Unorm8(rgba.w));
    coordinate = stq;
    fog_out = Unorm8((q0.w >> 4) & 0xFFu);
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}
