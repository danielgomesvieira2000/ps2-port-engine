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
const uint kEnvStream = 1u, kGuardPlace = 2u, kSkin = 4u;
const int kHeaderQuads = 23;        // (rw_native.h's kRwHeaderQuads: the skinning's palette rows after it)

vec4 PaletteRow(uint address) {
    const uint first = floatBitsToUint(header.q[0].w);
    return header.q[kHeaderQuads + int((address - first) & 0x3FFu)];
}

float Madd(float acc, float a, float b) { return Add(acc, Mul(a, b)); }
float Msub(float acc, float a, float b) { return Sub(acc, Mul(a, b)); }

// The GS's vertex from its packet's words.
void Emit(uint x, uint y, uint z, vec3 stq, uvec4 rgba, uint w) {
    const vec2 position = (vec2(float(x), float(y)) - header.q[kMisc].xy) / 16.0;
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(Unorm8(rgba.x), Unorm8(rgba.y), Unorm8(rgba.z), Unorm8(rgba.w));
    coordinate = stq;
    fog_out = Unorm8((w >> 4) & 0xFFu);
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}

// The instanced strips (VU1 D4B61373; rw_native.h's InstancedVertex).
const uint kInstanced = 1u << 8, kPathB = 1u, kPathC = 2u;
const int kIM = 1, kIS = 5, kIOff = 6, kIColour = 7, kIAB = 8;
void Instanced(uint mode) {
    const bool path_b = (mode & kPathB) != 0u;
    const uint j = uint(gl_VertexIndex);
    const uvec4 q0 = mesh.quad[3u * j];
    const vec2 uv = uintBitsToFloat(mesh.quad[3u * j + 1u].xy);
    const vec3 p = uintBitsToFloat(q0.xyz);
    vec4 c;
    for (int l = 0; l < 4; ++l) {
        float acc = Mul(header.q[kIM + 3][l], 1.0);
        if (path_b || l != 2) {
            acc = Madd(acc, header.q[kIM][l], p.x);
            acc = Madd(acc, header.q[kIM + 1][l], p.y);
        }
        c[l] = Madd(acc, header.q[kIM + 2][l], p.z);
    }
    const float q = Recip(c.w);
    const vec4 S = header.q[kIS], off = header.q[kIOff];
    uint x, y, z;
    if (path_b) {
        x = uint(Ftoi(Add(Mul(Mul(c.x, 1024.0), q), off.x) * 16.0)) & 0xFFFFu;
        y = uint(Ftoi(Add(Mul(Mul(c.y, 1024.0), q), off.y) * 16.0)) & 0xFFFFu;
        z = (uint(Ftoi(Add(Mul(S.z, q), off.z) * 16.0)) >> 4) & 0xFFFFFFu;
    } else {
        x = uint(Ftoi(Add(Mul(c.x, q), off.x) * 16.0)) & 0xFFFFu;
        y = uint(Ftoi(Add(Mul(c.y, q), off.y) * 16.0)) & 0xFFFFu;
        z = (uint(Ftoi(Add(Mul(c.z, q), off.z) * 16.0)) >> 4) & 0xFFFFFFu;
    }
    const vec3 stq = vec3(Mul(uv.x, q), Mul(uv.y, q), q);
    const vec4 colour = header.q[kIColour];
    uvec4 rgba;
    if ((mode & kPathC) != 0u) {
        const uint g = j / 4u, lane = j % 4u;
        const float nx = uintBitsToFloat(mesh.quad[3u * (4u * g) + 2u][lane]);
        const float ny = uintBitsToFloat(mesh.quad[3u * (4u * g + 1u) + 2u][lane]);
        const float nz = uintBitsToFloat(mesh.quad[3u * (4u * g + 2u) + 2u][lane]);
        const float t = Madd(Madd(Mul(nx, colour.x), ny, colour.y), nz, colour.z);
        const vec4 ab = header.q[kIAB];
        rgba.xyz = uvec3(floatBitsToUint(Madd(ab.x, ab.y, t)));
        rgba.w = floatBitsToUint(Madd(colour.w, 0.0, t));
    } else {
        rgba = floatBitsToUint(colour);
    }
    Emit(x, y, z, stq, rgba, q0.w);
}

// The world's strips (VU1 2678EB26, 1332C99F, 01CE30D5; rw_native.h's World*): the index a corner of strip
// triangle t (3t + c), its triangle decided by every corner alike - one the microcode leaves undrawn (ADC) a point.
const uint kWorld = 1u << 9, kWCull = 1u, kWGuard = 2u, kWLit = 4u, kWAlphaZero = 8u, kWFog = 16u;
const int kWT = 1, kWO = 5, kWScale = 6, kWOff = 7, kWG0 = 8, kWG1 = 9, kWL = 10, kWE = 11, kWMin = 12, kWK = 13, kWBase = 14;

vec4 WorldClip(vec3 p) {
    vec4 c;
    for (int l = 0; l < 4; ++l) {
        if (l == 2) continue;
        float acc = Mul(header.q[kWT + 3][l], 1.0);
        acc = Madd(acc, header.q[kWT][l], p.x);
        acc = Madd(acc, header.q[kWT + 1][l], p.y);
        c[l] = Madd(acc, header.q[kWT + 2][l], p.z);
    }
    c.z = header.q[kWT + 3].z;
    return c;
}

vec2 WorldScreen(uint v) {
    const vec4 c = WorldClip(uintBitsToFloat(mesh.quad[3u * v].xyz));
    const float q = Recip(c.w);
    const vec4 O = header.q[kWO];
    return vec2(Madd(O.x, c.x, q), Madd(O.y, c.y, q));
}

uint Judge(vec4 g) {
    const float aw = abs(g.w);
    uint j = 0u;
    if (g.x > aw) j |= 1u;
    if (g.x < -aw) j |= 2u;
    if (g.y > aw) j |= 4u;
    if (g.y < -aw) j |= 8u;
    if (g.z > aw) j |= 16u;
    if (g.z < -aw) j |= 32u;
    return j;
}

uint WorldJudge(uint v) {
    const vec4 c = WorldClip(uintBitsToFloat(mesh.quad[3u * v].xyz));
    const vec4 g0 = header.q[kWG0], g1 = header.q[kWG1];
    return Judge(vec4(Mul(c.x, g0.y), Mul(c.y, g1.y), Madd(g0.z, g1.z, c.w), c.w));
}

bool WorldDrawn(uint mode, uint k) {
    if ((mesh.quad[3u * k].w & 0x8000u) != 0u) return false;
    if ((mode & kWCull) != 0u) {
        const vec2 a = WorldScreen(k - 2u), b = WorldScreen(k - 1u), c = WorldScreen(k);
        vec2 u, w;
        if ((k & 1u) == 0u) {
            u = vec2(Sub(b.x, a.x), Sub(b.y, a.y));
            w = vec2(Sub(b.x, c.x), Sub(b.y, c.y));
        } else {
            u = vec2(Sub(c.x, b.x), Sub(c.y, b.y));
            w = vec2(Sub(a.x, b.x), Sub(a.y, b.y));
        }
        const float cz = Sub(Mul(u.x, w.y), Mul(w.x, u.y));
        const uint base = floatBitsToUint(header.q[kWBase].x);
        if (((base + ((floatBitsToUint(cz) >> 31) != 0u ? 0x20u : 0u)) & 0x8000u) != 0u) return false;
    }
    if ((mode & kWGuard) != 0u)
        for (uint j = k - 2u; j <= k; ++j)
            if (WorldJudge(j) != 0u) return false;
    return true;
}

void World(uint mode) {
    const uint index = uint(gl_VertexIndex);
    const uint t = index / 3u, v = t + index % 3u;
    if (!WorldDrawn(mode, t + 2u)) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        shade = vec4(0.0);
        coordinate = vec3(0.0);
        fog_out = 0.0;
        range_out = vec4(-1e9, -1e9, 1e9, 1e9);
        return;
    }
    const uvec4 q0 = mesh.quad[3u * v], q1 = mesh.quad[3u * v + 1u];
    const vec3 p = uintBitsToFloat(q0.xyz);
    const vec4 c = WorldClip(p);
    const float q = Recip(c.w);
    const vec4 O = header.q[kWO];
    const uint x = uint(Ftoi(Madd(O.x, c.x, q) * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(Madd(O.y, c.y, q) * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(Madd(O.z, c.z, q) * 16.0)) >> 4) & 0xFFFFFFu;
    const vec3 stq = vec3(Mul(Itof12(q1.x), q), Mul(Itof12(q1.y), q), q);
    uvec4 rgba = mesh.quad[3u * v + 2u];
    if ((mode & kWAlphaZero) != 0u) {
        rgba.w = 0u;
    } else if ((mode & kWLit) != 0u) {
        // 1332's: the reflected light's cosine to the eye, raised to the power e by the float's bits, times the mask
        const uint nb = floatBitsToUint(header.q[kMisc].w) + 4u * (v / 4u);
        const uint lane = v % 4u;
        const vec4 L = header.q[kWL], E = header.q[kWE], K = header.q[kWK];
        const float dx = Sub(p.x, E.x), dy = Sub(p.y, E.y), dz = Sub(p.z, E.z);
        const float d2 = Madd(Madd(Mul(dy, dy), dx, dx), dz, dz);
        const float inv = Rsqrt(d2);
        const float nx = ItofTZ(int(mesh.quad[nb][lane])), ny = ItofTZ(int(mesh.quad[nb + 1u][lane]));
        const float nz = ItofTZ(int(mesh.quad[nb + 2u][lane])), m = ItofTZ(int(mesh.quad[nb + 3u][lane]));
        const float nl = Madd(Madd(Mul(nx, K.x), ny, K.y), nz, K.z);
        const float rx = Madd(Sub(0.0, L.x), nl, nx), ry = Madd(Sub(0.0, L.y), nl, ny), rz = Madd(Sub(0.0, L.z), nl, nz);
        const float rd = Madd(Madd(Mul(rx, dx), ry, dy), rz, dz);
        const float cs = VuMax(Mul(rd, inv), K.w);
        const float pw = Madd(Add(0.0, header.q[kWBase].y), ItofTZ(int(floatBitsToUint(cs))), L.w);
        const float s1 = VuMax(Mul(m, uintBitsToFloat(uint(Ftoi(pw)))), header.q[kWMin].x);
        rgba.w = uint(Ftoi(s1));
    }
    // path C's fog (XYZF2's F): -B F + W F, at least 0, at most 255 (FGE: the GS's PRMODE may say it)
    uint fw = q0.w;
    if ((mode & kWFog) != 0u) {
        const float F = header.q[kWScale].w, Bd = header.q[kWOff].w;
        float f = Add(Sub(0.0, Mul(Bd, F)), Mul(c.w, F));
        f = VuMin(VuMax(f, 0.0), 255.0);
        fw = uint(Ftoi(f * 16.0));
    }
    Emit(x, y, z, stq, rgba, fw);
}

void Cars(uint mode) {
    const uint j = uint(gl_VertexIndex);
    const uvec4 q0 = mesh.quad[3u * j];
    const uvec4 q2 = mesh.quad[3u * j + 2u];
    vec3 p = uintBitsToFloat(q0.xyz);
    const uint g = j / 4u, lane = j % 4u;
    const uvec4 q1 = mesh.quad[3u * (4u * g + ((lane & 1u) != 0u ? 2u : 0u)) + 1u];
    const uvec2 st = (lane & 2u) != 0u ? q1.zw : q1.xy;
    const vec4 L = header.q[kL];
    vec3 n = vec3(Sub(uintBitsToFloat(q2.x), L.w), Sub(uintBitsToFloat(q2.y), L.w), Sub(uintBitsToFloat(q2.z), L.w));
    if ((mode & kSkin) != 0u) {
        // two bones a vertex (rw_native.h's CarsSkin): weights /65535, the identity by what is left
        const uint wa = mesh.quad[3u * (4u * g + 1u) + 1u][lane], wb = mesh.quad[3u * (4u * g + 3u) + 1u][lane];
        const float q = Recip(65535.0);
        const float a = Mul(ItofTZ(int(wa)), q), b = Mul(ItofTZ(int(wb)), q);
        const float w0 = Sub(Sub(Mul(1.0, 1.0), Mul(a, 1.0)), Mul(b, 1.0));
        const uint ba = (974u + (wa & 0xFFFFu)) & 0x3FFu, bb = (974u + (wb & 0xFFFFu)) & 0x3FFu;
        vec3 R[4];
        for (uint r = 0u; r < 4u; ++r) {
            const vec4 A = PaletteRow(ba + r), Bm = PaletteRow(bb + r);
            for (int l = 0; l < 3; ++l) {
                if (r < 3u) R[r][l] = Madd(Madd(Mul(A[l], a), uint(l) == r ? 1.0 : 0.0, w0), Bm[l], b);
                else R[r][l] = Madd(Mul(A[l], a), Bm[l], b);
            }
        }
        const vec3 P = p, N = n;
        for (int l = 0; l < 3; ++l) {
            p[l] = Madd(Madd(Madd(Mul(R[0][l], P.x), R[1][l], P.y), R[2][l], P.z), R[3][l], 1.0);
            n[l] = Madd(Madd(Mul(R[0][l], N.x), R[1][l], N.y), R[2][l], N.z);
        }
    }
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
    Emit(x, y, z, stq, rgba, q0.w);
}

void main() {
    const uint mode = floatBitsToUint(header.q[kMisc].z);
    if ((mode & kWorld) != 0u) World(mode);
    else if ((mode & kInstanced) != 0u) Instanced(mode);
    else Cars(mode);
}
