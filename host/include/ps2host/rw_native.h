#pragma once
/*
 * RenderWare's VU1 programs on the CPU (docs/NATIVE_RENDERER.md, level B): what
 * gs_mesh_rw.vert computes for a vertex, as the emulated VU computes it - the
 * caller under the VU's rounding (toward zero: VuRounding, or
 * _MM_ROUND_TOWARD_ZERO), each operation one float operation (a MADD two: the
 * product, then the sum), DIV and SQRT the float's own under that rounding, the
 * reciprocal square roots through a double (as the emulated RSQRT and ERSQRT).
 * The shader's twin: a draw the renderer declines is drawn from it, and the
 * port's check compares it with the microcode's packets.
 *
 * Burnout 3's car program (VU1 image 7E86891B; games/burnout3/docs/NATIVE_RENDERER.md):
 * a vertex three quadwords - q0 the place (w 0 or 0x8000: the strip's ADC),
 * q1 texture coordinates packed two vertices a quadword (vertex 4g+lane's in
 * q1 of 4g or 4g+2, 12-bit fixed point), q2 the normal (0x47800000 + a byte,
 * less 970.w). The header (quadwords, kRw*): misc (OFX, OFY, the mode, -), the
 * transform vf28-vf31 and vf02, data memory 966-973 (the lighting's quadratic
 * form, the Fresnel direction, the sphere map's rows and the eye), 1010-1012
 * (the colours, the env pass's RGBAQ), 1022-1023 (the guard band) and the guard
 * band's matrix (G, the T4 path's).
 */

#include <cmath>
#include <cstdint>
#include <cstring>

#include "ps2rt/vu_math.h"

namespace ps2host {
namespace rw_native {

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

using u32 = std::uint32_t;

// The header's quadwords
constexpr int kRwMisc = 0;      // x OFX, y OFY (sixteenths: the bridge's), z the mode (bits), w -
constexpr int kRwC = 1;         // vf28-vf31: the transform (columns, scaled)
constexpr int kRwO = 5;         // vf02: the screen's offset (1019)
constexpr int kRwM = 6;         // 966-969: the lighting's quadratic form
constexpr int kRwL = 10;        // 970: the Fresnel direction; w the normal's bias
constexpr int kRwR = 11;        // 971-973: the sphere map's rows; their w the eye
constexpr int kRwC0 = 14;       // 1010: the colour's base
constexpr int kRwC1 = 15;       // 1011: its top
constexpr int kRwEnv = 16;      // 1012: the env pass's RGBAQ (words)
constexpr int kRwGuard = 17;    // 1022-1023: the guard band
constexpr int kRwG = 19;        // the guard band's matrix G0-G3 (T4)
constexpr int kRwHeaderQuads = 23;

// The mode (misc.z)
constexpr u32 kRwEnvStream = 1;     // the env pass: the sphere map's coordinates, 1012's colour
constexpr u32 kRwGuardPlace = 2;    // T4: the place through G, then the guard band's scale

inline float F(u32 b) { float f; std::memcpy(&f, &b, 4); return f; }
inline u32 B(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
inline float Mul(float a, float b) { return a * b; }
inline float Add(float a, float b) { return a + b; }
inline float Sub(float a, float b) { return a - b; }
inline float Madd(float acc, float a, float b) { return acc + a * b; }
inline float Msub(float acc, float a, float b) { return acc - a * b; }
inline float Recip(float x) { return 1.0f / x; }
inline float Sqrt(float x) { return std::sqrt(std::fabs(x)); }
inline float Rsqrt(float x) {
    const double root = std::sqrt(std::fabs(double(x)));
    return root == 0.0 ? 3.4028234663852886e38f : float(1.0 / root);
}
inline u32 Ftoi(float y) {
    if (y >= 2147483648.0f) return 0x7FFFFFFFu;
    if (y < -2147483648.0f) return 0x80000000u;
    return u32(std::int32_t(y));
}
inline float Itof12(u32 i) { return float(std::int32_t(i)) / 4096.0f; }
// MAX's order: the bits as sign and magnitude
inline float Max(float a, float b) {
    auto order = [](float f) { const u32 b = B(f); const std::int32_t m = std::int32_t(b & 0x7FFFFFFFu); return (b >> 31) ? -m : m; };
    return order(a) < order(b) ? b : a;
}

struct GsVertex {
    u32 stq[3];         // ST's S, T and Q (floats' bits)
    u32 rgba[4];        // RGBAQ's words (the GS their low bytes)
    u32 xyz[3];         // XYZF2's X, Y, Z words (FTOI4)
    u32 fog = 0;        // XYZF2's F (the world's path C; else 0)
};

inline const float* Q(const float* h, int quad) { return h + 4 * quad; }

/* The guard band's judgement of a place (T2: from `clip`; T4: `g`, the place through G), the CLIP instruction's
   six bits. */
inline u32 Judge(float x, float y, float z, float w) {
    const float aw = std::fabs(w);
    u32 j = 0;
    if (x > aw) j |= 1;
    if (x < -aw) j |= 2;
    if (y > aw) j |= 4;
    if (y < -aw) j |= 8;
    if (z > aw) j |= 16;
    if (z < -aw) j |= 32;
    return j;
}

/* The place through the transform (T1-T3: x, y, w; z c3.z). */
inline void ClipOf(const float* h, const float* p, float* clip) {
    const float* c0 = Q(h, kRwC), *c1 = Q(h, kRwC + 1), *c2 = Q(h, kRwC + 2), *c3 = Q(h, kRwC + 3);
    for (int l = 0; l < 4; ++l) {
        if (l == 2) continue;
        float acc = Mul(c3[l], 1.0f);
        acc = Madd(acc, c0[l], p[0]);
        acc = Madd(acc, c1[l], p[1]);
        clip[l] = Madd(acc, c2[l], p[2]);
    }
    clip[2] = Madd(Mul(c3[2], 1.0f), c2[2], p[2]);
}

/* T4's: the place through G (all four lanes). */
inline void GuardOf(const float* h, const float* p, float* g) {
    const float* g0 = Q(h, kRwG), *g1 = Q(h, kRwG + 1), *g2 = Q(h, kRwG + 2), *g3 = Q(h, kRwG + 3);
    for (int l = 0; l < 4; ++l) {
        float acc = Mul(g3[l], 1.0f);
        acc = Madd(acc, g0[l], p[0]);
        acc = Madd(acc, g1[l], p[1]);
        g[l] = Madd(acc, g2[l], p[2]);
    }
}

/* T2's judgement vector from the plain transform's place. */
inline u32 JudgeT2(const float* h, const float* clip) {
    const float* gb0 = Q(h, kRwGuard), *gb1 = Q(h, kRwGuard + 1);
    const float gx = Mul(clip[0], gb0[1]), gy = Mul(clip[1], gb1[1]);
    const float gz = Madd(Add(gb0[2], 0.0f), gb1[2], clip[3]);
    const float gw = Mul(clip[3], 1.0f);
    return Judge(gx, gy, gz, gw);
}

/* Vertex j of a car batch (the mesh's quadwords as words), for the header's stream. */
inline void CarsVertex(const float* h, const u32* mesh, u32 j, GsVertex* o) {
    u32 mode;
    std::memcpy(&mode, Q(h, kRwMisc) + 2, 4);
    const u32* q0 = mesh + 12 * j;
    const u32* q2 = q0 + 8;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    const u32 g = j / 4, lane = j % 4;
    // (s, t): lane 0 q1(4g).xy, 1 q1(4g+2).xy, 2 q1(4g).zw, 3 q1(4g+2).zw
    const u32* st = mesh + 12 * (4 * g + ((lane & 1) ? 2 : 0)) + 4 + ((lane & 2) ? 2 : 0);
    const float* L = Q(h, kRwL);
    const float X = Sub(F(q2[0]), L[3]), Y = Sub(F(q2[1]), L[3]), Z = Sub(F(q2[2]), L[3]);
    // the place, and Q
    float clip[4], q;
    if (mode & kRwGuardPlace) {
        float gg[4];
        GuardOf(h, p, gg);
        q = Recip(gg[3]);
        const float* gb0 = Q(h, kRwGuard), *gb1 = Q(h, kRwGuard + 1);
        clip[0] = Mul(gg[0], gb0[0]);
        clip[1] = Mul(gg[1], Add(0.0f, gb1[0]));
        clip[2] = Q(h, kRwC + 3)[2];
        clip[3] = Mul(gg[3], 1.0f);
    } else {
        ClipOf(h, p, clip);
        q = Recip(clip[3]);
    }
    const float* O = Q(h, kRwO);
    for (int l = 0; l < 3; ++l) o->xyz[l] = Ftoi(Madd(Mul(O[l], 1.0f), clip[l], q) * 16.0f);
    if (mode & kRwEnvStream) {
        // the reflection about the normal, through the sphere map's rows
        const float* R0 = Q(h, kRwR), *R1 = Q(h, kRwR + 1), *R2 = Q(h, kRwR + 2);
        const float dx = Sub(p[0], R0[3]), dy = Sub(p[1], R1[3]), dz = Sub(p[2], R2[3]);
        const float l2 = g == 0 ? Madd(Madd(Mul(dy, dy), dx, dx), dz, dz) : Madd(Madd(Mul(dx, dx), dz, dz), dy, dy);
        const float dn = Madd(Madd(Mul(dx, X), dy, Y), dz, Z);
        const float k = Add(dn, dn);
        const float rx = Sub(dx, Mul(k, X)), ry = Sub(dy, Mul(k, Y)), rz = Sub(dz, Mul(k, Z));
        const float ex = Madd(Madd(Mul(rx, R0[0]), ry, R1[0]), rz, R2[0]);
        const float ey = Madd(Madd(Mul(rx, R0[1]), ry, R1[1]), rz, R2[1]);
        const float ez = Madd(Madd(Madd(Add(Sqrt(l2), 0.0f), rx, R0[2]), ry, R1[2]), rz, R2[2]);
        const float e2 = Madd(Madd(Mul(ex, ex), ey, ey), ez, ez);
        const float s = Mul(Rsqrt(e2), 0.5f);
        const float u = Madd(0.5f, ex, s), v = Madd(0.5f, ey, s);
        o->stq[0] = B(Mul(Add(0.0f, u), q)), o->stq[1] = B(Mul(Add(0.0f, v), q)), o->stq[2] = B(Mul(1.0f, q));
        std::memcpy(o->rgba, Q(h, kRwEnv), 16);
        return;
    }
    o->stq[0] = B(Mul(Itof12(st[0]), q)), o->stq[1] = B(Mul(Itof12(st[1]), q)), o->stq[2] = B(Mul(1.0f, q));
    // the lighting: a quadratic form of the normal; alpha 2|d| - d^2 of the Fresnel direction's
    const float* M0 = Q(h, kRwM), *M1 = Q(h, kRwM + 1), *M2 = Q(h, kRwM + 2), *M3 = Q(h, kRwM + 3);
    float r[3];
    for (int l = 0; l < 3; ++l) r[l] = Madd(Madd(Madd(Add(0.0f, M3[l]), X, M0[l]), Z, M2[l]), Y, M1[l]);
    const float d = Madd(Madd(Mul(X, L[0]), Y, L[1]), Z, L[2]);
    const float a = std::fabs(d);
    const float f = Msub(Add(a, a), a, a);
    float e = Add(0.0f, M3[3]);
    e = Madd(e, X, M0[3]);
    e = Madd(e, Z, M2[3]);
    e = Madd(e, Y, M1[3]);
    e = Madd(e, r[0], X);
    e = Madd(e, r[1], Y);
    e = Madd(e, r[2], Z);
    const float* C0 = Q(h, kRwC0), *C1 = Q(h, kRwC1);
    for (int l = 0; l < 3; ++l) o->rgba[l] = Ftoi(Madd(Add(C0[l], 0.0f), Sub(C1[l], C0[l]), e));
    o->rgba[3] = Ftoi(Msub(Add(Max(0.0f, C1[3]), 0.0f), Sub(C1[3], C0[3]), f));
}

// ---------------------------------------------------------------- the instanced strips (VU1 D4B61373, 34C3CF67)
//
// A batch of strip vertices (three quadwords: q0 the place, w its flags - bit 15 the strip's ADC; q1 U, V; q2 the
// normals, SoA a group of four - nx in q2 of 4g, ny of 4g+1, nz of 4g+2) drawn once an instance, each with its own
// matrix (the camera's rows by the instance's: InstanceMatrix) and colour (or light: path C). The header
// (quadwords): misc (OFX, OFY, the mode: kRwInstanced | path bits, -), M0-M3, 1018 (the screen's scale, Zs),
// 1019 (its offset), the instance's 938 (RGBA words, or path C's light and alpha), its A and B (934.z, 935.z).

constexpr u32 kRwInstanced = 1u << 8;   // the mode's program: the instanced strips
constexpr u32 kRwPathB = 1;             // the guard band's test (937.z): the place's x, y scaled, z the depth row
constexpr u32 kRwPathC = 2;             // the lighting (932.y): the colour from the normal
constexpr int kRwIM = 1, kRwIS = 5, kRwIOff = 6, kRwIColour = 7, kRwIAB = 8, kRwIHeaderQuads = 9;

/* The instance's matrix (into header quadwords 1-4): V the camera's rows (1014-1017), R the instance's
   (934+5i-937+5i: lanes x, y, w their coefficients), S 1018, `far`/`near` 1022.w/1023.w (path B). */
inline void InstanceMatrix(const float (*V)[4], const float (*R)[4], const float* S, float far, float near,
                           bool path_b, float (*M)[4]) {
    const int lanes[3] = {0, 1, 3};
    for (int n = 0; n < 4; ++n) {
        for (int l : lanes) {
            float acc = Mul(V[0][l], R[n][0]);
            acc = Madd(acc, V[1][l], R[n][1]);
            acc = Madd(acc, V[2][l], R[n][3]);
            if (n == 3) acc = Madd(acc, V[3][l], 1.0f);
            M[n][l] = acc;
        }
        M[n][2] = n == 3 ? 1.0f : 0.0f;
    }
    for (int n = 0; n < 4; ++n) {
        const float h = Mul(M[n][3], 0.5f);
        M[n][0] = Sub(M[n][0], h);
        M[n][1] = Sub(M[n][1], h);
    }
    if (!path_b) {
        for (int n = 0; n < 4; ++n) M[n][0] = Mul(M[n][0], S[0]), M[n][1] = Mul(M[n][1], S[1]);
        M[3][2] = Mul(1.0f, S[2]);
        return;
    }
    const float sx = Mul(S[0], 1.0f / 1024.0f), sy = Mul(S[1], 1.0f / 1024.0f);
    const float q = Recip(Add(far, near));
    const float za = Mul(Sub(far, near), q);
    const float zb = Mul(Mul(Mul(far, near), -2.0f), q);
    for (int n = 0; n < 4; ++n) M[n][0] = Mul(M[n][0], sx);
    for (int n = 0; n < 4; ++n) M[n][1] = Mul(M[n][1], sy);
    for (int n = 0; n < 4; ++n) M[n][2] = Mul(za, M[n][3]);
    M[3][2] = Add(M[3][2], zb);
}

/* The place through the instance's matrix (path B: all four lanes; else x, y, w - z Zs). */
inline void InstancedClip(const float* h, const float* p, bool path_b, float* c) {
    const float* M0 = Q(h, kRwIM), *M1 = Q(h, kRwIM + 1), *M2 = Q(h, kRwIM + 2), *M3 = Q(h, kRwIM + 3);
    for (int l = 0; l < 4; ++l) {
        float acc = Mul(M3[l], 1.0f);
        if (path_b || l != 2) {
            acc = Madd(acc, M0[l], p[0]);
            acc = Madd(acc, M1[l], p[1]);
        }
        c[l] = Madd(acc, M2[l], p[2]);
    }
}

/* The same with the VU's own arithmetic (ps2rt/vu_math.h: operands cleaned, results saturated - for places that
   are not floats: memory past a batch), the caller under the VU's rounding; `p` and `c` bits. */
inline void InstancedClipVu(const float* h, const u32* p, bool path_b, u32* c) {
    using namespace lifted;
    const float* M0 = Q(h, kRwIM), *M1 = Q(h, kRwIM + 1), *M2 = Q(h, kRwIM + 2), *M3 = Q(h, kRwIM + 3);
    for (int l = 0; l < 4; ++l) {
        u32 acc = BinaryOf<kMul>(B(M3[l]), kOne, 0);
        if (path_b || l != 2) {
            acc = BinaryOf<kMadd>(B(M0[l]), p[0], acc);
            acc = BinaryOf<kMadd>(B(M1[l]), p[1], acc);
        }
        c[l] = BinaryOf<kMadd>(B(M2[l]), p[2], acc);
    }
}

/* CLIP's judgement as the VU makes it (vu_core.h's ClipJudge: clean magnitudes). */
inline u32 JudgeVu(const u32* c) {
    auto magnitude = [](u32 bits) {
        const u32 e = bits & 0x7F800000u;
        return e == 0 ? 0u : e == 0x7F800000u ? 0x7F7FFFFFu : bits & 0x7FFFFFFFu;
    };
    const u32 w = magnitude(c[3]);
    u32 j = 0;
    for (int lane = 0; lane < 3; ++lane)
        if (magnitude(c[lane]) > w) j |= (c[lane] >> 31 ? 2u : 1u) << (2 * lane);
    return j;
}

/* Vertex j of an instanced batch, for the header's instance. */
inline void InstancedVertex(const float* h, const u32* mesh, u32 j, GsVertex* o) {
    u32 mode;
    std::memcpy(&mode, Q(h, kRwMisc) + 2, 4);
    const bool path_b = (mode & kRwPathB) != 0;
    const u32* q0 = mesh + 12 * j;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    float c[4];
    InstancedClip(h, p, path_b, c);
    const float q = Recip(c[3]);
    const float* S = Q(h, kRwIS), *off = Q(h, kRwIOff);
    if (path_b) {
        o->xyz[0] = Ftoi(Add(Mul(Mul(c[0], 1024.0f), q), off[0]) * 16.0f);
        o->xyz[1] = Ftoi(Add(Mul(Mul(c[1], 1024.0f), q), off[1]) * 16.0f);
        o->xyz[2] = Ftoi(Add(Mul(S[2], q), off[2]) * 16.0f);
    } else {
        for (int l = 0; l < 3; ++l) o->xyz[l] = Ftoi(Add(Mul(c[l], q), off[l]) * 16.0f);
    }
    o->stq[0] = B(Mul(F(q0[4]), q)), o->stq[1] = B(Mul(F(q0[5]), q)), o->stq[2] = B(q);
    const float* colour = Q(h, kRwIColour);
    if (mode & kRwPathC) {
        const u32 g = j / 4, lane = j % 4;
        const float nx = F(mesh[12 * (4 * g) + 8 + lane]), ny = F(mesh[12 * (4 * g + 1) + 8 + lane]);
        const float nz = F(mesh[12 * (4 * g + 2) + 8 + lane]);
        const float t = Madd(Madd(Mul(nx, colour[0]), ny, colour[1]), nz, colour[2]);
        const float* ab = Q(h, kRwIAB);
        o->rgba[0] = o->rgba[1] = o->rgba[2] = B(Madd(ab[0], ab[1], t));
        o->rgba[3] = B(Madd(colour[3], 0.0f, t));
    } else {
        std::memcpy(o->rgba, colour, 16);
    }
}

// ---------------------------------------------------------------- the world's strips (VU1 2678EB26, 1332C99F, 01CE30D5)
//
// RenderWare's strip pipeline (games/burnout3/build/native/spec_2678EB26.md and its family): a vertex three
// quadwords - q0 the place (w its flag: bit 15 the strip's ADC), q1 U, V (12-bit fixed point), q2 RGBA (words) -
// through the transform vf28-vf31 (x, y, w; z the depth scale vf31.z), the screen's offset vf02. Paths A and B:
// a triangle (k-2, k-1, k) of the strip drawn unless its last vertex's flag, its screen winding (kRwWCull: the
// cross product's sign, by the strip's parity; 2678's and 1332's) or (kRwWGuard: path B) any of its three
// vertices out of the guard band (1022-1023) says ADC. 1332's lighting (kRwWLit) puts a specular term in each
// vertex's alpha from its normal (a block of them past the batch: misc.w the mesh's quadword it starts at).
//
// The draw's indices are its triangles' corners, 3t + c (vertex t + c of the strip's triangle t): each corner
// decides its triangle (WorldDrawn) - one not drawn collapses to a point.

constexpr u32 kRwWorld = 1u << 9;       // the mode's program: the world's strips
constexpr u32 kRwWCull = 1;             // the screen winding's ADC (base: the header's kRwWBase.x)
constexpr u32 kRwWGuard = 2;            // path B's guard band
constexpr u32 kRwWLit = 4;              // 1332's specular alpha
constexpr u32 kRwWAlphaZero = 8;        // 1332 with 1013.z: alpha 0
constexpr u32 kRwWFog = 16;             // path C's fog in XYZF2's F
constexpr int kRwWT = 1, kRwWO = 5, kRwWScale = 6, kRwWOff = 7, kRwWG0 = 8, kRwWG1 = 9, kRwWL = 10, kRwWE = 11,
              kRwWMin = 12, kRwWK = 13, kRwWBase = 14, kRwWHeaderQuads = 15;

inline u32 Mode(const float* h) {
    u32 mode;
    std::memcpy(&mode, Q(h, kRwMisc) + 2, 4);
    return mode;
}

// The place through the transform: x, y, w (z: vf31.z)
inline void WorldClip(const float* h, const float* p, float* c) {
    const float* t0 = Q(h, kRwWT), *t1 = Q(h, kRwWT + 1), *t2 = Q(h, kRwWT + 2), *t3 = Q(h, kRwWT + 3);
    for (int l : {0, 1, 3}) {
        float acc = Mul(t3[l], 1.0f);
        acc = Madd(acc, t0[l], p[0]);
        acc = Madd(acc, t1[l], p[1]);
        c[l] = Madd(acc, t2[l], p[2]);
    }
    c[2] = t3[2];
}

// ITOF0 under the VU's rounding (toward zero)
inline float Itof0(u32 i) { return float(std::int32_t(i)); }

/* 1332's alpha of vertex v (its lighting pre-pass): the reflected light's cosine to the eye, raised to the
   power e by the float's bits, times the mask; 0 with kRwWAlphaZero. */
inline u32 WorldAlpha(const float* h, const u32* mesh, u32 v) {
    if (Mode(h) & kRwWAlphaZero) return 0;
    const u32* q0 = mesh + 12 * v;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    u32 first;
    std::memcpy(&first, Q(h, kRwMisc) + 3, 4);
    const u32* nb = mesh + 4 * (first + 4 * (v / 4));
    const u32 lane = v % 4;
    const float* L = Q(h, kRwWL), *E = Q(h, kRwWE), *K = Q(h, kRwWK);
    const float dx = Sub(p[0], E[0]), dy = Sub(p[1], E[1]), dz = Sub(p[2], E[2]);
    const float d2 = Madd(Madd(Mul(dy, dy), dx, dx), dz, dz);
    const float inv = Rsqrt(d2);
    const float nx = Itof0(nb[lane]), ny = Itof0(nb[4 + lane]), nz = Itof0(nb[8 + lane]), m = Itof0(nb[12 + lane]);
    const float nl = Madd(Madd(Mul(nx, K[0]), ny, K[1]), nz, K[2]);
    const float rx = Madd(Sub(0.0f, L[0]), nl, nx), ry = Madd(Sub(0.0f, L[1]), nl, ny), rz = Madd(Sub(0.0f, L[2]), nl, nz);
    const float rd = Madd(Madd(Mul(rx, dx), ry, dy), rz, dz);
    const float cs = Max(Mul(rd, inv), K[3]);
    const float pw = Madd(Add(0.0f, Q(h, kRwWBase)[1]), Itof0(B(cs)), L[3]);
    const float s1 = Max(Mul(m, F(Ftoi(pw))), Q(h, kRwWMin)[0]);
    return Ftoi(s1);
}

/* Vertex v of a world batch. */
inline void WorldVertex(const float* h, const u32* mesh, u32 v, GsVertex* o) {
    const u32 mode = Mode(h);
    const u32* q0 = mesh + 12 * v;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    float c[4];
    WorldClip(h, p, c);
    const float q = Recip(c[3]);
    const float* O = Q(h, kRwWO);
    o->xyz[0] = Ftoi(Madd(O[0], c[0], q) * 16.0f);
    o->xyz[1] = Ftoi(Madd(O[1], c[1], q) * 16.0f);
    o->xyz[2] = Ftoi(Madd(O[2], c[2], q) * 16.0f);
    o->stq[0] = B(Mul(Itof12(q0[4]), q)), o->stq[1] = B(Mul(Itof12(q0[5]), q)), o->stq[2] = B(q);
    std::memcpy(o->rgba, q0 + 8, 16);
    if (mode & kRwWFog) {
        const float fs = Q(h, kRwWScale)[3], fb = Q(h, kRwWOff)[3];
        float f = Add(Sub(0.0f, Mul(fb, fs)), Mul(c[3], fs));
        f = Max(f, 0.0f);
        if (f > 255.0f) f = 255.0f;
        o->fog = (Ftoi(f * 16.0f) >> 4) & 0xFF;
    }
    if (mode & kRwWAlphaZero) {
        o->rgba[3] = 0;
    } else if (mode & kRwWLit) {
        // 1332's: the reflected light's cosine to the eye, raised to the power e by the float's bits, times the mask
        u32 first;
        std::memcpy(&first, Q(h, kRwMisc) + 3, 4);
        const u32* nb = mesh + 4 * (first + 4 * (v / 4));
        const u32 lane = v % 4;
        const float* L = Q(h, kRwWL), *E = Q(h, kRwWE), *K = Q(h, kRwWK);
        const float dx = Sub(p[0], E[0]), dy = Sub(p[1], E[1]), dz = Sub(p[2], E[2]);
        const float d2 = Madd(Madd(Mul(dy, dy), dx, dx), dz, dz);
        const float inv = Rsqrt(d2);
        const float nx = Itof0(nb[lane]), ny = Itof0(nb[4 + lane]), nz = Itof0(nb[8 + lane]), m = Itof0(nb[12 + lane]);
        const float nl = Madd(Madd(Mul(nx, K[0]), ny, K[1]), nz, K[2]);
        const float rx = Madd(Sub(0.0f, L[0]), nl, nx), ry = Madd(Sub(0.0f, L[1]), nl, ny), rz = Madd(Sub(0.0f, L[2]), nl, nz);
        const float rd = Madd(Madd(Mul(rx, dx), ry, dy), rz, dz);
        const float cs = Max(Mul(rd, inv), K[3]);
        const float pw = Madd(Add(0.0f, Q(h, kRwWBase)[1]), Itof0(B(cs)), L[3]);
        const float s1 = Max(Mul(m, F(Ftoi(pw))), Q(h, kRwWMin)[0]);
        o->rgba[3] = Ftoi(s1);
    }
}

/* The screen's place before FTOI4 (the winding's). */
inline void WorldScreen(const float* h, const u32* mesh, u32 v, float* s) {
    const u32* q0 = mesh + 12 * v;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    float c[4];
    WorldClip(h, p, c);
    const float q = Recip(c[3]);
    const float* O = Q(h, kRwWO);
    s[0] = Madd(O[0], c[0], q);
    s[1] = Madd(O[1], c[1], q);
}

/* Path B's guard band's judgement of vertex v. */
inline u32 WorldJudge(const float* h, const u32* mesh, u32 v) {
    const u32* q0 = mesh + 12 * v;
    const float p[3] = {F(q0[0]), F(q0[1]), F(q0[2])};
    float c[4];
    WorldClip(h, p, c);
    const float* g0 = Q(h, kRwWG0), *g1 = Q(h, kRwWG1);
    return Judge(Mul(c[0], g0[1]), Mul(c[1], g1[1]), Madd(g0[2], g1[2], c[3]), c[3]);
}

/* The same with the VU's own arithmetic (memory past a batch: not floats), `q0` the quadword's bits; the caller
   under the VU's rounding. */
inline u32 WorldJudgeVu(const float* h, const u32* q0) {
    using namespace lifted;
    const float* t0 = Q(h, kRwWT), *t1 = Q(h, kRwWT + 1), *t2 = Q(h, kRwWT + 2), *t3 = Q(h, kRwWT + 3);
    u32 c[4];
    for (int l : {0, 1, 3}) {
        u32 acc = BinaryOf<kMul>(B(t3[l]), kOne, 0);
        acc = BinaryOf<kMadd>(B(t0[l]), q0[0], acc);
        acc = BinaryOf<kMadd>(B(t1[l]), q0[1], acc);
        c[l] = BinaryOf<kMadd>(B(t2[l]), q0[2], acc);
    }
    const float* g0 = Q(h, kRwWG0), *g1 = Q(h, kRwWG1);
    const u32 g[4] = {BinaryOf<kMul>(c[0], B(g0[1]), 0), BinaryOf<kMul>(c[1], B(g1[1]), 0),
                      BinaryOf<kMadd>(B(g1[2]), c[3], B(g0[2])), c[3]};
    return JudgeVu(g);
}

/* Whether the strip's triangle ending at vertex k is drawn (its ADC as the microcode makes it). */
inline bool WorldDrawn(const float* h, const u32* mesh, u32 k) {
    const u32 mode = Mode(h);
    if (mesh[12 * k + 3] & 0x8000) return false;
    if (mode & kRwWCull) {
        float a[2], b[2], c[2];
        WorldScreen(h, mesh, k - 2, a);
        WorldScreen(h, mesh, k - 1, b);
        WorldScreen(h, mesh, k, c);
        float u[2], w[2];
        if ((k & 1) == 0) {
            u[0] = Sub(b[0], a[0]), u[1] = Sub(b[1], a[1]);
            w[0] = Sub(b[0], c[0]), w[1] = Sub(b[1], c[1]);
        } else {
            u[0] = Sub(c[0], b[0]), u[1] = Sub(c[1], b[1]);
            w[0] = Sub(a[0], b[0]), w[1] = Sub(a[1], b[1]);
        }
        const float cz = Sub(Mul(u[0], w[1]), Mul(w[0], u[1]));
        u32 base;
        std::memcpy(&base, Q(h, kRwWBase), 4);
        if (((base + ((B(cz) >> 31) ? 0x20u : 0u)) & 0x8000u) != 0) return false;
    }
    if (mode & kRwWGuard)
        for (u32 j = k - 2; j <= k; ++j)
            if (WorldJudge(h, mesh, j)) return false;
    return true;
}

/* The bridge's fallback: an index's vertex (the world's: a triangle's corner, 3t + c), whether its triangle is
   drawn, and the vertex of the program the header's mode says. */
inline u32 RwVertexOf(const float* h, u32 index) { return (Mode(h) & kRwWorld) ? index / 3 + index % 3 : index; }
inline bool RwDrawn(const float* h, const u32* mesh, u32 index) {
    return !(Mode(h) & kRwWorld) || WorldDrawn(h, mesh, index / 3 + 2);
}
inline void RwVertex(const float* h, const u32* mesh, u32 j, GsVertex* o) {
    const u32 mode = Mode(h);
    if (mode & kRwWorld) WorldVertex(h, mesh, j, o);
    else if (mode & kRwInstanced) InstancedVertex(h, mesh, j, o);
    else CarsVertex(h, mesh, j, o);
}

}  // namespace rw_native
}  // namespace ps2host
