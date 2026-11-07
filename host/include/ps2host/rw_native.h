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

}  // namespace rw_native
}  // namespace ps2host
