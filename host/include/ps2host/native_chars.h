#pragma once
/*
 * Outbreak's characters' rigid vertex on the CPU (docs/NATIVE_RENDERER.md): what
 * gs_mesh_chars.vert computes, as the emulated VU computes it - the caller under
 * the VU's rounding (toward zero: _MM_ROUND_TOWARD_ZERO), each operation one
 * float operation, the reciprocal square roots through a double (as the
 * emulated RSQRT and ERSQRT). The header as gs_mesh_chars.vert's (floats):
 * vf00-vf21 (0-87), rows 11-13 (88-99), misc (100-103: OFX, OFY, the first
 * matrix row's address, vi15), the kept vertices (104-127), the enhanced draws' ext (128-175), the matrices' rows
 * (from quadword kRows).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <xmmintrin.h>
#include <emmintrin.h>

namespace ps2host {
namespace native_chars {

#pragma clang fp contract(off)

constexpr int kRows = 44;      // (the header's quadword the matrices' rows start at)

struct Vertex {
    std::uint32_t stq[3];       // S, T, Q as floats' bits
    std::uint32_t rgb[3];       // FTOI0, as the packet's (all 32 bits)
    std::uint32_t xyz[3];       // FTOI4
    std::uint32_t fog4;         // the fog's FTOI4
    float guard[4];             // the place through vf05-vf08 (the clip test)
    float place[4];             // (full:) the place through its matrix - the lights'
    float normal[3];            // ...the normal through its own
    float view[4];              // ...the place through vf01-vf04, before the divide
};

inline float Rsqrt(float x) {
    const double root = std::sqrt(std::fabs(double(x)));
    const float vu = root == 0.0 ? 3.4028234663852886e38f : float(1.0 / root);
    // PS2HOST_NATIVE_RSQRT_CHECK=1 (a test): gs_mesh_common.glsl's Rsqrt (the nearest's arithmetic, its exact
    // errors) against the VU's, on each input.
    static const bool check = std::getenv("PS2HOST_NATIVE_RSQRT_CHECK") != nullptr;
    if (check && x > 0.0f) {
        static unsigned long long all = 0, differ = 0;
        const unsigned mode = _MM_GET_ROUNDING_MODE();
        _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
        auto bits = [](float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; };
        auto of = [](std::uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; };
        auto past = [](float y, float v) {
            const float pp = y * y;
            const float pe = std::fma(y, y, -pp);
            const float t = std::fma(pp, v, -1.0f);
            return std::fma(pe, v, t) > 0.0f;
        };
        float y = 1.0f / std::sqrt(x);
        const float hh = 0.5f * x * y;
        const float nn = std::fma(-hh, y, 1.5f);
        y = y * nn;
        for (int i = 0; i < 2; ++i)
            if (past(y, x)) y = of(bits(y) - 1);
        if (!past(of(bits(y) + 1), x)) y = of(bits(y) + 1);
        _MM_SET_ROUNDING_MODE(mode);
        ++all;
        if (y != vu && ++differ <= 8)
            std::fprintf(stderr, "[native] rsqrt(%08X): the VU's %08X, the shader's %08X (%llu of %llu)\n", bits(x), bits(vu),
                         bits(y), differ, all);
        if (all % 5000000 == 0) std::fprintf(stderr, "[native] rsqrt: %llu of %llu not the VU's\n", differ, all);
    }
    return vu;
}

inline std::uint32_t Ftoi(float y) {
    if (y >= 2147483648.0f) return 0x7FFFFFFFu;
    if (y < -2147483648.0f) return 0x80000000u;
    return std::uint32_t(std::int32_t(y));
}

// The header's first matrix row's address (misc.z's low 16 bits; bit 16: a blended batch's).
inline std::uint32_t First(const float* h) {
    std::uint32_t first;
    std::memcpy(&first, h + 102, 4);
    return first & 0xFFFF;
}
inline bool Blended(const float* h) {
    std::uint32_t first;
    std::memcpy(&first, h + 102, 4);
    return (first >> 16) & 1;
}

// The place and its guard band alone (the triangles' test), or everything (`full`), from the place's matrix rows
// `m` (four, each four lanes) and the normal's `nm` (three); `normalise`: the normal made unit length (RSQRT).
inline void MakeWith(const float* h, const float* pos, const float* nrm, const float* tx, const float (*m)[4],
                     const float (*nm)[4], bool normalise, bool full, Vertex* out) {
    auto vf = [&](int r, int l) { return h[r * 4 + l]; };
    float p[4], cp[4], g[4];
    for (int l = 0; l < 4; ++l) {
        float a = m[0][l] * pos[0];
        float b = m[1][l] * pos[1];
        float ab = a + b;
        float c = m[2][l] * pos[2];
        float abc = ab + c;
        p[l] = abc + m[3][l];
    }
    for (int l = 0; l < 4; ++l) {
        float a = vf(5, l) * p[0];
        float b = vf(6, l) * p[1];
        float ab = a + b;
        float c = vf(7, l) * p[2];
        float abc = ab + c;
        g[l] = abc + vf(8, l);
    }
    std::memcpy(out->guard, g, sizeof g);
    if (!full) return;
    for (int l = 0; l < 4; ++l) {
        float a = vf(1, l) * p[0];
        float b = vf(2, l) * p[1];
        float ab = a + b;
        float c = vf(3, l) * p[2];
        float abc = ab + c;
        cp[l] = abc + vf(4, l);
    }
    std::memcpy(out->place, p, sizeof p);
    std::memcpy(out->view, cp, sizeof cp);
    float n[3];
    for (int l = 0; l < 3; ++l) {
        float a = nm[0][l] * nrm[0];
        float b = nm[1][l] * nrm[1];
        float ab = a + b;
        float c = nm[2][l] * nrm[2];
        n[l] = ab + c;
    }
    if (normalise) {
        // (MUL.xyz, ADDY.x, ADDZ.x, RSQRT, MULQ.xyz)
        const float xx = n[0] * n[0], yy = n[1] * n[1], zz = n[2] * n[2];
        const float xy = xx + yy;
        const float r = Rsqrt(xy + zz);
        for (int l = 0; l < 3; ++l) n[l] = n[l] * r;
    }
    std::memcpy(out->normal, n, sizeof n);
    const float q = cp[3] == 0.0f ? (std::signbit(cp[3]) ? -3.4028234663852886e38f : 3.4028234663852886e38f)
                                  : 1.0f / cp[3];
    auto tex = [&](int l) {
        float a = h[88 + l] * tx[0];
        float b = h[92 + l] * tx[1];
        float ab = a + b;
        return float(ab + h[96 + l]);
    };
    const float s = tex(0) * q, t = tex(1) * q, qq = tx[2] * q;
    std::memcpy(&out->stq[0], &s, 4), std::memcpy(&out->stq[1], &t, 4), std::memcpy(&out->stq[2], &qq, 4);
    float fw = vf(10, 0) - cp[3];
    float fy = fw * vf(10, 1);
    const float fog = std::max(std::min(float(fy * 255.0f), 255.0f), 0.0f);
    out->fog4 = Ftoi(fog * 16.0f);
    for (int l = 0; l < 3; ++l) {
        float m = cp[l] * q;
        out->xyz[l] = Ftoi(m * 16.0f);
    }
    // the lights: x a direction, y and z points
    float lx[3] = {vf(11, 0), 0, 0}, ly[3] = {vf(12, 0), 0, 0}, lz[3] = {vf(13, 0), 0, 0}, atten[3] = {1, 1, 1};
    for (int k = 1; k <= 2; ++k) {
        lx[k] = vf(11, k) - p[0], ly[k] = vf(12, k) - p[1], lz[k] = vf(13, k) - p[2];
        float a = lx[k] * lx[k];
        float b = ly[k] * ly[k];
        float ab = a + b;
        float c = lz[k] * lz[k];
        const float r = Rsqrt(ab + c);
        lx[k] = lx[k] * r, ly[k] = ly[k] * r, lz[k] = lz[k] * r;
        float st = vf(14, k) * r;
        atten[k] = std::min(float(st), 1.0f);
    }
    float d[3];
    for (int k = 0; k < 3; ++k) {
        float a = lx[k] * n[0];
        float b = ly[k] * n[1];
        float ab = a + b;
        float c = lz[k] * n[2];
        float abc = ab + c;
        d[k] = std::max(float(abc), 0.0f);
        if (k) d[k] = d[k] * atten[k];
    }
    for (int l = 0; l < 3; ++l) {
        float acc = vf(15, l) * d[0];
        float t1 = vf(16, l) * d[1];
        acc = acc + t1;
        float t2 = vf(17, l) * d[2];
        acc = acc + t2;
        acc = acc + vf(18, l);
        float t3 = vf(19, l) * atten[1];
        acc = acc + t3;
        float t4 = vf(20, l) * atten[2];
        acc = acc + t4;
        acc = acc + vf(21, l);
        float c = std::min(float(acc), 128.0f) * vf(9, l);
        out->rgb[l] = Ftoi(c);
    }
}

// The rigid batch's vertex: a place matrix (four rows at `ma`), a normal one (three at `mb`).
inline void Make(const float* h, const float* pos, const float* nrm, const float* tx, std::uint32_t ma, std::uint32_t mb,
                 bool full, Vertex* out) {
    const std::uint32_t first = First(h);
    float m[4][4], nm[3][4];
    std::memcpy(m, h + (kRows + (ma - first)) * 4, sizeof m);
    std::memcpy(nm, h + (kRows + (mb - first)) * 4, sizeof nm);
    MakeWith(h, pos, nrm, tx, m, nm, false, full, out);
}

// The blended batch's vertex: four matrices (at `mats`' four addresses) by its weights, for the place and the
// normal (three rows of the same), the normal made unit length.
inline void MakeBlended(const float* h, const float* pos, const float* nrm, const float* tx, const std::uint32_t* mats,
                        const float* w, bool full, Vertex* out) {
    const std::uint32_t first = First(h);
    auto row = [&](std::uint32_t address, int r, int l) { return h[(kRows + (address - first) + r) * 4 + l]; };
    float m[4][4];
    for (int r = 0; r < 4; ++r)
        for (int l = 0; l < 4; ++l) {
            // (MULAX, MADDAY, MADDAZ, MADDW: the weights x, y, z, w)
            const float a = row(mats[0] & 0xFFFF, r, l) * w[0];
            const float b = row(mats[1] & 0xFFFF, r, l) * w[1];
            const float ab = a + b;
            const float c = row(mats[2] & 0xFFFF, r, l) * w[2];
            const float abc = ab + c;
            const float d = row(mats[3] & 0xFFFF, r, l) * w[3];
            m[r][l] = abc + d;
        }
    MakeWith(h, pos, nrm, tx, m, m, true, full, out);
}

// The guard band's place alone (the triangles' test), four lanes at once: the same operations in the same order
// as MakeWith's (each product and sum rounded on its own, under the caller's rounding) - the same bits.
inline void Guard(const float* h, const float* q, float* g) {
    const std::uint32_t first = First(h);
    std::uint32_t idx[4];
    std::memcpy(idx, q + 12, 16);
    auto row = [&](std::uint32_t address, int r) { return _mm_loadu_ps(h + (kRows + (address - first) + r) * 4); };
    __m128 m[4];
    if (Blended(h)) {
        const __m128 wx = _mm_set1_ps(q[16]), wy = _mm_set1_ps(q[17]), wz = _mm_set1_ps(q[18]), ww = _mm_set1_ps(q[19]);
        for (int r = 0; r < 4; ++r) {
            const __m128 a = _mm_mul_ps(row(idx[0] & 0xFFFF, r), wx);
            const __m128 b = _mm_mul_ps(row(idx[1] & 0xFFFF, r), wy);
            const __m128 ab = _mm_add_ps(a, b);
            const __m128 c = _mm_mul_ps(row(idx[2] & 0xFFFF, r), wz);
            const __m128 abc = _mm_add_ps(ab, c);
            const __m128 d = _mm_mul_ps(row(idx[3] & 0xFFFF, r), ww);
            m[r] = _mm_add_ps(abc, d);
        }
    } else {
        for (int r = 0; r < 4; ++r) m[r] = row(idx[0] & 0xFFFF, r);
    }
    const __m128 px = _mm_set1_ps(q[0]), py = _mm_set1_ps(q[1]), pz = _mm_set1_ps(q[2]);
    const __m128 p = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(m[0], px), _mm_mul_ps(m[1], py)), _mm_mul_ps(m[2], pz)), m[3]);
    alignas(16) float pl[4];
    _mm_store_ps(pl, p);
    const __m128 gx = _mm_set1_ps(pl[0]), gy = _mm_set1_ps(pl[1]), gz = _mm_set1_ps(pl[2]);
    const __m128 gd = _mm_add_ps(
        _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(h + 20), gx), _mm_mul_ps(_mm_loadu_ps(h + 24), gy)),
                   _mm_mul_ps(_mm_loadu_ps(h + 28), gz)),
        _mm_loadu_ps(h + 32));
    _mm_storeu_ps(g, gd);
}

// A vertex of either kind (the header's), its five quadwords `q`.
inline void MakeVertex(const float* h, const float* q, bool full, Vertex* out) {
    std::uint32_t idx[4];
    std::memcpy(idx, q + 12, 16);
    if (Blended(h)) MakeBlended(h, q, q + 4, q + 8, idx, q + 16, full, out);
    else Make(h, q, q + 4, q + 8, idx[0] & 0xFFFF, idx[1] & 0xFFFF, full, out);
}

#pragma clang fp contract(on)

}  // namespace native_chars
}  // namespace ps2host
