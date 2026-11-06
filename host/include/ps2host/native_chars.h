#pragma once
/*
 * Outbreak's characters' rigid vertex on the CPU (docs/NATIVE_RENDERER.md): what
 * gs_mesh_chars.vert computes, as the emulated VU computes it - the caller under
 * the VU's rounding (toward zero: _MM_ROUND_TOWARD_ZERO), each operation one
 * float operation, the reciprocal square roots through a double (as the
 * emulated RSQRT and ERSQRT). The header as gs_mesh_chars.vert's (floats):
 * vf00-vf21 (0-87), rows 11-13 (88-99), misc (100-103: OFX, OFY, the first
 * matrix row's address, vi15), the kept vertices (104-127), the matrices' rows.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <xmmintrin.h>

namespace ps2host {
namespace native_chars {

#pragma clang fp contract(off)

struct Vertex {
    std::uint32_t stq[3];       // S, T, Q as floats' bits
    std::uint32_t rgb[3];       // FTOI0, as the packet's (all 32 bits)
    std::uint32_t xyz[3];       // FTOI4
    std::uint32_t fog4;         // the fog's FTOI4
    float guard[4];             // the place through vf05-vf08 (the clip test)
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

// The place and its guard band alone (the triangles' test), or everything (`full`).
inline void Make(const float* h, const float* pos, const float* nrm, const float* tx, std::uint32_t ma, std::uint32_t mb,
                 bool full, Vertex* out) {
    auto vf = [&](int r, int l) { return h[r * 4 + l]; };
    std::uint32_t first;
    std::memcpy(&first, h + 102, 4);
    auto row = [&](std::uint32_t address, int l) { return h[(32 + (address - first)) * 4 + l]; };
    float p[4], cp[4], g[4];
    for (int l = 0; l < 4; ++l) {
        float a = row(ma, l) * pos[0];
        float b = row(ma + 1, l) * pos[1];
        float ab = a + b;
        float c = row(ma + 2, l) * pos[2];
        float abc = ab + c;
        p[l] = abc + row(ma + 3, l);
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
    float n[3];
    for (int l = 0; l < 3; ++l) {
        float a = row(mb, l) * nrm[0];
        float b = row(mb + 1, l) * nrm[1];
        float ab = a + b;
        float c = row(mb + 2, l) * nrm[2];
        n[l] = ab + c;
    }
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

#pragma clang fp contract(on)

}  // namespace native_chars
}  // namespace ps2host
