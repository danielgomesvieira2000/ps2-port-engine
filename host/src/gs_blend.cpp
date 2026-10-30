/* A picture between two from their GS input (ps2host/gs_blend.h). */

#include "ps2host/gs_blend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace ps2host {

namespace {

struct Packet {
    u32 first;          // its tag's first word in the path's words
    u32 size;           // words, the tag's four with them
    u64 lo, hi;         // the tag
};

void Cut(const std::vector<u32>& words, std::vector<Packet>& out) {
    std::size_t at = 0;
    while (words.size() - at >= 4) {
        const u64 lo = words[at] | u64(words[at + 1]) << 32, hi = words[at + 2] | u64(words[at + 3]) << 32;
        const u32 nloop = u32(lo & 0x7FFF), flg = u32(lo >> 58) & 3, nreg = u32(lo >> 60) ? u32(lo >> 60) : 16;
        const u64 data = flg == 0 ? u64(nloop) * nreg * 4 : flg == 1 ? (u64(nloop) * nreg + 1) / 2 * 4 : u64(nloop) * 4;
        if (words.size() - at - 4 < data) break;
        out.push_back({u32(at), u32(4 + data), lo, hi});
        at += 4 + data;
    }
}

/* NLOOP, FLG, NREG, REGS (as many as NREG) and PRIM where PRE: not EOP nor
   the tag's unused bits (microcode leaves its own values there). */
u64 ShapeLo(u64 lo) {
    const u64 pre = (lo >> 46) & 1;
    return (lo & (0x7FFFull | 0xFull << 60 | 3ull << 58)) | (pre ? lo & (0xFFFull << 46) : 0);
}
u64 ShapeHi(u64 lo, u64 hi) {
    const u32 nreg = u32(lo >> 60) ? u32(lo >> 60) : 16;
    return nreg == 16 ? hi : hi & ((1ull << (4 * nreg)) - 1);
}

/* The packets of `a` and `b` that are the same thing, in order (patience
   alignment): equal ends taken first; shapes found once in each are anchors,
   the longest run of them in order in both lines the two up, and the gaps
   between are aligned the same way; a gap with no anchors, small enough,
   by the longest common subsequence. (As difflib's on recorded pictures:
   93-97% of the vertices, where a nearest-way-back walk lost a third.) */
struct ShapeKey {
    u64 lo, hi;
    bool operator==(const ShapeKey& o) const { return lo == o.lo && hi == o.hi; }
};
struct ShapeHash {
    std::size_t operator()(const ShapeKey& k) const { return std::size_t(k.lo * 0x9E3779B97F4A7C15ull ^ k.hi); }
};

class Aligner {
public:
    Aligner(const std::vector<Packet>& a, const std::vector<Packet>& b, std::vector<std::pair<u32, u32>>& out)
        : out_(out) {
        for (const Packet& p : a) a_.push_back({ShapeLo(p.lo), ShapeHi(p.lo, p.hi)});
        for (const Packet& p : b) b_.push_back({ShapeLo(p.lo), ShapeHi(p.lo, p.hi)});
        Go(0, u32(a_.size()), 0, u32(b_.size()));
    }

private:
    std::vector<ShapeKey> a_, b_;
    std::vector<std::pair<u32, u32>>& out_;

    void Go(u32 alo, u32 ahi, u32 blo, u32 bhi) {
        while (alo < ahi && blo < bhi && a_[alo] == b_[blo]) out_.push_back({alo++, blo++});
        u32 tail = 0;
        while (alo < ahi && blo < bhi && a_[ahi - 1] == b_[bhi - 1]) { --ahi; --bhi; ++tail; }
        if (alo < ahi && blo < bhi) Middle(alo, ahi, blo, bhi);
        for (u32 k = 0; k < tail; ++k) out_.push_back({ahi + k, bhi + k});
    }

    void Middle(u32 alo, u32 ahi, u32 blo, u32 bhi) {
        // Anchors: shapes once in each.
        std::unordered_map<ShapeKey, std::pair<int, u32>, ShapeHash> in_a, in_b;     // count, where
        for (u32 i = alo; i < ahi; ++i) { auto& e = in_a[a_[i]]; ++e.first; e.second = i; }
        for (u32 j = blo; j < bhi; ++j) { auto& e = in_b[b_[j]]; ++e.first; e.second = j; }
        std::vector<std::pair<u32, u32>> anchors;
        for (u32 i = alo; i < ahi; ++i) {
            const auto& ea = in_a[a_[i]];
            if (ea.first != 1) continue;
            auto found = in_b.find(a_[i]);
            if (found != in_b.end() && found->second.first == 1) anchors.push_back({i, found->second.second});
        }
        if (anchors.empty()) {
            Common(alo, ahi, blo, bhi);
            return;
        }
        // The longest run of anchors in order in both (patience sorting).
        std::vector<u32> tails, tail_at, back(anchors.size(), ~0u);
        for (u32 n = 0; n < anchors.size(); ++n) {
            const u32 j = anchors[n].second;
            const u32 k = u32(std::lower_bound(tails.begin(), tails.end(), j) - tails.begin());
            if (k == tails.size()) { tails.push_back(j); tail_at.push_back(n); }
            else { tails[k] = j; tail_at[k] = n; }
            back[n] = k ? tail_at[k - 1] : ~0u;
        }
        std::vector<u32> chain;
        for (u32 n = tail_at.back(); n != ~0u; n = back[n]) chain.push_back(n);
        u32 pa = alo, pb = blo;
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            const auto [i, j] = anchors[*it];
            Go(pa, i, pb, j);
            out_.push_back({i, j});
            pa = i + 1;
            pb = j + 1;
        }
        Go(pa, ahi, pb, bhi);
    }

    /* No anchors: the longest common subsequence where the gap is small. */
    void Common(u32 alo, u32 ahi, u32 blo, u32 bhi) {
        const u32 n = ahi - alo, m = bhi - blo;
        if (u64(n) * m > 65536) return;
        std::vector<u16> l((n + 1) * (m + 1), 0);
        auto at = [&](u32 x, u32 y) -> u16& { return l[x * (m + 1) + y]; };
        for (u32 x = n; x-- > 0;)
            for (u32 y = m; y-- > 0;)
                at(x, y) = a_[alo + x] == b_[blo + y] ? u16(at(x + 1, y + 1) + 1) : std::max(at(x + 1, y), at(x, y + 1));
        u32 x = 0, y = 0;
        while (x < n && y < m) {
            if (a_[alo + x] == b_[blo + y]) { out_.push_back({alo + x, blo + y}); ++x; ++y; }
            else if (at(x + 1, y) >= at(x, y + 1)) ++x;
            else ++y;
        }
    }
};

void Align(const std::vector<Packet>& a, const std::vector<Packet>& b, std::vector<std::pair<u32, u32>>& out) {
    Aligner(a, b, out);
}

constexpr bool IsXyz(u32 r) { return r == 4 || r == 5 || r == 0xC || r == 0xD; }

float Lerp(float a, float b, float t) { return a + (b - a) * t; }
/* Integers: t in 256ths (a million of them a picture - a double's rounding
   each was most of the blend's time). */
thread_local int g_weight = 128;
inline u32 LerpU(u32 a, u32 b, float) {
    return u32(std::int64_t(a) + ((std::int64_t(b) - std::int64_t(a)) * g_weight + 128 >> 8));
}
float AsFloat(u32 w) { float f; std::memcpy(&f, &w, 4); return f; }
u32 AsWord(float f) { u32 w; std::memcpy(&w, &f, 4); return w; }

/* A matched PACKED packet's vertex values blended into `to` (newer's copy);
   false where it jumps. */
bool BlendPacked(const u32* a, const u32* b, u32* to, const Packet& p, float t, float jump, u32* vertices) {
    const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
    u32 regs[16];
    for (u32 r = 0; r < nreg; ++r) regs[r] = u32(p.hi >> (4 * r)) & 0xF;
    const float limit = jump * 16.0f;
    u32 count = 0;
    for (u32 q = 0; q < nloop * nreg; ++q) {
        if (!IsXyz(regs[q % nreg])) continue;
        const u32* x = a + 4 + 4 * q;
        const u32* y = b + 4 + 4 * q;
        const float dx = std::fabs(float(x[0] & 0xFFFF) - float(y[0] & 0xFFFF));
        const float dy = std::fabs(float(x[1] & 0xFFFF) - float(y[1] & 0xFFFF));
        if (dx > limit || dy > limit) return false;
        ++count;
    }
    *vertices = count;
    for (u32 q = 0; q < nloop * nreg; ++q) {
        const u32 reg = regs[q % nreg];
        const u32* x = a + 4 + 4 * q;
        const u32* y = b + 4 + 4 * q;
        u32* o = to + 4 + 4 * q;
        switch (reg) {
            case 1:             // RGBAQ: each byte
                for (int i = 0; i < 4; ++i) o[i] = (y[i] & ~0xFFu) | LerpU(x[i] & 0xFF, y[i] & 0xFF, t);
                break;
            case 2:             // ST (and Q)
                for (int i = 0; i < 3; ++i) o[i] = AsWord(Lerp(AsFloat(x[i]), AsFloat(y[i]), t));
                break;
            case 3:             // UV: 14 bits each
                o[0] = (y[0] & ~0x3FFFu) | LerpU(x[0] & 0x3FFF, y[0] & 0x3FFF, t);
                o[1] = (y[1] & ~0x3FFFu) | LerpU(x[1] & 0x3FFF, y[1] & 0x3FFF, t);
                break;
            case 4: case 0xC:   // XYZF: x, y 16 bits, z 24 (<<4), f 8 (<<4 of the fourth)
                o[0] = (y[0] & ~0xFFFFu) | LerpU(x[0] & 0xFFFF, y[0] & 0xFFFF, t);
                o[1] = (y[1] & ~0xFFFFu) | LerpU(x[1] & 0xFFFF, y[1] & 0xFFFF, t);
                o[2] = (y[2] & ~0xFFFFFF0u) | LerpU((x[2] >> 4) & 0xFFFFFF, (y[2] >> 4) & 0xFFFFFF, t) << 4;
                o[3] = (y[3] & ~0xFF0u) | LerpU((x[3] >> 4) & 0xFF, (y[3] >> 4) & 0xFF, t) << 4;
                break;
            case 5: case 0xD:   // XYZ: x, y 16 bits, z 32
                o[0] = (y[0] & ~0xFFFFu) | LerpU(x[0] & 0xFFFF, y[0] & 0xFFFF, t);
                o[1] = (y[1] & ~0xFFFFu) | LerpU(x[1] & 0xFFFF, y[1] & 0xFFFF, t);
                o[2] = LerpU(x[2], y[2], t);
                break;
            case 0xA:           // FOG: 8 bits (<<4 of the fourth)
                o[3] = (y[3] & ~0xFF0u) | LerpU((x[3] >> 4) & 0xFF, (y[3] >> 4) & 0xFF, t) << 4;
                break;
            default: break;     // A+D, PRIM, NOP: the newer's
        }
    }
    return true;
}

}  // namespace

GsBlendStats BlendGsPictures(const GsPicture& older, const GsPicture& newer, float t, GsPicture& out, float jump) {
    GsBlendStats stats;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point from) { return std::chrono::duration<double, std::milli>(Clock::now() - from).count(); };
    g_weight = int(std::lround(std::clamp(t, 0.0f, 1.0f) * 256.0f));
    auto began = Clock::now();
    out = newer;
    stats.ms_copy = ms(began);
    for (int path = 0; path < 3; ++path) {
        began = Clock::now();
        std::vector<Packet> a, b;
        Cut(older.paths[path], a);
        Cut(newer.paths[path], b);
        stats.packets += u32(b.size());
        std::vector<std::pair<u32, u32>> pairs;
        Align(a, b, pairs);
        stats.ms_align += ms(began);
        began = Clock::now();
        for (const auto& [i, j] : pairs) {
            const Packet& p = b[j];
            ++stats.matched;
            const u32 flg = u32(p.lo >> 58) & 3;
            const u32* x = older.paths[path].data() + a[i].first;
            const u32* y = newer.paths[path].data() + p.first;
            if (flg == 0) {
                const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
                u32 xyz = 0;
                for (u32 r = 0; r < nreg; ++r) xyz += IsXyz(u32(p.hi >> (4 * r)) & 0xF) ? nloop : 0;
                u32 vertices = 0;
                if (std::memcmp(x, y, p.size * 4) == 0) stats.still += xyz;
                else if (BlendPacked(x, y, out.paths[path].data() + p.first, p, t, jump, &vertices)) stats.blended += vertices;
                else { ++stats.jumped; stats.jumped_vertices += xyz; }
            }
        }
        stats.ms_blend += ms(began);
        for (const Packet& p : b) {
            if ((u32(p.lo >> 58) & 3) != 0) continue;
            const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
            for (u32 r = 0; r < nreg; ++r)
                if (IsXyz(u32(p.hi >> (4 * r)) & 0xF)) stats.vertices += nloop;
        }
    }
    return stats;
}

}  // namespace ps2host
