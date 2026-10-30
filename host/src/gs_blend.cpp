/* A picture between two from their GS input (ps2host/gs_blend.h). */

#include "ps2host/gs_blend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

/* A screen coordinate (1/16 pixels) between two, through the projection:
   (c - centre) * w blended with w, then divided again. */
u32 Projected(u32 a, u32 b, double centre, double wa, double wb, double wm, float t) {
    const double ua = (double(a) - centre) * wa, ub = (double(b) - centre) * wb;
    return u32(std::lround(std::clamp(centre + (ua + (ub - ua) * t) / wm, 0.0, 65535.0)));
}
/* Z, affine in Q: where the blended Q falls between the two's (linear in
   time where the Qs are the same). */
double AlongQ(u32 a, u32 b, double qa, double qb, double qm, float t, u32 most) {
    const double k = std::fabs(qb - qa) > 1e-12 * std::max(qa, qb) ? (qm - qa) / (qb - qa) : double(t);
    return std::clamp(std::round(double(a) + (double(b) - double(a)) * k), 0.0, double(most));
}

/* The affine map of the screen taking `from` to `to` (least squares, about
   the means): to = m * (from - mf) + mt. A translation where the points are
   too few or in a line. */
struct Affine {
    double m00 = 1, m01 = 0, m10 = 0, m11 = 1, fx = 0, fy = 0, tx = 0, ty = 0;
    void Apply(double x, double y, double* ox, double* oy) const {
        *ox = tx + m00 * (x - fx) + m01 * (y - fy);
        *oy = ty + m10 * (x - fx) + m11 * (y - fy);
    }
};
Affine Fit(const std::vector<std::pair<double, double>>& from, const std::vector<std::pair<double, double>>& to) {
    Affine f;
    const double count = double(from.size());
    for (std::size_t i = 0; i < from.size(); ++i) {
        f.fx += from[i].first; f.fy += from[i].second; f.tx += to[i].first; f.ty += to[i].second;
    }
    f.fx /= count; f.fy /= count; f.tx /= count; f.ty /= count;
    double sxx = 0, sxy = 0, syy = 0, bxx = 0, bxy = 0, byx = 0, byy = 0;
    for (std::size_t i = 0; i < from.size(); ++i) {
        const double x = from[i].first - f.fx, y = from[i].second - f.fy;
        const double u = to[i].first - f.tx, v = to[i].second - f.ty;
        sxx += x * x; sxy += x * y; syy += y * y;
        bxx += x * u; bxy += y * u; byx += x * v; byy += y * v;
    }
    const double det = sxx * syy - sxy * sxy;
    if (from.size() >= 3 && det > 1e-6 * (sxx + syy) * (sxx + syy) && det > 1.0) {
        f.m00 = (bxx * syy - bxy * sxy) / det;
        f.m01 = (bxy * sxx - bxx * sxy) / det;
        f.m10 = (byx * syy - byy * sxy) / det;
        f.m11 = (byy * sxx - byx * sxy) / det;
    }
    return f;
}

/* A matched PACKED packet's vertex values blended into `to` (newer's copy):
   kBlended, kJumped (moved too far), or kLoose - its vertices' moves not one
   motion of the screen (VU1's clipper cut the polygon anew, as many vertices
   but other points of it: their texture coordinates blended across the
   texture drew a strip of the crossing's stripes). */
enum { kJumped, kBlended, kLoose };
int BlendPacked(const u32* a, const u32* b, u32* to, const Packet& p, float t, const GsBlendOptions& o, u32* vertices) {
    const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
    u32 regs[16];
    for (u32 r = 0; r < nreg; ++r) regs[r] = u32(p.hi >> (4 * r)) & 0xF;
    const float limit = o.jump * 16.0f;
    auto shown = [&o](const u32* v) {
        const u32 x = v[0] & 0xFFFF, y = v[1] & 0xFFFF;
        return x >= o.window[0] && x <= o.window[2] && y >= o.window[1] && y <= o.window[3];
    };
    u32 count = 0;
    for (u32 q = 0; q < nloop * nreg; ++q) {
        const u32* x = a + 4 + 4 * q;
        const u32* y = b + 4 + 4 * q;
        // State among the vertices (A+D, PRIM) that differs: another thing.
        if ((regs[q % nreg] == 0xE || regs[q % nreg] == 0) && std::memcmp(x, y, 16) != 0) return kJumped;
        if (!IsXyz(regs[q % nreg])) continue;
        const float dx = std::fabs(float(x[0] & 0xFFFF) - float(y[0] & 0xFFFF));
        const float dy = std::fabs(float(x[1] & 0xFFFF) - float(y[1] & 0xFFFF));
        // (Only where it is seen, in one picture or the other: a polygon
        // reaching far off the screen - the road under the camera - swings
        // its far corners by thousands of pixels as it nears the camera;
        // blended through its 1/w below, they come out right.)
        if ((dx > limit || dy > limit) && (shown(x) || shown(y))) return kJumped;
        ++count;
    }
    // One motion: the vertices on the screen, older to newer, an affine map's
    // to within 4 pixels or a quarter of their mean move.
    {
        std::vector<std::pair<double, double>> from, onto;
        for (u32 q = 0; q < nloop * nreg; ++q) {
            if (!IsXyz(regs[q % nreg])) continue;
            const u32* x = a + 4 + 4 * q;
            const u32* y = b + 4 + 4 * q;
            if (!shown(x) && !shown(y)) continue;
            from.push_back({double(x[0] & 0xFFFF), double(x[1] & 0xFFFF)});
            onto.push_back({double(y[0] & 0xFFFF), double(y[1] & 0xFFFF)});
        }
        if (from.size() >= 4) {
            const Affine f = Fit(from, onto);
            double worst = 0, moved = 0;
            for (std::size_t i = 0; i < from.size(); ++i) {
                double ex, ey;
                f.Apply(from[i].first, from[i].second, &ex, &ey);
                worst = std::max(worst, std::max(std::fabs(ex - onto[i].first), std::fabs(ey - onto[i].second)));
                moved += std::max(std::fabs(from[i].first - onto[i].first), std::fabs(from[i].second - onto[i].second));
            }
            moved /= double(from.size());
            if (worst > std::max(4.0 * 16.0, 0.25 * moved)) return kLoose;
        }
    }
    *vertices = count;
    // In perspective where the vertex has its Q (ST's third: 1/w): screen
    // positions are not linear in time - a corner near the camera swings far
    // as its w nears 0, and its edge, blended on the screen, became a thin
    // line across it. Blended as the projection's own (homogeneous) values -
    // (x - centre) * w and w, then divided - they are, for things moving
    // steadily before the camera; S and T kept per the vertex's s = S/Q and
    // put on the blended Q; Z (affine in Q) taken along with Q.
    int st = -1;
    for (u32 r = 0; r < nreg; ++r)
        if (regs[r] == 2) st = int(r);
    const double cx = (double(o.window[0]) + double(o.window[2])) / 2, cy = (double(o.window[1]) + double(o.window[3])) / 2;
    bool persp = false;
    double wa = 0, wb = 0, wm = 0, qa = 0, qb = 0, qm = 0;
    const u32* const window = o.window;
    for (u32 q = 0; q < nloop * nreg; ++q) {
        const u32 reg = regs[q % nreg];
        if (q % nreg == 0) {                // a vertex's registers begin: its Q, both pictures'
            persp = false;
            if (st >= 0) {
                qa = AsFloat(a[4 + 4 * (q + u32(st)) + 2]);
                qb = AsFloat(b[4 + 4 * (q + u32(st)) + 2]);
                persp = qa > 1e-9 && qb > 1e-9 && qa < 1e9 && qb < 1e9;
                if (persp) {
                    wa = 1.0 / qa;
                    wb = 1.0 / qb;
                    wm = wa + (wb - wa) * t;
                    qm = 1.0 / wm;
                }
            }
        }
        const u32* x = a + 4 + 4 * q;
        const u32* y = b + 4 + 4 * q;
        u32* o = to + 4 + 4 * q;
        // Off the screen in both, but on its other side, in front of the
        // camera both times (behind it, the projection itself throws a vertex
        // over - the road under the car): another vertex of a strip in that
        // place (a road's line reaching far both ways, its far ends swapped).
        // Blended, it crossed the screen as a thin line; it stays the newer's.
        if (IsXyz(reg) && st >= 0 && AsFloat(a[4 + 4 * (q - q % nreg + u32(st)) + 2]) > 0.0f &&
            AsFloat(b[4 + 4 * (q - q % nreg + u32(st)) + 2]) > 0.0f) {
            const u32 ax = x[0] & 0xFFFF, ay = x[1] & 0xFFFF, bx = y[0] & 0xFFFF, by = y[1] & 0xFFFF;
            if ((ax < window[0] && bx > window[2]) || (ax > window[2] && bx < window[0]) ||
                (ay < window[1] && by > window[3]) || (ay > window[3] && by < window[1]))
                continue;                   // (`to` holds the newer's)
        }
        switch (reg) {
            case 1:             // RGBAQ: each byte
                for (int i = 0; i < 4; ++i) o[i] = (y[i] & ~0xFFu) | LerpU(x[i] & 0xFF, y[i] & 0xFF, t);
                break;
            case 2:             // ST (and Q)
                if (persp) {
                    o[0] = AsWord(float(double(AsFloat(y[0])) / qb * qm));
                    o[1] = AsWord(float(double(AsFloat(y[1])) / qb * qm));
                    o[2] = AsWord(float(qm));
                } else {
                    for (int i = 0; i < 3; ++i) o[i] = AsWord(Lerp(AsFloat(x[i]), AsFloat(y[i]), t));
                }
                break;
            case 3:             // UV: 14 bits each
                o[0] = (y[0] & ~0x3FFFu) | LerpU(x[0] & 0x3FFF, y[0] & 0x3FFF, t);
                o[1] = (y[1] & ~0x3FFFu) | LerpU(x[1] & 0x3FFF, y[1] & 0x3FFF, t);
                break;
            case 4: case 0xC:   // XYZF: x, y 16 bits, z 24 (<<4), f 8 (<<4 of the fourth)
                if (persp) {
                    o[0] = (y[0] & ~0xFFFFu) | Projected(x[0] & 0xFFFF, y[0] & 0xFFFF, cx, wa, wb, wm, t);
                    o[1] = (y[1] & ~0xFFFFu) | Projected(x[1] & 0xFFFF, y[1] & 0xFFFF, cy, wa, wb, wm, t);
                    o[2] = (y[2] & ~0xFFFFFF0u) | u32(AlongQ((x[2] >> 4) & 0xFFFFFF, (y[2] >> 4) & 0xFFFFFF, qa, qb, qm, t, 0xFFFFFF)) << 4;
                } else {
                    o[0] = (y[0] & ~0xFFFFu) | LerpU(x[0] & 0xFFFF, y[0] & 0xFFFF, t);
                    o[1] = (y[1] & ~0xFFFFu) | LerpU(x[1] & 0xFFFF, y[1] & 0xFFFF, t);
                    o[2] = (y[2] & ~0xFFFFFF0u) | LerpU((x[2] >> 4) & 0xFFFFFF, (y[2] >> 4) & 0xFFFFFF, t) << 4;
                }
                o[3] = (y[3] & ~0xFF0u) | LerpU((x[3] >> 4) & 0xFF, (y[3] >> 4) & 0xFF, t) << 4;
                break;
            case 5: case 0xD:   // XYZ: x, y 16 bits, z 32
                if (persp) {
                    o[0] = (y[0] & ~0xFFFFu) | Projected(x[0] & 0xFFFF, y[0] & 0xFFFF, cx, wa, wb, wm, t);
                    o[1] = (y[1] & ~0xFFFFu) | Projected(x[1] & 0xFFFF, y[1] & 0xFFFF, cy, wa, wb, wm, t);
                    o[2] = u32(AlongQ(x[2], y[2], qa, qb, qm, t, 0xFFFFFFFFu));
                } else {
                    o[0] = (y[0] & ~0xFFFFu) | LerpU(x[0] & 0xFFFF, y[0] & 0xFFFF, t);
                    o[1] = (y[1] & ~0xFFFFu) | LerpU(x[1] & 0xFFFF, y[1] & 0xFFFF, t);
                    o[2] = LerpU(x[2], y[2], t);
                }
                break;
            case 0xA:           // FOG: 8 bits (<<4 of the fourth)
                o[3] = (y[3] & ~0xFF0u) | LerpU((x[3] >> 4) & 0xFF, (y[3] >> 4) & 0xFF, t) << 4;
                break;
            default: break;     // A+D, PRIM, NOP: the newer's
        }
    }
    return kBlended;
}

/* Each XYZ register's x, y (1/16 pixels) in a PACKED packet. */
void Points(const u32* words, const Packet& p, std::vector<std::pair<double, double>>& out) {
    const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
    out.clear();
    for (u32 q = 0; q < nloop * nreg; ++q)
        if (IsXyz(u32(p.hi >> (4 * (q % nreg))) & 0xF))
            out.push_back({double(words[4 + 4 * q] & 0xFFFF), double(words[4 + 4 * q + 1] & 0xFFFF)});
}

/* A packet's vertices' box on the screen (1/16 pixels). */
struct Box {
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
};
Box BoxOf(const u32* words, const Packet& p) {
    std::vector<std::pair<double, double>> points;
    Points(words, p, points);
    Box b;
    for (const auto& [x, y] : points) {
        b.x0 = std::min(b.x0, x); b.x1 = std::max(b.x1, x);
        b.y0 = std::min(b.y0, y); b.y1 = std::max(b.y1, y);
    }
    return b;
}

/* `to` (an unmatched packet, the newer's copy) moved back `t` of the way
   along the screen motion of its neighbour (older `na`, newer `nb`). */
bool Carry(const u32* na, const u32* nb, const Packet& n, u32* to, const Packet& p, float t,
           const GsBlendOptions& o) {
    std::vector<std::pair<double, double>> from, back;
    Points(nb, n, from);        // newer
    Points(na, n, back);        // older
    if (from.empty() || from.size() != back.size()) return false;
    const Affine f = Fit(from, back);       // newer to older
    const double limit = o.jump * 16.0;
    const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
    for (int pass = 0; pass < 2; ++pass) {          // (looked at whole first: none moving past `jump`)
        for (u32 q = 0; q < nloop * nreg; ++q) {
            if (!IsXyz(u32(p.hi >> (4 * (q % nreg))) & 0xF)) continue;
            u32* w = to + 4 + 4 * q;
            const double x = double(w[0] & 0xFFFF), y = double(w[1] & 0xFFFF);
            double bx, by;
            f.Apply(x, y, &bx, &by);
            if (pass == 0) {
                if (std::fabs(bx - x) > limit || std::fabs(by - y) > limit) return false;
                continue;
            }
            const double nx = std::clamp(bx + (x - bx) * t, 0.0, 65535.0), ny = std::clamp(by + (y - by) * t, 0.0, 65535.0);
            w[0] = (w[0] & ~0xFFFFu) | u32(std::lround(nx));
            w[1] = (w[1] & ~0xFFFFu) | u32(std::lround(ny));
        }
    }
    return true;
}

/* BlendGsPictures' last step on a path: the older's unmatched vertex
   packets moved forward and put into `out` (its words and the 'W' entries
   that hold them). */
void Add(const GsPicture& older, const GsPicture& newer, const std::vector<Packet>& a, const std::vector<Packet>& b,
         const std::vector<std::pair<u32, u32>>& pairs, const std::vector<int>& kinds, int path, float t,
         const GsBlendOptions& options, GsPicture& out, GsBlendStats& stats) {
    auto vertices = [](const Packet& p) {
        if ((u32(p.lo >> 58) & 3) != 0) return false;
        const u32 nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
        for (u32 r = 0; r < nreg; ++r)
            if (IsXyz(u32(p.hi >> (4 * r)) & 0xF)) return true;
        return false;
    };
    std::vector<int> partner(a.size(), -1);
    for (const auto& [i, j] : pairs) partner[i] = int(j);
    std::vector<u32> run(a.size(), 0);
    for (std::size_t i = 0, r = 0; i < a.size(); ++i) {
        run[i] = u32(r);
        if (!vertices(a[i]) && (u32(a[i].lo >> 58) & 3) != 2) ++r;
    }
    struct Insert {
        u32 at;                 // after the newer's packet ending here (its words' offset)
        std::size_t after;      // ...which is b[after]
        std::vector<u32> words;
    };
    std::vector<Insert> inserts;
    const u32* ow = older.paths[path].data();
    const u32* nw = newer.paths[path].data();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (partner[i] >= 0 || !vertices(a[i])) continue;
        const Box mine = BoxOf(ow + a[i].first, a[i]);
        int near = -1;
        double best = 1e300;
        for (std::size_t d = 1; d <= 8; ++d) {
            for (const std::size_t n : {i - d, i + d}) {
                if (n >= a.size() || partner[n] < 0 || kinds[partner[n]] != 4 || run[n] != run[i]) continue;
                const Box theirs = BoxOf(ow + a[n].first, a[n]);
                const double w = theirs.x1 - theirs.x0, h = theirs.y1 - theirs.y0;
                if (mine.x0 < theirs.x0 - w || mine.x1 > theirs.x1 + w || mine.y0 < theirs.y0 - h || mine.y1 > theirs.y1 + h)
                    continue;
                const double dx = (mine.x0 + mine.x1 - theirs.x0 - theirs.x1) / 2,
                             dy = (mine.y0 + mine.y1 - theirs.y0 - theirs.y1) / 2;
                if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; near = int(n); }
            }
        }
        if (near < 0) continue;
        // In after the neighbour's newer packet where it came before this one,
        // else after the packet before that (the same run: no state between).
        const std::size_t j = std::size_t(partner[near]);
        std::size_t after;
        if (std::size_t(near) < i) after = j;
        else if (j > 0) after = j - 1;
        else continue;
        Insert in{b[after].first + b[after].size, after, std::vector<u32>(ow + a[i].first, ow + a[i].first + a[i].size)};
        // Older to newer: `nb` the older's, `na` the newer's, moved 1 - t back from where it is (the older's).
        if (!Carry(nw + b[j].first, ow + a[near].first, a[near], in.words.data(), a[i], 1.0f - t, options)) continue;
        inserts.push_back(std::move(in));
    }
    if (inserts.empty()) return;
    std::stable_sort(inserts.begin(), inserts.end(), [](const Insert& x, const Insert& y) { return x.at < y.at; });
    // EOP: the packet before an insert's moves to the insert (the last of them).
    std::vector<u32>& words = out.paths[path];
    for (std::size_t k = 0; k < inserts.size(); ++k) {
        const bool last = k + 1 == inserts.size() || inserts[k + 1].at != inserts[k].at;
        u32& tag = words[b[inserts[k].after].first];
        if (last && (tag & 0x8000u)) {
            tag &= ~0x8000u;
            inserts[k].words[0] |= 0x8000u;
        } else {
            inserts[k].words[0] &= ~0x8000u;
        }
    }
    std::vector<u32> merged;
    merged.reserve(words.size() + 1024);
    std::size_t k = 0;
    for (u32 at = 0; at <= words.size(); ++at) {
        while (k < inserts.size() && inserts[k].at == at) {
            merged.insert(merged.end(), inserts[k].words.begin(), inserts[k].words.end());
            stats.added += 1;
            ++k;
        }
        if (at < words.size()) merged.push_back(words[at]);
    }
    // The entries' words: an insert at `p` belongs to the entry holding the
    // word before it (first < p <= first + count).
    auto shift = [&inserts](u32 offset) {
        std::size_t n = 0;
        for (const Insert& in : inserts)
            if (in.at <= offset) n += in.words.size();
        return u32(offset + n);
    };
    for (GsPicture::Entry& e : out.entries) {
        if (e.kind != 'W' || e.path != path + 1) continue;
        const u32 first = e.first == 0 ? 0 : shift(e.first), end = shift(e.first + e.count);
        e.first = first;
        e.count = end - first;
    }
    words.swap(merged);
}

}  // namespace

GsBlendStats BlendGsPictures(const GsPicture& older, const GsPicture& newer, float t, GsPicture& out,
                             const GsBlendOptions& options) {
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
        // PS2HOST_BLEND_HIDE (a test's): packets of these kinds drawn as
        // nothing (every vertex at the first's) - 1 not matched, 2 jumped,
        // 4 blended, 8 the same in both, 16 carried (no match, moved as a
        // neighbour moved), 64 snapped (its thing the newer's whole); 32: the older's
        // with no match not added.
        static const int hide = std::getenv("PS2HOST_BLEND_HIDE") ? std::atoi(std::getenv("PS2HOST_BLEND_HIDE")) : 0;
        auto collapse = [&](const Packet& p) {
            if ((u32(p.lo >> 58) & 3) != 0) return;
            const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
            u32* w = out.paths[path].data() + p.first + 4;
            const u32* first = nullptr;
            for (u32 q = 0; q < nloop * nreg; ++q) {
                if (!IsXyz(u32(p.hi >> (4 * (q % nreg))) & 0xF)) continue;
                if (!first) first = w + 4 * q;
                else { w[4 * q] = first[0]; w[4 * q + 1] = first[1]; }
            }
        };
        std::vector<int> match(b.size(), -1);
        for (const auto& [i, j] : pairs) match[j] = int(i);
        std::vector<int> kinds(b.size(), 0);
        std::vector<u32> xyzs(b.size(), 0);
        std::vector<u32> run(b.size(), 0);      // which run of vertex packets between state packets
        u32 runs = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const Packet& p = b[j];
            const u32 flg = u32(p.lo >> 58) & 3;
            const u32 nloop = u32(p.lo & 0x7FFF), nreg = u32(p.lo >> 60) ? u32(p.lo >> 60) : 16;
            u32 xyz = 0;
            if (flg == 0)
                for (u32 r = 0; r < nreg; ++r) xyz += IsXyz(u32(p.hi >> (4 * r)) & 0xF) ? nloop : 0;
            const u32* y = newer.paths[path].data() + p.first;
            const u32* x = match[j] >= 0 ? older.paths[path].data() + a[match[j]].first : nullptr;
            if (x) ++stats.matched;
            run[j] = runs;
            if (!xyz) {
                if (flg != 2) ++runs;           // a state packet (an IMAGE's data is none)
                continue;
            }
            int kind;
            u32 vertices = 0;
            if (!x) kind = 1;
            else if (std::memcmp(x, y, p.size * 4) == 0) { stats.still += xyz; kind = 8; }
            else {
                const int how = BlendPacked(x, y, out.paths[path].data() + p.first, p, t, options, &vertices);
                if (how == kBlended) { stats.blended += vertices; kind = 4; }
                else if (how == kLoose) { ++stats.loose; kind = 1; }     // (carried below, as one with no match)
                else { ++stats.jumped; stats.jumped_vertices += xyz; kind = 2; }
            }
            kinds[j] = kind;
            xyzs[j] = xyz;
        }
        // A thing that cannot be blended whole is the newer's whole: in a run
        // of vertex packets under one state (a material of a thing) with
        // under 70% of its vertices blended or still, none is blended, carried
        // or added - drawn as the newer picture has it, it steps once; in
        // pieces, some blended and some not, it tore (a white car coming in
        // fast at the screen's corner, its panels strewn over the road).
        {
            std::vector<u32> total(runs + 1, 0), clean(runs + 1, 0);
            for (std::size_t j = 0; j < b.size(); ++j) {
                total[run[j]] += xyzs[j];
                if (kinds[j] == 4 || kinds[j] == 8) clean[run[j]] += xyzs[j];
            }
            // PS2HOST_BLEND_SNAP: the share blended or still, in percent, a thing needs (70:
            // the pile-up's strays -60% for 10% of its vertices stepping; 90 stepped half).
            static const u32 need = std::getenv("PS2HOST_BLEND_SNAP") ? u32(std::atoi(std::getenv("PS2HOST_BLEND_SNAP"))) : 70;
            for (std::size_t j = 0; j < b.size(); ++j) {
                if (!xyzs[j] || u64(100) * clean[run[j]] >= u64(need) * total[run[j]]) continue;
                if (kinds[j] == 4) {
                    std::memcpy(out.paths[path].data() + b[j].first, newer.paths[path].data() + b[j].first, b[j].size * 4);
                    stats.blended -= xyzs[j];
                }
                stats.snapped += xyzs[j];
                kinds[j] = 64;
            }
        }
        // A packet with no match - most often a piece VU1's clipper cut
        // anew at the screen's edge, its vertices not the older's - moved
        // as the nearest blended packet beside it in the list moved (the
        // same mesh's next piece, mostly): that one's vertices, newer to
        // older, fitted as an affine map of the screen; this one's vertices
        // taken back `t` of the way along it. Left where it was in the newer
        // picture, its texture stood out of line with the blended road
        // round it (a strip of the crossing's stripes across the road).
        for (std::size_t j = 0; j < b.size(); ++j) {
            if (kinds[j] != 1) continue;
            // Under the same state (no state packet between: the same
            // material - carried by another thing's motion, a crossing's
            // stripe went 40 pixels down the road), the one nearest on the
            // screen within eight, and only where this one lies within it
            // widened by its own size: an affine map taken far past the
            // points it was fitted to is not the perspective's (a piece of
            // pavement thrown up over the kerb).
            const Box mine = BoxOf(newer.paths[path].data() + b[j].first, b[j]);
            int near = -1;
            double best = 1e300;
            for (std::size_t d = 1; d <= 8; ++d) {
                for (const std::size_t n : {j - d, j + d}) {
                    if (n >= b.size() || kinds[n] != 4 || run[n] != run[j]) continue;      // (j - d wraps past 0)
                    const Box theirs = BoxOf(newer.paths[path].data() + b[n].first, b[n]);
                    const double w = theirs.x1 - theirs.x0, h = theirs.y1 - theirs.y0;
                    if (mine.x0 < theirs.x0 - w || mine.x1 > theirs.x1 + w || mine.y0 < theirs.y0 - h ||
                        mine.y1 > theirs.y1 + h)
                        continue;
                    const double dx = (mine.x0 + mine.x1 - theirs.x0 - theirs.x1) / 2,
                                 dy = (mine.y0 + mine.y1 - theirs.y0 - theirs.y1) / 2;
                    if (dx * dx + dy * dy < best) { best = dx * dx + dy * dy; near = int(n); }
                }
            }
            if (near >= 0 && Carry(older.paths[path].data() + a[match[near]].first,
                                   newer.paths[path].data() + b[near].first, b[near],
                                   out.paths[path].data() + b[j].first, b[j], t, options))
                kinds[j] = 16, ++stats.carried;
        }
        for (std::size_t j = 0; j < b.size(); ++j)
            if (hide & kinds[j]) collapse(b[j]);
        // The older's packets with no match drawn too, moved forward `t` of
        // the way along a blended neighbour's motion (as carried, older to
        // newer): what goes out of the picture, or VU1's clipper cut anew,
        // covers its part of the screen still (the camera turning, a strip
        // along the screen's edge was left with nothing drawn - the glow
        // behind it showing white). Each goes in after the newer packet its
        // neighbour's place in the list is (its state the same).
        if (!(hide & 32)) Add(older, newer, a, b, pairs, kinds, path, t, options, out, stats);
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
