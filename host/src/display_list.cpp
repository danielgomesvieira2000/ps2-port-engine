/*
 * Display lists blended (ps2host/display_list.h).
 */

#include "ps2host/display_list.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2host {
namespace {

struct Code {
    u32 at;         // the code's word
    u32 code;
    u32 words;      // its data's, after it
    bool floats;    // an UNPACK V4-32: four floats a vector
};

struct Segment {
    u64 key;        // its codes'
    u32 first, end; // its codes
    // Its work alone: its UNPACKs, STCYCLs and microprogram calls (MSCAL,
    // MSCALF, MSCNT), to its last call - not the next thing's set-up that
    // may come at its end (its registers' DIRECT, a texture's DIRECTHL, an
    // MPG with BASE and OFFSET, a last UNPACK), which changes its key from
    // step to step though the thing is the same (File #1's floor items,
    // lamp glows, shadows: unmatched, stepped while the room blended).
    u64 work;
};

bool Unpack(u32 code);

u64 WorkKey(const std::vector<Code>& codes, u32 first, u32 end) {
    u32 last = end;
    for (u32 j = first; j < end; ++j) {
        const u32 cmd = (codes[j].code >> 24) & 0x7F;
        if (cmd == 0x14 || cmd == 0x15 || cmd == 0x17) last = j + 1;
    }
    u64 key = 1469598103934665603ull;
    for (u32 j = first; j < last; ++j) {
        const u32 code = codes[j].code, cmd = (code >> 24) & 0x7F;
        if (Unpack(code) || cmd == 0x01 || cmd == 0x14 || cmd == 0x15 || cmd == 0x17) key = (key ^ code) * 1099511628211ull;
    }
    return key;
}

/* A VIF code's data, in words after it. `cl`, `wl`: STCYCL's cycle - an
   UNPACK writing more than it reads (WL > CL, filling: Mister Mosquito's)
   reads CL vectors of each WL it writes. */
u32 DataWords(u32 code, u32 cl, u32 wl) {
    const u32 cmd = (code >> 24) & 0x7F;
    const u32 num = (code >> 16) & 0xFF;
    const u32 imm = code & 0xFFFF;
    if ((cmd & 0x60) == 0x60) {         // UNPACK: vn, vl - num vectors written
        const u32 vl = cmd & 3, vn = (cmd >> 2) & 3;
        u32 vectors = num ? num : 256;
        if (wl > cl) vectors = cl * (vectors / wl) + std::min(vectors % wl, cl);
        const u32 bits = (32u >> vl) * (vn + 1) * vectors;
        return (bits + 31) / 32;
    }
    switch (cmd) {
    case 0x20: return 1;                // STMASK
    case 0x30: case 0x31: return 4;     // STROW, STCOL
    case 0x4A: return 2 * (num ? num : 256);                        // MPG
    case 0x50: case 0x51: return 4 * (imm ? imm : 65536);           // DIRECT, DIRECTHL
    default: return 0;
    }
}

bool Unpack(u32 code) { return ((code >> 24) & 0x60) == 0x60; }

void Parse(const std::vector<u32>& w, std::vector<Code>& codes) {
    codes.clear();
    size_t i = 0;
    u32 cl = 1, wl = 1;     // STCYCL's (the VIF's reset values are 0; 1/1 reads as written)
    while (i < w.size()) {
        const u32 code = w[i];
        const u32 cmd = (code >> 24) & 0x7F;
        if (cmd == 0x01) {
            cl = code & 0xFF;
            wl = (code >> 8) & 0xFF;
        }
        const u32 words = DataWords(code, cl, wl);
        if (i + 1 + words > w.size()) break;
        codes.push_back({u32(i), code, words, Unpack(code) && (cmd & 0x0F) == 0x0C});
        i += 1 + words;
    }
}

void Split(const std::vector<Code>& codes, int address, std::vector<Segment>& segments) {
    segments.clear();
    u32 first = 0;
    u64 key = 1469598103934665603ull;
    for (u32 j = 0; j < codes.size(); ++j) {
        const u32 code = codes[j].code;
        // (an UNPACK's address with FLG set is the double buffer's: a batch of
        // vertices within a thing, not a thing's start)
        if (j > first && address >= 0 && Unpack(code) && !(code & 0x8000) && int(code & 0x3FF) == address) {
            segments.push_back({key, first, j, WorkKey(codes, first, j)});
            first = j;
            key = 1469598103934665603ull;
        }
        key = (key ^ code) * 1099511628211ull;
    }
    if (first < codes.size()) segments.push_back({key, first, u32(codes.size()), WorkKey(codes, first, u32(codes.size()))});
}

bool Plausible(float v) { return v == 0.0f || (std::fabs(v) > 1e-6f && std::fabs(v) < 1e9f); }

float Float(u32 w) {
    float v;
    std::memcpy(&v, &w, 4);
    return v;
}

/* How far apart two segments' first UNPACKs are (each thing's matrices):
   0 the same, ~1 a value a word unlike; `words` the words that differ. */
float Distance(const std::vector<u32>& a, const Code& ca, const std::vector<u32>& b, const Code& cb, u32* words) {
    *words = 0;
    if (!ca.floats || ca.words != cb.words) return 0.0f;
    float d = 0.0f;
    for (u32 k = 1; k <= ca.words; ++k) {
        const u32 wa = a[ca.at + k], wb = b[cb.at + k];
        if (wa == wb) continue;
        ++*words;
        const float x = Float(wa), y = Float(wb);
        if (!Plausible(x) || !Plausible(y)) { d += 1.0f; continue; }
        d += std::fabs(y - x) / (std::fabs(x) + std::fabs(y) + 1.0f);
    }
    return d;
}

/* A matched pair's words to blend, added to `plan` - false (none added) if
   one jumps. The newer list's words [skip_lo, skip_hi) are left out (the
   camera's view, planned apart: PlanCamera). */
// The plan before's changes (ListBlendPlan::changed), by the older list's
// word (the plan before's newer list): this step's PlanListBlend's.
thread_local const std::vector<std::pair<u32, float>>* t_before = nullptr;

bool ChangedBefore(u32 at, float* change) {
    if (!t_before || t_before->empty()) return false;
    const auto p = std::lower_bound(t_before->begin(), t_before->end(), std::make_pair(at, -1e30f));
    if (p == t_before->end() || p->first != at) return false;
    *change = p->second;
    return true;
}

bool PlanPair(const std::vector<u32>& a, const std::vector<Code>& ca, u32 sa, const std::vector<u32>& b,
              const std::vector<Code>& cb, const Segment& s, const ListBlendOptions& o, ListBlendPlan& plan,
              const std::vector<u32>* moving, u32 skip_lo = 0, u32 skip_hi = 0, float* worst = nullptr,
              u32 segment = 0, const std::vector<u32>* pairing = nullptr) {
    // The older list's code for the newer's j: the same place in the
    // segment, or `pairing`'s (~0u: none).
    const auto older = [&](u32 j) { return pairing ? (*pairing)[j - s.first] : sa + (j - s.first); };
    const size_t mark = plan.at.size(), changed_mark = plan.changed.size();
    float most = 0.0f;              // the largest move, as a share of its jump
    const auto skipped = [&](u32 at) { return at >= skip_lo && at < skip_hi; };
    for (u32 j = s.first; j < s.end; ++j) {
        if (older(j) == ~0u) continue;
        const Code& y = cb[j];
        const Code& x = ca[older(j)];
        // (Not an UNPACK to the double buffer, FLG: a batch's own data -
        // model-space vertices, the same each step, and where not, values
        // the program reads as other than coordinates: Outbreak's decal
        // under a character took its CLUT from one, blended a black, then a
        // white square on the floor. ListBlendOptions::blend_flg: blended.)
        if (!y.floats || !y.words || ((y.code & 0x8000) && !o.blend_flg)) continue;
        if (std::memcmp(&a[x.at + 1], &b[y.at + 1], y.words * 4) == 0) continue;
        float scale[4] = {0, 0, 0, 0};
        for (u32 k = 0; k < y.words; ++k) {
            if (skipped(y.at + 1 + k)) continue;
            const float fa = Float(a[x.at + 1 + k]), fb = Float(b[y.at + 1 + k]);
            if (Plausible(fa)) scale[k & 3] = std::max(scale[k & 3], std::fabs(fa));
            if (Plausible(fb)) scale[k & 3] = std::max(scale[k & 3], std::fabs(fb));
        }
        u32 row_skipped = 0;        // a row stepped alone: its words up to here left the newer step's
        const auto steps_alone = [&](u32 row) {
            for (const auto& r : o.step_rows)
                if (r.words == y.words && row >= r.first_row && row < r.end_row) return true;
            return false;
        };
        for (u32 k = 0; k < y.words; ++k) {
            const u32 wa = a[x.at + 1 + k], wb = b[y.at + 1 + k];
            if (k < row_skipped || wa == wb || skipped(y.at + 1 + k)) continue;
            const float fa = Float(wa), fb = Float(wb);
            if (!Plausible(fa) || !Plausible(fb)) continue;
            float before = 0.0f;
            const bool had_before = ChangedBefore(x.at + 1 + k, &before);
            const float change = fb - fa;
            bool past_cut = !(std::fabs(change) < o.cut);
            if (past_cut && o.cut_steady && had_before && (before > 0) == (change > 0) &&
                std::fabs(before) * 3.0f >= std::fabs(change) && std::fabs(before) <= 3.0f * std::fabs(change))
                past_cut = false;           // (moving on steadily: blended)
            if (past_cut && !o.cut_steady) continue;
            plan.changed.push_back({y.at + 1 + k, change});
            // A texture's cell (a multiple of 1/64 within 0-1 at both ends:
            // an animated sprite's cell in its atlas, File #2's smoke, 0.25
            // and 0.125 steps) is the newer step's: blended, half of two
            // cells, and at the atlas's wrap an unrelated one. The rest of
            // the thing blends on (the puff rising smoothly).
            if (fa >= 0.0f && fa <= 1.0f && fb >= 0.0f && fb <= 1.0f && std::floor(fa * 64.0f) == fa * 64.0f &&
                std::floor(fb * 64.0f) == fb * 64.0f)
                continue;
            // A value moving on as it moved the step before is motion, not a jump.
            const bool steady = had_before && (before > 0) == (fb - fa > 0) &&
                                std::fabs((fb - fa) - before) <= 0.25f * std::fabs(fb - fa);
            // (one word so is enough: a flickering light's billboard, its w
            // jumping, blended is a beam across the room). A value within 0-1
            // at both ends, moving the step before, that moves by more than
            // half is an offset wrapping round - a texture scrolled 1/24 a
            // step, from 23/24 to 0 (Below Freezing Point's lights): blended,
            // half a turn away for a picture, a flash. Its lane's largest (a
            // translation's 1e6 in the same UNPACK) would never call it a
            // jump. Still the step before, it is a switch (a light on, 0 to
            // 1, in every lit thing's values): blended with the rest - stepped
            // in some things and blended in others, the layers of one wall
            // apart for a picture.
            // (A wrap turns a small steady motion back - +1/24, then -23/24 - not
            // any value of 0-1 that moved and moves by more than half: a bone's
            // rotation, a light's direction, a prop's row moving fast stepped
            // whole characters.)
            (void)moving;
            const bool wraps = fa >= 0.0f && fa <= 1.0f && fb >= 0.0f && fb <= 1.0f && std::fabs(fb - fa) > 0.5f &&
                               had_before && (before > 0) != (fb - fa > 0) && std::fabs(before) < 0.5f;
            const bool jumps = wraps || past_cut || (!steady && std::fabs(fb - fa) > o.jump * scale[k & 3]);
            if (jumps && steps_alone(k / 4)) {
                // This row alone the newer step's (its words already planned taken back).
                const u32 row_at = y.at + 1 + (k / 4) * 4;
                while (plan.at.size() > mark && plan.at.back() >= row_at) {
                    plan.at.pop_back();
                    plan.from.pop_back();
                    plan.to.pop_back();
                    plan.segment.pop_back();
                }
                row_skipped = (k / 4 + 1) * 4;
                continue;
            }
            if (jumps) {
                static const bool jump_log = std::getenv("PS2HOST_BLEND_JUMPS") != nullptr;
                if (jump_log)
                    std::fprintf(stderr, "[jump] segment %u words %u word %u: %g -> %g (scale %g, before %g%s)%s%s\n", segment, y.words, k, fa, fb,
                                 scale[k & 3], had_before ? before : 0.0f, had_before ? "" : " none", wraps ? " wraps" : "", past_cut ? " past the cut" : "");
                plan.at.resize(mark);
                plan.from.resize(mark);
                plan.to.resize(mark);
                plan.segment.resize(mark);
                // (its changes kept all the same: the next step's steady test)
                for (u32 q = j; q < s.end; ++q) {
                    if (older(q) == ~0u) continue;
                    const Code& yq = cb[q];
                    const Code& xq = ca[older(q)];
                    if (!yq.floats || !yq.words || ((yq.code & 0x8000) && !o.blend_flg)) continue;
                    for (u32 r = q == j ? k + 1 : 0; r < yq.words; ++r) {
                        const float ga = Float(a[xq.at + 1 + r]), gb = Float(b[yq.at + 1 + r]);
                        if (a[xq.at + 1 + r] != b[yq.at + 1 + r] && Plausible(ga) && Plausible(gb))
                            plan.changed.push_back({yq.at + 1 + r, gb - ga});
                    }
                }
                (void)changed_mark;
                return false;
            }
            if (scale[k & 3] > 0.0f && !steady) most = std::max(most, std::fabs(fb - fa) / (o.jump * scale[k & 3]));
            plan.at.push_back(y.at + 1 + k);
            plan.from.push_back(fa);
            plan.to.push_back(fb);
            plan.segment.push_back(segment);
        }
    }
    if (worst) *worst = most;
    return true;
}

/* The camera's segment: its view (rows view_first to view_end of its first
   UNPACK of floats) blended whatever it does - its cut is judged apart
   (ViewJumps) - and the rest as any thing's, left as the game steps it where
   a value jumps: its lights' rows. A light switched on (0 to 1 in every
   lit thing's segment, stepping there) blended in the camera's alone was
   between on and off for a picture: Below Freezing Point's gate at 144. */
void PlanCamera(const std::vector<u32>& a, const std::vector<Code>& ca, u32 sa, const std::vector<u32>& b,
                const std::vector<Code>& cb, const Segment& s, const ListBlendOptions& o, ListBlendPlan& plan,
                const std::vector<u32>* moving, float* worst, u32 segment) {
    u32 lo = 0, hi = 0;
    for (u32 j = s.first; j < s.end; ++j) {
        const Code& y = cb[j];
        const Code& x = ca[sa + (j - s.first)];
        if (!y.floats || !y.words || (y.code & 0x8000)) continue;
        const u32 rows = y.words / 4;
        lo = y.at + 1 + 4 * std::min(o.view_first, rows);
        hi = y.at + 1 + 4 * std::min(o.view_end, rows);
        for (u32 at = lo; at < hi; ++at) {
            const u32 wa = a[x.at + (at - y.at)], wb = b[at];
            if (wa == wb) continue;
            const float fa = Float(wa), fb = Float(wb);
            // (with cut_steady, the view whatever its move: its cut is ViewJumps')
            if (!Plausible(fa) || !Plausible(fb) || (!o.cut_steady && !(std::fabs(fb - fa) < o.cut))) continue;
            plan.at.push_back(at);
            plan.from.push_back(fa);
            plan.to.push_back(fb);
            plan.segment.push_back(segment);
        }
        break;
    }
    PlanPair(a, ca, sa, b, cb, s, o, plan, moving, lo, hi, worst, segment);
}

/* The camera's segment's view (rows `first` to `end` of its first UNPACK of
   floats) turned or moved by more than `jump` of a row's own length - its
   axes' rows (`first` to `axes_end`) by their length together: the camera
   cut. Its lanes' largest are its translations' (a projection's reach 2e7),
   by which no turn is a jump - a cut from one corner of a room to another
   blended through the walls. */
bool ViewJumps(const std::vector<u32>& a, const std::vector<Code>& ca, u32 sa, const std::vector<u32>& b,
               const std::vector<Code>& cb, const Segment& s, float jump, u32 first, u32 axes_end, u32 end,
               float* share) {
    for (u32 j = s.first; j < s.end; ++j) {
        const Code& y = cb[j];
        const Code& x = ca[sa + (j - s.first)];
        if (!y.floats || !y.words || x.words != y.words) continue;
        bool cut = false;
        // (rows of one group: the axes, then each row on alone)
        const auto judge = [&](u32 from, u32 to) {
            float d = 0.0f, la = 0.0f, lb = 0.0f;
            bool plausible = true;
            for (u32 k = from * 4; k < to * 4 && k < y.words; ++k) {
                const float fa = Float(a[x.at + 1 + k]), fb = Float(b[y.at + 1 + k]);
                plausible &= Plausible(fa) && Plausible(fb);
                d += (fb - fa) * (fb - fa);
                la += fa * fa;
                lb += fb * fb;
            }
            if (!plausible || std::max(la, lb) <= 1e-6f) return;
            *share = std::max(*share, std::sqrt(d / std::max(la, lb)) / jump);
            if (d > jump * jump * std::max(la, lb)) cut = true;
        };
        u32 row = first;
        if (axes_end > first) {
            if (u32(axes_end) * 4 <= y.words) judge(first, std::min(axes_end, end));
            row = std::min(axes_end, end);
        }
        for (; row < end && (row + 1) * 4 <= y.words; ++row) judge(row, row + 1);
        return cut;
    }
    return false;
}

}  // namespace

ListBlend PlanListBlend(const Vif1Chain& a, const Vif1Chain& b, const ListBlendOptions& o, ListBlendPlan& plan,
                        const std::vector<u32>* moving) {
    plan.at.clear();
    plan.from.clear();
    plan.to.clear();
    plan.segment.clear();
    // The changes the plan before found (this plan's, a step ago) for its
    // steady test, then this step's anew.
    static thread_local std::vector<std::pair<u32, float>> before;
    before.swap(plan.changed);
    plan.changed.clear();
    t_before = &before;
    struct Forget {
        ~Forget() { t_before = nullptr; }
    } forget;
    ListBlend& r = plan.stats;
    const float view_before = r.view;     // (this plan's a step ago: the camera's move then)
    r = ListBlend{};
    static thread_local std::vector<Code> ca, cb;
    static thread_local std::vector<Segment> sa, sb;
    Parse(a.words, ca);
    Parse(b.words, cb);
    Split(ca, o.segment_address, sa);
    Split(cb, o.segment_address, sb);
    r.codes = cb.size();
    r.segments = u32(sb.size());
    r.whole = ca.size() == cb.size() && std::equal(ca.begin(), ca.end(), cb.begin(),
                                                   [](const Code& x, const Code& y) { return x.code == y.code; });

    // The older list's segments by key, in order: (key, index).
    static thread_local std::vector<std::pair<u64, u32>> keys;
    keys.clear();
    for (u32 i = 0; i < sa.size(); ++i) keys.push_back({sa[i].key, i});
    std::sort(keys.begin(), keys.end());
    static thread_local std::vector<bool> used;
    used.assign(sa.size(), false);

    u32 expected = 0, matched = 0;
    bool camera_cut = false;        // (options.cut_segment's)
    static thread_local std::vector<std::pair<u32, u32>> unmatched;     // (newer segment, where its match was expected)
    unmatched.clear();
    for (u32 jb = 0; jb < sb.size(); ++jb) {
        const Segment& s = sb[jb];
        const bool camera = int(jb) == o.cut_segment;
        auto lo = std::lower_bound(keys.begin(), keys.end(), std::make_pair(s.key, 0u));
        auto hi = std::upper_bound(keys.begin(), keys.end(), std::make_pair(s.key, ~0u));
        // Its place first; else the nearest values among the unused nearby.
        u32 best = ~0u, best_words = 0;
        float best_d = 0.0f;
        const auto at = std::lower_bound(lo, hi, std::make_pair(s.key, expected));
        auto try_one = [&](u32 i) {
            if (used[i]) return;
            u32 words = 0;
            const float d = Distance(a.words, ca[sa[i].first], b.words, cb[s.first], &words);
            const bool nearer = best == ~0u || d < best_d ||
                                (d == best_d && (i > expected ? i - expected : expected - i) <
                                                    (best > expected ? best - expected : expected - best));
            if (nearer) { best = i; best_d = d; best_words = words; }
        };
        // The list's order first: the first unused one of its codes from where
        // the last match left off, a few places on at most.
        u32 in_order = ~0u;
        for (auto p = at; p != hi && p->second <= expected + o.skip_window; ++p)
            if (!used[p->second]) {
                in_order = p->second;
                break;
            }
        int n = 0;
        for (auto p = at; p != hi && n < 16; ++p, ++n) try_one(p->second);
        n = 0;
        for (auto p = at; p != lo && n < 16; ++n) try_one((--p)->second);
        // Out of its place, only a near match: one far (a thing whose own
        // match drew with other codes this step - more batches, its detail -
        // paired with another of the codes it now has) is not one. (The
        // camera is one of a kind, and its lights may flash.)
        const auto near = [&](float d, u32 words) { return d <= o.moved_distance * float(std::max(1u, words)); };
        if (in_order != ~0u && in_order != best) {
            u32 words = 0;
            const float d = Distance(a.words, ca[sa[in_order].first], b.words, cb[s.first], &words);
            // In order wins, unless it is out of its place and far while a near one is elsewhere.
            const bool near_elsewhere = best != ~0u && near(best_d, best_words);
            if (in_order == expected || near(d, words) || !near_elsewhere) best = in_order, best_d = d, best_words = words;
        }
        if (best != ~0u && best != expected && best != in_order && !camera && !near(best_d, best_words))
            best = ~0u;
        if (best == ~0u) {
            ++r.unmatched;
            if (camera) camera_cut = true;
            else unmatched.push_back({jb, expected});
            continue;
        }
        used[best] = true;
        ++matched;
        const bool moved = best != expected;
        if (moved) ++r.moved;
        expected = best + 1;
        if (camera) {
            // The camera judged by its view's rows alone (ViewJumps): by its
            // lanes, a translation passing through 0 - the camera moving on
            // steadily - was a jump, and every such step a cut (the camera
            // still for a step, then on).
            // (Out of its place is no cut: a thing before it came or went -
            // the virus gauge.)
            camera_cut |= ViewJumps(a.words, ca, sa[best].first, b.words, cb, s, o.jump, o.view_first,
                                    o.view_axes_end, o.view_end, &r.view);
            float worst = 0.0f;
            PlanCamera(a.words, ca, sa[best].first, b.words, cb, s, o, plan, moving, &worst, jb);
            if (worst > r.worst) r.worst = worst, r.worst_segment = jb;
            continue;
        }
        float worst = 0.0f;
        if (!PlanPair(a.words, ca, sa[best].first, b.words, cb, s, o, plan, moving, 0, 0, &worst, jb)) {
            ++r.jumps;
            continue;
        }
        if (worst > 0.5f) ++r.near;
        if (worst > r.worst) r.worst = worst, r.worst_segment = jb;
    }
    // Second: the things left unmatched by their work alone (Segment::work),
    // among the unused near where each was expected - the nearest values,
    // near enough - their UNPACKs paired in order.
    static thread_local std::vector<u32> pairing, older_unpacks;
    for (const auto& [jb, near_at] : unmatched) {
        const Segment& s = sb[jb];
        u32 best = ~0u, best_words = 0;
        float best_d = 0.0f;
        const u32 from = near_at > o.skip_window ? near_at - o.skip_window : 0;
        for (u32 i = from; i < sa.size() && i <= near_at + o.skip_window; ++i) {
            if (used[i] || sa[i].work != s.work || ca[sa[i].first].code != cb[s.first].code ||
                ca[sa[i].first].words != cb[s.first].words || !cb[s.first].floats)
                continue;
            u32 words = 0;
            const float d = Distance(a.words, ca[sa[i].first], b.words, cb[s.first], &words);
            if (best == ~0u || d < best_d) best = i, best_d = d, best_words = words;
        }
        if (best == ~0u || best_d > 0.1f * float(std::max(1u, best_words))) continue;
        older_unpacks.clear();
        for (u32 i = sa[best].first; i < sa[best].end; ++i)
            if (Unpack(ca[i].code)) older_unpacks.push_back(i);
        pairing.assign(s.end - s.first, ~0u);
        size_t n = 0;
        for (u32 j = s.first; j < s.end && n < older_unpacks.size(); ++j) {
            if (!Unpack(cb[j].code)) continue;
            if (ca[older_unpacks[n]].code == cb[j].code && ca[older_unpacks[n]].words == cb[j].words)
                pairing[j - s.first] = older_unpacks[n];
            ++n;
        }
        used[best] = true;
        ++matched;
        ++r.moved;
        --r.unmatched;
        float worst = 0.0f;
        if (!PlanPair(a.words, ca, sa[best].first, b.words, cb, s, o, plan, moving, 0, 0, &worst, jb, &pairing)) {
            ++r.jumps;
            continue;
        }
        if (worst > 0.5f) ++r.near;
        if (worst > r.worst) r.worst = worst, r.worst_segment = jb;
    }
    if (!unmatched.empty()) {
        // (the plan in the newer list's order again: its words, and its changes searched)
        static thread_local std::vector<u32> order;
        order.resize(plan.at.size());
        for (u32 i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](u32 x, u32 y) { return plan.at[x] < plan.at[y]; });
        ListBlendPlan sorted;
        sorted.at.reserve(order.size());
        for (u32 i : order) {
            sorted.at.push_back(plan.at[i]);
            sorted.from.push_back(plan.from[i]);
            sorted.to.push_back(plan.to[i]);
            sorted.segment.push_back(plan.segment[i]);
        }
        plan.at.swap(sorted.at);
        plan.from.swap(sorted.from);
        plan.to.swap(sorted.to);
        plan.segment.swap(sorted.segment);
        std::sort(plan.changed.begin(), plan.changed.end());
    }
    if (o.cut_segment >= 0 && r.view >= o.new_view && float(r.unmatched) >= o.new_share * float(r.segments))
        camera_cut = true;
    if (o.cut_segment >= 0 && o.view_sudden > 0.0f && r.view >= o.view_sudden_least &&
        r.view >= o.view_sudden * (view_before + 0.01f))
        camera_cut = true;
    if (o.cut_segment == -2 ? false
                            : o.cut_segment >= 0 ? camera_cut : matched && float(r.jumps + r.moved) > o.cut_share * float(matched)) {
        r.cut = true;
        plan.at.clear();
        plan.from.clear();
        plan.to.clear();
        plan.segment.clear();
    }
    r.blended = plan.at.size();
    return r;
}

void ApplyListBlend(const ListBlendPlan& plan, float t, std::vector<u32>& words, size_t offset) {
    for (size_t i = 0; i < plan.at.size(); ++i) {
        const size_t at = offset + plan.at[i];
        if (at >= words.size()) continue;
        const float v = plan.from[i] + t * (plan.to[i] - plan.from[i]);
        std::memcpy(&words[at], &v, 4);
    }
}

void GsRegisterWrites(const std::vector<u32>& words, const std::vector<u32>& registers, std::vector<u32>& at) {
    at.clear();
    std::vector<Code> codes;
    Parse(words, codes);
    for (const Code& c : codes) {
        const u32 cmd = (c.code >> 24) & 0x7F;
        if (cmd != 0x50 && cmd != 0x51) continue;
        // The DIRECT's GIF packets: their tags, and in a PACKED one each
        // A+D (descriptor 0xE) - its value's two words, its register's byte.
        u32 i = c.at + 1;
        const u32 end = c.at + 1 + c.words;
        while (i + 4 <= end) {
            const u64 tag = words[i] | u64(words[i + 1]) << 32;
            const u64 regs = words[i + 2] | u64(words[i + 3]) << 32;
            i += 4;
            const u32 nloop = u32(tag & 0x7FFF), flg = u32(tag >> 58) & 3;
            const u32 nreg = u32(tag >> 60) ? u32(tag >> 60) : 16;
            if (flg == 0) {
                for (u32 l = 0; l < nloop && i + 4 <= end; ++l)
                    for (u32 r = 0; r < nreg && i + 4 <= end; ++r, i += 4) {
                        const u32 reg = words[i + 2] & 0xFF;
                        if (((regs >> (4 * r)) & 0xF) == 0xE &&
                            std::find(registers.begin(), registers.end(), reg) != registers.end())
                            at.push_back(i);
                    }
            } else if (flg == 1) {
                i += (nloop * nreg * 2 + 3) & ~3u;
            } else {
                i += nloop * 4;
            }
        }
    }
}

void ListMotion::Picture(const std::vector<u32>& words, size_t end, int segment_address, unsigned long long frame,
                         float t, bool step, FILE* file) {
    if (!file) return;
    static thread_local std::vector<u32> part;
    part.assign(words.begin(), words.begin() + std::min(end, words.size()));
    static thread_local std::vector<Code> codes;
    static thread_local std::vector<Segment> segments;
    Parse(part, codes);
    Split(codes, segment_address, segments);
    std::unordered_map<u64, std::vector<float>> now;
    std::unordered_map<u64, u32> copies;
    for (u32 i = 0; i < segments.size(); ++i) {
        const Segment& s = segments[i];
        const u32 copy = copies[s.key]++;
        const u64 id = s.key * 31 + copy;
        const Code* first = nullptr;
        for (u32 j = s.first; j < s.end && !first; ++j)
            if (codes[j].floats && codes[j].words && !(codes[j].code & 0x8000)) first = &codes[j];
        if (!first) continue;
        std::vector<float>& v = now[id];
        v.resize(first->words);
        for (u32 k = 0; k < first->words; ++k) {
            const float f = Float(part[first->at + 1 + k]);
            v[k] = Plausible(f) && std::fabs(f) < 1e8f ? f : 0.0f;
        }
        const auto was = last_.find(id);
        if (was == last_.end() || was->second.size() != v.size()) continue;
        double d = 0.0, n = 0.0;
        for (size_t k = 0; k < v.size(); ++k) {
            d += double(v[k] - was->second[k]) * (v[k] - was->second[k]);
            n += double(v[k]) * v[k];
        }
        std::fprintf(file, "%llu,%u,%016llx,%u,%u,%.6g,%.6g,%.3f,%d\n", frame, i, (unsigned long long)s.key, copy,
                     first->words, std::sqrt(d), std::sqrt(n), t, step ? 1 : 0);
    }
    last_.swap(now);
}

ListBlend BlendVif1Chains(const Vif1Chain& a, const Vif1Chain& b, float t, float cut, Vif1Chain& out) {
    ListBlendOptions options;
    options.cut = cut;
    ListBlendPlan plan;
    PlanListBlend(a, b, options, plan);
    out = b;
    ApplyListBlend(plan, t, out.words);
    return plan.stats;
}

}  // namespace ps2host
