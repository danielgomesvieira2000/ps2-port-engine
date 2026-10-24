#pragma once
/*
 * A game's display lists - the VIF1 chains it sends each picture (Vif1Chain,
 * SetVif1ChainFilter) - blended: a picture between two of its steps made from
 * the lists of the two, for a game whose drawing is written into one list
 * while its update runs (Resident Evil Outbreak), where no draw can be run
 * apart from the step (skill ps2-high-fps).
 *
 * Each list is cut into the things it draws - a segment from each UNPACK to
 * one VU address (`segment_address`: where the game puts each thing's
 * matrices; not one to the double buffer, FLG) to the next. Each of the newer
 * list's segments is matched with one of the older list's of the same VIF
 * codes - the one in its place, or, where the game drew its things in another
 * order (two billboards that traded places), the one whose first values are
 * nearest. In a matched pair the UNPACK data (not those to the double buffer,
 * FLG: a batch's own, read as more than coordinates) that are plausible
 * floats (0, or 1e-6 to 1e9 either way - a projection's translations reach
 * 2e7: a GIF tag, a count, a colour read as a far smaller or odd float) and
 * differ by less than `cut` are blended - unless one of the pair's values
 * jumps by more than `jump` of the largest in its lane (x, y, z or w) of its
 * UNPACK: a different thing in that place, a sign turned over (w through 0:
 * vertices thrown across the screen), a flickering light's billboard - or if
 * a value within 0-1 that was moving moves by more than half (a texture's
 * offset wrapping round). That segment (stepping as the game does), a segment with no match
 * (what came into the picture) and every other word are the newer list's.
 * Where the camera's own segment jumps (`cut_segment`: its view's rows, each
 * by its own length), or else where many segments jump or move at once
 * (`cut_share` of them), the camera has cut: nothing is blended.
 *
 * A pair's blend is planned once (PlanListBlend: the words to blend, from and
 * to) and made for each picture between the two steps (ApplyListBlend).
 */

#include "ps2host/run.h"

namespace ps2host {

struct ListBlendOptions {
    float cut = 100000.0f;      // a value changing by more: not motion (not blended)
    int segment_address = 0;    // a segment starts at each UNPACK to this VU address (-1: one segment)
    float jump = 0.5f;          // of a lane's largest value: a jump - the segment not blended
    float cut_share = 0.2f;     // segments jumping or moved, of those matched: the camera cut
    float moved_distance = 0.02f;   // a match out of its place: its first values this far a word at most
    // The segment that is the camera (its view's matrices: Outbreak's 1, the
    // first after the list's start): its jump the cut, and only it (a flash
    // of light makes many things jump at once); -1: `cut_share` decides.
    int cut_segment = -1;
    // Its view: these rows (quadwords) of its first UNPACK of floats, each
    // turned or moved by more than `jump` of its own length the cut (not its
    // lights' rows: a shot's flash). The rest of it blended whole.
    unsigned view_first = 0, view_end = ~0u;
};

struct ListBlend {
    u64 codes = 0;          // VIF codes in the newer list
    u64 blended = 0;        // words blended
    u32 segments = 0;       // the newer list's segments
    u32 moved = 0;          // matched out of their place
    u32 unmatched = 0;      // with no match
    u32 jumps = 0;          // matched but jumping: not blended
    bool cut = false;       // the camera cut: nothing blended
    bool whole = false;     // the lists' codes the same to their end
    // How near the blend came to its limits (a trace's, to find the moves
    // that look wrong): the largest move of a blended value, as a share of
    // the jump that would have stepped its segment (1: it would have), and
    // whose segment; segments blended with a move above half of it; the
    // camera's view's largest row move as a share of the cut's (1: a cut).
    float worst = 0.0f;
    u32 worst_segment = 0;
    u32 near = 0;
    float view = 0.0f;
};

struct ListBlendPlan {
    std::vector<u32> at;    // the newer list's words to blend
    std::vector<float> from, to;
    ListBlend stats;
};

/* The blend of `a` (older) into `b` (newer), planned. `moving`: the words of
   `a` blended the step before (the plan that made `a`'s `at`, in order) -
   a value within 0-1 moving by more than half jumps only where it was moving
   then (a texture's offset scrolling round), not where it was still (a
   light switched on, 0 to 1: blended with the rest of its light). None: no
   such jumps. */
ListBlend PlanListBlend(const Vif1Chain& a, const Vif1Chain& b, const ListBlendOptions& options, ListBlendPlan& plan,
                        const std::vector<u32>* moving = nullptr);

/* `words` (b's own, from `offset` in them) made `t` of the way from a's (0:
   a's values, 1: b's). */
void ApplyListBlend(const ListBlendPlan& plan, float t, std::vector<u32>& words, size_t offset = 0);

/* Where `words` (a VIF1 list) writes the GS registers `registers` (0x4C:
   FRAME_1, and so on) with A+D in its DIRECTs' GIF packets: each value's
   first word. A list made for one of a double buffer's halves draws to its
   own half where it names it (Outbreak's light composited after its shadow
   pass): a picture between steps drawn into the other half has them
   changed. */
void GsRegisterWrites(const std::vector<u32>& words, const std::vector<u32>& registers, std::vector<u32>& at);

/* Both at once: `out` = `b` blended `t` of the way from `a`. */
ListBlend BlendVif1Chains(const Vif1Chain& a, const Vif1Chain& b, float t, float cut, Vif1Chain& out);

}  // namespace ps2host
