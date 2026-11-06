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
 * PS2HOST_BLEND_JUMPS=1 logs each thing that jumps: the word, its values,
 * its lane's scale and its change the step before.
 */

#include "ps2host/run.h"

#include <cstdio>
#include <unordered_map>
#include <vector>

namespace ps2host {

struct ListBlendOptions {
    float cut = 100000.0f;      // a value changing by more: not motion (not blended)
    // A value changing by more than `cut` blended all the same where it moves
    // on as it moved the step before (the same way, a third to three times
    // as far) - else its thing jumps (steps whole). A camera turning behind
    // the shoulder (Outbreak's RE4 style) moves every thing's screen
    // translations by 1e6 a step, steadily: past the cut, each prop stepped
    // while the room turned. A lamp's glow given another lamp's place moves
    // as far at once: left out alone, it was put between the two.
    bool cut_steady = false;
    // A value moving on as it moved the step before - the same way, its
    // change within this share of it of the change before - is motion, not
    // a jump (ListBlendPlan::changed). A camera easing to a stop (Outbreak's
    // RE4 camera, 0.8 of its move a step) changes by ~20% a step and more
    // where it passes 0: a quarter stepped its things as it came to rest.
    float steady = 0.25f;
    int segment_address = 0;    // a segment starts at each UNPACK to this VU address (-1: one segment)
    float jump = 0.5f;          // of a lane's largest value: a jump - the segment not blended
    float cut_share = 0.2f;     // segments jumping or moved, of those matched: the camera cut
    float moved_distance = 0.02f;   // a match out of its place: its first values this far a word at most
    // A segment's match in the list's order first: the first unused one of its
    // codes at most this many places on from where the last match left off
    // (things come and go before it), whatever its values - the nearest
    // values took another copy of the same model and lost the list's order
    // for every thing after it (Outbreak File #2's street: 19 of 63 things
    // stepped at once). Out of its place and far, a near one elsewhere wins.
    u32 skip_window = 8;
    // The segment that is the camera (its view's matrices: Outbreak's 1, the
    // first after the list's start): its jump the cut, and only it (a flash
    // of light makes many things jump at once); -1: `cut_share` decides; -2:
    // never a cut (a list whose cut another list decides: Outbreak's others,
    // their camera list 0's - File #2's characters' list's segment 1 is a
    // tree, and its going unmatched cancelled the list's blend).
    int cut_segment = -1;
    // Its view: these rows (quadwords) of its first UNPACK of floats, each
    // turned or moved by more than `jump` of its own length the cut (not its
    // lights' rows: a shot's flash). The rest of it blended whole.
    unsigned view_first = 0, view_end = ~0u;
    // Of those, the rows from view_first to here are its axes (a projection
    // times a rotation: x, y and z), judged together - moved by more than
    // `jump` of their length as one (a turn keeps it). Each alone, a turning
    // camera's axis passing its depth (Outbreak's x and z: 2,000 across the
    // screen, ~1 in depth) was short, its step a "jump": a cut every 60
    // degrees of a turn, the whole picture stepped (RE4's camera behind the
    // shoulder). view_first: none - every row alone.
    unsigned view_axes_end = 0;
    // A sudden move of the view the cut: its move (ListBlend::view, a share
    // of a cut's) `view_sudden` times the step before's (+0.01) or more, and
    // at least `view_sudden_least` - a still camera put somewhere else,
    // nearer than a cut's length (Outbreak File #2's street: 0.28 after
    // 0.0001, blended, a picture swung between the two views). A turn or a
    // pan comes on gradually: Outbreak's RE4 camera eases in (3x at most).
    // 0: off.
    float view_sudden = 0.0f, view_sudden_least = 0.15f;
    // A cut the view's rows miss: most of the things new (`new_share` of the
    // segments: another room's or shot's) while the camera's view moved by
    // `new_view` of a cut or more. Outbreak (played 2026-10-24): two steps
    // in 22 minutes, a picture between two shots each; a still camera over
    // a room whose things come and go (a fire's) is no cut.
    float new_share = 0.5f, new_view = 0.3f;
    // The double buffer's data (UNPACK with FLG) blended too, as any thing's:
    // for a game that writes vertices it moved itself there (Mister
    // Mosquito's bodies, deformed by the EE each step). Off: a batch's own
    // data may be read as other than coordinates (Outbreak's decal CLUTs).
    bool blend_flg = false;
    // Rows (quadwords) a jump steps alone, the rest of the thing blended:
    // UNPACKs of `words` words, rows first_row to end_row. Outbreak's things'
    // 992-word headers have their lights at rows 14-23 (bones from 24): a
    // light's slot given to another lamp stepped the whole character.
    struct StepRows {
        u32 words, first_row, end_row;
    };
    std::vector<StepRows> step_rows;
    // The colours (RGBAQ's R, G, B and A) of a matched pair's GIF packets
    // (its DIRECTs and DIRECTHLs, their tags the same in both) blended too:
    // a fade is a sprite over the screen whose alpha the game steps (File
    // #2's, 0x10 a step: the picture brightened at 30 a second at 120).
    bool blend_colors = false;
    // The two lists made in different buffers (a double buffer's halves, its
    // memory each half's own): the last list made in the newer's buffer
    // before it. A value the newer has as that one had it, at the same place
    // under the same code, is the newer buffer's own - the newer's, neither
    // blended nor a jump; the thing blends on. Outbreak's lists keep each
    // half's leftovers in words the VU never reads (a character's 992-word
    // header: 0 in one half, 128 in the other); at 144 a step's half
    // alternated (4.8 pictures a step), and those words "jumped" every step:
    // the characters, 19 of the 59 things, stepped at 30.
    const std::vector<u32>* buffer_before = nullptr;
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
    std::vector<u32> segment;   // each word's segment (the newer list's), in order
    ListBlend stats;
    // Every value of a matched pair that changed (blended or not): the newer
    // list's word and its change - the next plan's (the same plan planned
    // again a step on) to tell steady motion from a jump. A value moving on
    // as it moved (within a quarter of its change, the same way) is motion:
    // a thing's translation passing through 0 as the camera pans steadily
    // was a jump by its lane's largest, the thing stepped while the room
    // blended (File #1's bar: props shaking at 30 a second).
    std::vector<std::pair<u32, float>> changed;
    // Set before planning: this step's moves are not motion (a camera's snap
    // the game told of, the step shown whole) - `changed` carries each
    // value's change of the step before on, not this step's. Its moves as
    // the next step's history, every thing moving on ordinarily after it was
    // "unsteady" and stepped (Outbreak's RE4 camera pulled in at a hedge: 44
    // things the step after).
    bool carry = false;
    // The colours to blend (ListBlendOptions::blend_colors): the newer list's
    // word, its value in each list, and its bytes blended (4: an RGBA word;
    // 1: a PACKED RGBAQ's colour, its low byte).
    struct Color {
        u32 at, from, to, bytes;
    };
    std::vector<Color> colors;
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

/* A profile of the pictures' motion, thing by thing: each picture's list as
   sent (its first `end` words), cut as the blend cuts it, each thing's first
   UNPACK of floats (its matrices) against the picture before's same thing
   (its codes', and the how-manyth of them). A line a thing a picture to
   `file` (CSV: frame, segment, key, copy, words, change, size, t, step): its
   change the length of its values' move, `size` theirs. Blended, a thing's
   change is even from picture to picture; stepping as the game's, it moves
   on a step's pictures and stands on the others' - a judder
   (engine/tools/list_motion.py ranks them). */
class ListMotion {
public:
    void Picture(const std::vector<u32>& words, size_t end, int segment_address, unsigned long long frame, float t,
                 bool step, FILE* file);

private:
    std::unordered_map<u64, std::vector<float>> last_;
};

}  // namespace ps2host
