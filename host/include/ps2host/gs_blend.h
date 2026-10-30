#pragma once
/*
 * A picture between two of a game's pictures, made from the GS's input for
 * them - the GIF packets after VU1 (PATH1's XGKICKs, PATH2, PATH3), as the GS
 * thread takes them - rather than drawn by the game again (high frame rates
 * on a weak machine: games/burnout3/docs/PERFORMANCE_PLAN.md, option 2).
 *
 * Each path's words are cut into GIF packets (a tag and its data). The newer
 * picture's packets are lined up with the older's in order by their shape
 * (FLG, NLOOP, NREG, REGS, PRIM where PRE): where the two differ, the next
 * few of each are looked through for the other's (a thing come or gone). In
 * a matched PACKED packet the vertex values - XYZ (its ADC bit the newer's),
 * Z, ST and Q, UV, RGBA, FOG - are made `t` of the way from the older's to
 * the newer's; everything else is the newer's. A packet whose vertices move
 * further than `jump` pixels (a different thing in that place, a vertex
 * thrown round the guard band) is the newer's whole, as is one with no match.
 * engine/tools/gsmatch.py measured the matching on recorded pictures.
 */

#include <vector>

#include "ps2rt/types.h"

namespace ps2host {

/* A picture's GS input, as replay.bin records it ('W', 'R', 'P'). */
struct GsPicture {
    struct Entry {
        char kind;          // 'W' words, 'R' a register, 'P' a privileged register
        u8 path;            // 'W': 1-3
        u32 address;        // 'R', 'P'
        u64 value;
        u32 first, count;   // 'W': the path's words (paths[path - 1]) from `first`
    };
    std::vector<Entry> entries;
    std::vector<u32> paths[3];      // each path's words, in order
};

struct GsBlendStats {
    u32 packets = 0;        // the newer's
    u32 matched = 0;
    u32 jumped = 0;         // matched, but moved further than `jump`: the newer's
    u32 vertices = 0;       // the newer's
    u32 blended = 0;        // vertices blended
    u32 still = 0;          // vertices in matched packets the same in both
    u32 jumped_vertices = 0;    // vertices in jumped packets
    u32 carried = 0;        // packets with no match moved as their blended neighbour moved
    u32 snapped = 0;        // vertices of things not blendable whole: the newer's
    u32 added = 0;          // the older's packets with no match drawn too, moved forward
    u32 loose = 0;          // matched packets whose vertices' moves are not one motion (carried, or the newer's)
    double ms_copy = 0, ms_align = 0, ms_blend = 0;     // the time: copying the newer, cutting and lining up, blending
};

struct GsBlendOptions {
    float jump = 256.0f;    // pixels: a vertex moving further is another thing in its place
    // Where the screen is, in the GS's coordinates (XYOFFSET's units, 1/16
    // pixel), with a margin: only a vertex here, in one picture or the
    // other, can jump. Everywhere, by default.
    u32 window[4] = {0, 0, 0xFFFF, 0xFFFF};
};

/* `out`: `newer` with its matched packets' vertex values `t` of the way from
   `older`'s (0: older's, 1: newer's). */
GsBlendStats BlendGsPictures(const GsPicture& older, const GsPicture& newer, float t, GsPicture& out,
                             const GsBlendOptions& options = {});

}  // namespace ps2host
