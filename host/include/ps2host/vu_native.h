#pragma once
/*
 * Native microprograms (docs/NATIVE_RENDERER.md, level B): a VU1 microprogram's
 * work done by the port's own code, from the same inputs, in place of running
 * its microcode.
 *
 * A native program is registered with a test of VU1's instruction memory (does
 * it hold the microcode it replaces? checked once each time that memory
 * changes) and a run: from a start address, it does what the microprogram's run
 * would - data memory, XGKICKs, registers and flags as it leaves them, `resume`
 * set - and returns true; or it declines (a case it does not do: a triangle for
 * the clipper), returning false with nothing changed, and the microcode runs.
 *
 * PS2HOST_VU_NATIVE (or the setting vu_native): on (the default: a native
 * program is registered once checked exact in every scripted run), off, or check - each run a native program takes is run both ways:
 * the native run's results kept apart, the VU put back as it was, the run
 * emulated (whose results stand), and the two compared: data memory, the
 * kicked words, the vector and integer registers, ACC, Q and the clip flags.
 * A difference is logged ([vu native]: the first ones in full) and counted;
 * PS2HOST_VU_NATIVE_REPORT at the run's end (VuNativeReport) says how many
 * runs agreed.
 */

#include <cstdint>

#include "ps2host/vu.h"

namespace ps2host {

struct VuNativeProgram {
    const char* name;
    /* Whether VU1's instruction memory holds the microcode this replaces. */
    bool (*holds)(const Vu& vu);
    /* The run from `pc` (bytes), as the microcode's: true, `*pairs` the pairs it
       would have run (for the statistics); false: declined, nothing changed. */
    bool (*run)(Vu& vu, Vu::u32 pc, Vu::u64* pairs);
};

/* A port's native program (at start-up, before the game runs). */
void RegisterVuNative(const VuNativeProgram& program);

/* For a native program's code: VU state a run leaves that the class keeps to
   itself - the clip flags (landed and latest: a run ends with both the same). */
struct VuNativeAccess {
    static void SetClip(Vu& vu, Vu::u32 clip);
    /* XGKICK: the GIF packet at `address` (quadwords) sent, as the microcode's. */
    static void Kick(Vu& vu, Vu::u32 address);
};

/* The runs taken, declined and (check) agreeing, by program: written at the
   run's end. */
void VuNativeReport();

}  // namespace ps2host
