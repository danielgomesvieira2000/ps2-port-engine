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

#include <cstddef>
#include <cstdint>
#include <cstring>

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

/* A fast hash for a native program's tests (its microcode, in Holds: every time instruction memory changes) and its
   meshes' keys: four lanes of 64-bit words, each multiplied on (FNV's prime), then mixed (MurmurHash3's finish).
   `bytes` a multiple of 4. (Burnout 3's build/native/codehash.py computes the same.) */
inline std::uint64_t VuNativeHash(const void* data, std::size_t bytes, std::uint64_t seed) {
    constexpr std::uint64_t kPrime = 1099511628211ull;
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t lane[4] = {seed, seed ^ 0x9E3779B97F4A7C15ull, seed ^ 0xC2B2AE3D27D4EB4Full, seed ^ 0x165667B19E3779F9ull};
    std::size_t i = 0;
    for (; i + 32 <= bytes; i += 32)
        for (int l = 0; l < 4; ++l) {
            std::uint64_t w;
            std::memcpy(&w, p + i + 8 * l, 8);
            lane[l] = (lane[l] ^ w) * kPrime;
        }
    for (int l = 0; i + 4 <= bytes; i += 4, l = (l + 1) & 3) {
        std::uint32_t w;
        std::memcpy(&w, p + i, 4);
        lane[l] = (lane[l] ^ w) * kPrime;
    }
    std::uint64_t h = bytes;
    for (int l = 0; l < 4; ++l) {
        std::uint64_t k = lane[l];
        k ^= k >> 33, k *= 0xFF51AFD7ED558CCDull, k ^= k >> 33, k *= 0xC4CEB9FE1A85EC53ull, k ^= k >> 33;
        h = (h ^ k) * kPrime;
    }
    h ^= h >> 29;
    return h;
}

/* A mesh's key from its vertices (the native programs' mesh caches): XXH3, several times VuNativeHash's speed - a
   key is only compared within a run, never written down (VuNativeHash stays the microcode's test, which tools
   compute too). */
std::uint64_t VuMeshHash(const void* data, std::size_t bytes, std::uint64_t seed);

}  // namespace ps2host
