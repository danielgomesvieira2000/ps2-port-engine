/*
 * Host implementations of Sony SDK library routines, applied by name.
 *
 * Library routines that drive the hardware - wait for the graphics paths, poll
 * a DMA channel - are better replaced whole than emulated register by register:
 * the routine's contract is small, and on the host the hardware it waits for is
 * already done. A game's analysis names such routines (config/names.csv, from
 * diagnostic strings and the cross-game signature library); every routine named
 * here is overridden wherever the game's dispatch table gives that name, so a new
 * game gets these for free once its routines are named.
 *
 * Some are found by their instructions instead (ApplySdkPatterns): routines of
 * the SDK's that games link unchanged, found in the executable once it is
 * loaded - by a hash of their words with addresses left out, so no game's code
 * is kept here.
 */

#include <cstdio>
#include <cstring>
#include <string>

#include "ps2host/host.h"
#include "ps2host/memory.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

void Return(lifted::Context& c, u64 value) { c.r[2] = u64(s64(s32(u32(value)))); }

// sceGsSyncPath(mode, timeout): wait until the paths to the GS are idle - VIF1,
// GIF and VU1 - and return 0, or a negative value on timeout. The host's
// transfers complete when they are started, so the paths are always idle.
void GsSyncPath(lifted::Context& c) { Return(c, 0); }

/* The SDK's kernel-mode store: it disables interrupts, enters kernel mode
   (Status.ERL), stores $a1 at $a0 and returns through `eret` with interrupts as
   they were; the timer-register setters after it jump to it. The host has no
   modes, so the store is all there is to it. (Burnout 3's 0x001156C0, Resident
   Evil Outbreak's 0x0011DEC0.) */
void KernelStore(lifted::Context& c) { generated::Write32(u32(c.r[4]), u32(c.r[5])); }

/* The SDK's TTY write over DECI2 (buffer $a0, length $a1): what the game prints
   with printf. A retail console has no debugger to send it to; the port logs
   it, the game telling what it is doing. (Burnout 3's 0x0010D3E8, Resident Evil
   Outbreak's 0x00115BE8.) */
void TtyWrite(lifted::Context& c) {
    const u32 buffer = u32(c.r[4]), length = u32(c.r[5]);
    std::string text;
    for (u32 i = 0; i < length && i < 4096; ++i) text += char(generated::ReadU8(buffer + i));
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    if (!text.empty()) std::fprintf(stderr, "[tty] %s\n", text.c_str());
    c.r[2] = length;
}

/* A routine known by its words: how many, the first (masked as below), and
   the FNV-1a hash of all of them, masked - jump targets and 16-bit
   immediates (addresses, offsets) left out. */
struct SdkPattern {
    const char* name;
    u32 words;
    u32 first;
    u64 hash;
    GuestFunction function;
};

const SdkPattern kPatterns[] = {
    {"the SDK's kernel-mode store", 27, 0x40036000u, 0x017368F0F4A27350ull, KernelStore},
    {"the SDK's TTY write (DECI2)", 93, 0x27BD0000u, 0x30483F2551ACEA61ull, TtyWrite},
};

u32 Masked(u32 word) {
    const u32 op = word >> 26;
    if (op == 2 || op == 3) return word & 0xFC000000u;                     // j, jal
    if (op == 0x0F || op == 0x09 || op == 0x0D || (op >= 0x20 && op < 0x30) || op == 0x31 || op == 0x37 ||
        op == 0x39 || op == 0x3F)
        return word & 0xFFFF0000u;                                          // lui, addiu, ori, loads, stores
    return word;
}

u64 Hash(const u8* at, u32 words) {
    u64 h = 0xCBF29CE484222325ull;
    for (u32 i = 0; i < words; ++i) {
        u32 word;
        std::memcpy(&word, at + 4 * i, 4);
        word = Masked(word);
        for (int b = 0; b < 4; ++b) {
            h ^= (word >> (8 * b)) & 0xFFu;
            h *= 0x100000001B3ull;
        }
    }
    return h;
}

struct SdkRoutine {
    const char* name;
    GuestFunction function;
};

const SdkRoutine kRoutines[] = {
    {"sceGsSyncPath", GsSyncPath},
};

}  // namespace

void ApplySdkPatterns(u32 start, u32 end) {
    if (end <= start) return;
    const u8* image = GuestMemory().Translate(start, end - start);
    if (!image) return;
    for (const SdkPattern& pattern : kPatterns) {
        const u32 bytes = 4 * pattern.words;
        u32 found = 0, matches = 0;
        for (u32 at = 0; at + bytes <= end - start; at += 4) {
            u32 word;
            std::memcpy(&word, image + at, 4);
            if (Masked(word) != pattern.first || Hash(image + at, pattern.words) != pattern.hash) continue;
            found = start + at;
            ++matches;
        }
        // Only where it is the one routine so: never a guess.
        if (matches != 1 || Overridden(found)) continue;
        Override(found, pattern.function, pattern.name);
        char address[16];
        std::snprintf(address, sizeof address, "0x%08X", found);
        Note(std::string("sdk ") + pattern.name, std::string("SDK routine ") + pattern.name + " at " + address +
                                                      " runs on the host");
    }
}

void ApplySdkOverrides(const FunctionEntry* table, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!table[i].name) continue;
        for (const SdkRoutine& routine : kRoutines) {
            if (std::strcmp(table[i].name, routine.name) == 0) {
                Override(table[i].address, routine.function, routine.name);
                Note(std::string("sdk ") + routine.name,
                     std::string("SDK routine ") + routine.name + " runs on the host");
            }
        }
    }
}

}  // namespace ps2host
