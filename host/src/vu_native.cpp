/*
 * Native microprograms (ps2host/vu_native.h): the registry, the dispatch from
 * Vu::Run, and the check mode.
 */

#include "ps2host/vu_native.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "ps2host/settings.h"
#define XXH_INLINE_ALL
#include "xxhash.h"

namespace ps2host {
namespace {

struct Registered {
    VuNativeProgram program;
    std::uint64_t taken = 0, declined = 0, agreed = 0, differed = 0;
};

std::vector<Registered>& Programs() {
    static std::vector<Registered> programs;
    return programs;
}

enum class Mode { kOff, kOn, kCheck };

Mode NativeMode() {
    static const Mode mode = [] {
        const char* e = std::getenv("PS2HOST_VU_NATIVE");
        // (on by default: each native program is exact - checked, PS2HOST_VU_NATIVE=check, in every
        // scripted run before it was registered)
        const std::string v = e ? std::string(e) : Setting("vu_native", "on");
        return v == "off" || v == "0" ? Mode::kOff : v == "check" ? Mode::kCheck : Mode::kOn;
    }();
    return mode;
}

/* A run's results, to compare. */
struct Outcome {
    std::vector<std::uint8_t> data;
    std::vector<std::uint32_t> kicked;
    std::uint32_t vf[32][4];
    std::uint32_t vc[32];
    std::uint32_t vacc[4];
    std::uint32_t clip = 0, resume = 0;
};

void Take(const Vu& vu, Outcome& o) {
    o.data.assign(vu.data.data(), vu.data.data() + vu.data.size());
    std::memcpy(o.vf, vu.regs.vf, sizeof o.vf);
    std::memcpy(o.vc, vu.regs.vc, sizeof o.vc);
    std::memcpy(o.vacc, vu.regs.vacc, sizeof o.vacc);
    o.clip = vu.clip;
    o.resume = vu.resume;
}

/* What differs, said in full for the first few runs. */
bool Compare(const char* name, Vu::u32 pc, const Outcome& native, const Outcome& emulated) {
    std::string what;
    char line[256];
    if (native.data != emulated.data) {
        int shown = 0;
        size_t count = 0;
        for (size_t q = 0; q * 16 < native.data.size(); ++q)
            if (std::memcmp(&native.data[q * 16], &emulated.data[q * 16], 16) != 0) {
                ++count;
                if (shown++ < 4) {
                    const std::uint32_t* a = reinterpret_cast<const std::uint32_t*>(&native.data[q * 16]);
                    const std::uint32_t* b = reinterpret_cast<const std::uint32_t*>(&emulated.data[q * 16]);
                    std::snprintf(line, sizeof line, " data %zu: native %08X %08X %08X %08X, emulated %08X %08X %08X %08X;", q,
                                  a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
                    what += line;
                }
            }
        std::snprintf(line, sizeof line, " (%zu quadwords of data differ)", count);
        what += line;
    }
    if (native.kicked != emulated.kicked) {
        size_t i = 0;
        while (i < native.kicked.size() && i < emulated.kicked.size() && native.kicked[i] == emulated.kicked[i]) ++i;
        std::snprintf(line, sizeof line, " kicked %zu words against %zu, apart from word %zu;", native.kicked.size(),
                      emulated.kicked.size(), i);
        what += line;
    }
    for (int r = 0; r < 32; ++r)
        if (std::memcmp(native.vf[r], emulated.vf[r], 16) != 0) {
            std::snprintf(line, sizeof line, " vf%02d %08X %08X %08X %08X / %08X %08X %08X %08X;", r, native.vf[r][0],
                          native.vf[r][1], native.vf[r][2], native.vf[r][3], emulated.vf[r][0], emulated.vf[r][1],
                          emulated.vf[r][2], emulated.vf[r][3]);
            what += line;
        }
    for (int r : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 22})
        if (native.vc[r] != emulated.vc[r]) {
            std::snprintf(line, sizeof line, " vc%d %X / %X;", r, native.vc[r], emulated.vc[r]);
            what += line;
        }
    if (std::memcmp(native.vacc, emulated.vacc, 16) != 0) what += " ACC;";
    if (native.clip != emulated.clip) {
        std::snprintf(line, sizeof line, " clip %06X / %06X;", native.clip, emulated.clip);
        what += line;
    }
    if (native.resume != emulated.resume) {
        std::snprintf(line, sizeof line, " resume %X / %X;", native.resume, emulated.resume);
        what += line;
    }
    if (what.empty()) return true;
    static int said = 0;
    if (said++ < 12) std::fprintf(stderr, "[vu native] %s from 0x%X differs:%s\n", name, unsigned(pc), what.c_str());
    return false;
}

}  // namespace

void RegisterVuNative(const VuNativeProgram& program) { Programs().push_back(Registered{program}); }

void VuNativeAccess::SetClip(Vu& vu, Vu::u32 clip) {
    vu.clip = clip;
    vu.clip_latest_ = clip;
}

void VuNativeAccess::Kick(Vu& vu, Vu::u32 address) { vu.Kick(address); }

/* Vu::RunTimed's: true when a native program took the run from `pc` (its
   pairs in *pairs); `cache` the program found for this instruction memory. */
bool VuNativeRun(Vu& vu, Vu::u32 pc, Vu::u64 limit, Vu::u64* pairs, Vu::u64 generation, int* cache,
                 Vu::u64* cache_generation) {
    if (NativeMode() == Mode::kOff || Programs().empty()) return false;
    if (*cache_generation != generation) {
        *cache_generation = generation;
        *cache = -1;
        for (size_t i = 0; i < Programs().size(); ++i)
            if (Programs()[i].program.holds(vu)) {
                *cache = int(i);
                break;
            }
    }
    if (*cache < 0) return false;
    Registered& r = Programs()[size_t(*cache)];
    if (NativeMode() == Mode::kOn) {
        if (r.program.run(vu, pc, pairs)) {
            ++r.taken;
            return true;
        }
        ++r.declined;
        return false;
    }
    // check: the native run first, its results kept, the VU put back...
    static thread_local Outcome before, native, emulated;
    Take(vu, before);
    const lifted::Context regs = vu.regs;
    const Vu::u32 mac = vu.mac, status = vu.status, p = vu.p;
    auto block = vu.on_kick_block;
    auto quad = vu.on_kick;
    native.kicked.clear();
    vu.on_kick_block = [](const Vu::u32* words, Vu::u32 count) { native.kicked.insert(native.kicked.end(), words, words + count); };
    vu.on_kick = nullptr;
    Vu::u64 native_pairs = 0;
    const bool taken = r.program.run(vu, pc, &native_pairs);
    if (!taken) {
        vu.on_kick_block = block;
        vu.on_kick = quad;
        ++r.declined;
        return false;
    }
    ++r.taken;
    Take(vu, native);
    std::memcpy(vu.data.data(), before.data.data(), before.data.size());
    vu.regs = regs;
    vu.mac = mac, vu.status = status, vu.p = p;
    VuNativeAccess::SetClip(vu, before.clip);
    vu.resume = before.resume;
    // ...then emulated, its kicks seen and passed on: its results stand.
    emulated.kicked.clear();
    vu.on_kick_block = [&](const Vu::u32* words, Vu::u32 count) {
        emulated.kicked.insert(emulated.kicked.end(), words, words + count);
        if (block) block(words, count);
        else if (quad)
            for (Vu::u32 i = 0; i + 4 <= count; i += 4)
                quad(Vu::u64(words[i]) | Vu::u64(words[i + 1]) << 32, Vu::u64(words[i + 2]) | Vu::u64(words[i + 3]) << 32);
    };
    vu.on_kick = nullptr;
    *pairs = vu.RunEmulated(pc, limit);
    vu.on_kick_block = block;
    vu.on_kick = quad;
    Take(vu, emulated);
    if (Compare(r.program.name, pc, native, emulated)) ++r.agreed;
    else ++r.differed;
    return true;
}

std::uint64_t VuMeshHash(const void* data, std::size_t bytes, std::uint64_t seed) {
    return XXH3_64bits_withSeed(data, bytes, seed);
}

void VuNativeReport() {
    for (const Registered& r : Programs()) {
        if (!r.taken && !r.declined) continue;
        std::fprintf(stderr, "[vu native] %s: %llu runs taken, %llu declined", r.program.name,
                     (unsigned long long)r.taken, (unsigned long long)r.declined);
        if (NativeMode() == Mode::kCheck)
            std::fprintf(stderr, "; checked: %llu the same, %llu different", (unsigned long long)r.agreed,
                         (unsigned long long)r.differed);
        std::fprintf(stderr, "\n");
    }
}

}  // namespace ps2host
