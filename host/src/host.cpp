#include "ps2host/host.h"

#include "ps2host/platform.h"
#ifndef _WIN32
#include <time.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "ps2host/disc.h"
#include "ps2host/gs.h"
#include "ps2host/iop_vm.h"
#include "ps2host/vif.h"
#include "ps2host/vu.h"
#include "ps2host/mc2.h"
#include "ps2host/memory.h"
#include "ps2host/profile.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/timeline.h"

namespace ps2host {

/* --gs-replay <file> <dir>: a GS stream (engine/tools/pcsx2dump.py's, from a
   PCSX2 GS dump) through the GS model alone: local memory and registers set,
   then GIF packets, privileged registers and vertical blanks in turn - each
   blank's picture to <dir>/replay_<n>.bmp, the GS trace to <dir>/trace.txt.
   What the model draws from the console's own commands, beside what PCSX2
   drew (the dump's picture): a GS fault or one before it, told apart.
   PS2HOST_REPLAY_GPU=1: the GPU renderer draws instead (as --gpu), each
   picture to <dir>/gpu_<n>/ (screen.bmp at the internal resolution,
   PS2HOST_REPLAY_SCALE, default 1; screen_1x.bmp).
   PS2HOST_REPLAY_REPEAT=n: then the stream n times more, its pictures not
   written, timed - the GS thread's work for a capture's frames (the GS model,
   the GPU bridge, the renderer's recording), the same every run; with
   PS2HOST_REPLAY_SAMPLE=1 sampled too (the sampler's report after). */
/* The calling thread's CPU time (user and kernel), in milliseconds. */
double ThreadCpuMs() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return 0;
    auto ms = [](const FILETIME& t) { return double(u64(t.dwHighDateTime) << 32 | t.dwLowDateTime) / 10000.0; };
    return ms(kernel) + ms(user);
#else
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) != 0) return 0;
    return double(t.tv_sec) * 1000.0 + double(t.tv_nsec) / 1e6;
#endif
}

int GsReplay(const std::string& file, const std::string& out) {
    std::FILE* f = std::fopen(file.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "gs-replay: cannot open %s\n", file.c_str());
        return 66;
    }
    std::vector<unsigned char> data;
    {
        unsigned char buffer[1 << 16];
        std::size_t n;
        while ((n = std::fread(buffer, 1, sizeof buffer, f)) > 0) data.insert(data.end(), buffer, buffer + n);
        std::fclose(f);
    }
    const std::size_t kMemory = 4u << 20;
    if (data.size() < 5 + kMemory || std::memcmp(data.data(), "GSRP1", 5) != 0) {
        std::fprintf(stderr, "gs-replay: not a replay stream\n");
        return 65;
    }
    std::filesystem::create_directories(out);
    gs::Gs& g = gs::TheGs();
    std::memcpy(g.memory.data(), data.data() + 5, kMemory);
    const bool gpu = std::getenv("PS2HOST_REPLAY_GPU") != nullptr;
    // PS2HOST_REPLAY_SAMPLE=renderer: the renderer's own thread sampled
    // instead (its recording of what the bridge hands it).
    const char* sample_what = std::getenv("PS2HOST_REPLAY_SAMPLE");
    const bool sample_renderer = sample_what && std::string(sample_what) == "renderer";
    if (sample_renderer) SetSampleRendererThread(true);
    if (gpu) {
        std::string error;
        if (!StartGpuRenderer(&error, GpuMode::Primary, false)) {
            std::fprintf(stderr, "gs-replay: no GPU (%s)\n", error.c_str());
            return 69;
        }
        const char* scale = std::getenv("PS2HOST_REPLAY_SCALE");
        if (scale) GpuSetScale(u32(std::max(1, std::atoi(scale))));
        if (g.on_memory) g.on_memory(0, 16384, true);       // local memory as the dump has it, on the GPU too
    }
    std::FILE* trace = std::fopen((out + "/trace.txt").c_str(), "w");
    if (trace) {
        g.trace_out = trace;
        g.trace = true;
    }
    // PS2HOST_GS_SPLIT=1: through the GS's threads as a game's run feeds them
    // (the decoding on its own), synced at each vertical blank.
    const bool threaded = GsSplit();
    if (threaded) {
        if (sample_what && !sample_renderer) SetSampleGsThread(true);
        StartGsThread();
    }
    std::size_t at = 5 + kMemory;
    auto u32_at = [&](std::size_t o) { u32 v; std::memcpy(&v, data.data() + o, 4); return v; };
    auto u64_at = [&](std::size_t o) { u64 v; std::memcpy(&v, data.data() + o, 8); return v; };
    int pictures = 0;
    std::vector<u32> words;
    const std::size_t stream = at;
    bool timing = false;                // PS2HOST_REPLAY_REPEAT's passes: nothing written
    const int repeat = std::getenv("PS2HOST_REPLAY_REPEAT") ? std::atoi(std::getenv("PS2HOST_REPLAY_REPEAT")) : 0;
    int pass = 0;
    std::chrono::steady_clock::time_point started;
    double cpu_started = 0, renderer_started = 0;
    for (;;) {
    while (at < data.size()) {
        const char kind = char(data[at++]);
        if (kind == 'R') {
            if (threaded) GsRegister(u32_at(at), u64_at(at + 4));
            else g.WriteRegister(u32_at(at), u64_at(at + 4));
            at += 12;
        } else if (kind == 'P') {
            if (threaded) GsPrivileged(u32_at(at), u64_at(at + 4));
            else g.SetPrivileged(u32_at(at), u64_at(at + 4));
            at += 12;
        } else if (kind == 'W') {
            const int path = data[at];
            const u32 bytes = u32_at(at + 1);
            at += 5;
            words.resize(bytes / 4);
            std::memcpy(words.data(), data.data() + at, words.size() * 4);
            at += bytes;
            // PS2HOST_REPLAY_STOP_PRIM=n: the replay ends as primitive n is drawn
            // (memory then, to compare two renderers mid-frame).
            static const long long stop = std::getenv("PS2HOST_REPLAY_STOP_PRIM")
                                              ? std::atoll(std::getenv("PS2HOST_REPLAY_STOP_PRIM")) : -1;
            if (threaded) {
                GsWords(path, words.data(), u32(words.size()));
            } else if (stop < 0) {
                g.Words(path, words.data(), u32(words.size()));
            } else {
                bool stopped = false;
                for (std::size_t w = 0; w < words.size() && !stopped; w += 4) {
                    g.Words(path, words.data() + w, u32(std::min<std::size_t>(4, words.size() - w)));
                    stopped = (long long)g.primitives() >= stop;
                }
                if (stopped) {
                    at = data.size();
                    break;
                }
            }
        } else if (kind == 'V') {
            if (threaded) SyncGs();
            g.Finish();
            if (timing) {
                if (gpu) GpuShadowFrame(u64(pictures), std::string());
                continue;
            }
            if (trace) std::fprintf(trace, "[gs] ---- vertical blank %d\n", pictures);
            if (gpu) {
                GpuShadowFrame(u64(pictures), std::string());
                const std::string dir = out + "/gpu_" + std::to_string(pictures++);
                std::filesystem::create_directories(dir);
                GpuCaptureScreen(dir);
            } else {
                g.WriteBmp(out + "/replay_" + std::to_string(pictures++) + ".bmp");
            }
        } else {
            std::fprintf(stderr, "gs-replay: bad record at %zu\n", at - 1);
            return 65;
        }
    }
    if (pass == 0 && trace) {           // the trace of the first pass alone
        if (threaded) SyncGs();
        g.trace = false;
        g.trace_out = stderr;
        std::fclose(trace);
        trace = nullptr;
    }
    if (pass == 0) {
        if (repeat > 0 && std::getenv("PS2HOST_REPLAY_SAMPLE")) {
            if (!threaded && !sample_renderer) StartSampler();      // (split: the GS thread's, started with it; the renderer's: with it)
            SetSampleFrom(0);
            SampleFrame(0);
        }
        started = std::chrono::steady_clock::now();
        cpu_started = ThreadCpuMs();
        renderer_started = RendererThreadCpuMs();
    }
    if (pass++ >= repeat) break;
    timing = true;
    at = stream;
    }
    if (threaded) StopGsThread();
    g.Finish();
    if (repeat > 0) {
        // To the last drawn: a block read back waits for the GPU (and for a
        // renderer on a thread of its own, its queue).
        if (gpu && g.on_memory) g.on_memory(0, 1, false);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // The thread's own CPU as well: with a renderer on a thread of its
        // own the passes' time may be that thread's (or the GPU's), not this.
        std::fprintf(stderr, "gs-replay: %d passes, %.2f ms a pass (%d pictures each), this thread's CPU %.2f ms a pass, the renderer's %.2f\n",
                     repeat, ms / repeat, pictures, (ThreadCpuMs() - cpu_started) / repeat,
                     (RendererThreadCpuMs() - renderer_started) / repeat);
        if (std::getenv("PS2HOST_REPLAY_SAMPLE"))
            std::printf("%s", SampleReport(std::getenv("PS2HOST_SAMPLE_TOP") ? std::atoi(std::getenv("PS2HOST_SAMPLE_TOP")) : 30).c_str());
    }
    if (gpu && g.on_memory) g.on_memory(0, 16384, false);     // what the GPU drew, into local memory first
    if (std::FILE* m = std::fopen((out + "/gsmem.bin").c_str(), "wb")) {      // local memory at the end
        std::fwrite(g.memory.data(), 1, kMemory, m);
        std::fclose(m);
    }
    if (trace) {
        g.trace = false;
        g.trace_out = stderr;
        std::fclose(trace);
    }
    std::fprintf(stderr, "gs-replay: %d pictures to %s\n", pictures, out.c_str());
    return 0;
}

namespace {

const FunctionEntry* g_table = nullptr;
std::size_t g_count = 0;
u32 g_image_start = 0, g_image_end = 0;       // the executable's bytes, as loaded
std::unordered_map<u32, FunctionEntry> g_overrides;
bool g_trace = false;
bool g_all_notes = false;
std::vector<u32> g_watch;
bool g_watching = false;        // !g_watch.empty(), for Dispatch's fast path (a vector's load a call)
// Each call counted (the report's "most called": a scripted run's, or PS2HOST_CALL_COUNTS=1): the count's
// increment missed the cache a call - at 144 pictures a second a tenth of the EE's time.
bool g_count_calls = false;
std::set<std::string> g_noted;

// Calls already made, by address: the function and its call count, so a call
// costs a lookup in a small table rather than a map's (a frame makes hundreds
// of thousands). Cleared when an override replaces a function.
struct CachedCall {
    u32 address = 0;
    GuestFunction function = nullptr;   // (the entry's own, here: no second load from the table)
    u64* count = nullptr;
};
constexpr std::size_t kCallCache = 1u << 14;
CachedCall g_call_cache[kCallCache];

const OverlayEntry* g_overlays = nullptr;
std::size_t g_overlay_count = 0;

// The function of the overlay whose code is in memory at `address`, if the
// address is in an overlay and one is: its first bytes there are the original's.
// OverrideInOverlay's: (overlay name, address) -> the host's function.
std::map<std::pair<std::string, u32>, FunctionEntry> g_overlay_overrides;

const FunctionEntry* FindInOverlay(u32 address, bool* in_overlay) {
    for (std::size_t i = 0; i < g_overlay_count; ++i) {
        const OverlayEntry& o = g_overlays[i];
        if (address < o.start || address >= o.end) continue;
        *in_overlay = true;
        const FunctionEntry* end = o.table + o.count;
        const FunctionEntry* at = std::lower_bound(
            o.table, end, address, [](const FunctionEntry& e, u32 a) { return e.address < a; });
        if (at == end || at->address != address) continue;
        const u32* check = o.checks + 4 * (at - o.table);
        const u32 length = at->size && at->size < 16 ? at->size : 16;
        const u8* here = GuestMemory().Translate(address, length);
        if (here && std::memcmp(here, check, length) == 0) {
            if (!g_overlay_overrides.empty()) {
                const auto mine = g_overlay_overrides.find({o.name, address});
                if (mine != g_overlay_overrides.end()) return &mine->second;
            }
            return at;
        }
    }
    return nullptr;
}

const FunctionEntry* Find(u32 address) {
    auto found = g_overrides.find(address);
    if (found != g_overrides.end()) return &found->second;
    if (g_overlay_count) {
        bool in_overlay = false;
        const FunctionEntry* entry = FindInOverlay(address, &in_overlay);
        if (entry || in_overlay) return entry;
    }
    const FunctionEntry* end = g_table + g_count;
    const FunctionEntry* at = std::lower_bound(
        g_table, end, address, [](const FunctionEntry& e, u32 a) { return e.address < a; });
    return (at != end && at->address == address) ? at : nullptr;
}

std::string Hex(u32 value) {
    char text[16];
    std::snprintf(text, sizeof text, "0x%08X", value);
    return text;
}

std::string Describe(u32 address) {
    const char* name = FunctionName(address);
    return name ? Hex(address) + " " + name : Hex(address);
}

// The running thread's guest call stack, kept as the host unwinds through
// translated code.
struct Frame {
    explicit Frame(u32 address) : calls(CurrentCalls()) { calls.push_back(address); }
    ~Frame() { calls.pop_back(); }
    std::vector<u32>& calls;
};

}  // namespace

namespace {

// Code the program copied elsewhere and runs there - start-up code installs
// routines in kernel RAM as syscall handlers. A call to an address with no
// translation is matched against the program's functions by content: a copy of
// one, byte for byte, runs that function's translation. The copies are
// position-independent (their branches are relative and their calls absolute, so
// they reach the same targets), so the translation behaves the same there.
const FunctionEntry* FindCopy(u32 address) {
    static std::unordered_map<u32, const FunctionEntry*> copies;
    auto known = copies.find(address);
    if (known != copies.end()) return known->second;
    Memory& memory = GuestMemory();
    const FunctionEntry* found = nullptr;
    const u8* here = memory.Translate(address, 8);
    for (std::size_t i = 0; here && i < g_count && !found; ++i) {
        const FunctionEntry& e = g_table[i];
        if (!e.function || e.size < 8) continue;
        const u8* there = memory.Translate(address, e.size);
        const u8* original = memory.Translate(e.address, e.size);
        if (there && original && std::memcmp(there, original, e.size) == 0) found = &e;
    }
    copies[address] = found;
    if (found) {
        Note("copy " + Hex(address), "code at " + Hex(address) + " is a copy of " +
             Describe(found->address) + "; running its translation");
    }
    return found;
}

// Where in the loaded executable the code at `address` came from, or 0: the
// executable is searched for the bytes there. Code kept as data and copied out at
// run time (a kernel extension, an overlay) has no translation yet; finding its
// source lets engine/tools/port.py translate it for the next build.
u32 FindSource(u32 address) {
    // Code in the executable itself that no function starts at - a tail call's
    // target the analysis missed - is its own source.
    if (address >= g_image_start && address < g_image_end) return address;
    // So is code in an overlay's range: port.py translates it from the overlay.
    for (std::size_t i = 0; i < g_overlay_count; ++i)
        if (address >= g_overlays[i].start && address < g_overlays[i].end) return address;
    constexpr u32 kPattern = 32;
    Memory& memory = GuestMemory();
    const u8* here = memory.Translate(address, kPattern);
    const u8* image = memory.Translate(g_image_start, g_image_end - g_image_start);
    if (!here || !image || g_image_end - g_image_start < kPattern) return 0;
    const u8* end = image + (g_image_end - g_image_start) - kPattern;
    for (const u8* at = image; at <= end; at += 4) {
        const u32 source = g_image_start + u32(at - image);
        if (source != address && std::memcmp(at, here, kPattern) == 0) return source;
    }
    return 0;
}

}  // namespace

void RegisterFunctions(const FunctionEntry* table, std::size_t count) {
    g_table = table;
    g_count = count;
    // Before the game's own overrides, which may replace these in turn.
    ApplySdkOverrides(table, count);
}

void RegisterOverlays(const OverlayEntry* overlays, std::size_t count) {
    g_overlays = overlays;
    g_overlay_count = count;
    CodeChanged();
}

void CodeChanged() {
    if (!g_overlay_count) return;           // the executable's code never moves
    for (CachedCall& line : g_call_cache) line = CachedCall{};
}

void Override(u32 address, GuestFunction function, const char* name) {
    g_overrides[address] = FunctionEntry{address, function, name, 0};
    for (CachedCall& line : g_call_cache) line = CachedCall{};
}

void OverrideInOverlay(const char* overlay, u32 address, GuestFunction function, const char* name) {
    g_overlay_overrides[{overlay, address}] = FunctionEntry{address, function, name, 0};
    for (CachedCall& line : g_call_cache) line = CachedCall{};
}

const char* FunctionName(u32 address) {
    const FunctionEntry* entry = Find(address);
    return entry ? entry->name : nullptr;
}

bool Tracing() { return g_trace; }

bool NotingAll() { return g_all_notes; }

namespace {
bool g_fast_disc = false, g_fast_loading_default = false;
}  // namespace

bool FastDisc() { return g_fast_disc; }
void SetFastDisc(bool on) { g_fast_disc = on; }
void SetFastLoadingDefault(bool on) { g_fast_loading_default = on; }
bool g_vu_fast_default = false;
void SetVuFastDefault(bool on) { g_vu_fast_default = on; }
bool g_frame_pacing_default = true;
void SetFramePacingDefault(bool on) { g_frame_pacing_default = on; }
bool FramePacingDefault() { return g_frame_pacing_default; }

Stats& RunStats() {
    static Stats stats;
    return stats;
}

std::mutex g_notes_mutex;               // VIF1's worker notes too

bool Noting(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_notes_mutex);
    return g_all_notes || !g_noted.count(key);
}

void Note(const std::string& key, const std::string& text) {
    std::lock_guard<std::mutex> lock(g_notes_mutex);
    if (g_noted.insert(key).second) {
        RunStats().notes.push_back(text);
        std::fprintf(stderr, "[host] %s\n", text.c_str());
    } else if (g_all_notes) {
        std::fprintf(stderr, "[host] %s (again)\n", text.c_str());
    }
}

void Dispatch(lifted::Context& c, u32 address) {
    CachedCall& line = g_call_cache[(address >> 2) & (kCallCache - 1)];
    if (line.address == address && line.function && !g_watching) {
        if (g_count_calls) ++*line.count;
        if (g_trace) std::fprintf(stderr, "[call] %s\n", Describe(address).c_str());
        Frame frame(address);
        try {
            line.function(c);
        } catch (...) {
            if (RunStats().stopped_in.empty()) {
                RunStats().stopped_in = CurrentCalls();
                RunStats().interrupted_in = InterruptedCalls();
            }
            throw;
        }
        return;
    }
    const FunctionEntry* entry = Find(address);
    const bool cacheable = entry && entry->function;     // the program's own, or an override
    if (!entry || !entry->function) {
        if (const FunctionEntry* copy = FindCopy(address)) entry = copy;
    }
    if (!entry || !entry->function) {
        if (const u32 source = FindSource(address)) {
            RunStats().discovered.emplace_back(address, source);
            throw Unimplemented{"a call to " + Hex(address) + ", code " +
                                (source == address ? std::string("no function covers")
                                                   : "copied from " + Hex(source) + " at run time") +
                                ", not translated yet (port.py translates it for the next build)"};
        }
        throw Unimplemented{"a call to " + Hex(address) + ", which has no translated function"};
    }
    u64& count = RunStats().calls[address];
    if (++count == 1 || g_trace)
        std::fprintf(stderr, "[call] %s\n", Describe(address).c_str());
    if (cacheable) line = CachedCall{address, entry->function, &count};
    if (g_watching && std::find(g_watch.begin(), g_watch.end(), address) != g_watch.end()) {
        std::fprintf(stderr, "[watch] %s a0=%016llX a1=%016llX a2=%016llX a3=%016llX "
                     "f12=%08X f13=%08X ra=%08X s0=%08X s1=%08X sp=%08X gp=%08X t0=%08X\n",
                     Describe(address).c_str(),
                     (unsigned long long)c.r[4], (unsigned long long)c.r[5],
                     (unsigned long long)c.r[6], (unsigned long long)c.r[7], c.f[12], c.f[13],
                     unsigned(c.r[31]), unsigned(c.r[16]), unsigned(c.r[17]), unsigned(c.r[29]),
                     unsigned(c.r[28]), unsigned(c.r[8]));
    }
    Frame frame(address);
    try {
        entry->function(c);
    } catch (...) {
        // The innermost frame keeps the stack as it was where the run stopped;
        // unwinding empties it on the way out.
        if (RunStats().stopped_in.empty()) {
            RunStats().stopped_in = CurrentCalls();
            RunStats().interrupted_in = InterruptedCalls();
        }
        throw;
    }
    if (g_watching && std::find(g_watch.begin(), g_watch.end(), address) != g_watch.end())
        std::fprintf(stderr, "[watch] %s returned v0=%016llX f0=%08X s0=%08X s1=%08X sp=%08X\n",
                     Describe(address).c_str(), (unsigned long long)c.r[2], c.f[0],
                     unsigned(c.r[16]), unsigned(c.r[17]), unsigned(c.r[29]));
}

namespace {

// ELF32 little-endian, the PS2's executable format: load every PT_LOAD segment.
u32 LoadElf(const std::string& path, std::vector<u8> data) {
    auto u16at = [&](std::size_t at) { u16 v; std::memcpy(&v, &data.at(at + 1) - 1, 2); return v; };
    auto u32at = [&](std::size_t at) { u32 v; std::memcpy(&v, &data.at(at + 3) - 3, 4); return v; };
    if (data.size() < 52 || std::memcmp(data.data(), "\x7F" "ELF", 4) != 0)
        throw Unimplemented{path + " is not an ELF file"};
    const u32 entry = u32at(24), phoff = u32at(28);
    const u16 phentsize = u16at(42), phnum = u16at(44);
    Memory& memory = GuestMemory();
    for (u16 i = 0; i < phnum; ++i) {
        const std::size_t ph = phoff + std::size_t(i) * phentsize;
        if (u32at(ph) != 1) continue;                       // PT_LOAD
        const u32 offset = u32at(ph + 4), vaddr = u32at(ph + 8);
        const u32 filesz = u32at(ph + 16), memsz = u32at(ph + 20);
        if (!memsz) continue;
        u8* target = memory.Translate(vaddr, memsz);
        if (!target || std::size_t(offset) + filesz > data.size())
            throw Unimplemented{"a segment of " + path + " lies outside guest memory"};
        std::memcpy(target, data.data() + offset, filesz);
        std::memset(target + filesz, 0, memsz - filesz);
        if (filesz && !g_image_start) {
            g_image_start = vaddr;
            g_image_end = vaddr + filesz;
        }
    }
    return entry;
}

u32 LoadElf(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw Unimplemented{"cannot open the executable " + path};
    return LoadElf(path, std::vector<u8>((std::istreambuf_iterator<char>(file)),
                                         std::istreambuf_iterator<char>()));
}

// No executable named: the one the disc boots (SYSTEM.CNF), read from it.
u32 LoadBootElf() {
    const std::string boot = GuestDisc().BootPath();
    std::vector<u8> data;
    if (boot.empty() || !GuestDisc().ReadFile(boot, &data))
        throw Unimplemented{"the disc image has no executable to boot (SYSTEM.CNF)"};
    return LoadElf(boot, std::move(data));
}

void Report(const Options& options, const std::string& outcome) {
    const Stats& stats = RunStats();
    std::printf("\n=== host report ===\n%s\n", outcome.c_str());
    std::printf("functions executed: %zu distinct\n", stats.calls.size());
    const std::vector<u32>& stack = stats.stopped_in;
    if (!stack.empty()) {
        std::printf("guest call stack where it stopped (innermost last):\n");
        const std::size_t from = stack.size() > std::size_t(options.report_limit)
                                     ? stack.size() - options.report_limit : 0;
        for (std::size_t i = from; i < stack.size(); ++i)
            std::printf("  %s\n", Describe(stack[i]).c_str());
    }
    if (!stats.interrupted_in.empty()) {
        std::printf("in an interrupt handler; the thread it interrupted was in:\n");
        const std::vector<u32>& thread = stats.interrupted_in;
        const std::size_t from = thread.size() > std::size_t(options.report_limit)
                                     ? thread.size() - options.report_limit : 0;
        for (std::size_t i = from; i < thread.size(); ++i)
            std::printf("  %s\n", Describe(thread[i]).c_str());
    }
    std::printf("threads:\n%s%s", ThreadReport().c_str(), iop::Report().c_str());
    if (!options.dump_ram.empty()) {
        if (std::FILE* f = std::fopen(options.dump_ram.c_str(), "wb")) {
            std::fwrite(GuestMemory().rdram(), 1, 32u << 20, f);
            std::fclose(f);
            std::printf("EE RAM written to %s\n", options.dump_ram.c_str());
        }
    }
    for (u32 address : options.peek) {
        std::printf("memory at 0x%08X:", unsigned(address));
        for (u32 i = 0; i < 64; i += 4) {
            if (i % 16 == 0) std::printf("\n  +%02X", unsigned(i));
            const u8* bytes = GuestMemory().Translate(address + i, 4);
            u32 word = 0;
            if (bytes) std::memcpy(&word, bytes, 4);
            std::printf(bytes ? "  %08X" : "  --------", unsigned(word));
        }
        std::printf("\n");
    }
    std::printf("VIF1 commands:%s\n", Vif1().Summary().c_str());
    std::printf("GS: %llu primitives, %llu pixels written\n",
                static_cast<unsigned long long>(gs::TheGs().primitives()),
                static_cast<unsigned long long>(gs::TheGs().pixels_drawn()));
    std::printf("%s\n", gs::TheGs().Summary().c_str());
    std::printf("%s\n", ProfileSummary().c_str());
    const std::string samples = SampleReport(std::getenv("PS2HOST_SAMPLE_TOP") ? std::atoi(std::getenv("PS2HOST_SAMPLE_TOP")) : 30);
    if (!samples.empty()) std::printf("%s", samples.c_str());
    if (!GpuShadowSummary().empty()) std::printf("%s\n", GpuShadowSummary().c_str());
    std::printf("kernel services used:\n");
    std::map<s32, u64> services = stats.syscalls;
    for (s32 i = 0; i < 0x200; ++i)
        if (stats.syscall_counts[i]) services[i - 0x100] += stats.syscall_counts[i];
    for (const auto& [service, count] : services)
        std::printf("  %4d  %-24s x%llu\n", service, SyscallName(service),
                    (unsigned long long)count);
    auto registers = [&](const char* title, const std::map<u32, u64>& seen) {
        std::printf("%s: %zu\n", title, seen.size());
        int shown = 0;
        for (const auto& [address, count] : seen) {
            if (shown++ == options.report_limit) break;
            std::printf("  0x%08X x%llu\n", address, (unsigned long long)count);
        }
    };
    {
        // Where the time goes: the functions called most - a loop that never
        // ends shows here as the functions it keeps calling.
        std::vector<std::pair<u64, u32>> hot;
        for (const auto& [address, count] : stats.calls) hot.push_back({count, address});
        std::sort(hot.rbegin(), hot.rend());
        std::printf("most called:\n");
        for (std::size_t i = 0; i < hot.size() && i < std::size_t(options.report_limit); ++i)
            std::printf("  %-40s x%llu\n", Describe(hot[i].second).c_str(), (unsigned long long)hot[i].first);
    }
    registers("device registers read", GuestMemory().RegisterCounts(false));
    registers("device registers written", GuestMemory().RegisterCounts(true));
    for (const auto& [run, source] : stats.discovered)
        std::printf("discovered: code at %s is a copy of %s\n", Hex(run).c_str(), Hex(source).c_str());
    if (!options.discover_path.empty()) {
        // Merged with what earlier runs found: each run stops at the first one.
        std::FILE* out = std::fopen(options.discover_path.c_str(), "a");
        if (out) {
            for (const auto& [run, source] : stats.discovered)
                std::fprintf(out, "0x%08X 0x%08X\n", run, source);
            std::fclose(out);
        }
    }
}

}  // namespace

namespace {

// Stops a run that goes on too long, so it ends with a report rather than being
// killed without one: the running context's loop budget is exhausted, and the
// translated code stops at its next loop head (lifted::Tick) with the guest call
// stack intact.
class Watchdog {
public:
    explicit Watchdog(int seconds) : thread_([this, seconds] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!done_.wait_for(lock, std::chrono::seconds(seconds), [this] { return finished_; })) {
            fired = true;
            StopRequested() = true;
            ForEachContext([](lifted::Context& each) {
                reinterpret_cast<std::atomic<s64>*>(&each.budget)->store(-1);
            });
            // A run that does not stop is stuck in host code (no guest loop
            // head comes): say where the guest was, rather than be killed
            // without a word.
            if (!done_.wait_for(lock, std::chrono::seconds(20), [this] { return finished_; })) {
                u32 calls[32];
                const int n = GuestCallsNow(calls, 32);
                std::fprintf(stderr, "\n=== host report ===\nstuck: the run did not stop 20 s after its time "
                                     "limit - the host is in a loop of its own\nguest call stack (innermost first):");
                for (int i = 0; i < n; ++i) std::fprintf(stderr, " 0x%08X", unsigned(calls[i]));
                std::fprintf(stderr, "\n");
                std::fflush(nullptr);
                // PS2HOST_STUCK_RAM=file: EE RAM as it stands, to look into.
                if (const char* path = std::getenv("PS2HOST_STUCK_RAM")) {
                    if (std::FILE* f = std::fopen(path, "wb")) {
                        std::fwrite(GuestMemory().Translate(0, 32u << 20), 1, 32u << 20, f);
                        std::fclose(f);
                        std::fprintf(stderr, "EE RAM written to %s\n", path);
                    }
                }
                // The IOP's threads too (read while it runs: a diagnostic, last).
                std::fprintf(stderr, "%s", iop::Report().c_str());
                std::fflush(nullptr);
                std::_Exit(3);
            }
        }
    }) {}
    ~Watchdog() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            finished_ = true;
        }
        done_.notify_all();
        thread_.join();
    }
    std::atomic<bool> fired{false};

private:
    std::mutex mutex_;
    std::condition_variable done_;
    bool finished_ = false;
    std::thread thread_;
};

}  // namespace

int Run(const Options& options) {
    SetVirtualClock(options.virtual_time);  // before anything reads the clock
    g_trace = options.trace_calls;
    g_all_notes = options.all_notes;
    GuestMemory().WatchWrites(options.watch_writes);
    InstallCrashReport();
    SetVif1Worker(!options.vif1_sync);
    SetCheckGuesses(options.check_guesses);
    if (!options.vu_snapshot.empty())
        Vu::Snapshots(options.vu_snapshot, u64(options.vu_snapshot_from), options.vu_snapshot_count);
    // A scripted run (--virtual-time) draws the same pictures every time: not
    // with the settings a player keeps beside the executable - the defaults,
    // unless it brings a memory card (and so settings) of its own, or goes on
    // in real time (--real-time-from: measured as played).
    if (!options.virtual_time || !options.memcard.empty() || options.real_time_from) {
        // The PC settings: beside the memory card (its directory's parent),
        // or the executable, so they go where the player's saves go.
        std::filesystem::path where;
        if (!options.memcard.empty()) {
            where = std::filesystem::absolute(options.memcard).parent_path();
        } else {
#ifdef _WIN32
            wchar_t exe[MAX_PATH] = {};
            GetModuleFileNameW(nullptr, exe, MAX_PATH);
            where = std::filesystem::path(exe).parent_path();
#else
            where = std::filesystem::current_path();
#endif
        }
        LoadSettings((where / "pc_settings.ini").string());
    }
    g_fast_disc = options.fast_disc || (!options.virtual_time && SettingOn("fast_loading", g_fast_loading_default));
    if (g_fast_disc) std::fprintf(stderr, "[host] disc reads fast (no drive time)\n");
    // --setting KEY=VALUE: a scripted run's own (a wider picture), for this run.
    for (const std::string& setting : options.settings) {
        const auto equals = setting.find('=');
        if (equals != std::string::npos) RunSetting(setting.substr(0, equals), setting.substr(equals + 1));
    }
    if (options.window) OpenWindow(L"PS2 port", 1280, 896);      // or the game's title (SetGameTitle)
    iop::SetTrace(options.iop_trace);
    gs::TheGs().trace = options.gs_trace;
    SetGsTraceFrame(options.gs_trace_frame);
    SetGsDumpFrame(options.gs_dump_frame);
    if (options.sample) {
        SetSampleFrom(u64(options.sample_from));
        if (options.sample_gs) SetSampleGsThread(true);
        else if (options.sample_vif1) SetSampleVif1Thread(true);
        else if (options.sample_renderer) SetSampleRendererThread(true);
        else StartSampler();
    }
    if (!options.vu_record.empty()) Vu::RecordPrograms(options.vu_record);
    if (!options.memcard.empty()) mc2::SetCardDirectory(options.memcard);
    if (options.vu_interpret) Vu1().use_compiled = Vu0().use_compiled = false;
    Vu1().fast = options.vu_fast || (g_vu_fast_default && !options.vu_exact && SettingOn("vu_fast", true));
    if (Vu1().fast && !options.vu_fast) std::fprintf(stderr, "[host] VU1 fast mode (the game's default; --vu-exact for the exact flags)\n");
    SetUntilFrame(options.until_frame);
    SetFrameTimes(options.frame_times);
    if (options.gpu || options.gpu_shadow) {
        std::string error;
        const GpuMode mode = options.gpu ? GpuMode::Primary : options.gpu_shadow_rtt ? GpuMode::ShadowRtt : GpuMode::Shadow;
        if (!StartGpuRenderer(&error, mode, options.gpu_show))
            std::fprintf(stderr, "%s: %s\n", options.gpu ? "--gpu" : "--gpu-shadow", error.c_str());
    }
    bool sound = false;
    if (options.audio) {
        sound = OpenAudio();
        if (!sound) std::fprintf(stderr, "no sound device: running silent\n");
    }
    if (!options.audio_wav.empty()) {
        if (OpenWavDump(options.audio_wav)) sound = true;
        else std::fprintf(stderr, "cannot write %s\n", options.audio_wav.c_str());
    }
    if (sound) iop::EnableSound();
    for (const auto& press : options.presses)
        if (!ScriptPress(press.button, press.from, press.to, press.pressure))
            std::fprintf(stderr, "--press: no button %s\n", press.button.c_str());
    if (!options.record_input.empty() && !RecordInput(options.record_input))
        std::fprintf(stderr, "--record-input: cannot write %s\n", options.record_input.c_str());
    if (!options.replay_input.empty() && !ReplayInput(options.replay_input))
        std::fprintf(stderr, "--replay-input: nothing to read in %s\n", options.replay_input.c_str());
    SetPaced(options.paced);
    if (!options.frames_dir.empty())
        SetFrameDumps(options.frames_dir, options.frames_every, options.frames_limit, u64(options.frames_from));
    g_watch = options.watch;
    g_watching = !g_watch.empty();
    g_count_calls = (options.virtual_time && !options.real_time_from) || std::getenv("PS2HOST_CALL_COUNTS") != nullptr;
    InstallDevices();
    InstallIopServers();
    if (!options.gs_inline) StartGsThread();
    lifted::Context c = NewContext();
    timeline::NameThread("EE");
    Watchdog watchdog(options.timeout_seconds);
    std::string outcome;
    int code = 2;
    try {
        if (!options.disc_path.empty() && !GuestDisc().Open(options.disc_path))
            throw Unimplemented{"cannot open the disc image " + options.disc_path};
        const u32 entry = options.elf_path.empty() ? LoadBootElf() : LoadElf(options.elf_path);
        // The kernel enters a program with a stack at the top of RAM; the program's
        // own start-up code then asks for its real one (SetupThread).
        c.r[29] = kRdramSize - 0x10;
        c.r[31] = 0;
        std::fprintf(stderr, "[host] entry %s\n", Describe(entry).c_str());
        RunThreads(c, entry, options.loop_budget);
    } catch (const GuestExit& exit) {
        outcome = "the guest exited with code " + std::to_string(exit.code);
        code = 0;
    } catch (const Unimplemented& missing) {
        outcome = "stopped: " + missing.what;
    } catch (const MemoryFault& fault) {
        outcome = std::string("stopped: a ") + (fault.write ? "write" : "read") +
                  " at " + Hex(fault.address) + ", outside guest memory";
    } catch (const lifted::Abort& abort) {
        static const char* reasons[] = {"", "a trap", "a loop that did not finish within the budget",
                                        "a callee that failed", "a memory fault", "a COP2 operation"};
        const char* reason = abort.reason < 6 ? reasons[abort.reason] : "an abort";
        if (abort.reason == 2 && watchdog.fired)
            reason = "the time limit, in the loop";
        outcome = std::string("stopped: ") + reason + " at " + Hex(abort.address);
        if (abort.reason == 2 && watchdog.fired && abort.address == 0)
            outcome = "stopped: the time limit, between threads (no thread was in a loop)";
        if (abort.reason == 2 && UntilFrameReached())
            outcome = "stopped: frame " + std::to_string(options.until_frame) + " reached (--until-frame)";
    }
    // The GS takes what it was sent before the report reads it.
    try {
        StopGsThread();
    } catch (const Unimplemented& missing) {
        if (outcome.empty()) outcome = "stopped: " + missing.what;
    }
    timeline::Finish();
    Report(options, outcome);
    return code;
}

int Main(int argc, char** argv) {
    // PS2HOST_CORES=N: the process held to its first N logical cores from
    // the start, before any thread (the GPU driver's pools too) is made - a
    // machine with that many, to measure a low-end one on a big one. Or a
    // mask (0x5: logical cores 0 and 2 - two physical cores where the first
    // ones are hyperthreaded pairs).
#ifdef _WIN32
    if (const char* cores = std::getenv("PS2HOST_CORES")) {
        DWORD_PTR mask = 0;
        if (cores[0] == '0' && (cores[1] == 'x' || cores[1] == 'X')) {
            mask = DWORD_PTR(std::strtoull(cores + 2, nullptr, 16));
        } else {
            const int n = std::atoi(cores);
            if (n > 0 && n < int(8 * sizeof(DWORD_PTR))) mask = (DWORD_PTR(1) << n) - 1;
        }
        if (mask && SetProcessAffinityMask(GetCurrentProcess(), mask))
            std::fprintf(stderr, "[host] held to cores 0x%llX (PS2HOST_CORES)\n", (unsigned long long)mask);
    }
#endif
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace") options.trace_calls = true;
        else if (arg == "--notes") options.all_notes = true;
        else if (arg == "--iop-trace") options.iop_trace = true;
        else if (arg == "--window") options.window = true;
        else if (arg == "--gs-trace") options.gs_trace = true;
        else if (arg == "--gpu") options.gpu = true;
        else if (arg == "--gpu-shadow") options.gpu_shadow = true;
        else if (arg == "--gpu-shadow-rtt") options.gpu_shadow = options.gpu_shadow_rtt = true;
        else if (arg == "--gpu-show") options.gpu_shadow = options.gpu_shadow_rtt = options.gpu_show = true;
        else if (arg == "--virtual-time") options.virtual_time = true;
        else if (arg == "--until-frame" && i + 1 < argc) options.until_frame = std::stoull(argv[++i]);
        else if (arg == "--frame-times" && i + 1 < argc) options.frame_times = unsigned(std::stoul(argv[++i]));
        else if (arg == "--gs-inline") options.gs_inline = true;
        else if (arg == "--gs-trace-frame" && i + 1 < argc) options.gs_trace_frame = std::stoll(argv[++i]);
        else if (arg == "--gs-dump-frame" && i + 1 < argc) options.gs_dump_frame = std::stoll(argv[++i]);
        else if (arg == "--sample") options.sample = true;
        else if (arg == "--sample-gs") options.sample = options.sample_gs = true;
        else if (arg == "--sample-vif1") options.sample = options.sample_vif1 = true;
        else if (arg == "--sample-renderer") options.sample = options.sample_renderer = true;
        else if (arg == "--vif1-sync") options.vif1_sync = true;
        else if (arg == "--fast-disc") options.fast_disc = true;
        else if (arg == "--check-guesses") options.check_guesses = true;
        else if (arg == "--vu-snapshot" && i + 3 < argc) {
            options.vu_snapshot = argv[++i];
            options.vu_snapshot_from = std::stoll(argv[++i]);
            options.vu_snapshot_count = unsigned(std::stoul(argv[++i]));
        }
        else if (arg == "--vu-bench" && i + 1 < argc) options.vu_bench = argv[++i];
        else if (arg == "--gs-replay" && i + 2 < argc) {
            const std::string file = argv[++i], out = argv[++i];
            return GsReplay(file, out);
        }
        else if (arg == "--sample-from" && i + 1 < argc) {
            options.sample = true;
            options.sample_from = std::stoll(argv[++i]);
        }
        else if (arg == "--vu-record" && i + 1 < argc) options.vu_record = argv[++i];
        else if (arg == "--vu-interpret") options.vu_interpret = true;
        else if (arg == "--vu-fast") options.vu_fast = true;
        else if (arg == "--vu-exact") options.vu_exact = true;
        else if (arg == "--memcard" && i + 1 < argc) options.memcard = argv[++i];
        else if (arg == "--no-memcard") options.memcard.clear();     // (a test on Android: no card, as on a PC)
        else if (arg == "--setting" && i + 1 < argc) options.settings.push_back(argv[++i]);
        else if (arg == "--setting-at" && i + 1 < argc) {
            // --setting-at KEY=VALUE@FRAME: changed mid-run (settings.h).
            const std::string spec = argv[++i];
            const std::size_t equals = spec.find('='), at = spec.rfind('@');
            if (equals == std::string::npos || at == std::string::npos || at < equals)
                std::fprintf(stderr, "--setting-at %s: KEY=VALUE@FRAME\n", spec.c_str());
            else
                ScheduleSetting(spec.substr(0, equals), spec.substr(equals + 1, at - equals - 1),
                                std::stoull(spec.substr(at + 1)));
        }
        else if (arg == "--audio") options.audio = true;
        else if (arg == "--audio-wav" && i + 1 < argc) options.audio_wav = argv[++i];
        else if (arg == "--press" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const std::size_t at = spec.find('@');
            if (at != std::string::npos) {
                std::string when = spec.substr(at + 1);
                u8 pressure = 0xFF;
                const std::size_t colon = when.find(':');
                if (colon != std::string::npos) {
                    pressure = u8(std::min(255, std::max(1, std::atoi(when.c_str() + colon + 1))));
                    when.resize(colon);
                }
                const std::size_t dash = when.find('-');
                const u64 from = std::stoull(when.substr(0, dash));
                const u64 to = dash == std::string::npos ? from + 5 : std::stoull(when.substr(dash + 1));
                options.presses.push_back({spec.substr(0, at), from, to, pressure});
            }
        }
        else if (arg == "--key" && i + 1 < argc) {
            // --key NAME@FROM[-TO]: a key held (six frames without TO).
            const std::string spec = argv[++i];
            const std::size_t at = spec.find('@');
            bool ok = at != std::string::npos;
            if (ok) {
                const std::string when = spec.substr(at + 1);
                const std::size_t dash = when.find('-');
                const u64 from = std::stoull(when.substr(0, dash));
                const u64 to = dash == std::string::npos ? from + 5 : std::stoull(when.substr(dash + 1));
                ok = ScriptKey(spec.substr(0, at), from, to);
            }
            if (!ok) std::fprintf(stderr, "--key %s: KEY@FROM[-TO]\n", spec.c_str());
        }
        else if (arg == "--record-input" && i + 1 < argc) options.record_input = argv[++i];
        else if (arg == "--replay-input" && i + 1 < argc) options.replay_input = argv[++i];
        else if (arg == "--paced") options.paced = true;
        else if (arg == "--real-time-from" && i + 1 < argc) {
            options.real_time_from = std::stoull(argv[++i]);
            SetRealTimeFrom(options.real_time_from);
        }
        else if (arg == "--fast-until" && i + 1 < argc) {
            const std::string spec = argv[++i];
            const std::size_t colon = spec.find(':');
            SetFastUntil(std::stoull(spec.substr(0, colon)), colon == std::string::npos ? 8u : unsigned(std::stoul(spec.substr(colon + 1))));
        }
        else if (arg == "--mouse" && i + 1 < argc) {
            if (!ScriptMouse(argv[++i])) std::fprintf(stderr, "--mouse %s: FRAME:X,Y[,L|R|U|D]\n", argv[i]);
        }
        else if (arg == "--frames" && i + 1 < argc) options.frames_dir = argv[++i];
        else if (arg == "--frames-every" && i + 1 < argc) options.frames_every = std::stoi(argv[++i]);
        else if (arg == "--frames-from" && i + 1 < argc) options.frames_from = std::stoll(argv[++i]);
        else if (arg == "--frames-limit" && i + 1 < argc) options.frames_limit = std::stoi(argv[++i]);
        else if (arg == "--budget" && i + 1 < argc) options.loop_budget = std::stoll(argv[++i]);
        else if (arg == "--limit" && i + 1 < argc) options.report_limit = std::stoi(argv[++i]);
        else if (arg == "--timeout" && i + 1 < argc) options.timeout_seconds = std::stoi(argv[++i]);
        else if (arg == "--discover" && i + 1 < argc) options.discover_path = argv[++i];
        else if (arg == "--disc" && i + 1 < argc) options.disc_path = argv[++i];
        else if (arg == "--watch" && i + 1 < argc)
            options.watch.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg == "--watch-write" && i + 1 < argc)
            options.watch_writes.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg == "--dump-ram" && i + 1 < argc) options.dump_ram = argv[++i];
        else if (arg == "--peek" && i + 1 < argc)
            options.peek.push_back(u32(std::stoul(argv[++i], nullptr, 16)));
        else if (arg.rfind("--", 0) != 0) options.elf_path = arg;
        else {
            std::fprintf(stderr, "unknown option %s\n", arg.c_str());
            return 64;
        }
    }
    if (!options.vu_bench.empty()) {
        // With --sample, a hundred passes under the sampler and its report.
        if (options.sample) {
            StartSampler();
            SetSampleFrom(0);
            SampleFrame(0);
        }
        const int status = Vu::Bench(options.vu_bench, options.sample ? 100 : 5, options.vu_fast);
        if (options.sample) std::printf("%s", SampleReport().c_str());
        return status;
    }
    if (options.elf_path.empty() && options.disc_path.empty()) {
        std::fprintf(stderr, "usage: %s <executable from the game disc> | --disc <image> [--trace] [--notes] [--watch addr] [--peek addr] "
                             "[--budget N] [--limit N] [--timeout seconds]\n", argv[0]);
        return 64;
    }
    return Run(options);
}

}  // namespace ps2host

// The hooks lifted code calls through.
namespace lifted {

void Call(Context& c, u32 target) { ps2host::Dispatch(c, target); }

void TailCall(Context& c, u32 target, u32 /*return_address*/, u32 /*entry_sp*/) {
    // The guest jumps and never comes back here; natively, the callee runs and
    // the translated function returns straight after (it emits `return;`).
    ps2host::Dispatch(c, target);
}

void Syscall(Context& c) { ps2host::KernelSyscall(c); }

// vcallms: VU0's registers are the ones macro mode uses, so the microprogram
// runs on the calling thread's and leaves its results there.
void Vcallms(Context& c, u32 address) {
    ps2host::Vu& vu = ps2host::Vu0();
    std::memcpy(vu.regs.vf, c.vf, sizeof c.vf);
    std::memcpy(vu.regs.vc, c.vc, sizeof c.vc);
    std::memcpy(vu.regs.vacc, c.vacc, sizeof c.vacc);
    vu.Run(address);
    std::memcpy(c.vf, vu.regs.vf, sizeof c.vf);
    std::memcpy(c.vc, vu.regs.vc, sizeof c.vc);
    std::memcpy(c.vacc, vu.regs.vacc, sizeof c.vacc);
}

}  // namespace lifted

// Derived code's hooks: a guest stack for out-parameters, and pointer calls.
namespace generated {

namespace {
u32 g_stack = 0x01F00000u;      // below the guest stack the kernel hands out
}

u64 StackEnter(u32 size) { g_stack -= size; return g_stack; }
void StackLeave(u32 size) { g_stack += size; }

u64 CallPointer(u32 address, u64, u64, u64, u64, u64, u64, u64, u64,
                u64, u64, u64, u64, u64, u64, u64, u64) {
    throw ps2host::Unimplemented{"a derived pointer call to " + std::to_string(address)};
}

}  // namespace generated
