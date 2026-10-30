/*
 * The GS on a thread of its own (run.h: StartGsThread).
 *
 * On the console the GS draws while the EE and VU1 go on; to the EE it is a
 * sink - the host answers CSR reads itself, and nothing the GS does comes back
 * before the frame is shown. So the host's GS model (and the GPU bridge behind
 * it) can take its packets on another thread: the EE side writes each call -
 * a quadword, a run of words, a privileged register - into a ring, in order,
 * and the GS thread makes the same calls in the same order. Nothing reads the
 * GS's state but at the end of a frame (the picture, the GPU's frame, the
 * dumps) and of the run, and there SyncGs waits for the ring to empty: the GS
 * sees the same calls as without the thread, and a scripted run draws the
 * same pictures.
 *
 * One producer (the EE thread, which runs the DMA, VIF and VU1 that feed the
 * GS) and one consumer. An exception on the GS thread is handed to the EE
 * thread at its next call here.
 *
 * Split (PS2HOST_GS_SPLIT=1): the GIF's decoding - tags, PACKED data, the
 * vertices the XYZ registers make (gs_decode.h) - on a thread of its own in
 * front of the GS thread. It takes the ring as the GS thread did and hands
 * what it decodes on through a second ring - register writes the GS keeps,
 * vertices, IMAGE data, privileged registers, frame ends - which the GS
 * thread takes in order: the same calls on the GS model, the same pictures.
 * In the crash-breaker's explosion (~98,000 primitives a picture) the
 * decoding was a quarter of the GS thread's time.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "ps2host/gs.h"
#include "gs_decode.h"
#include "ps2host/timeline.h"
#include "ps2host/profile.h"
#include "ps2host/refresh_lock.h"
#include "ps2host/run.h"

namespace ps2host {

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kRing = 1u << 22;                 // words (16 MB)
// What wakes the GS thread at once: this much waiting (16 KB), or a frame's
// end, a privileged register, the producer waiting for it. Less waits for
// its next look (kLookMs): woken for every few packets, it slept and woke
// 10,000-13,000 times a second - 30% of a core in waits through gameplay.
constexpr u32 kWakeWords = 4096;
constexpr int kLookMs = 1;
constexpr u32 kMaxRun = kRing / 8;              // the longest run of words one entry carries
// The feed's entries, and (split) the second ring's: a register write the
// decoding takes (kRegister: the replay's), and what it hands on - a
// register the GS keeps (kApply), a vertex (kVertex, the path bits 1 if it
// may draw), IMAGE data (kImage, quadwords), a line of the trace (kTrace).
enum : u32 { kQword = 1, kWord = 2, kWords = 3, kPrivileged = 4, kWrap = 5, kFrame = 6, kRegister = 7, kApply = 8,
             kVertex = 9, kImage = 10, kTrace = 11 };
constexpr u32 kVertexWords = sizeof(gs::Vertex) / 4;
static_assert(sizeof(gs::Vertex) % 4 == 0, "a vertex is whole words");

std::atomic<u64> g_gs_busy_us{0};       // the GS thread at work, not waiting for more (--frame-times)

// GsFrame's work, on the GS thread at the frame's place in the stream.
std::function<void(u64)> g_frame_handler;

// When the EE ended each of the last frames (GsFrame), by frame number.
std::atomic<u64> g_frame_ended[256];

// PS2HOST_HITCH: each frame's way to the screen - when the EE ended it (and
// its idle time and VIF1's worker's busy time then), when the worker passed
// it on, when the GS thread took and finished it - and a frame that came a
// period late, or took long on its way, told.
struct FrameWay { std::atomic<u64> ee{0}, worker{0}, idle{0}, worker_us{0}; };
FrameWay g_way[256];
const bool g_hitch = std::getenv("PS2HOST_HITCH") != nullptr;

void HitchFrame(u64 frame, u64 took, u64 done) {
    static u64 last_blank = 0, last_busy = 0;
    const u64 blank = g_frame_ended[frame & 255].load(std::memory_order_relaxed);
    const u64 busy = g_gs_busy_us.load(std::memory_order_relaxed);
    const u64 gap = blank && last_blank && blank > last_blank ? blank - last_blank : 0;
    const auto ms = [&](u64 t) { return t && blank ? (double(t) - double(blank)) / 1e6 : -1.0; };
    const FrameWay& w = g_way[frame & 255];
    const FrameWay& before = g_way[(frame - 1) & 255];
    const u64 idle = w.idle.load(std::memory_order_relaxed), idle_before = before.idle.load(std::memory_order_relaxed);
    const u64 vu = w.worker_us.load(std::memory_order_relaxed), vu_before = before.worker_us.load(std::memory_order_relaxed);
    const u64 ee = w.ee.load(std::memory_order_relaxed), ee_before = before.ee.load(std::memory_order_relaxed);
    if (ms(done) > 22.0 || gap > 20'000'000)
        std::fprintf(stderr, "[hitch] frame %llu: blank %+.1f ms after the last; EE %.1f, VIF1 %.1f, GS took %.1f done %.1f ms after it; GS busy %.1f ms; EE frame %.1f ms, idle %.1f, VIF1 worker busy %.1f\n",
                     (unsigned long long)frame, last_blank ? gap / 1e6 : 0.0, ms(ee),
                     ms(w.worker.load(std::memory_order_relaxed)), ms(took), ms(done),
                     (busy - last_busy) / 1e3, ee > ee_before ? (ee - ee_before) / 1e6 : -1.0,
                     idle >= idle_before ? (idle - idle_before) / 1e6 : -1.0,
                     vu >= vu_before ? (vu - vu_before) / 1e3 : -1.0);
    last_blank = blank;
    last_busy = busy;
}

// With VIF1 on its worker (vif1_worker.cpp) the worker feeds the GS: what
// the EE would feed it goes through the worker's queue, in turn (privileged
// registers, frame ends), or waits for the queue to empty first.
struct Producing {
    Producing() {
        if (Vif1Async() && !OnVif1Worker()) JoinVif1("producing");
    }
};

u32 Header(u32 type, int path, u32 count) { return type << 28 | u32(path & 0xF) << 24 | (count & 0xFFFFFF); }

void Pause() {
#if defined(_MSC_VER)
    _mm_pause();
#endif
}

struct Feed {
    std::vector<u32> ring = std::vector<u32>(kRing);
    // Positions in words, counted from the start (the ring index is mod kRing).
    std::atomic<u64> head{0};                   // published up to, by the EE thread
    std::atomic<u64> tail{0};                   // done up to, by the GS thread
    u64 written = 0;                            // written up to (the producer's own; head trails it)
    std::atomic<bool> stop{false};
    std::atomic<bool> sleeping{false};
    std::mutex mutex;
    std::condition_variable wake;
    // The producer asleep for room (or for the ring to empty: Sync), the
    // GS thread waking it as it takes entries (Room).
    std::atomic<bool> waiting{false};
    std::mutex room_mutex;
    std::condition_variable room;
    std::exception_ptr failed;                  // set by the GS thread, taken by the EE's
    std::atomic<bool> has_failed{false};
    std::thread thread;

    ~Feed() { Stop(); }

    void Stop() {
        if (!thread.joinable()) return;
        stop = true;
        {
            std::lock_guard<std::mutex> lock(mutex);
            wake.notify_one();
        }
        thread.join();
    }

    void Rethrow() {
        if (has_failed.load()) {
            has_failed = false;
            std::rethrow_exception(failed);
        }
    }

    /* Room for `words` contiguous words: their index in the ring. */
    u32* Reserve(u32 words) {
        Rethrow();
        u64 at = written;
        if (at % kRing + words > kRing) {       // no room before the end: wrap
            WaitFor(kRing - at % kRing);
            ring[at % kRing] = Header(kWrap, 0, 0);
            at += kRing - at % kRing;
            written = at;
            head.store(at, std::memory_order_release);
        }
        WaitFor(words);
        return &ring[at % kRing];
    }

    void WaitFor(u32 words) {
        const u64 at = written;
        if (at + words - tail.load(std::memory_order_acquire) <= kRing) return;
        head.store(at, std::memory_order_release);      // all that is written, for the consumer to make room
        Wake();
        Timed timed(Section::GsWait);
        Await([&] { return at + words - tail.load(std::memory_order_acquire) <= kRing; });
    }

    /* Until `done`: a moment spinning, then asleep until the GS thread has
       taken more (Room). Spinning the whole wait (yield) held a core while
       the GS thread was full - VIF1's worker 53-62% of a core on two cores,
       taken from the GS thread it waited for. */
    template <class Done>
    void Await(Done done) {
        for (int spins = 0; !done(); ++spins) {
            Rethrow();
            if (spins < 256) {
                Pause();
            } else if (spins < 256 + 16) {
                std::this_thread::yield();
            } else {
                std::unique_lock<std::mutex> lock(room_mutex);
                waiting.store(true);
                room.wait_for(lock, std::chrono::milliseconds(1), [&] { return done() || has_failed.load(); });
                waiting.store(false);
            }
        }
    }

    /* The GS thread, having taken entries: a producer waiting woken. */
    void Room() {
        if (!waiting.load(std::memory_order_relaxed)) return;
        std::lock_guard<std::mutex> lock(room_mutex);
        room.notify_all();
    }

    void Publish(u32 words, bool now = false) {
        written += words;
        Flush(now);
    }

    /* Written, published later (Flush) - or as soon as there is enough for
       the consumer to go on with. */
    void Append(u32 words) {
        written += words;
        if (written - head.load(std::memory_order_relaxed) >= kWakeWords) Flush(false);
    }

    void Flush(bool now) {
        const u64 h = written;
        if (h != head.load(std::memory_order_relaxed)) head.store(h, std::memory_order_release);
        if (sleeping.load() && (now || h - tail.load(std::memory_order_acquire) >= kWakeWords)) Wake();
    }

    void Wake() {
        if (!sleeping.load()) return;
        std::lock_guard<std::mutex> lock(mutex);
        wake.notify_one();
    }

    void Sync() {
        const u64 target = head.load(std::memory_order_relaxed);
        if (tail.load(std::memory_order_acquire) >= target) return Rethrow();
        Wake();
        Timed timed(Section::GsWait);
        Await([&] { return tail.load(std::memory_order_acquire) >= target; });
        Rethrow();
    }

    /* Takes entries until stopped: `handle(entry, index)` does one and
       says how many words it was. `busy`: its time is the GS's (--frame-
       times' share). */
    template <class Handle>
    void Run(Handle handle, bool busy_counts) {
        while (true) {
            u64 t = tail.load(std::memory_order_relaxed);
            u64 h = head.load(std::memory_order_acquire);
            if (t == h) {
                // Briefly busy, then asleep until there is more. Briefly: a
                // pause is up to 140 cycles (Skylake on), and 200 of them each
                // time the ring ran dry were a third of the thread's time in
                // gameplay (20-26% of a core -> 14-16% with 16, the pictures'
                // pacing the same). PS2HOST_GS_SPIN to measure another.
                static const int kSpins = [] {
                    const char* n = std::getenv("PS2HOST_GS_SPIN");
                    return n ? std::atoi(n) : 16;
                }();
                for (int spins = 0; spins < kSpins && t == h; ++spins) {
                    Pause();
                    h = head.load(std::memory_order_acquire);
                }
                if (t == h) {
                    std::unique_lock<std::mutex> lock(mutex);
                    sleeping = true;
                    wake.wait_for(lock, std::chrono::milliseconds(kLookMs),
                                  [&] { return stop.load() || head.load() != tail.load(); });
                    sleeping = false;
                    if (stop.load() && head.load() == tail.load()) return;
                    continue;
                }
            }
            const auto began = std::chrono::steady_clock::now();
            struct Busy {
                std::chrono::steady_clock::time_point began;
                bool counts;
                ~Busy() {
                    const auto now = std::chrono::steady_clock::now();
                    if (counts)
                        g_gs_busy_us.fetch_add(u64(std::chrono::duration_cast<std::chrono::microseconds>(now - began).count()),
                                               std::memory_order_relaxed);
                    if (timeline::On())
                        timeline::Span("busy", u64(std::chrono::duration_cast<std::chrono::nanoseconds>(began.time_since_epoch()).count()),
                                       u64(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()));
                }
            } busy{began, busy_counts};
            StallWorkBegan();
            try {
                while (t < h) {
                    const u32 at = u32(t % kRing);
                    t += handle(&ring[at], at);
                    tail.store(t, std::memory_order_release);
                    Room();
                }
            } catch (...) {
                failed = std::current_exception();
                has_failed = true;
                // Skip what is left: the EE thread stops at its next call.
                tail.store(head.load(std::memory_order_acquire), std::memory_order_release);
                Room();
            }
            StallWorkEnded();
        }
    }
};

Feed* g_feed = nullptr;
Feed* g_back = nullptr;         // split: the decoding's to the GS thread

Feed& TheFeed() {
    static Feed feed;
    return feed;
}

Feed& TheBack() {
    static Feed back;
    return back;
}

u64 Value(const u32* at) { return u64(at[0]) | u64(at[1]) << 32; }

void FrameEnd(const u32* e) {
    gs::TheGs().RecordBlank();          // (a capture's replay.bin: the frames apart)
    if (!g_frame_handler) return;
    const u64 frame = Value(e + 1);
    const u64 took = g_hitch ? ClockNs() : 0;
    g_frame_handler(frame);
    if (g_hitch) HitchFrame(frame, took, ClockNs());
}

/* The GS thread's entries, unsplit: the calls the EE side made. */
u32 Whole(const u32* e, u32 at) {
    gs::Gs& g = gs::TheGs();
    const u32 type = e[0] >> 28, count = e[0] & 0xFFFFFF;
    const int path = int((e[0] >> 24) & 0xF);
    switch (type) {
        case kQword: g.Qword(path, Value(e + 1), Value(e + 3)); return 5;
        case kWord: g.Word(path, e[1]); return 2;
        case kWords: g.Words(path, e + 2, count, e[1]); return 2 + count;
        case kPrivileged: g.SetPrivileged(e[1], Value(e + 2)); return 4;
        case kRegister: g.WriteRegister(e[1], Value(e + 2)); return 4;
        case kFrame: FrameEnd(e); return 3;
        case kWrap: return kRing - at;
        default: return 1;
    }
}

/* The decoding's sink (split): what it decodes, into the second ring. */
struct RingSink {
    Feed& f;
    void Apply(u32 address, u64 value) {
        u32* e = f.Reserve(4);
        e[0] = Header(kApply, 0, 0);
        e[1] = address;
        e[2] = u32(value);
        e[3] = u32(value >> 32);
        f.Append(4);
    }
    void PushVertex(const gs::Vertex& v, bool draw) {
        u32* e = f.Reserve(1 + kVertexWords);
        e[0] = Header(kVertex, draw ? 1 : 0, 0);
        std::memcpy(e + 1, &v, sizeof v);
        f.Append(1 + kVertexWords);
    }
    void Image(const gs::u8* data, u32 qwords) {
        while (qwords) {
            const u32 run = std::min(qwords, kMaxRun / 4);
            u32* e = f.Reserve(1 + 4 * run);
            e[0] = Header(kImage, 0, run);
            std::memcpy(e + 1, data, std::size_t(run) * 16);
            f.Append(1 + 4 * run);
            data += std::size_t(run) * 16;
            qwords -= run;
        }
    }
    void ApplyPrivileged(u32 address, u64 value) {
        u32* e = f.Reserve(4);
        e[0] = Header(kPrivileged, 0, 0);
        e[1] = address;
        e[2] = u32(value);
        e[3] = u32(value >> 32);
        f.Append(4);
    }
    // The recording's file is the GS model's, written by the decoding alone
    // while it records (begun and ended with the GS synced).
    void Record(int path, const u32* words, u32 count) { gs::TheGs().Record(path, words, count); }
    void RecordPrivileged(u32 address, u64 value) { gs::TheGs().RecordPrivileged(address, value); }
    bool Tracing() const { return gs::TheGs().trace; }
    void TraceLine(const char* text) {
        const u32 bytes = u32(std::strlen(text)) + 1, words = (bytes + 3) / 4;
        u32* e = f.Reserve(1 + words);
        e[0] = Header(kTrace, 0, words);
        e[words] = 0;
        std::memcpy(e + 1, text, bytes);
        f.Append(1 + words);
    }
};

/* The decoding's thread (split): the EE side's calls decoded, handed on. */
u32 Decode(const u32* e, u32 at) {
    gs::Gs& g = gs::TheGs();
    RingSink sink{*g_back};
    using D = gs::GifDecode<RingSink>;
    const u32 type = e[0] >> 28, count = e[0] & 0xFFFFFF;
    const int path = int((e[0] >> 24) & 0xF);
    u32 size = 1;
    bool now = false;
    switch (type) {
        case kQword: D::Qword(g.gif, sink, path, Value(e + 1), Value(e + 3)); size = 5; break;
        case kWord: D::Word(g.gif, sink, path, e[1]); size = 2; break;
        case kWords: D::Words(g.gif, sink, path, e + 2, count, e[1]); size = 2 + count; break;
        case kPrivileged: D::Privileged(g.gif, sink, e[1], Value(e + 2)); size = 4; now = true; break;
        case kRegister: D::Register(g.gif, sink, e[1], Value(e + 2)); size = 4; break;
        case kFrame: {
            u32* out = g_back->Reserve(3);
            std::memcpy(out, e, 12);
            g_back->Append(3);
            size = 3;
            now = true;
            break;
        }
        case kWrap: size = kRing - at; break;
        default: break;
    }
    // All it made for this entry, published before the entry counts as
    // taken (SyncGs waits for the feed, then for this ring).
    g_back->Flush(now);
    return size;
}

/* The GS thread's entries (split): the decoding's. */
u32 Back(const u32* e, u32 at) {
    gs::Gs& g = gs::TheGs();
    const u32 type = e[0] >> 28, count = e[0] & 0xFFFFFF;
    switch (type) {
        case kApply: g.Apply(e[1], Value(e + 2)); return 4;
        case kVertex: {
            gs::Vertex v;
            std::memcpy(&v, e + 1, sizeof v);
            g.PushVertex(v, ((e[0] >> 24) & 1) != 0);
            return 1 + kVertexWords;
        }
        case kImage: g.Image(reinterpret_cast<const gs::u8*>(e + 1), count); return 1 + 4 * count;
        case kPrivileged: g.ApplyPrivileged(e[1], Value(e + 2)); return 4;
        case kFrame: FrameEnd(e); return 3;
        case kTrace: g.TraceLine(reinterpret_cast<const char*>(e + 1)); return 1 + count;
        case kWrap: return kRing - at;
        default: return 1;
    }
}

}  // namespace

bool GsSplit() {
    static const bool on = [] {
        const char* v = std::getenv("PS2HOST_GS_SPLIT");
        return v && *v == '1';
    }();
    return on;
}

void StartGsThread() {
    if (g_feed) return;
    g_feed = &TheFeed();
    if (GsSplit()) {
        g_back = &TheBack();
        g_back->thread = std::thread([] {
            timeline::NameThread("GS");
            FastThread();
            if (SampleGsThread()) StartSampler();
            g_back->Run(Back, true);
        });
        g_feed->thread = std::thread([] {
            timeline::NameThread("GS decode");
            FastThread();
            g_feed->Run(Decode, false);
        });
        std::fprintf(stderr, "[host] GS split: the GIF's decoding on a thread of its own\n");
        return;
    }
    StartStallWatch();
    g_feed->thread = std::thread([] {
        timeline::NameThread("GS");
        FastThread();
        if (SampleGsThread()) StartSampler();
        g_feed->Run(Whole, true);
    });
}

void GsDrain() {
    if (Vif1Async() && !OnVif1Worker()) JoinVif1("drain");
    if (g_feed) g_feed->Sync();
    if (g_back) g_back->Sync();
}

void StopGsThread() {
    if (!g_feed) return;
    // VIF1's worker first: what it still has queued (a VU1 program's XGKICKs,
    // DIRECT) feeds the GS thread, and a feed gone under it was a crash at the
    // run's end (GsWords between its check of g_feed and its use; seen with
    // the car select's garage at 60 pictures a second keeping VU1 busy). The
    // EE has stopped: nothing more is queued. A failure of the worker's at
    // this point changes nothing - the run is over.
    if (Vif1Async() && !OnVif1Worker()) {
        try {
            JoinVif1("stop");
        } catch (...) {
        }
    }
    g_feed->Sync();
    if (g_back) g_back->Sync();
    g_feed->Stop();
    g_feed = nullptr;
    if (g_back) {
        g_back->Stop();
        g_back = nullptr;
    }
}

void SyncGs() {
    JoinVif1("syncgs");
    if (g_feed) g_feed->Sync();
    if (g_back) g_back->Sync();
}

void GsRegister(u32 address, u64 value) {
    if (!g_feed) return gs::TheGs().WriteRegister(address, value);
    u32* e = g_feed->Reserve(4);
    e[0] = Header(kRegister, 0, 0);
    e[1] = address;
    e[2] = u32(value);
    e[3] = u32(value >> 32);
    g_feed->Publish(4);
}

void GsQword(int path, u64 low, u64 high) {
    if (!g_feed) return gs::TheGs().Qword(path, low, high);
    Producing producing;
    u32* e = g_feed->Reserve(5);
    e[0] = Header(kQword, path, 0);
    e[1] = u32(low);
    e[2] = u32(low >> 32);
    e[3] = u32(high);
    e[4] = u32(high >> 32);
    g_feed->Publish(5);
}

void GsWord(int path, u32 word) {
    if (!g_feed) return gs::TheGs().Word(path, word);
    Producing producing;
    u32* e = g_feed->Reserve(2);
    e[0] = Header(kWord, path, 0);
    e[1] = word;
    g_feed->Publish(2);
}

void GsWords(int path, const u32* words, u32 count, u32 from) {
    if (!g_feed) return gs::TheGs().Words(path, words, count, from);
    Producing producing;
    while (count) {
        const u32 run = count < kMaxRun ? count : kMaxRun;
        u32* e = g_feed->Reserve(2 + run);
        e[0] = Header(kWords, path, run);
        e[1] = from;
        std::memcpy(e + 2, words, std::size_t(run) * 4);
        g_feed->Publish(2 + run);
        words += run;
        count -= run;
        if (from != gs::Gs::kNoSource) from += 4 * run;
    }
}

void SetGsFrameHandler(std::function<void(u64)> handler) { g_frame_handler = std::move(handler); }


u64 FrameEndedNs(u64 frame) { return g_frame_ended[frame & 255].load(std::memory_order_relaxed); }

/* The steady clock, as the vertical blank's timer and the display keep time
   (ClockNs goes on from a virtual clock's time after --real-time-from). */
u64 SteadyNowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
}

u64 FrameDue(u64 frame) {
    // The delay each frame came with, the longest of the last 120 (two
    // seconds: a race's heavy moments come together) the one
    // every frame keeps to (and half a millisecond more), so frames go out
    // a frame's period apart however much each took on its way; never more
    // than a period and a half behind - a frame later than that goes at once.
    static u64 delays[120] = {};
    static int next = 0;
    // The blanks' rate changed: the delays at the old forgotten (the frame
    // across the change came a period late; kept, every picture after it
    // waited that long, the most, for two seconds).
    static u64 period = 0;
    if (BlankPeriodNs() != period) {
        period = BlankPeriodNs();
        std::fill(std::begin(delays), std::end(delays), u64(0));
    }
    const u64 ended = FrameEndedNs(frame);
    const u64 now = SteadyNowNs();
    if (!ended || now < ended) return now;
    delays[next] = now - ended;
    next = (next + 1) % 120;
    if (timeline::On()) timeline::Instant("frame's delay (us)", timeline::Now(), (now - ended) / 1000);
    u64 keep = 0;
    for (u64 d : delays) keep = std::max(keep, d);
    // At least 6 ms: a frame that alone takes longer on its way (a heavy
    // frame's VU1 work, 12 ms past its blank) still goes out on time, for
    // under a third of a frame's delay where all is quick.
    // And under two periods of the blanks: the display keeps three pictures,
    // one shown - two waiting; held back more (at 144 a second, 25 ms is
    // 3.6 periods), the oldest waiting was dropped unshown, 3 a second.
    const u64 most = std::min<u64>(25'000'000, 2 * BlankPeriodNs() - 1'000'000);
    keep = std::min<u64>(std::max<u64>(keep + 500'000, std::min<u64>(6'000'000, most)), most);
    u64 due = std::max(now, ended + keep);
    // The blanks locked to the screen's (RefreshLocked): a picture on time
    // handed over in the middle of a refresh, then shown at the next one
    // whichever way a wake-up errs - at a refresh's very end it was now that
    // one, now the next (a picture shown twice, then one dropped). Never more
    // than `most` behind its blank.
    double refresh = 0;
    u64 screen_blank = 0;
    if (due > now && RefreshLocked(&refresh, &screen_blank)) {
        u64 middle = due + u64(refresh_lock::ToMiddle(double(due) - double(screen_blank), refresh));
        if (middle - ended > most) middle -= u64(refresh);      // (the middle before)
        due = std::max(now, middle);
    }
    // Its blank to its time to be shown: the host's share of what a press
    // takes to reach the screen (the display's refresh and compositor after).
    if (timeline::On()) timeline::Instant("due after its blank (us)", timeline::Now(), (due - ended) / 1000);
    return due;
}

u64 GsBusyMicroseconds() { return g_gs_busy_us.load(std::memory_order_relaxed); }

bool GsFrame(u64 frame) {
    if (!g_frame_handler) return false;
    if (!OnVif1Worker()) {
        // The frame's vertical blank as the timer had it: frames a period
        // apart however late the EE took each (a long loop runs on to its
        // next loop head first).
        const u64 blank = VirtualClock() ? 0 : LastVblankNs();
        g_frame_ended[frame & 255].store(blank ? blank : SteadyNowNs(), std::memory_order_relaxed);
        if (g_hitch) {
            g_way[frame & 255].ee.store(ClockNs(), std::memory_order_relaxed);
            g_way[frame & 255].idle.store(IdleNanoseconds(), std::memory_order_relaxed);
            g_way[frame & 255].worker_us.store(Vif1BusyMicroseconds(), std::memory_order_relaxed);
        }
    } else if (g_hitch) {
        g_way[frame & 255].worker.store(ClockNs(), std::memory_order_relaxed);
    }
    if (Vif1Async() && !OnVif1Worker()) {
        Vif1RunFrame([frame] { GsFrame(frame); });
        return true;
    }
    if (!g_feed) {
        g_frame_handler(frame);
        return true;
    }
    u32* e = g_feed->Reserve(3);
    e[0] = Header(kFrame, 0, 0);
    e[1] = u32(frame);
    e[2] = u32(frame >> 32);
    g_feed->Publish(3, true);
    return true;
}

void GsPrivileged(u32 address, u64 value) {
    if (Vif1Async() && !OnVif1Worker()) {
        Vif1Run([address, value] { GsPrivileged(address, value); });
        return;
    }
    if (!g_feed) return gs::TheGs().SetPrivileged(address, value);
    u32* e = g_feed->Reserve(4);
    e[0] = Header(kPrivileged, 0, 0);
    e[1] = address;
    e[2] = u32(value);
    e[3] = u32(value >> 32);
    g_feed->Publish(4, true);
}

}  // namespace ps2host
