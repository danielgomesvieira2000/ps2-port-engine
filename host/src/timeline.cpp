/*
 * The frame-pacing profiler (timeline.h): per-thread buffers of spans and
 * moments, written at the end as a Chrome trace, with a summary of the
 * frames - their pacing, the dropped ones and what was busy before them.
 */

#include "ps2host/timeline.h"
#include "ps2host/run.h"          // NoteHeavyThread

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace ps2host::timeline {

namespace {

using u64 = std::uint64_t;

struct Event {
    u64 from, to;               // to == from: a moment
    const char* name;
    u64 value;
};

/* A lane's events in fixed chunks: a vector's growth copied the GS thread's
   ~100 MB of them at once, a 16 ms stall in the middle of the run - the
   profile's own worst "late picture". */
class Events {
public:
    static constexpr std::size_t kChunk = 1 << 16;
    bool empty() const { return size_ == 0; }
    Event& front() { return chunks_[0][0]; }
    const Event& front() const { return chunks_[0][0]; }
    Event& back() { return chunks_[(size_ - 1) / kChunk][(size_ - 1) % kChunk]; }
    void push_back(const Event& e) {
        if (size_ % kChunk == 0) chunks_.emplace_back(new Event[kChunk]);
        chunks_.back()[size_ % kChunk] = e;
        ++size_;
    }
    struct Iter {
        const Events* of;
        std::size_t i;
        const Event& operator*() const { return of->chunks_[i / kChunk][i % kChunk]; }
        Iter& operator++() { ++i; return *this; }
        bool operator!=(const Iter& o) const { return i != o.i; }
    };
    Iter begin() const { return {this, 0}; }
    Iter end() const { return {this, size_}; }

private:
    std::vector<std::unique_ptr<Event[]>> chunks_;
    std::size_t size_ = 0;
};

struct Lane {
    std::string name;
    Events events;
};

std::mutex g_lanes_mutex;
std::vector<Lane*>& Lanes() {
    static std::vector<Lane*> lanes;            // kept to the end (a thread's lane outlives it)
    return lanes;
}

const char* Path() {
    static const char* path = std::getenv("PS2HOST_PROFILE");
    return path && *path ? path : nullptr;
}

u64 FromFrame() {
    static const u64 from = std::getenv("PS2HOST_PROFILE_FROM") ? std::strtoull(std::getenv("PS2HOST_PROFILE_FROM"), nullptr, 10) : 0;
    return from;
}

std::atomic<bool> g_on{false};
std::atomic<u64> g_frame{0};

// The frames and pictures, from every thread: rare, so under a lock.
std::mutex g_frames_mutex;
struct Picture { u64 due, shown; bool dropped; };
std::vector<u64> g_blanks;                      // when each frame began (Frame)
std::vector<u64> g_blank_cpu;                   // the process's CPU time then (ns)
std::vector<u64> g_lost;                        // when a blank was lost
std::vector<Picture> g_pictures;

/* The CPU time every thread of the process has had (ns): over a stall, how
   many cores the process kept busy - all it was given (its own work), or few
   (something outside held the cores). */
u64 ProcessCpuNs() {
#ifdef _WIN32
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    return ((u64(kernel.dwHighDateTime) << 32 | kernel.dwLowDateTime) +
            (u64(user.dwHighDateTime) << 32 | user.dwLowDateTime)) * 100;
#else
    return 0;
#endif
}

Lane& ThisLane() {
    thread_local Lane* lane = nullptr;
    if (!lane) {
        lane = new Lane;
        std::lock_guard<std::mutex> hold(g_lanes_mutex);
        lane->name = "thread " + std::to_string(Lanes().size());
        Lanes().push_back(lane);
    }
    return *lane;
}

}  // namespace

bool On() { return g_on.load(std::memory_order_relaxed); }

u64 Now() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count());
}

namespace {
// The named threads, for ThreadCpu: a handle each, and their CPU time and the
// clock at the last report.
struct Named {
    std::string name;
#ifdef _WIN32
    HANDLE handle = nullptr;
#endif
    u64 last_cpu_100ns = 0, last_kernel_100ns = 0;
};
std::mutex g_named_mutex;
std::vector<Named> g_named;
std::chrono::steady_clock::time_point g_named_since = std::chrono::steady_clock::now();
}  // namespace

std::string ThreadCpu() {
    std::lock_guard<std::mutex> hold(g_named_mutex);
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - g_named_since).count();
    g_named_since = now;
    std::string out;
#ifdef _WIN32
    for (Named& n : g_named) {
        FILETIME created, exited, kernel, user;
        if (!GetThreadTimes(n.handle, &created, &exited, &kernel, &user)) continue;
        const u64 cpu = (u64(kernel.dwHighDateTime) << 32 | kernel.dwLowDateTime) +
                        (u64(user.dwHighDateTime) << 32 | user.dwLowDateTime);
        const u64 in_kernel = u64(kernel.dwHighDateTime) << 32 | kernel.dwLowDateTime;
        const u64 used = cpu - n.last_cpu_100ns, used_kernel = in_kernel - n.last_kernel_100ns;
        n.last_cpu_100ns = cpu;
        n.last_kernel_100ns = in_kernel;
        if (seconds <= 0) continue;
        // PS2HOST_CPU_KERNEL=1: the kernel's part of each as well (waits, wakes).
        static const bool kernel_too = std::getenv("PS2HOST_CPU_KERNEL") != nullptr;
        char one[128];
        if (kernel_too)
            std::snprintf(one, sizeof one, "%s%s %d%% (kernel %d%%)", out.empty() ? "" : ", ", n.name.c_str(),
                          int(100.0 * double(used) / 1e7 / seconds + 0.5),
                          int(100.0 * double(used_kernel) / 1e7 / seconds + 0.5));
        else
            std::snprintf(one, sizeof one, "%s%s %d%%", out.empty() ? "" : ", ", n.name.c_str(),
                          int(100.0 * double(used) / 1e7 / seconds + 0.5));
        out += one;
    }
#endif
    return out;
}

void NameThread(const char* name) {
    ps2host::NoteHeavyThread(name);
#ifdef _WIN32
    {
        wchar_t wide[64] = {};
        for (int i = 0; i < 63 && name[i]; ++i) wide[i] = wchar_t(name[i]);
        SetThreadDescription(GetCurrentThread(), wide);
        HANDLE self = nullptr;
        DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
                        THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0);
        std::lock_guard<std::mutex> hold(g_named_mutex);
        Named n;
        n.name = name;
        n.handle = self;
        g_named.push_back(n);
    }
#endif
    if (!Path()) return;
    Lane& lane = ThisLane();
    std::lock_guard<std::mutex> hold(g_lanes_mutex);
    lane.name = name;
}

void Span(const char* name, u64 from_ns, u64 to_ns) {
    if (!On() || to_ns <= from_ns) return;
    Lane& lane = ThisLane();
    if (!lane.events.empty()) {
        Event& last = lane.events.back();
        if (last.name == name && last.to <= from_ns && from_ns - last.to < 20'000 && last.to != last.from) {
            last.to = to_ns;
            return;
        }
    }
    lane.events.push_back({from_ns, to_ns, name, 0});
}

void Instant(const char* name, u64 at_ns, u64 value) {
    if (!On()) return;
    ThisLane().events.push_back({at_ns, at_ns, name, value});
}

void Frame(u64 frame) {
    if (!Path()) return;
    g_frame.store(frame, std::memory_order_relaxed);
    if (!On() && frame >= FromFrame()) g_on.store(true);
    if (!On()) return;
    const u64 now = Now();
    Instant("frame", now, frame);
    const u64 cpu = ProcessCpuNs();
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_blanks.push_back(now);
    g_blank_cpu.push_back(cpu);
}

void BlankLost() {
    if (!On()) return;
    const u64 now = Now();
    Instant("blank lost", now, g_frame.load(std::memory_order_relaxed));
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_lost.push_back(now);
}

void Presented(u64 due_ns, u64 shown_ns) {
    if (!On()) return;
    Span("present", shown_ns > 2'000'000 ? shown_ns - 2'000'000 : shown_ns, shown_ns);   // marked as the 2 ms before it
    if (shown_ns > due_ns + 1'000'000) Instant("shown after its time (us)", shown_ns, (shown_ns - due_ns) / 1000);
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_pictures.push_back({due_ns, shown_ns, false});
}

void Dropped(u64 due_ns) {
    if (!On()) return;
    Instant("picture dropped", Now());
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    g_pictures.push_back({due_ns, 0, true});
}

namespace {

std::string Escape(const std::string& s) {
    std::string out;
    for (char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        out += ch;
    }
    return out;
}

/* How long each lane was busy in [from, to): its spans' overlap, idle
   waits not counted. */
std::vector<std::pair<std::string, double>> BusyIn(u64 from, u64 to) {
    std::map<std::string, double> busy;
    for (const Lane* lane : Lanes())
        for (const Event& e : lane->events) {
            if (e.to == e.from || e.to <= from || e.from >= to || std::strcmp(e.name, "idle") == 0) continue;
            const u64 a = std::max(e.from, from), b = std::min(e.to, to);
            busy[lane->name + ": " + e.name] += double(b - a) / 1e6;
        }
    std::vector<std::pair<std::string, double>> sorted(busy.begin(), busy.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return sorted;
}

}  // namespace

void Finish() {
    if (!Path()) return;
    g_on.store(false);
    std::lock_guard<std::mutex> hold_lanes(g_lanes_mutex);
    std::lock_guard<std::mutex> hold(g_frames_mutex);
    u64 origin = ~u64(0);
    for (const Lane* lane : Lanes())
        if (!lane->events.empty()) origin = std::min(origin, lane->events.front().from);
    if (origin == ~u64(0)) {
        std::fprintf(stderr, "[profile] nothing recorded (PS2HOST_PROFILE_FROM past the run's end?)\n");
        return;
    }
    if (std::FILE* f = std::fopen(Path(), "w")) {
        std::fprintf(f, "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
        bool first = true;
        int tid = 0;
        for (const Lane* lane : Lanes()) {
            ++tid;
            std::fprintf(f, "%s{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s\"}}",
                         first ? "" : ",\n", tid, Escape(lane->name).c_str());
            first = false;
            for (const Event& e : lane->events) {
                const double ts = double(e.from - origin) / 1e3;
                if (e.to == e.from)
                    std::fprintf(f, ",\n{\"ph\":\"i\",\"s\":\"t\",\"name\":\"%s\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"args\":{\"value\":%llu}}",
                                 e.name, tid, ts, (unsigned long long)e.value);
                else
                    std::fprintf(f, ",\n{\"ph\":\"X\",\"name\":\"%s\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f}",
                                 e.name, tid, ts, double(e.to - e.from) / 1e3);
            }
        }
        std::fprintf(f, "\n]}\n");
        std::fclose(f);
    }
    // The summary: the pictures as shown.
    std::vector<u64> shown;
    unsigned dropped = 0;
    for (const Picture& p : g_pictures) {
        if (p.dropped) ++dropped;
        else shown.push_back(p.shown);
    }
    std::sort(shown.begin(), shown.end());
    std::vector<double> intervals;
    for (std::size_t i = 1; i < shown.size(); ++i) intervals.push_back(double(shown[i] - shown[i - 1]) / 1e6);
    std::fprintf(stderr, "\n=== frame pacing (PS2HOST_PROFILE) ===\n");
    std::fprintf(stderr, "%zu blanks, %zu pictures shown, %u dropped before they were shown, %zu blanks lost by the EE\n",
                 g_blanks.size(), shown.size(), dropped, g_lost.size());
    if (intervals.empty()) {
        std::fprintf(stderr, "no pictures shown (no window, or the software GS's): the blanks alone\n");
        std::vector<double> gaps;
        for (std::size_t i = 1; i < g_blanks.size(); ++i) gaps.push_back(double(g_blanks[i] - g_blanks[i - 1]) / 1e6);
        intervals = gaps;
    }
    if (intervals.empty()) return;
    std::vector<double> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    const auto at = [&](double q) { return sorted[std::min(sorted.size() - 1, std::size_t(q * double(sorted.size())))]; };
    const double nominal = at(0.5);
    std::fprintf(stderr, "interval: median %.2f ms, 95%% %.2f, 99%% %.2f, 99.9%% %.2f, longest %.2f\n", nominal, at(0.95),
                 at(0.99), at(0.999), sorted.back());
    // Each interval judged by the blanks of its moment (the last eight's
    // median): a game whose blanks change rate - 144 a second in play, 60 in
    // its cutscenes and menus - is late by its own, not by the run's median
    // (every cutscene picture counted late at 144). A loading screen's blanks
    // run faster than any screen (SetFastForward): not judged.
    const std::vector<u64>& times = shown.size() > 1 ? shown : g_blanks;
    const auto period = [&](std::size_t i) {
        const auto end = std::upper_bound(g_blanks.begin(), g_blanks.end(), times[i + 1]);
        const std::size_t b = std::size_t(end - g_blanks.begin());
        if (b < 3) return nominal;
        double gaps[8];
        std::size_t n = 0;
        for (std::size_t k = b - 1; k > 0 && n < 8; --k) gaps[n++] = double(g_blanks[k] - g_blanks[k - 1]) / 1e6;
        std::sort(gaps, gaps + n);
        return gaps[n / 2];
    };
    const auto late_at = [&](std::size_t i) {
        const double own = period(i);
        return own > 4.0 && intervals[i] > own * 1.5;
    };
    unsigned late = 0, repeats = 0;
    for (std::size_t i = 0; i < intervals.size(); ++i)
        if (late_at(i)) {
            ++late;
            repeats += unsigned(intervals[i] / period(i) + 0.5) - 1;
        }
    std::fprintf(stderr, "frames shown late: %u (%u frames' time the picture repeated)\n", late, repeats);
    // The worst: what was busy in the frame before each.
    std::vector<std::size_t> worst;
    for (std::size_t i = 0; i < intervals.size(); ++i)
        if (late_at(i)) worst.push_back(i);
    std::sort(worst.begin(), worst.end(), [&](std::size_t a, std::size_t b) { return intervals[a] > intervals[b]; });
    if (worst.size() > 12) worst.resize(12);
    for (std::size_t i : worst) {
        const u64 to = times[i + 1], from = to - u64(intervals[i] * 1e6) - u64(nominal * 1e6);
        std::fprintf(stderr, "  at %.3f s: %.1f ms; busy then:", double(to - origin) / 1e9, intervals[i]);
        int n = 0;
        for (const auto& [what, ms] : BusyIn(from, to)) {
            if (n++ == 5) break;
            std::fprintf(stderr, " %s %.1f ms;", what.c_str(), ms);
        }
        // The process's CPU over the blanks around it, in cores.
        const auto first = std::upper_bound(g_blanks.begin(), g_blanks.end(), from);
        const auto last = std::lower_bound(g_blanks.begin(), g_blanks.end(), to);
        if (first != g_blanks.begin() && last != g_blanks.end()) {
            const std::size_t a = std::size_t(first - g_blanks.begin()) - 1, b = std::size_t(last - g_blanks.begin());
            if (b < g_blank_cpu.size() && g_blank_cpu[b] > g_blank_cpu[a])
                std::fprintf(stderr, " process: %.2f cores", double(g_blank_cpu[b] - g_blank_cpu[a]) / double(g_blanks[b] - g_blanks[a]));
        }
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "the timeline: %s (ui.perfetto.dev)\n", Path());
}

}  // namespace ps2host::timeline
