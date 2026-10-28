/*
 * What the platform the port runs on has (ps2host/run.h): a PC's window and
 * keyboard, or a phone's touch controls - for a port's settings, which show
 * only the rows that mean something there. Android's through the activity
 * (PortActivity's methods, over JNI).
 */

#include "ps2host/run.h"

#ifdef __ANDROID__
#include <jni.h>
#include <SDL3/SDL.h>
#include "ps2gpu/display.h"

#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <time.h>
#include <vector>
#endif

namespace ps2host {

#ifdef __ANDROID__
namespace {
std::atomic<bool> g_keyboard{false};    // the activity's nativeKeyboard
std::atomic<float> g_tilt{0.0f};        // the activity's nativeTilt, and when (steady clock, ns)
std::atomic<long long> g_tilt_at{0};
long long Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* The activity's setTouchShown(shown), from a thread of its own: the game's
   run on fibers, whose stacks ART takes for an overflow (a StackOverflowError
   left pending, and the next call into Java aborts). */
void TellActivity(bool shown) {
    std::thread([shown] {
        JNIEnv* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
        jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
        if (!env || !activity) return;
        jclass type = env->GetObjectClass(activity);
        jmethodID method = env->GetMethodID(type, "setTouchShown", "(Z)V");
        if (method) env->CallVoidMethod(activity, method, jboolean(shown));
        if (env->ExceptionCheck()) env->ExceptionClear();
        env->DeleteLocalRef(type);
        env->DeleteLocalRef(activity);
    }).join();
}
}  // namespace

namespace {

/* The port's heavy threads on a phone (NoteHeavyThread): on the big cores,
   and in a performance hint session (Android 13's ADPF) told each frame how
   long the busiest of them worked - the governor then clocks them for the
   frame's budget at once, rather than ramping up late after a scene's work
   grows (or putting a thread that waits each frame on a little core). */
struct HeavyThread {
    std::string name;
    long long load_ns = 0;              // its CPU time when HeavyThreadLoad last looked
    pid_t tid;
    clockid_t clock;
    long long cpu_ns = 0;               // its CPU time at the last frame
};
std::mutex g_heavy_lock;
std::vector<HeavyThread> g_heavy;
bool g_heavy_changed = false;

long long ThreadCpuNs(clockid_t clock) {
    timespec t{};
    if (clock_gettime(clock, &t) != 0) return -1;
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

/* The cores of the fastest clusters: those whose highest clock is at least
   three quarters of the fastest's (Tensor G2: the X1s and the A78s, not the
   A55s). Empty where the system says nothing. */
cpu_set_t BigCores(int* count) {
    cpu_set_t set;
    CPU_ZERO(&set);
    *count = 0;
    std::vector<long> max_khz;
    const long cpus = std::min<long>(sysconf(_SC_NPROCESSORS_CONF), CPU_SETSIZE);
    for (int cpu = 0; cpu < cpus; ++cpu) {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        long khz = 0;
        if (FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &khz) != 1) khz = 0;
            std::fclose(f);
        }
        max_khz.push_back(khz);
    }
    const long top = max_khz.empty() ? 0 : *std::max_element(max_khz.begin(), max_khz.end());
    if (top <= 0) return set;
    int fewer = 0;
    for (std::size_t cpu = 0; cpu < max_khz.size(); ++cpu) {
        if (max_khz[cpu] * 4 >= top * 3) {
            CPU_SET(int(cpu), &set);
            ++*count;
        } else if (max_khz[cpu] > 0) {
            ++fewer;
        }
    }
    if (!fewer) *count = 0;             // one cluster: nothing to choose
    return set;
}

// ADPF (android/performance_hint.h, API 33), looked up: older systems lack it.
struct Adpf {
    void* (*get_manager)() = nullptr;
    void* (*create)(void*, const int32_t*, size_t, int64_t) = nullptr;
    int (*report)(void*, int64_t) = nullptr;
    void (*close)(void*) = nullptr;
    void* manager = nullptr;
    void* session = nullptr;
    bool tried = false;
};
Adpf g_adpf;
constexpr long long kTargetNs = 13300000;   // the busiest thread's work a frame: 80% of a 60 Hz frame

bool OpenAdpf() {
    if (g_adpf.tried) return g_adpf.manager != nullptr;
    g_adpf.tried = true;
    if (const char* off = std::getenv("PS2HOST_ADPF"); off && std::strcmp(off, "0") == 0) return false;
    void* library = dlopen("libandroid.so", RTLD_NOW);
    if (!library) return false;
    g_adpf.get_manager = reinterpret_cast<void* (*)()>(dlsym(library, "APerformanceHint_getManager"));
    g_adpf.create = reinterpret_cast<void* (*)(void*, const int32_t*, size_t, int64_t)>(
        dlsym(library, "APerformanceHint_createSession"));
    g_adpf.report = reinterpret_cast<int (*)(void*, int64_t)>(dlsym(library, "APerformanceHint_reportActualWorkDuration"));
    g_adpf.close = reinterpret_cast<void (*)(void*)>(dlsym(library, "APerformanceHint_closeSession"));
    if (!g_adpf.get_manager || !g_adpf.create || !g_adpf.report || !g_adpf.close) return false;
    g_adpf.manager = g_adpf.get_manager();
    return g_adpf.manager != nullptr;
}
}  // namespace

void NoteHeavyThread(const char* name) {
    static const char* const kHeavy[] = {"EE", "GS", "GS decode", "VIF1 worker", "GPU renderer"};
    pthread_setname_np(pthread_self(), std::string(name).substr(0, 15).c_str());   // (for the system's profilers)
    bool heavy = false;
    for (const char* h : kHeavy) heavy = heavy || std::strcmp(h, name) == 0;
    if (!heavy) return;
    static int big_count = -1;
    static cpu_set_t big;
    static std::once_flag once;
    std::call_once(once, [] {
        big = BigCores(&big_count);
        std::fprintf(stderr, "[android] %d big cores for the heavy threads%s\n", big_count,
                     big_count ? "" : " (none told apart: left to the system)");
    });
    const char* pin = std::getenv("PS2HOST_PIN");
    if (big_count > 0 && !(pin && std::strcmp(pin, "0") == 0)) sched_setaffinity(0, sizeof big, &big);
    HeavyThread t;
    t.name = name;
    t.tid = gettid();
    if (pthread_getcpuclockid(pthread_self(), &t.clock) != 0) return;
    t.cpu_ns = t.load_ns = ThreadCpuNs(t.clock);
    std::lock_guard<std::mutex> hold(g_heavy_lock);
    g_heavy.push_back(t);
    g_heavy_changed = true;
}

void HeavyThreadsFrame() {
    if (!OpenAdpf()) return;
    std::lock_guard<std::mutex> hold(g_heavy_lock);
    if (g_heavy.empty()) return;
    if (g_heavy_changed) {
        g_heavy_changed = false;
        if (g_adpf.session) g_adpf.close(g_adpf.session);
        std::vector<int32_t> tids;
        for (const HeavyThread& t : g_heavy) tids.push_back(t.tid);
        g_adpf.session = g_adpf.create(g_adpf.manager, tids.data(), tids.size(), kTargetNs);
        std::fprintf(stderr, "[android] performance hint session for %zu threads: %s\n", tids.size(),
                     g_adpf.session ? "open" : "refused");
    }
    if (!g_adpf.session) return;
    long long busiest = 0;
    for (HeavyThread& t : g_heavy) {
        const long long now = ThreadCpuNs(t.clock);
        if (now < 0) continue;          // (gone)
        busiest = std::max(busiest, now - t.cpu_ns);
        t.cpu_ns = now;
    }
    if (busiest > 0) g_adpf.report(g_adpf.session, busiest);
}

std::string HeavyThreadLoad() {
    static auto since = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - since).count();
    since = now;
    std::lock_guard<std::mutex> hold(g_heavy_lock);
    std::string out;
    for (HeavyThread& t : g_heavy) {
        const long long cpu = ThreadCpuNs(t.clock);
        if (cpu < 0) continue;
        const long long used = cpu - t.load_ns;
        t.load_ns = cpu;
        if (seconds <= 0) continue;
        const char* shown = t.name == "VIF1 worker" ? "VU1" : t.name == "GPU renderer" ? "RENDER" : t.name.c_str();
        char one[48];
        std::snprintf(one, sizeof one, "%s%s %d%%", out.empty() ? "CPU " : "  ", shown,
                      int(100.0 * double(used) / 1e9 / seconds + 0.5));
        out += one;
    }
    return out;
}

bool OnPhone() { return true; }
bool KeyboardPresent() { return g_keyboard.load(std::memory_order_relaxed); }
bool TouchControlsShown() { return ps2gpu::TouchPadShown(); }
void SetTouchControls(bool shown) {
    ps2gpu::ShowTouchPad(shown);        // at once: the activity's own note comes from its thread
    TellActivity(shown);
}
bool Tilt(float* degrees) {
    if (Now() - g_tilt_at.load(std::memory_order_relaxed) > 500000000LL) return false;
    *degrees = g_tilt.load(std::memory_order_relaxed);
    return true;
}
#else
void NoteHeavyThread(const char*) {}
void HeavyThreadsFrame() {}
std::string HeavyThreadLoad() { return {}; }
bool OnPhone() { return false; }
bool KeyboardPresent() { return true; }
bool TouchControlsShown() { return false; }
void SetTouchControls(bool) {}
bool Tilt(float*) { return false; }
#endif

}  // namespace ps2host

#ifdef __ANDROID__
extern "C" JNIEXPORT void JNICALL Java_org_ps2port_PortActivity_nativeKeyboard(JNIEnv*, jclass, jboolean present) {
    ps2host::g_keyboard.store(present != JNI_FALSE, std::memory_order_relaxed);
}
extern "C" JNIEXPORT void JNICALL Java_org_ps2port_PortActivity_nativeTilt(JNIEnv*, jclass, jfloat degrees) {
    ps2host::g_tilt.store(degrees, std::memory_order_relaxed);
    ps2host::g_tilt_at.store(ps2host::Now(), std::memory_order_relaxed);
}
#endif
