/*
 * High frame rates (ps2host/high_fps.h): the rate, the steps, the pictures
 * between them. First made for Burnout 3 (games/burnout3/docs/
 * HIGH_FPS_PLAN.md: what was measured), then made the engine's.
 *
 * The game steps at its own rate (60 a second) and stays so. Above 60 the
 * vertical blanks come more often (SetFastForward's whole multiples, exact;
 * SetBlankRate's others: 144, 165) and the game draws a picture at each;
 * each blank moves its time on by 59.94 over the rate (a half at 120), a
 * step each time that takes it past the next one - 0, 1 or more a picture
 * (HighFpsSteps, the count the game's own frame runs). The picture's place
 * between its last two steps comes from what ran: the game's own count of
 * steps, less the time the pictures moved on, kept between 0 and 1 (where
 * the counting began sets no phase).
 *
 * After each frame that stepped, every transform (an RwMatrix by shape, in
 * the game's regions) and the cameras' values are taken down; for each
 * picture each one that moved is placed between where the step before and
 * the last left it, at the picture's time; at the next frame's start it is
 * put back - where the picture left it as placed (what the draw wrote anew
 * stays). So the pictures show the game evenly, less than a step late. One
 * that jumped further than `cut` in a step is not blended; a camera's own
 * cut (a jump, a reversal, a sharp turn against the step before) neither.
 */

#include "ps2host/high_fps.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2rt/generated.h"

namespace ps2host {
namespace {

HighFpsGame g_game;
bool g_configured = false;

// The pictures a second while above 60 (0: not), and the steps one blank
// moves the game's time on.
double g_rate = 0.0;
double g_per_picture = 1.0;
// The steps asked for: each blank adds g_per_picture; past 1 the game's next
// frame steps, and 1 is taken off (HighFpsSteps).
double g_time = 1.0;
// How far the picture is behind the last step the game ran (see above), and
// the picture's place between its last two steps: 1 less that.
double g_behind = 0.0;
float g_picture_time = 1.0f;
bool g_counted = false;                 // g_step_count holds the game's (since the rate changed)
u32 g_step_count = 0;
u32 g_frame_steps = 1;                  // the steps this frame of the game's ran
unsigned g_frame_blanks = 1;            // the blanks the picture before took (the trace's)
// Blanks since the game's frame last ran (HighFpsFrameSteps): past `gone`,
// it has stopped - a loading screen, a movie - and the rate is its own again.
// (HighFpsPicture, where the rate is chosen, runs only in its frames.)
std::atomic<int> g_blanks_since{0};
unsigned long long g_frames = 0;        // the game's frames (HighFpsFrameSteps)

struct Matrix {
    float m[16];
};
std::unordered_map<u32, Matrix> g_last, g_now;      // the transforms after the last two steps
std::vector<u32> g_known;           // where the last whole read found them (TakeDown)
unsigned g_steps = 0;
// What the in-between placing covered: as the step left it, and as placed.
struct Placed {
    u32 address;
    Matrix was, placed;
};
std::vector<Placed> g_placed;

struct CameraValues {
    std::vector<float> last, now, placed, was;
    bool placing = false;               // `placed` is in RAM (PutBack)
    float step = 0.0f, turn = 0.0f;     // the last step's
    float way[3] = {};                  // the last step's move
    bool cut = false;                   // the last step a cut
    bool flipped = false;               // the last step only turned its way of looking round
};
std::vector<CameraValues> g_cameras;

constexpr double kStepRate = 1e6 / 16683.0;    // the blanks' own, 59.94 (threads.cpp's VblankTimer)

// What this machine kept up with in this run of the game's frames (0: no
// limit found): a rate whose pictures were often shown late (LatePictures)
// steps down a multiple of 60. Looked at every two seconds or so of the
// game's frames, the first such stretch after a change let go (its catching
// up).
double g_cap = 0.0;
int g_sampled = 0;                      // frames since the last look

/* Into a rate above the game's own, the rate rises to it in eighths
   (PS2HOST_HIGH_FPS_RAMP ms for all eight at the least, 600 by default -
   0, at once, where the GPU's boost keeps its clocks up; Rise): a laptop's GPU
   idles clocked down through the loading and the 60-a-second screens
   before a race (an RTX 4070 Laptop at 600-900 MHz, its memory at 810),
   and at once at 144 it took half a second to clock up, the race's first
   dozen pictures shown late. Rising, its load rises with the rate and its
   clocks with it. */
double g_target = 0.0;                  // the rate risen to (0: not rising)
std::chrono::steady_clock::time_point g_rise_from;
bool g_rise_new = false;                // a rise begun: Rise starts its steps

double RiseMs() {
    static const double ms = [] {
        const char* e = std::getenv("PS2HOST_HIGH_FPS_RAMP");
        return e ? std::max(0.0, std::atof(e)) : -1.0;
    }();
    if (VirtualClock()) return 0.0;     // (a GPU's clocks are real time's: a scripted run's pictures as ever)
    if (ms >= 0.0) return ms;
    // The GPU's clocks kept up (WindowGpuBoosted: NVIDIA's boost): no rise -
    // the rate at once (race start at 144: none late either way, the rate
    // there 0.75 s sooner).
    return WindowGpuBoosted() ? 0.0 : 600.0;
}
int g_looks = 0;                        // looks since the rate changed
u64 g_late_seen = 0;                    // LatePictures at the last look

/* FRAME RATE, as pictures a second (60: none above the game's own):
   PS2HOST_FRAME_RATE, else the setting - 120, or "display", the screen's
   refresh (up to 240; 60 where it is not known or below 90) - never above
   the screen's refresh. */
/* The game's own picture rate: 60 for a game that steps each blank, 30 for
   one that steps every second one. */
double Own() { return 60.0 * g_game.steps_per_blank; }

double Wanted() {
    static const char* env = std::getenv("PS2HOST_FRAME_RATE");
    const std::string rate = env ? std::string(env) : Setting("frame_rate", g_game.default_rate);
    double hz = rate == "display" ? double(WindowRefreshRate()) : std::atof(rate.c_str());
    // On a VRR screen (WindowVrr), DISPLAY 3% under its refresh (139 on 144,
    // 160 on 165): at the very top of its range a picture a little early
    // waits for the next refresh as with a fixed one, and the queue behind it.
    if (rate == "display" && WindowVrr()) hz = std::floor(hz * 0.97);
    // Never above the screen's refresh: a picture more than it shows is drawn
    // and thrown away (MAILBOX keeps the newest), its input no newer - the
    // game steps at its own rate. 120 on a 60 Hz screen (a fullscreen mode
    // 60 divides) was twice the work for nothing - at 3x the GS thread full,
    // pictures 70 ms behind. (A rate within 2% of it is its.)
    const double screen = double(WindowRefreshRate());
    if (screen > 0.0 && hz > screen * 1.02) hz = WindowVrr() ? std::floor(screen * 0.97) : screen;
    double want = hz >= 1.5 * Own() ? std::min(hz, 240.0) : Own();
    // PS2HOST_HIGH_FPS_FROM=<frame>: 60 until then (a scripted run's test:
    // its presses land as at 60, and the same screen at 60 and above is
    // compared at the same times).
    // (Or the setting high_fps_from: a run's --setting, in its .args.)
    static const unsigned long long from = [] {
        const char* e = std::getenv("PS2HOST_HIGH_FPS_FROM");
        const std::string set = Setting("high_fps_from", "");
        return e ? std::strtoull(e, nullptr, 10) : set.empty() ? 0ull : std::strtoull(set.c_str(), nullptr, 10);
    }();
    if (FrameNumber() < from) want = Own();
    // Minimised: pictures nobody sees - the game's own rate (at 144 a race
    // took 2 cores, at 60 0.7), back up when the window is.
    if (WindowMinimized()) want = Own();
    return g_cap > 0.0 ? std::min(want, g_cap) : want;
}

/* Each frame of the game's above 60: every two seconds or so, the pictures
   shown late since; more than 5% and the rate steps down (to the multiple
   of 60 below it). */
void KeepUp() {
    const int sample = int(2.0 * g_rate);
    if (++g_sampled < sample) return;
    const u64 late_now = LatePictures(), late = late_now - g_late_seen;
    g_late_seen = late_now;
    g_sampled = 0;
    if (g_looks++ == 0) return;         // (the change's own catching up)
    if (late * 20 > u64(sample)) {
        g_cap = std::max(Own(), 60.0 * std::ceil(g_rate / 60.0 - 1.0 - 1e-6));
        std::fprintf(stderr,
                     "[high fps] %llu of %d pictures late at %.0f a second: %.0f until the game's frames next stop\n",
                     static_cast<unsigned long long>(late), sample, g_rate, g_cap);
    }
}

/* The blanks at `hz`: a whole multiple of 60 (within a hertz for each 60:
   120, 180, 240) as SetFastForward's multiple, exact; another (144, 165) at
   its own rate. The steps a blank moves the game's time on. */
double Pace(double hz) {
    const int multiple = std::max(int(hz / 60.0 + 0.5), 1);
    if (std::fabs(hz - 60.0 * multiple) < multiple) {
        SetBlankRate(0.0);
        SetFastForward(unsigned(multiple));
        return g_game.steps_per_blank / multiple;
    }
    SetFastForward(1);
    SetBlankRate(hz);
    return g_game.steps_per_blank * kStepRate / hz;
}

/* The game running: its frames (HighFpsFrameSteps) went on within the last
   few HighFpsPicture calls. */
bool Running() {
    static unsigned long long seen = 0;
    static int still = 99;
    still = g_frames != seen ? 0 : still + 1;
    seen = g_frames;
    return still <= g_game.running;
}

float Dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/* An RwMatrix: its right, up and at rows of length 1 and square to each
   other, its position finite. */
bool Shaped(const float* m) {
    constexpr float kTolerance = 0.01f;
    for (int row = 0; row < 3; ++row)
        if (!(std::fabs(Dot3(m + 4 * row, m + 4 * row) - 1.0f) < kTolerance)) return false;
    if (!(std::fabs(Dot3(m, m + 4)) < kTolerance && std::fabs(Dot3(m, m + 8)) < kTolerance &&
          std::fabs(Dot3(m + 4, m + 8)) < kTolerance))
        return false;
    for (int k = 12; k < 15; ++k)
        if (!(std::fabs(m[k]) < 1e6f)) return false;
    return true;
}

/* A camera's own cut, step to step: a jump - its position moved more than
   3 times as far as in the step before (and over a unit), or over a unit
   against the way it moved in the step before, or the way it looks turned
   more than 60 degrees and twice as far as in the step before (the game cut
   to another camera, or turned it round). Such a step is shown as it is,
   not blended: a picture between the two showed neither, a one-picture
   lurch. But a way of looking only turned round (its values negated: Burnout
   3's at speed, for a step, now and then) while the camera moved on as
   before is no cut - blended as ever, those values as the step left them
   (blended they would pass through 0); taken for one, the camera stood
   still for the step's pictures, then went on: a jitter. */
void CameraTakenDown(const HighFpsCamera& camera, CameraValues& v) {
    const float* a = v.last.data() + camera.position;
    const float* b = v.now.data() + camera.position;
    const float d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float step = std::sqrt(Dot3(d, d));
    float turn = 0.0f;
    if (camera.direction >= 0 && camera.quaternion) {
        // The rotation between the two (its sign none of it: q and -q are one).
        const float* la = v.last.data() + camera.direction;
        const float* na = v.now.data() + camera.direction;
        const float dot = Dot3(la, na) + la[3] * na[3];
        const float lengths = std::sqrt((Dot3(la, la) + la[3] * la[3]) * (Dot3(na, na) + na[3] * na[3]));
        turn = lengths > 1e-6f ? 2.0f * std::acos(std::min(1.0f, std::fabs(dot) / lengths)) * 57.29578f : 0.0f;
    } else if (camera.direction >= 0) {
        const float* la = v.last.data() + camera.direction;
        const float* na = v.now.data() + camera.direction;
        // (only where the two are of a length: Burnout 3's +0x20 values
        // shrink to a few hundredths now and then, and their turns are noise)
        const float lengths = std::sqrt(Dot3(la, la) * Dot3(na, na));
        turn = lengths > 0.01f ? std::acos(std::min(1.0f, std::max(-1.0f, Dot3(la, na) / lengths))) * 57.29578f
                               : 0.0f;
    }
    const bool reversed = step > 1.0f && Dot3(d, v.way) < 0.0f;
    const bool jumped = step > 1.0f && step > 3.0f * v.step;
    v.flipped = !camera.quaternion && turn > 170.0f && !jumped && !reversed;
    v.cut = jumped || reversed || (turn > 60.0f && turn > 2.0f * v.turn && !v.flipped);
    std::memcpy(v.way, d, sizeof v.way);
    v.step = step;
    if (!v.flipped) v.turn = turn;
}

/* PS2HOST_HIGH_FPS_BLENDS=<file>: each step, the transforms whose step
   stands out - a blend between the two would show it (CSV): `jump` moved
   over 4 times its step before and over a unit, `turn` turned over 30
   degrees, `flip` changed handedness, `far` moved over 10 units (blended
   unless over the game's cut). With the step's frame, the transform's
   place and its step before. */
std::FILE* BlendsFile() {
    static std::FILE* f = [] {
        const char* path = std::getenv("PS2HOST_HIGH_FPS_BLENDS");
        std::FILE* out = path ? std::fopen(path, "w") : nullptr;
        if (out) std::fprintf(out, "ms,frame,step,address,why,moved,before,turn,x,y,z,cut\n");
        return out;
    }();
    return f;
}

float Determinant(const float* m) {
    return m[0] * (m[5] * m[10] - m[6] * m[9]) - m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

/* Each transform's last step and the one before: how far it moved and how
   far it turned (degrees). */
struct Stepped {
    float moved = 0.0f, turn = 0.0f;
};
std::unordered_map<u32, Stepped> g_stepped, g_stepped_before;

void Measure() {
    g_stepped_before.swap(g_stepped);
    g_stepped.clear();
    for (const auto& [address, now] : g_now) {
        const auto last = g_last.find(address);
        if (last == g_last.end()) continue;
        const float* a = last->second.m;
        const float* b = now.m;
        const float d[3] = {b[12] - a[12], b[13] - a[13], b[14] - a[14]};
        const float trace = Dot3(a, b) + Dot3(a + 4, b + 4) + Dot3(a + 8, b + 8);
        g_stepped[address] = {std::sqrt(Dot3(d, d)),
                              std::acos(std::min(1.0f, std::max(-1.0f, (trace - 1.0f) / 2.0f))) * 57.29578f};
    }
}

/* A transform put somewhere else in a step, not moved there: over 2 units
   and 4 times its step before, or over 60 degrees turned when it was all
   but still - a wrecked car back on the road a few metres on (under the
   game's cut), turned round. Blended, it slid there through the scene; it
   is shown as the step left it. */
bool Teleported(u32 address) {
    const auto now = g_stepped.find(address);
    const auto before = g_stepped_before.find(address);
    if (now == g_stepped.end() || before == g_stepped_before.end()) return false;
    const Stepped& n = now->second;
    const Stepped& b = before->second;
    return (n.moved > 2.0f && n.moved > 4.0f * b.moved) || (n.turn > 60.0f && n.turn > 4.0f * b.turn + 10.0f);
}

void LogBlends() {
    std::FILE* f = BlendsFile();
    if (!f) return;
    static std::unordered_map<u32, float> before;           // each transform's step before
    static const auto start = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::unordered_map<u32, float> moved_now;
    for (const auto& [address, now] : g_now) {
        const auto last = g_last.find(address);
        if (last == g_last.end()) continue;
        const float* a = last->second.m;
        const float* b = now.m;
        const float d[3] = {b[12] - a[12], b[13] - a[13], b[14] - a[14]};
        const float moved = std::sqrt(Dot3(d, d));
        const float trace = Dot3(a, b) + Dot3(a + 4, b + 4) + Dot3(a + 8, b + 8);
        const float turn = std::acos(std::min(1.0f, std::max(-1.0f, (trace - 1.0f) / 2.0f))) * 57.29578f;
        const bool flip = (Determinant(a) < 0.0f) != (Determinant(b) < 0.0f);
        const auto was = before.find(address);
        const float prior = was != before.end() ? was->second : -1.0f;
        moved_now.emplace(address, moved);
        const char* why = flip ? "flip"
                          : (prior >= 0.0f && moved > 1.0f && moved > 4.0f * prior) ? "jump"
                          : moved > 10.0f ? "far"
                          : turn > 30.0f ? "turn"
                                         : nullptr;
        if (!why) continue;
        std::fprintf(f, "%.1f,%llu,%u,%08X,%s,%.3f,%.3f,%.1f,%.2f,%.2f,%.2f,%d\n", ms, g_frames, g_steps,
                     unsigned(address), why, moved, prior, turn, b[12], b[13], b[14], moved > g_game.cut ? 1 : 0);
    }
    before.swap(moved_now);
}

/* The transforms and the cameras' values as the step left them. */
void TakeDown() {
    for (size_t k = 0; k < g_game.cameras.size(); ++k) {
        const HighFpsCamera& camera = g_game.cameras[k];
        CameraValues& v = g_cameras[k];
        const size_t bytes = v.now.size() * sizeof(float);
        if (const u8* p = generated::RamPointer(camera.address, u32(bytes))) {
            std::memcpy(v.last.data(), v.now.data(), bytes);
            std::memcpy(v.now.data(), p, bytes);
            CameraTakenDown(camera, v);
        }
    }
    g_last.swap(g_now);
    g_now.clear();
    // The regions read whole every kRescan steps; between, only where a
    // transform was (Burnout 3's ~850 of 2 MB) - a transform made since is
    // blended from the next look on, a few steps later.
    constexpr unsigned kRescan = 4;
    if (++g_steps % kRescan != 0 && !g_known.empty()) {
        for (u32 address : g_known) {
            const u8* p = generated::RamPointer(address, 64);
            Matrix m;
            if (!p) continue;
            std::memcpy(m.m, p, sizeof m.m);
            if (Shaped(m.m)) g_now.emplace(address, m);
        }
        return;
    }
    g_known.clear();
    for (const HighFpsRegion& region : g_game.transforms) {
        const u8* base = generated::RamPointer(region.begin, region.end - region.begin);
        if (!base) continue;
        for (u32 at = 0; at + 64 <= region.end - region.begin; at += 16) {
            // The right row's length first, in place: most of 2 MB a step is
            // no matrix, and the copy of each candidate was most of the scan.
            float right[3];
            std::memcpy(right, base + at, sizeof right);
            if (!(std::fabs(Dot3(right, right) - 1.0f) < 0.01f)) continue;
            Matrix m;
            std::memcpy(m.m, base + at, sizeof m.m);
            if (!Shaped(m.m)) continue;
            g_now.emplace(region.begin + at, m);
            g_known.push_back(region.begin + at);
            at += 48;                   // (its rows are not another's)
        }
    }
}

void Normalise(float* v) {
    const float length = std::sqrt(Dot3(v, v));
    if (length > 0.0f)
        for (int k = 0; k < 3; ++k) v[k] /= length;
}

/* A transform `t` of the way from a to b: the position between, the
   rotation the rows' blend made orthonormal again (at, then up square to
   it, then right from the two, its handedness the original's). */
Matrix Between(const Matrix& a, const Matrix& b, float t) {
    Matrix out = b;
    float right[3], up[3], at[3];
    const float s = 1.0f - t;
    for (int k = 0; k < 3; ++k) {
        right[k] = s * a.m[k] + t * b.m[k];
        up[k] = s * a.m[4 + k] + t * b.m[4 + k];
        at[k] = s * a.m[8 + k] + t * b.m[8 + k];
        out.m[12 + k] = a.m[12 + k] + t * (b.m[12 + k] - a.m[12 + k]);
    }
    Normalise(at);
    const float d = Dot3(up, at);
    for (int k = 0; k < 3; ++k) up[k] -= d * at[k];
    Normalise(up);
    float cross[3] = {up[1] * at[2] - up[2] * at[1], up[2] * at[0] - up[0] * at[2], up[0] * at[1] - up[1] * at[0]};
    if (Dot3(cross, right) < 0.0f)
        for (float& x : cross) x = -x;
    for (int k = 0; k < 3; ++k) {
        out.m[k] = cross[k];
        out.m[4 + k] = up[k];
        out.m[8 + k] = at[k];
    }
    return out;
}

/* A transform that moved, placed `t` of the way from the step before to
   the last. */
void Place(u32 address, const Matrix& last, const Matrix& now, float t) {
    float d[3];
    for (int k = 0; k < 3; ++k) d[k] = now.m[12 + k] - last.m[12 + k];
    const float moved = std::sqrt(Dot3(d, d));
    bool turned = false;
    for (int k = 0; k < 12 && !turned; ++k) turned = now.m[k] != last.m[k];
    if ((moved == 0.0f && !turned) || moved > g_game.cut || Teleported(address)) return;
    u8* p = generated::RamPointer(address, 64);
    if (!p) return;
    Placed placed{address, {}, {}};
    std::memcpy(placed.was.m, p, sizeof placed.was.m);
    placed.placed = placed.was;
    const Matrix between = Between(last, now, t);
    std::memcpy(placed.placed.m, between.m, 12 * 4);     // the rows (their fourth words as they were)
    std::memcpy(placed.placed.m + 12, between.m + 12, 3 * 4);
    std::memcpy(p, placed.placed.m, sizeof placed.placed.m);
    g_placed.push_back(placed);
}

bool AValue(float v) { return std::fabs(v) > 1e-6f && std::fabs(v) < 1e6f; }

/* For the picture drawn now: the cameras' values and each transform that
   moved, `t` of the way from the step before to the last (1: as the last
   left them). */
void PlaceBetween(float t) {
    if (t >= 1.0f || g_last.empty()) return;
    for (size_t k = 0; k < g_game.cameras.size(); ++k) {
        CameraValues& v = g_cameras[k];
        const size_t count = v.now.size();
        float* camera = reinterpret_cast<float*>(generated::RamPointer(g_game.cameras[k].address,
                                                                       u32(count * sizeof(float))));
        if (!camera || v.cut) continue;
        std::memcpy(v.was.data(), camera, count * sizeof(float));
        std::memcpy(v.placed.data(), camera, count * sizeof(float));
        const HighFpsCamera& c = g_game.cameras[k];
        const int way = v.flipped ? c.direction : -1;
        const int q = c.quaternion ? c.direction : -1;
        for (size_t i = 0; i < count; ++i) {
            const float a = v.last[i], b = v.now[i];
            if (q >= 0 && int(i) >= q && int(i) < q + 4) continue;           // (the quaternion: below)
            if (a == b || !AValue(a) || !AValue(b) || !(std::fabs(b - a) < g_game.cut)) continue;
            if (way >= 0 && int(i) >= way && int(i) < way + 3) continue;    // (turned round: the step's)
            v.placed[i] = a + t * (b - a);
        }
        if (q >= 0) {
            // The rotation between, as one: the step before's turned to the
            // same side as the last's (q and -q are one rotation), blended,
            // its length 1 again - at t 1 the last's as it is.
            const float* a = v.last.data() + q;
            const float* b = v.now.data() + q;
            const float side = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0.0f ? -1.0f : 1.0f;
            float r[4], length = 0.0f;
            for (int i = 0; i < 4; ++i) {
                r[i] = side * a[i] + t * (b[i] - side * a[i]);
                length += r[i] * r[i];
            }
            const float lb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3];
            if (length > 1e-12f && lb > 1e-12f) {
                const float scale = std::sqrt(lb / length);     // (the last's length, as the game had it)
                for (int i = 0; i < 4; ++i) v.placed[size_t(q + i)] = r[i] * scale;
            }
        }
        std::memcpy(camera, v.placed.data(), count * sizeof(float));
        v.placing = true;
    }
    // A camera cut in this step (shown as the step left it): the transforms
    // too - blended, a car lagged the camera's new place by up to its step
    // for the step's pictures, then caught up (a twitch at every cut).
    for (const CameraValues& v : g_cameras)
        if (v.cut) return;
    for (const auto& [address, now] : g_now) {
        const auto last = g_last.find(address);
        if (last != g_last.end()) Place(address, last->second, now, t);
    }
}

/* What the placing covered, back as the step left it - where the picture
   left it as placed (what the draw wrote anew stays). */
void PutBack() {
    for (const Placed& placed : g_placed) {
        u8* p = generated::RamPointer(placed.address, 64);
        if (p && std::memcmp(p, placed.placed.m, sizeof placed.placed.m) == 0)
            std::memcpy(p, placed.was.m, sizeof placed.was.m);
    }
    g_placed.clear();
    for (size_t k = 0; k < g_game.cameras.size(); ++k) {
        CameraValues& v = g_cameras[k];
        if (!v.placing) continue;
        const size_t bytes = v.now.size() * sizeof(float);
        u8* camera = generated::RamPointer(g_game.cameras[k].address, u32(bytes));
        if (camera && std::memcmp(camera, v.placed.data(), bytes) == 0) std::memcpy(camera, v.was.data(), bytes);
        v.placing = false;
    }
}

/* PS2HOST_HIGH_FPS_TRACE=<file>: a line (CSV) for each frame of the game's
   above 60, as its picture is placed - its time, the steps it ran, where
   it is between them, how many transforms were placed, and each camera's
   position as the last two steps left it and as placed, how far it went in
   the step and how far its way of looking turned (the dot of the two
   steps' directions). engine/tools/high_fps_trace.py finds where a
   camera's path jumps. */
std::FILE* TraceFile() {
    static std::FILE* file = [] {
        const char* path = std::getenv("PS2HOST_HIGH_FPS_TRACE");
        std::FILE* f = path ? std::fopen(path, "w") : nullptr;
        if (f) {
            std::fprintf(f, "frame,ms,rate,ticks,t,placed,blanks,known");
            for (size_t k = 0; k < g_game.cameras.size(); ++k)
                std::fprintf(f, ",c%zu_last_x,c%zu_last_y,c%zu_last_z,c%zu_now_x,c%zu_now_y,c%zu_now_z,"
                             "c%zu_x,c%zu_y,c%zu_z,c%zu_step,c%zu_turn", k, k, k, k, k, k, k, k, k, k, k);
            for (size_t k = 0; k < g_game.subjects.size(); ++k)
                std::fprintf(f, ",s%zu_addr,s%zu_off,s%zu_last_x,s%zu_last_y,s%zu_last_z,s%zu_now_x,s%zu_now_y,"
                             "s%zu_now_z,s%zu_x,s%zu_y,s%zu_z,s%zu_placed", k, k, k, k, k, k, k, k, k, k, k, k);
            std::fprintf(f, "\n");
        }
        return f;
    }();
    return file;
}

/* Each subject's transform: the one nearest the position the game keeps
   (kept while within 3 units: a car's wheels are near its body), where the
   last two steps left it and where it was placed (as the last left it: not
   placed - it did not move, or it was not found since the last look). */
void TraceSubjects(std::FILE* f) {
    static std::vector<u32> chosen;
    chosen.resize(g_game.subjects.size(), 0);
    for (size_t k = 0; k < g_game.subjects.size(); ++k) {
        float at[3] = {};
        if (const u8* p = generated::RamPointer(g_game.subjects[k], 12)) std::memcpy(at, p, sizeof at);
        auto distance = [&](const Matrix& m) {
            const float d[3] = {m.m[12] - at[0], m.m[13] - at[1], m.m[14] - at[2]};
            return std::sqrt(Dot3(d, d));
        };
        const auto kept = g_now.find(chosen[k]);
        if (kept == g_now.end() || distance(kept->second) > 3.0f) {
            float best = 1e30f;
            chosen[k] = 0;
            for (const auto& [address, m] : g_now) {
                const float d = distance(m);
                if (d < best) {
                    best = d;
                    chosen[k] = address;
                }
            }
        }
        const auto now = g_now.find(chosen[k]);
        const auto last = g_last.find(chosen[k]);
        if (now == g_now.end()) {
            std::fprintf(f, ",0,0,0,0,0,0,0,0,0,0,0,0");
            continue;
        }
        const Matrix& b = now->second;
        const Matrix& a = last != g_last.end() ? last->second : b;
        const float* placed = b.m + 12;
        bool was_placed = false;
        for (const Placed& p : g_placed)
            if (p.address == chosen[k]) {
                placed = p.placed.m + 12;
                was_placed = true;
            }
        std::fprintf(f, ",%08X,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d", chosen[k], distance(b),
                     a.m[12], a.m[13], a.m[14], b.m[12], b.m[13], b.m[14], placed[0], placed[1], placed[2],
                     was_placed ? 1 : 0);
    }
}

void Trace(float t) {
    std::FILE* f = TraceFile();
    static unsigned long long traced = 0;
    if (!f || traced == g_frames) return;
    traced = g_frames;
    std::fprintf(f, "%llu,%.3f,%.0f,%u,%.4f,%zu,%u,%zu", g_frames, double(ClockNs()) / 1e6, g_rate, g_frame_steps,
                 double(t), g_placed.size(), g_frame_blanks, g_now.size());
    for (size_t k = 0; k < g_game.cameras.size(); ++k) {
        const HighFpsCamera& camera = g_game.cameras[k];
        const CameraValues& v = g_cameras[k];
        const float* a = v.last.data() + camera.position;
        const float* b = v.now.data() + camera.position;
        const float* placed = (v.placing ? v.placed.data() : v.now.data()) + camera.position;
        const float d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        // The cosine of the way it looks turned in the step (a quaternion's:
        // its rotation's angle, q and -q one; the 3 values' else).
        float dot = 1.0f;
        if (camera.direction >= 0 && camera.quaternion) {
            const float* la = v.last.data() + camera.direction;
            const float* na = v.now.data() + camera.direction;
            const float lengths = std::sqrt((Dot3(la, la) + la[3] * la[3]) * (Dot3(na, na) + na[3] * na[3]));
            const float half = lengths > 1e-6f ? std::min(1.0f, std::fabs(Dot3(la, na) + la[3] * na[3]) / lengths) : 1.0f;
            dot = 2.0f * half * half - 1.0f;       // cos(2 acos(half))
        } else if (camera.direction >= 0) {
            dot = Dot3(v.last.data() + camera.direction, v.now.data() + camera.direction);
        }
        std::fprintf(f, ",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.5f", a[0], a[1], a[2], b[0], b[1], b[2],
                     placed[0], placed[1], placed[2], std::sqrt(Dot3(d, d)), dot);
    }
    TraceSubjects(f);
    std::fprintf(f, "\n");
    if (g_frames % 120 == 0) std::fflush(f);
}

/* PS2HOST_HIGH_FPS_POKE=<hex address>: the float there raised by 0.5 after
   each picture's placing, before its draw - a picture that changes says the
   draw reads it (a camera's input, not a value the draw makes anew). */
void Poke() {
    static const u32 address = [] {
        const char* e = std::getenv("PS2HOST_HIGH_FPS_POKE");
        return e ? u32(std::strtoul(e, nullptr, 16)) : 0u;
    }();
    if (!address) return;
    if (u8* p = generated::RamPointer(address, 4)) {
        float v;
        std::memcpy(&v, p, 4);
        v += 0.5f;
        std::memcpy(p, &v, 4);
    }
}

bool Blend() {
    static const bool blend = std::getenv("PS2HOST_HIGH_FPS_NO_BLEND") == nullptr;   // to compare
    return blend;
}

/* Back to the game's own rate, the placing forgotten. */
void Reset() {
    g_last.clear();
    g_now.clear();
    g_known.clear();
    g_stepped.clear();
    g_stepped_before.clear();
    g_time = 1.0;
    g_behind = 0.0;
    g_counted = false;
}

}  // namespace

void HighFpsConfigure(const HighFpsGame& game) {
    g_game = game;
    g_configured = true;
    g_cameras.assign(game.cameras.size(), CameraValues{});
    for (size_t k = 0; k < game.cameras.size(); ++k) {
        CameraValues& v = g_cameras[k];
        const size_t count = size_t(std::max(game.cameras[k].floats, 1));
        v.last.assign(count, 0.0f);
        v.now.assign(count, 0.0f);
        v.placed.assign(count, 0.0f);
        v.was.assign(count, 0.0f);
    }
}

void HighFpsFrameStart() { PutBack(); }

bool HighFpsFrameSteps(u32 steps_so_far, unsigned blanks) {
    g_blanks_since = 0;
    ++g_frames;
    if (g_rate > 0.0) KeepUp();
    g_frame_steps = g_counted ? steps_so_far - g_step_count : 1;
    g_frame_blanks = blanks;
    g_step_count = steps_so_far;
    g_counted = true;
    if (g_rate > 0.0) {
        g_behind += double(g_frame_steps) - g_per_picture * std::max(1u, blanks);
        if (g_behind >= 1.0) g_behind -= std::floor(g_behind);
        if (g_behind < 0.0) g_behind = 0.0;
        g_picture_time = float(1.0 - g_behind);
    } else {
        g_picture_time = 1.0f;
    }
    return !(g_rate > 0.0 && g_frame_steps == 0);
}

void HighFpsStepped() {
    if (g_rate > 0.0) {
        TakeDown();
        Measure();
        LogBlends();
    }
}

/* Rising (g_target): an eighth of the way from the game's own rate at a
   time, each after RiseMs / 8 - and only when none of the pictures since the
   last eighth was shown late (LatePictures): a GPU still clocked down holds
   the rate where it keeps up until its clocks come up (an RTX 4070 Laptop at
   P5 drew a race's view at about 60 a second, and boosted only after half a
   second near 100%); never lower. Past 8 RiseMs, the target whatever came. */
void Rise() {
    if (g_target <= 0.0 || g_rate <= 0.0) return;
    static std::chrono::steady_clock::time_point step_from;
    static u64 late_from = 0;
    static int part = 1;
    const auto now = std::chrono::steady_clock::now();
    if (g_rise_new) {
        g_rise_new = false;
        step_from = now;
        late_from = LatePictures();
        part = 1;
    }
    const double step_ms = std::chrono::duration<double, std::milli>(now - step_from).count();
    const double all_ms = std::chrono::duration<double, std::milli>(now - g_rise_from).count();
    if (all_ms >= 8.0 * RiseMs()) part = 8;
    else if (step_ms >= RiseMs() / 8.0) {
        if (LatePictures() == late_from) ++part;
        step_from = now;
        late_from = LatePictures();
    }
    const double rate = part >= 8 ? g_target : Own() + (g_target - Own()) * part / 8.0;
    if (rate > g_rate + 0.5) {
        g_rate = rate;
        g_per_picture = Pace(rate);
    }
    if (part >= 8) g_target = 0.0;
}

void HighFpsPicture() {
    // Down to 60 only once the game has not run for a moment (`settle`
    // calls): a change drains the GS thread - a hitch.
    static int differing = 0;
    const bool running = Running();
    if (!running) g_cap = 0.0;          // (the next run tries the setting's rate again)
    const double want = running ? Wanted() : Own();
    const double now = g_target > 0.0 ? g_target : g_rate > 0.0 ? g_rate : Own();
    differing = std::fabs(want - now) >= 0.5 ? differing + 1 : 0;
    if (differing >= (want > Own() ? 1 : g_game.settle)) {
        differing = 0;
        std::fprintf(stderr, "[high fps] %.0f pictures a second\n", want);
        PutBack();
        Reset();
        g_sampled = 0;
        g_looks = 0;
        if (want > Own()) {
            // From the game's own rate: rising to it (Rise).
            const bool rising = g_rate <= 0.0 && RiseMs() > 0.0;
            g_target = rising ? want : 0.0;
            g_rise_from = std::chrono::steady_clock::now();
            g_rise_new = rising;
            g_rate = rising ? Own() + (want - Own()) / 8.0 : want;
            g_per_picture = Pace(g_rate);
        } else {
            g_target = 0.0;
            g_rate = 0.0;
            g_per_picture = 1.0;
            SetBlankRate(0.0);
            SetFastForward(1);
        }
    }
    Rise();
    if (g_rate > 0.0 && Blend()) {
        PutBack();
        PlaceBetween(g_picture_time);
        Poke();
        Trace(g_picture_time);
    }
}

int HighFpsSteps(unsigned blanks, int most) {
    if (g_rate <= 0.0) return -1;
    g_time += g_per_picture * std::max(1u, blanks);
    int steps = 0;
    while (g_time > 1.0 + 1e-6) {
        g_time -= 1.0;
        ++steps;
    }
    if (steps > most) {
        steps = most;
        g_time = 1.0;
    }
    return steps;
}

void HighFpsBlank() {
    if (g_rate <= 0.0 || ++g_blanks_since <= g_game.gone * std::max(1, int(g_rate / 60.0 + 0.5))) return;
    HighFpsLeave("the game's frames stopped");
}

void HighFpsLeave(const char* why) {
    if (g_rate <= 0.0) return;
    std::fprintf(stderr, "[high fps] %.0f pictures a second (%s)\n", Own(), why);
    g_cap = 0.0;
    g_target = 0.0;
    g_placed.clear();
    for (CameraValues& v : g_cameras) v.placing = false;
    g_rate = 0.0;
    g_per_picture = 1.0;
    Reset();
    SetBlankRate(0.0);
    SetFastForward(1);
}

double HighFpsRate() { return g_rate; }

float HighFpsPictureTime() { return g_picture_time; }

double HighFpsOwnRate() { return g_configured ? Own() : 0.0; }

}  // namespace ps2host
