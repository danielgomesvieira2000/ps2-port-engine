/*
 * The port's window (--window): the GS's displayed picture, every vertical
 * blank, scaled to the window, and the keyboard the controller reads.
 *
 * With the GPU display built in (PS2HOST_VULKAN, engine/gpu) the window is an
 * SDL3 window presenting through Vulkan, or Direct3D 11 where there is no
 * Vulkan 1.2 device (or the "renderer" setting asks for it: auto, vulkan,
 * d3d11; PS2HOST_RENDERER overrides it); without either, a plain Win32 window
 * drawn with GDI. Each is pumped on the host's thread at each frame
 * (FrameDone).
 */

#include "ps2host/platform.h"
#include "stb_easy_font.h"      // the overlay's text where there is no GDI; the loading screen's
#ifndef _WIN32
#include "fiber.h"              // OnThreadStack
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <functional>
#include <vector>

#include "ps2host/gs.h"
#include "ps2host/timeline.h"
#include "ps2host/run.h"
#include "ps2host/settings.h"
#include "ps2host/texture_pack.h"

#ifdef PS2HOST_VULKAN
#include <SDL3/SDL_scancode.h>

#include "ps2gpu/display.h"
#include "ps2gpu/gs_renderer.h"     // PipelineWarmProgress
#endif

namespace ps2host {

std::atomic<std::uint64_t> g_late_pictures{0};    // LatePictures

namespace {

#ifdef _WIN32
HWND g_window = nullptr;        // GDI's window, without a Vulkan display
#else
void* g_window = nullptr;       // (none: Vulkan's display or nothing)
#endif
bool g_closed = false;

#ifdef PS2HOST_VULKAN
std::unique_ptr<ps2gpu::Display> g_display;
bool g_pipelines_behind = false;    // the first start went on before its pipelines were all made

// The keys the controller reads, as Win32 virtual keys, by position.
/* A Win32 virtual key as SDL's scancode: the key in that place on a US
   keyboard (SDL names keys by place). The punctuation's virtual keys are a
   US layout's. */
int Scancode(int key) {
    switch (key) {
        case VK_BACK: return SDL_SCANCODE_BACKSPACE;
        case VK_TAB: return SDL_SCANCODE_TAB;
        case VK_RETURN: return SDL_SCANCODE_RETURN;
        case VK_ESCAPE: return SDL_SCANCODE_ESCAPE;      // the PC settings panel's
        case VK_SPACE: return SDL_SCANCODE_SPACE;
        case VK_UP: return SDL_SCANCODE_UP;
        case VK_DOWN: return SDL_SCANCODE_DOWN;
        case VK_LEFT: return SDL_SCANCODE_LEFT;
        case VK_RIGHT: return SDL_SCANCODE_RIGHT;
        case VK_PRIOR: return SDL_SCANCODE_PAGEUP;
        case VK_NEXT: return SDL_SCANCODE_PAGEDOWN;
        case VK_HOME: return SDL_SCANCODE_HOME;
        case VK_END: return SDL_SCANCODE_END;
        case VK_INSERT: return SDL_SCANCODE_INSERT;
        case VK_DELETE: return SDL_SCANCODE_DELETE;
        case VK_CAPITAL: return SDL_SCANCODE_CAPSLOCK;
        case VK_LSHIFT: return SDL_SCANCODE_LSHIFT;
        case VK_RSHIFT: return SDL_SCANCODE_RSHIFT;
        case VK_LCONTROL: return SDL_SCANCODE_LCTRL;
        case VK_RCONTROL: return SDL_SCANCODE_RCTRL;
        case VK_LMENU: return SDL_SCANCODE_LALT;
        case VK_RMENU: return SDL_SCANCODE_RALT;
        case VK_NUMPAD0: return SDL_SCANCODE_KP_0;
        case VK_DECIMAL: return SDL_SCANCODE_KP_PERIOD;
        case VK_ADD: return SDL_SCANCODE_KP_PLUS;
        case VK_SUBTRACT: return SDL_SCANCODE_KP_MINUS;
        case VK_MULTIPLY: return SDL_SCANCODE_KP_MULTIPLY;
        case VK_DIVIDE: return SDL_SCANCODE_KP_DIVIDE;
        case VK_OEM_1: return SDL_SCANCODE_SEMICOLON;
        case VK_OEM_PLUS: return SDL_SCANCODE_EQUALS;
        case VK_OEM_COMMA: return SDL_SCANCODE_COMMA;
        case VK_OEM_MINUS: return SDL_SCANCODE_MINUS;
        case VK_OEM_PERIOD: return SDL_SCANCODE_PERIOD;
        case VK_OEM_2: return SDL_SCANCODE_SLASH;
        case VK_OEM_3: return SDL_SCANCODE_GRAVE;
        case VK_OEM_4: return SDL_SCANCODE_LEFTBRACKET;
        case VK_OEM_5: return SDL_SCANCODE_BACKSLASH;
        case VK_OEM_6: return SDL_SCANCODE_RIGHTBRACKET;
        case VK_OEM_7: return SDL_SCANCODE_APOSTROPHE;
        default: break;
    }
    if (key >= VK_F1 && key <= VK_F12) return SDL_SCANCODE_F1 + (key - VK_F1);
    if (key >= 'A' && key <= 'Z') return SDL_SCANCODE_A + (key - 'A');
    if (key >= '1' && key <= '9') return SDL_SCANCODE_1 + (key - '1');
    if (key == '0') return SDL_SCANCODE_0;
    if (key >= VK_NUMPAD1 && key <= VK_NUMPAD9) return SDL_SCANCODE_KP_1 + (key - VK_NUMPAD1);
    return -1;
}
#endif

#ifdef _WIN32
LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM w, LPARAM l) {
    switch (message) {
        case WM_CLOSE:
            g_closed = true;
            DestroyWindow(window);
            return 0;
        case WM_DESTROY:
            g_window = nullptr;
            return 0;
        default:
            return DefWindowProcW(window, message, w, l);
    }
}

bool OpenGdiWindow(const wchar_t* title, int width, int height) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = L"ps2host";
    RegisterClassW(&wc);
    RECT r = {0, 0, width, height};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    g_window = CreateWindowW(L"ps2host", title, WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                             CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr,
                             wc.hInstance, nullptr);
    return g_window != nullptr;
}

void PresentGdi() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    if (!g_window) return;
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    if (!gs::TheGs().Picture(&pixels, &w, &h)) return;
    BITMAPINFO info = {};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = LONG(w);
    info.bmiHeader.biHeight = -LONG(h);                 // top-down rows
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    RECT client;
    GetClientRect(g_window, &client);
    HDC dc = GetDC(g_window);
    SetStretchBltMode(dc, HALFTONE);
    StretchDIBits(dc, 0, 0, client.right, client.bottom, 0, 0, LONG(w), LONG(h), pixels.data(), &info,
                  DIB_RGB_COLORS, SRCCOPY);
    ReleaseDC(g_window, dc);
}
#else
bool OpenGdiWindow(const wchar_t*, int, int) { return false; }
void PresentGdi() {}
#endif

#ifdef PS2HOST_VULKAN
/* The "antialiasing" setting as the display takes it: supersampling's
   pictures (drawn larger, SupersamplingFactor) averaged down. */
ps2gpu::Display::AntiAliasing DisplayAntiAliasing(const std::string& mode) {
    using AA = ps2gpu::Display::AntiAliasing;
    return mode == "fxaa"                        ? AA::kFxaa
           : mode == "smaa"                      ? AA::kSmaa
           : mode == "ssaa2" || mode == "ssaa4" ? AA::kDownsample
                                                 : AA::kOff;
}
#endif

/* The "aspect" setting as the display takes it: "4:3" (the default), "16:9",
   any other "W:H" (21:9, 32:9: an ultrawide screen's), or "auto" (0, the
   window's own shape). */
float SettingAspect() {
    const std::string aspect = Setting("aspect", "4:3");
    if (aspect == "auto") return 0.0f;
    float width = 0.0f, height = 0.0f;
    if (std::sscanf(aspect.c_str(), "%f:%f", &width, &height) == 2 && width > 0.0f && height > 0.0f) {
        const float shape = width / height;
        if (shape >= 1.0f && shape <= 4.0f) return shape;
    }
    return 4.0f / 3.0f;
}

}  // namespace

std::string g_title;                // the window's title as opened (SetWindowStatus adds to it)
void ApplyCursorArt();               // the game's pointer to a window opened after it was set

std::string& GameTitleText() {
    static std::string title;
    return title;
}

void SetGameTitle(const std::string& title) { GameTitleText() = title; }
const std::string& GameTitle() { return GameTitleText(); }

/* PS2HOST_REFRESH_LOG=1: every 600 pictures shown, the screen's refreshes
   they went to (each the first of its blanks after the picture was handed
   over, RefreshTiming's): pictures handed over in a refresh that already had
   one (a picture dropped, or shown a refresh late), refreshes none was handed
   over in (the one before shown twice - at the screen's own rate, a
   stutter), and the nearest a picture came to a blank (near one, a wake-up's
   error decides which refresh it goes to). */
void NoteRefreshes(std::uint64_t shown_ns) {
    static const bool on = std::getenv("PS2HOST_REFRESH_LOG") != nullptr;
    if (!on) return;
    double period = 0;
    u64 anchor = 0;
    if (!WindowRefreshTiming(&period, &anchor)) return;
    static double last_blank = 0;               // the blank the last picture went to
    static unsigned shown = 0, doubled = 0, missed = 0;
    static double nearest = 1e18;
    const double since = double(shown_ns) - double(anchor);
    const double into = since - std::floor(since / period) * period;
    const double blank = double(shown_ns) + (period - into);
    nearest = std::min(nearest, std::min(into, period - into));
    if (last_blank > 0) {
        const long long apart = std::llround((blank - last_blank) / period);
        if (apart <= 0) ++doubled;
        else missed += unsigned(apart - 1);
    }
    last_blank = blank;
    if (++shown < 600) return;
    double locked_period = 0;
    u64 locked_anchor = 0;
    std::fprintf(stderr,
                 "[refresh] 600 pictures on a %.4f ms refresh: %u to a refresh that had one, %u refreshes without "
                 "one; nearest a blank %.2f ms; blanks %s\n",
                 period / 1e6, doubled, missed, nearest / 1e6,
                 RefreshLocked(&locked_period, &locked_anchor) ? "locked to the screen's" : "by the processor's clock");
    shown = doubled = missed = 0;
    nearest = 1e18;
}

/* Each picture's GPU work done (Display::SetGpuDoneObserver), `tag` its
   blank: the lead the locked blanks keep (SetRefreshLead) - the longest
   blank-to-done of the last two seconds, a millisecond and a tenth more,
   as the screen's refreshes go (a 9 ms picture at 144: 3 ms - done before
   the second refresh after its blank, never on one). Every picture then
   waits that long, the longest's time and the margin: taken only where that
   is a millisecond under what they wait with the blanks on the screen's
   (each to the refresh after it is done) - a 60 Hz screen's pictures done
   3 ms after their blank: 6 ms against 16.7. At 144 a 3x picture's 5.5 ms,
   now and then 8.5, is quicker on the refresh after: none. Not with frame pacing
   (FrameDue times its pictures: a lead would only move them); nor with
   PS2HOST_REFRESH_LEAD=0. With
   PS2HOST_REFRESH_LOG, each 600 pictures: blank to shown (the screen's next
   refresh after done), what a press waits for beyond the game. */
void NoteGpuDone(std::uint64_t tag, std::uint64_t done_ns) {
    if (done_ns <= tag) return;
    double period = 0;
    u64 anchor = 0;
    if (!WindowRefreshTiming(&period, &anchor) || !(period > 0)) return;
    const u64 lag = done_ns - tag;
    struct Seen { u64 at, lag; };
    static std::vector<Seen> seen;              // the last two seconds'
    seen.push_back({done_ns, lag});
    std::size_t old = 0;
    while (old < seen.size() && seen[old].at + 2'000'000'000ull < done_ns) ++old;
    seen.erase(seen.begin(), seen.begin() + std::ptrdiff_t(old));
    u64 most = 0;
    double unled = 0;                           // waited on average with the blanks on the screen's
    for (const Seen& s : seen) {
        most = std::max(most, s.lag);
        unled += std::ceil(double(s.lag) / period) * period;
    }
    unled /= double(seen.size());
    const double led = double(most) * 1.1 + 1e6;   // waited by every picture with the lead
    static bool leading = false;
    leading = leading ? led < unled : led < unled - 1e6;
    const double lead = leading ? std::fmod(led, period) : 0.0;
    static const char* forced = std::getenv("PS2HOST_REFRESH_LEAD");     // (=0: none, to compare)
    const bool wanted = !(forced && forced[0] == '0') && !SettingOn("frame_pacing", FramePacingDefault());
    SetRefreshLead(wanted ? u64(lead) : 0);
    static const bool log = std::getenv("PS2HOST_REFRESH_LOG") != nullptr;
    if (!log) return;
    const double since = double(done_ns) - double(anchor);
    const double shown = double(done_ns) + (period - (since - std::floor(since / period) * period));
    static double sum = 0, longest = 0, lag_sum = 0;
    static unsigned n = 0;
    sum += shown - double(tag);
    longest = std::max(longest, shown - double(tag));
    lag_sum += double(lag);
    if (++n < 600) return;
    std::fprintf(stderr,
                 "[latency] 600 pictures: blank to shown %.2f ms (longest %.2f), blank to done on the GPU %.2f ms; "
                 "the blanks' lead %.2f ms\n",
                 sum / n / 1e6, longest / 1e6, lag_sum / n / 1e6, lead / 1e6);
    sum = longest = lag_sum = 0;
    n = 0;
}

bool OpenWindow(const wchar_t* title, int width, int height) {
#ifdef PS2HOST_VULKAN
    std::string narrow;
    for (const wchar_t* p = title; *p; ++p) narrow += char(*p < 128 ? *p : '?');
    if (!GameTitle().empty()) narrow = GameTitle();
    g_title = narrow;
    // The backend: the setting's, else Vulkan, then Direct3D 11 (Windows),
    // then SDL's 2D renderer with the software GS (sdl: that one alone).
    std::string want = Setting("renderer", "auto");
    if (const char* forced = std::getenv("PS2HOST_RENDERER")) want = forced;
    std::string error, why_not;
    const char* api = "Vulkan";
    if (want != "d3d11" && want != "sdl") g_display = ps2gpu::CreateVulkanDisplay(narrow.c_str(), width, height, &error);
#ifdef _WIN32
    if (!g_display && want != "vulkan" && want != "sdl") {
        if (want != "d3d11") why_not = " (no Vulkan display: " + error + ")";
        g_display = ps2gpu::CreateD3D11Display(narrow.c_str(), width, height, &error);
        api = "Direct3D 11";
    }
#endif
    if (!g_display && want != "vulkan" && want != "d3d11") {
        if (want != "sdl") why_not += " (no GPU display: " + error + ")";
        g_display = ps2gpu::CreateSdlDisplay(narrow.c_str(), width, height, &error);
        api = "SDL";
    }
    if (g_display) {
        Note("window", std::string("window: ") + api + " on " + g_display->Describe() + why_not);
        g_display->SetVrr(SettingOn("vrr"));
        g_display->SetGpuBoost(SettingOn("gpu_boost", true));      // (the GPU's clocks kept up)
        g_display->SetKeepRefresh(Setting("frame_rate", "60") == "display" || SettingOn("vrr"));
        if (SettingOn("fullscreen")) g_display->SetFullscreen(true);
        ApplyCursorArt();
        g_display->SetPresentObserver([](std::uint64_t due_ns, std::uint64_t shown_ns, bool dropped) {
            // Shown on the display's own thread (its lane named so); dropped
            // on the thread handing it a newer one.
            thread_local bool named = false;
            if (!dropped && !named) {
                timeline::NameThread("present");
                named = true;
            }
            if (dropped) timeline::Dropped(due_ns);
            else timeline::Presented(due_ns, shown_ns);
            // A picture dropped, or shown more than a period and a half
            // after the one before (the one before shown again meanwhile).
            static std::uint64_t last_shown = 0;
            if (dropped) {
                g_late_pictures.fetch_add(1, std::memory_order_relaxed);
            } else {
                if (last_shown && shown_ns - last_shown > BlankPeriodNs() * 3 / 2)
                    g_late_pictures.fetch_add(1, std::memory_order_relaxed);
                last_shown = shown_ns;
            }
        });
        if (std::getenv("PS2HOST_REFRESH_LOG")) g_display->SetShownObserver(NoteRefreshes);
        g_display->SetGpuDoneObserver(NoteGpuDone);
        if (SettingOn("vsync")) g_display->SetVsync(true);
        g_display->SetAspect(SettingAspect());
        g_display->SetAntiAliasing(DisplayAntiAliasing(Setting("antialiasing", "off")));
        return true;
    }
    Note("window", "window: no GPU display (" + error + ")" + why_not + "; drawing with GDI");
#endif
    return OpenGdiWindow(title, width, height);
}

bool WindowOpen() {
#ifdef PS2HOST_VULKAN
    if (g_display) return !g_closed;
#endif
    return g_window != nullptr;
}

bool WindowFocused() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Focused();
#endif
#ifdef _WIN32
    return g_window && GetForegroundWindow() == g_window;
#else
    return false;
#endif
}

bool WindowMinimized() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Minimized();
#endif
#ifdef _WIN32
    return g_window && IsIconic(g_window);
#else
    return false;
#endif
}

bool WindowClosed() { return g_closed; }

// A game's own EXIT GAME: as if the window had been closed.
void QuitGame() { g_closed = true; }

bool HostKeyDown(int key) {
    // --key: a run with scripted keys sees those alone - not whatever the
    // person at the PC types meanwhile (read with the window behind).
    if (KeysScripted()) return ScriptedKeyDown(key);
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->KeyDown(Scancode(key));
#endif
#ifdef _WIN32
    return (GetAsyncKeyState(key) & 0x8000) != 0;
#else
    return false;
#endif
}

void HostTriggerEffects(const HostTriggerEffect& left, const HostTriggerEffect& right) {
#ifdef PS2HOST_VULKAN
    if (g_display) {
        auto effect = [](const HostTriggerEffect& e) {
            ps2gpu::Display::TriggerEffect t;
            t.kind = ps2gpu::Display::TriggerEffect::Kind(e.kind);
            t.start = e.start;
            t.force = e.force;
            t.frequency = e.frequency;
            for (int i = 0; i < 10; ++i) t.zones[i] = e.zones[i];
            return t;
        };
        g_display->SetTriggerEffects(effect(left), effect(right));
    }
#else
    (void)left, (void)right;
#endif
}

void ReleaseController() {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->ReleaseController();
#endif
}

void HostRumble(u8 large, u8 small) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetRumble(u16(large * 257), u16(small * 257));
#else
    (void)large, (void)small;
#endif
}

bool TakeHostMouse(HostMouseState* state) {
    if (ScriptedMouse(state)) return true;      // --mouse
#ifdef PS2HOST_VULKAN
    ps2gpu::Display::MouseState m;
    if (g_display && g_display->TakeMouse(&m)) {
        state->over = m.over;
        state->x = m.x;
        state->y = m.y;
        state->held = m.held;
        state->pressed = m.pressed;
        state->wheel = m.wheel;
        state->moved = m.moved;
        return true;
    }
#endif
    (void)state;
    return false;
}

// The game's pointer (SetHostCursor), kept for a window opened after it is set.
struct CursorArt {
    std::vector<u32> rgba;
    u32 width = 0, height = 0, hot_x = 0, hot_y = 0;
};
CursorArt& TheCursorArt() {
    static CursorArt art;
    return art;
}

void ApplyCursorArt() {
#ifdef PS2HOST_VULKAN
    const CursorArt& a = TheCursorArt();
    if (g_display && !a.rgba.empty())
        g_display->SetCursorArt(std::vector<std::uint32_t>(a.rgba.begin(), a.rgba.end()), a.width, a.height,
                                a.hot_x, a.hot_y);
#endif
}

void SetHostCursor(const u32* rgba, u32 width, u32 height, u32 hot_x, u32 hot_y) {
    TheCursorArt() = CursorArt{std::vector<u32>(rgba, rgba + std::size_t(width) * height), width, height, hot_x, hot_y};
    ApplyCursorArt();
}

void ShowHostCursor(bool on) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->ShowGameCursor(on);
#else
    (void)on;
#endif
}

#ifdef PS2HOST_VULKAN
namespace {
ControllerFamily Family(std::uint8_t family) {
    return family == ps2gpu::Display::kPlayStationController ? ControllerFamily::kPlayStation
           : family == ps2gpu::Display::kXboxController      ? ControllerFamily::kXbox
                                                             : ControllerFamily::kOther;
}
void Convert(const ps2gpu::Display::PadState& pad, HostPadState* state) {
    state->buttons = pad.buttons;
    for (int i = 0; i < 4; ++i) state->sticks[i] = pad.sticks[i];
    state->l2 = pad.l2;
    state->r2 = pad.r2;
    state->family = Family(pad.family);
}
}  // namespace
#endif

bool HostPad(HostPadState* state) {
#ifdef PS2HOST_VULKAN
    ps2gpu::Display::PadState pad;
    if (g_display && g_display->Pad(&pad)) {
        Convert(pad, state);
        return true;
    }
#endif
    (void)state;
    return false;   // the GDI window has none
}

std::vector<HostDevice> WindowControllers() {
    std::vector<HostDevice> list;
#ifdef PS2HOST_VULKAN
    if (g_display)
        for (const auto& c : g_display->Controllers()) list.push_back({c.id, c.name, Family(c.family)});
#endif
    return list;
}

bool WindowControllerPad(u32 id, HostPadState* state) {
#ifdef PS2HOST_VULKAN
    ps2gpu::Display::PadState pad;
    if (g_display && g_display->PadOf(id, &pad)) {
        Convert(pad, state);
        return true;
    }
#endif
    (void)id, (void)state;
    return false;
}

void HostRumbleOf(u32 id, u8 large, u8 small) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetRumbleOf(id, u16(large * 257), u16(small * 257));
#else
    (void)id, (void)large, (void)small;
#endif
}

void HostTriggerTargets(const std::vector<u32>& ids) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetTriggerTargets(std::vector<std::uint32_t>(ids.begin(), ids.end()));
#else
    (void)ids;
#endif
}

namespace {
// --gpu-show: where the window's picture comes from instead of the software GS.
std::function<bool(std::vector<u32>*, u32*, u32*)> g_picture_source;
}  // namespace

#ifdef PS2HOST_VULKAN
ps2gpu::Gpu* WindowGpu() { return g_display ? g_display->SharedGpu() : nullptr; }
// A window that shows pixels alone (SDL's 2D renderer: no GPU display to be had).
bool WindowPixelsOnly() { return g_display && !g_display->SharedGpu(); }
#endif

void SetPictureSource(std::function<bool(std::vector<u32>*, u32*, u32*)> source) {
    g_picture_source = std::move(source);
}

namespace {
void RefreshOverlay();      // the overlay's lines and the FPS counter's, drawn
}  // namespace

// Alt+Enter changes fullscreen in the display's Pump: saved as the setting.
void KeepFullscreenSetting() {
#ifdef PS2HOST_VULKAN
    static int last = -1;
    const int now = g_display && g_display->Fullscreen() ? 1 : 0;
    if (last >= 0 && now != last) SetSettingOn("fullscreen", now != 0);
    last = now;
#endif
}

#ifdef PS2HOST_VULKAN
/* The display's events (SDL's, which on Android call into Java) on the
   thread's own stack, not an EE fiber's. */
static bool PumpDisplay() {
#ifdef _WIN32
    return g_display->Pump();
#else
    bool open = true;
    OnThreadStack([](void* p) { *static_cast<bool*>(p) = g_display->Pump(); }, &open);
    return open;
#endif
}
#endif

bool PumpWindow() {
#ifdef PS2HOST_VULKAN
    if (g_display) {
        if (!PumpDisplay()) g_closed = true;
        KeepFullscreenSetting();
        if (g_display->TakeCapture()) RequestCapture();
        UpdatePcMenu();                     // F9: the PC settings panel (pc_menu.cpp)
        // F1: the FPS counter on or off; its figure redrawn twice a second.
        static std::chrono::steady_clock::time_point fps_drawn{};
        const auto now = std::chrono::steady_clock::now();
        if (g_display->TakeFpsToggle()) {
            SetSettingOn("fps_counter", !SettingOn("fps_counter", false));
            RefreshOverlay();
            fps_drawn = now;
        } else if (now - fps_drawn >= std::chrono::milliseconds(500)) {
            RefreshOverlay();
            fps_drawn = now;
        }
        // Tab: the texture pack on or off, said on the screen for two seconds.
        static std::chrono::steady_clock::time_point notice_until{};
        static bool notice = false;
        if (g_display->TakeTexturePackToggle() && texture_pack::Count()) {
            const bool on = !texture_pack::Enabled();
            texture_pack::SetEnabled(on);
            SetSettingOn("texture_pack", on);
            SetWindowOverlay({on ? "TEXTURE PACK ON  (Tab)" : "TEXTURE PACK OFF  (Tab)"});
            notice = true;
            notice_until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        }
        if (notice && std::chrono::steady_clock::now() >= notice_until) {
            notice = false;
            if (!PcMenuOpen()) SetWindowOverlay({});
        }
        // A first start that went on before all its pipelines were made
        // (WaitForGpuPipelines' limit): said on the screen until the rest are
        // (a stutter meanwhile is why), its count redrawn twice a second.
        static std::chrono::steady_clock::time_point behind_drawn{};
        if (g_pipelines_behind && !notice && !PcMenuOpen() && now - behind_drawn >= std::chrono::milliseconds(500)) {
            behind_drawn = now;
            std::size_t made = 0, total = 0;
            if (ps2gpu::PipelineWarmProgress(&made, &total)) {
                char line[64];
                std::snprintf(line, sizeof line, "PREPARING GRAPHICS  %zu of %zu", made, total);
                SetWindowOverlay({line});
            } else {
                g_pipelines_behind = false;
                SetWindowOverlay({});
                std::fprintf(stderr, "[window] first start: all %zu GPU pipelines made\n", total);
            }
        }
        return !g_closed;
    }
#endif
    if (g_window) PresentGdi();
    return !g_closed;                  // a game's EXIT GAME (QuitGame) with no window too
}

void SetWindowFullscreen(bool on) {
    SetSettingOn("fullscreen", on);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetFullscreen(on);
#endif
}

namespace {

std::vector<std::string> g_overlay_lines;      // SetWindowOverlay's: the panel's, a notice's

// Draws the lines as the window's overlay (none when empty).
void DrawOverlay(const std::vector<std::string>& lines) {
#ifdef PS2HOST_VULKAN
    if (!g_display) return;
    if (lines.empty()) {
        g_display->SetOverlay({}, 0, 0);
        return;
    }
    // Text as Windows draws it, sized to the window (a 1080-line screen:
    // 22-pixel lines), white on a dark panel; a line starting with '>' in
    // yellow (the one chosen).
    const std::uint32_t screen = std::max(480u, g_display->PixelHeight());
#ifdef __ANDROID__
    const int line = std::max(14, int(screen / 30));      // (a phone: a small screen)
#else
    const int line = std::max(14, int(screen / 48));
#endif
#ifdef _WIN32
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT font = CreateFontA(-(line - 4), 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS,
                             CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN, "Consolas");
    HGDIOBJ old_font = SelectObject(dc, font);
    int width = 0;
    for (const std::string& text : lines) {
        SIZE size{};
        GetTextExtentPoint32A(dc, text.c_str(), int(text.size()), &size);
        width = std::max(width, int(size.cx));
    }
    width = std::min(1280, width + line);
    const int height = std::min(720, int(lines.size()) * line + line / 2);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof info.bmiHeader;
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    std::vector<std::uint32_t> pixels;
    if (bitmap && bits) {
        HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
        const RECT all{0, 0, width, height};
        HBRUSH panel = CreateSolidBrush(RGB(16, 20, 32));
        FillRect(dc, &all, panel);
        DeleteObject(panel);
        SetBkMode(dc, TRANSPARENT);
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const bool chosen = !lines[i].empty() && lines[i][0] == '>';
            SetTextColor(dc, chosen ? RGB(255, 210, 60) : i == 0 ? RGB(120, 190, 255) : RGB(235, 235, 235));
            TextOutA(dc, line / 2, int(i) * line + line / 4, lines[i].c_str(), int(lines[i].size()));
        }
        GdiFlush();
        pixels.assign(static_cast<const std::uint32_t*>(bits), static_cast<const std::uint32_t*>(bits) + width * height);
        for (std::uint32_t& p : pixels) p |= 0xFF000000u;
        SelectObject(dc, old_bitmap);
        DeleteObject(bitmap);
    }
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
#else
    // stb_easy_font's strokes (each glyph quads on a 7-unit-tall cell),
    // scaled to the line and filled in.
    const int scale = std::max(1, (line - 4) / 9);
    int width = 0;
    for (const std::string& text : lines)
        width = std::max(width, stb_easy_font_width(const_cast<char*>(text.c_str())) * scale);
    width = std::min(1280, width + line);
    const int height = std::min(720, int(lines.size()) * line + line / 2);
    std::vector<std::uint32_t> pixels(std::size_t(width) * height, 0xFF101420u);    // RGB(16, 20, 32), as GDI leaves it
    std::vector<float> quads(4 * 1024 * 4);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const bool chosen = !lines[i].empty() && lines[i][0] == '>';
        const std::uint32_t colour = chosen ? 0xFFFFD23Cu : i == 0 ? 0xFF78BEFFu : 0xFFEBEBEBu;  // (0xAARRGGBB, GDI's order)
        unsigned char unused[4] = {255, 255, 255, 255};
        const int n = stb_easy_font_print(0, 0, const_cast<char*>(lines[i].c_str()), unused, quads.data(),
                                          int(quads.size() * sizeof(float)));
        const int x0 = line / 2, y0 = int(i) * line + line / 4 + 1;
        for (int q = 0; q < n; ++q) {
            const float* v = &quads[std::size_t(q) * 16];      // 4 vertices: x y z colour
            const int left = x0 + int(v[0] * scale), top = y0 + int(v[1] * scale);
            const int right = x0 + int(v[8] * scale), bottom = y0 + int(v[9] * scale);
            for (int y = std::max(0, top); y < std::min(height, bottom); ++y)
                for (int x = std::max(0, left); x < std::min(width, right); ++x)
                    pixels[std::size_t(y) * width + x] = colour;
        }
    }
#endif
    g_display->SetOverlay(std::move(pixels), std::uint32_t(width), std::uint32_t(height));
#else
    (void)lines;
#endif
}

// The FPS counter (F1, setting "fps_counter", off unless turned on): the
// pictures shown a second, a line under whatever else the overlay says.
void RefreshOverlay() {
    std::vector<std::string> lines = g_overlay_lines;
#ifdef PS2HOST_VULKAN
    if (g_display && SettingOn("fps_counter", false)) {
        const double fps = g_display->PresentsPerSecond();
        char text[32];
        if (fps > 0) std::snprintf(text, sizeof text, "FPS %.1f", fps);
        else std::snprintf(text, sizeof text, "FPS --");
        lines.push_back(text);
        // "fps_detail" (a phone's PERFORMANCE OVERLAY at FPS + CPU): which
        // part is busy - the one near 100% the limit.
        if (SettingOn("fps_detail", false)) {
            const std::string load = HeavyThreadLoad();
            if (!load.empty()) lines.push_back(load);
            // The GPU's own time on the game's drawing (its timestamps): near
            // 100% it is the limit, not the processor.
            static std::uint64_t last_ns = 0;
            static auto last_at = std::chrono::steady_clock::now();
            const std::uint64_t busy = ps2gpu::GpuBusyNanoseconds();
            const auto now = std::chrono::steady_clock::now();
            const double seconds = std::chrono::duration<double>(now - last_at).count();
            if (last_ns && busy >= last_ns && seconds > 0) {
                char gpu[48];
                std::snprintf(gpu, sizeof gpu, "GPU %d%%", int(100.0 * double(busy - last_ns) / 1e9 / seconds + 0.5));
                lines.push_back(gpu);
            }
            last_ns = busy;
            last_at = now;
        }
    }
#endif
    static std::vector<std::string> shown;
    if (lines == shown) return;                 // drawn again only when it changes
    shown = lines;
    DrawOverlay(lines);
}

}  // namespace

void SetWindowOverlay(const std::vector<std::string>& lines) {
    g_overlay_lines = lines;
    RefreshOverlay();
}

void SetWindowStatus(const std::string& status) {
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetTitle(status.empty() ? g_title : g_title + "  -  " + status);
#else
    (void)status;
#endif
}

void SetWindowAspect(const std::string& aspect) {
    SetSetting("aspect", aspect);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetAspect(SettingAspect());
#endif
}

void SetWindowAntiAliasing(const std::string& mode) {
    SetSetting("antialiasing", mode);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetAntiAliasing(DisplayAntiAliasing(mode));
#endif
}

u32 SupersamplingFactor() {
    const std::string mode = Setting("antialiasing", "off");
    return mode == "ssaa2" ? 2 : mode == "ssaa4" ? 4 : 1;
}

float PictureAspect() {
    const float aspect = SettingAspect();
    if (aspect > 0.0f) return aspect;
#ifdef PS2HOST_VULKAN
    if (g_display) {
        const float window = g_display->WindowAspect();
        if (window > 0.0f) return window;
    }
#endif
    return 4.0f / 3.0f;
}

void SetWindowVsync(bool on) {
    SetSettingOn("vsync", on);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetVsync(on);
#endif
}

bool WindowVsync() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Vsync();
#endif
    return SettingOn("vsync");
}

bool WindowFullscreen() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->Fullscreen();
#endif
    return SettingOn("fullscreen");
}

void SetWindowKeepRefresh(bool keep) {
#ifdef PS2HOST_VULKAN
    // (On a VRR screen, always: its own refresh is the most it can follow.)
    if (g_display) g_display->SetKeepRefresh(keep || SettingOn("vrr"));
#endif
    (void)keep;
}

void SetWindowVrr(bool on) {
    SetSettingOn("vrr", on);
#ifdef PS2HOST_VULKAN
    if (g_display) {
        g_display->SetVrr(on);
        g_display->SetKeepRefresh(on || Setting("frame_rate", "60") == "display");
    }
#endif
}

bool WindowVrr() { return SettingOn("vrr"); }

void SetWindowGpuBoost(bool on) {
    SetSettingOn("gpu_boost", on);
#ifdef PS2HOST_VULKAN
    if (g_display) g_display->SetGpuBoost(on);
#endif
}

bool WindowGpuBoost() { return SettingOn("gpu_boost", true); }

bool WindowGpuBoosted() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->GpuBoosted();
#endif
    return false;
}

std::uint64_t LatePictures() { return g_late_pictures.load(std::memory_order_relaxed); }

float WindowRefreshRate() {
#ifdef PS2HOST_VULKAN
    if (g_display) return g_display->RefreshRate();
#endif
    return 0.0f;
}

bool WindowRefreshTiming(double* period_ns, u64* blank_ns) {
#ifdef PS2HOST_VULKAN
    std::uint64_t blank = 0;
    if (g_display && g_display->RefreshTiming(period_ns, &blank)) {
        *blank_ns = blank;
        return true;
    }
#endif
    (void)period_ns, (void)blank_ns;
    return false;
}

ps2gpu::Display* WindowDisplay() {
#ifdef PS2HOST_VULKAN
    return g_display.get();
#else
    return nullptr;
#endif
}

#ifdef PS2HOST_VULKAN
namespace {

/* `text` into a 640 x 448 picture (R in the low byte), centred on `y`,
   stb_easy_font's strokes `scale` times. */
void PictureText(std::vector<u32>& picture, const char* text, int y, int scale, u32 colour) {
    std::vector<float> quads(4 * 1024 * 4);
    unsigned char unused[4] = {255, 255, 255, 255};
    const int n = stb_easy_font_print(0, 0, const_cast<char*>(text), unused, quads.data(),
                                      int(quads.size() * sizeof(float)));
    const int x0 = (640 - stb_easy_font_width(const_cast<char*>(text)) * scale) / 2;
    for (int q = 0; q < n; ++q) {
        const float* v = &quads[std::size_t(q) * 16];
        const int left = x0 + int(v[0] * scale), top = y + int(v[1] * scale);
        const int right = x0 + int(v[8] * scale), bottom = y + int(v[9] * scale);
        for (int py = std::max(0, top); py < std::min(448, bottom); ++py)
            for (int px = std::max(0, left); px < std::min(640, right); ++px) picture[std::size_t(py) * 640 + px] = colour;
    }
}

}  // namespace
#endif

/* A first start on a device (or after a new driver): the GPU's pipelines
   compiled before the game starts, a loading screen saying so meanwhile -
   a phone's driver takes seconds to a minute, which would otherwise be a
   black screen and then stutter. A start whose pipelines come from the
   cache is through before anything is shown. On a PC the wait is at most
   10 s, the rest made in the background as the game starts (the way
   before the screen: a driver with no cache of its own took long enough
   that players took the game for hung - issue #29); the log says how far
   it got every 2 s. PS2HOST_PIPELINE_WAIT_S: that limit (0: none). */
void WaitForGpuPipelines() {
#ifdef PS2HOST_VULKAN
    if (!g_display) return;
    const auto began = std::chrono::steady_clock::now();
    std::size_t made = 0, total = 0;
    bool shown = false;
    // PS2HOST_LOADING_HOLD=ms: the screen shown at once and kept that long (to see it).
    const char* hold_env = std::getenv("PS2HOST_LOADING_HOLD");
    const auto hold = std::chrono::milliseconds(hold_env ? std::atoi(hold_env) : 0);
#ifdef __ANDROID__
    int limit_s = 0;
#else
    int limit_s = 10;
#endif
    if (const char* limit = std::getenv("PS2HOST_PIPELINE_WAIT_S")) limit_s = std::atoi(limit);
    auto said = began;
    while (ps2gpu::PipelineWarmProgress(&made, &total) || std::chrono::steady_clock::now() - began < hold) {
        const auto now = std::chrono::steady_clock::now();
        if (now - said >= std::chrono::seconds(2)) {
            said = now;
            std::fprintf(stderr, "[window] first start: %zu of %zu GPU pipelines made after %.0f s\n", made, total,
                         std::chrono::duration<double>(now - began).count());
            std::fflush(stderr);
        }
        if (limit_s > 0 && now - began >= std::chrono::seconds(limit_s) && now - began >= hold) {
            std::fprintf(stderr, "[window] first start: the game goes on after %d s, %zu of %zu GPU pipelines made "
                         "(the rest in the background)\n", limit_s, made, total);
            g_pipelines_behind = true;      // said on the screen until made (PumpWindow)
            break;
        }
        if (hold_env || now - began >= std::chrono::milliseconds(300)) {
            shown = true;
            std::vector<u32> picture(640 * 448, 0xFF201410u);         // a dark blue
            PictureText(picture, "PREPARING GRAPHICS", 150, 4, 0xFFFFFFFFu);
            PictureText(picture, "first start on this device only", 200, 2, 0xFFC0C0C0u);
            // The bar: made of total.
            const int bar_x = 120, bar_w = 400, bar_y = 250, bar_h = 14;
            const int filled = total ? int(std::int64_t(bar_w) * std::int64_t(made) / std::int64_t(total)) : 0;
            for (int y = bar_y; y < bar_y + bar_h; ++y)
                for (int x = bar_x; x < bar_x + bar_w; ++x) {
                    const bool edge = y == bar_y || y == bar_y + bar_h - 1 || x == bar_x || x == bar_x + bar_w - 1;
                    picture[std::size_t(y) * 640 + x] = edge ? 0xFFC0C0C0u : x - bar_x < filled ? 0xFF3C8CFFu : 0xFF302018u;
                }
            char count[64];
            std::snprintf(count, sizeof count, "%zu of %zu", made, total);
            PictureText(picture, count, 280, 2, 0xFFC0C0C0u);
            if (!g_display->Present(picture.data(), 640, 448)) break;
        }
        if (!g_display->Pump()) {
            g_closed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(shown ? 30 : 5));
    }
    if (shown && made == total)
        std::fprintf(stderr, "[window] first start: %zu GPU pipelines made in %.1f s behind the loading screen\n", total,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
#endif
}

void PresentFrame() {
#ifdef PS2HOST_VULKAN
    if (g_display) {
        if (!PumpDisplay()) {
            g_closed = true;
            return;
        }
        KeepFullscreenSetting();
        std::vector<u32> pixels;
        u32 w = 0, h = 0;
        const bool got = (g_picture_source && g_picture_source(&pixels, &w, &h)) || gs::TheGs().Picture(&pixels, &w, &h);
        if (got && !g_display->Present(pixels.data(), w, h)) {
            std::fprintf(stderr, "[window] a picture could not be shown: the window closed\n");
            g_closed = true;
        }
        return;
    }
#endif
    if (g_window) PresentGdi();
}

}  // namespace ps2host
