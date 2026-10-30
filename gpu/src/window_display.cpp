/*
 * SdlWindowDisplay (window_display.h): the window, its events and input, and
 * the screen's refresh - shared by every display backend.
 */

#include "window_display.h"

#include <algorithm>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#undef small            // rpcndr.h's, through the COM headers; SetRumble names a parameter so
#pragma comment(lib, "dxgi.lib")
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>

namespace ps2gpu {

std::uint64_t SteadyNs() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch()).count());
}

namespace {
#ifdef _WIN32
/* The DXGI output that shows `monitor`, on whichever adapter (a laptop's
   built-in screen is the integrated GPU's); null if none. */
IDXGIOutput* FindOutput(HMONITOR monitor) {
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return nullptr;
    IDXGIOutput* found = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    for (UINT a = 0; !found && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
        IDXGIOutput* output = nullptr;
        for (UINT o = 0; !found && adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
            DXGI_OUTPUT_DESC desc;
            if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == monitor) found = output;
            else output->Release();
        }
        adapter->Release();
    }
    factory->Release();
    return found;
}

void WatchVblanks(std::shared_ptr<VblankWatch> watch, HWND window) {
    // Woken at each blank, to note its time: the sooner it runs, the closer.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    IDXGIOutput* output = nullptr;
    HMONITOR on = nullptr;
    unsigned looks = 0, quick = 0;
    std::uint64_t before = 0;
    while (!watch->stop.load(std::memory_order_relaxed)) {
        // The window's screen, looked at again now and then (moved to another).
        if (looks++ % 256 == 0 || !output) {
            const HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
            if (monitor != on || !output) {
                if (output) output->Release();
                output = FindOutput(monitor);
                on = monitor;
                std::lock_guard<std::mutex> hold(watch->lock);
                watch->fit.Reset();
            }
        }
        if (!output || FAILED(output->WaitForVBlank())) {
            if (output) output->Release();
            output = nullptr;
            Sleep(250);
            continue;
        }
        const std::uint64_t now = SteadyNs();
        // A wait that does not wait (a screen asleep): not a blank, and no spin.
        if (now - before < 1'000'000) {
            if (++quick > 8) Sleep(50);
            before = now;
            continue;
        }
        quick = 0;
        before = now;
        const float hz = watch->nominal_hz.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> hold(watch->lock);
        watch->fit.Add(now, hz > 1.0f ? 1e9 / double(hz) : 0.0);
    }
    if (output) output->Release();
}
#endif

}  // namespace

bool SdlWindowDisplay::OpenWindow(const char* title, int width, int height, SDL_WindowFlags graphics) {
#ifdef __ANDROID__
    // A phone's game: on its side either way, the Back button the game's
    // (Escape's: KeyDown), not the system's (which would end it).
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    SDL_SetHint(SDL_HINT_ANDROID_TRAP_BACK_BUTTON, "1");
#endif
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        error_ = std::string("SDL_Init: ") + SDL_GetError();
        return false;
    }
    // Controllers: optional (the keyboard still plays), and read whichever
    // window has focus - the console's pad does not care either. With
    // PS2HOST_PAD_FOCUS=1, only while this window has it: two instances on
    // one PC each take the pad in turn.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, std::getenv("PS2HOST_PAD_FOCUS") ? "0" : "1");
    if (!SDL_InitSubSystem(SDL_INIT_GAMEPAD)) {
        std::fprintf(stderr, "[window] no game controllers: %s\n", SDL_GetError());
    } else {
        int joysticks = 0, gamepads = 0;
        // Wireless controllers can take a moment to be found. (Android: no
        // wait - a phone often has none, and one connected is taken as it comes.)
#ifdef __ANDROID__
        constexpr int kControllerWait = 0;
#else
        constexpr int kControllerWait = 1000;
#endif
        for (int waited = 0; waited < kControllerWait && joysticks == 0; waited += 50) {
            SDL_PumpEvents();
            SDL_free(SDL_GetJoysticks(&joysticks));
            if (joysticks == 0) SDL_Delay(50);
        }
        SDL_free(SDL_GetGamepads(&gamepads));
        std::fprintf(stderr, "[window] %d controller(s) plugged in, %d of them with a known layout\n", joysticks,
                     gamepads);
    }
#ifdef __ANDROID__
    // The whole screen, its bars hidden.
    window_ = SDL_CreateWindow(title, width, height, graphics | SDL_WINDOW_FULLSCREEN);
#else
    window_ = SDL_CreateWindow(title, width, height, graphics | SDL_WINDOW_RESIZABLE);
#endif
    if (!window_) {
        error_ = std::string("SDL_CreateWindow: ") + SDL_GetError();
        return false;
    }
#ifdef _WIN32
    // The screen's blanks, measured (RefreshTiming).
    if (HWND hwnd = static_cast<HWND>(
            SDL_GetPointerProperty(SDL_GetWindowProperties(window_), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr))) {
        vblank_watch_ = std::make_shared<VblankWatch>();
        std::thread(WatchVblanks, vblank_watch_, hwnd).detach();
    }
#endif
    return true;
}

SdlWindowDisplay::~SdlWindowDisplay() {
    if (vblank_watch_) vblank_watch_->stop = true;      // (its thread ends at its next blank)
    for (Gamepad& g : pads_) SDL_CloseGamepad(g.pad);
    if (cursor_) SDL_DestroyCursor(cursor_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_Quit();
}

/* The display mode fullscreen takes: the desktop's resolution at a refresh
   that the game's ~60 frames a second divide evenly (120 Hz, else 60 Hz),
   not above the desktop's - null for the desktop's own (borderless), when it
   is such a rate already or the display offers none. On a 144 Hz display a
   frame took two refreshes, then three (14 and 21 ms): a judder no pacing
   undoes. */
const SDL_DisplayMode* EvenMode(SDL_Window* window) {
    const SDL_DisplayID id = SDL_GetDisplayForWindow(window);
    const SDL_DisplayMode* desktop = id ? SDL_GetDesktopDisplayMode(id) : nullptr;
    if (!desktop) return nullptr;
    auto even = [](float hz) {
        const int n = int(hz / 60.0f + 0.5f);
        return n >= 1 && std::fabs(hz - 60.0f * float(n)) < 0.6f * float(n);
    };
    if (even(desktop->refresh_rate)) return nullptr;
    int count = 0;
    SDL_DisplayMode** modes = SDL_GetFullscreenDisplayModes(id, &count);
    static SDL_DisplayMode chosen;
    const SDL_DisplayMode* best = nullptr;
    for (int i = 0; modes && i < count; ++i) {
        const SDL_DisplayMode* m = modes[i];
        if (m->w != desktop->w || m->h != desktop->h || !even(m->refresh_rate)) continue;
        if (m->refresh_rate > desktop->refresh_rate + 0.5f) continue;
        if (!best || m->refresh_rate > best->refresh_rate) best = m;
    }
    if (best) chosen = *best;
    SDL_free(modes);
    return best ? &chosen : nullptr;
}

void SdlWindowDisplay::SetFullscreen(bool on) {
    if (!window_) return;
    if (on) {
        const SDL_DisplayMode* mode = keep_refresh_ ? nullptr : EvenMode(window_);
        SDL_SetWindowFullscreenMode(window_, mode);
        if (mode)
            std::fprintf(stderr, "[display] fullscreen at %dx%d, %.2f Hz\n", mode->w, mode->h,
                         double(mode->refresh_rate));
        else if (keep_refresh_)
            if (const SDL_DisplayMode* desktop = SDL_GetDesktopDisplayMode(SDL_GetDisplayForWindow(window_)))
                std::fprintf(stderr, "[display] fullscreen at the screen's own %.2f Hz (frame rate DISPLAY)\n",
                             double(desktop->refresh_rate));
    }
    SDL_SetWindowFullscreen(window_, on);
    fullscreen_cursor_stale_ = true;        // the system's pointer: hidden in fullscreen (UpdateCursor)
    cursor_showing_ = -1;
    UpdateCursor();
    stale_ = true;
}

bool SdlWindowDisplay::Fullscreen() const {
    return window_ && (SDL_GetWindowFlags(window_) & SDL_WINDOW_FULLSCREEN) != 0;
}

bool SdlWindowDisplay::Pump() {
    {
        std::lock_guard<std::mutex> hold(title_lock_);
        if (title_changed_ && window_) SDL_SetWindowTitle(window_, title_.c_str());
        title_changed_ = false;
    }
    // The screen's refresh, for RefreshRate on other threads: looked at
    // again every hundred pumps (the window moved to another screen, a
    // fullscreen mode taken).
    if (window_ && refresh_looks_++ % 100 == 0) {
        const SDL_DisplayID id = SDL_GetDisplayForWindow(window_);
        const SDL_DisplayMode* mode = id ? SDL_GetCurrentDisplayMode(id) : nullptr;
        refresh_hz_.store(mode ? mode->refresh_rate : 0.0f, std::memory_order_relaxed);
        if (vblank_watch_) vblank_watch_->nominal_hz.store(mode ? mode->refresh_rate : 0.0f, std::memory_order_relaxed);
    }
    // Each controller's: the triggers' effects as last set, when they change
    // (off on one SetTriggerTargets leaves out), and the motors as the game
    // last set them - renewed before SDL's duration runs out, stopped when
    // set to zero.
    {
        TriggerEffect wanted[2];
        std::vector<std::uint32_t> targets;
        {
            std::lock_guard<std::mutex> hold(triggers_lock_);
            wanted[0] = triggers_[0];
            wanted[1] = triggers_[1];
            targets = trigger_targets_;
        }
        const std::uint64_t now = SDL_GetTicks();
        std::lock_guard<std::mutex> hold(pads_lock_);
        for (Gamepad& g : pads_) {
            const bool target = targets.empty() || std::find(targets.begin(), targets.end(), g.id) != targets.end();
            const TriggerEffect left = target ? wanted[0] : TriggerEffect{}, right = target ? wanted[1] : TriggerEffect{};
            if (g.triggers_unsent || !(left == g.triggers_sent[0]) || !(right == g.triggers_sent[1])) {
                SendTriggers(g.pad, left, right);
                g.triggers_sent[0] = left;
                g.triggers_sent[1] = right;
                g.triggers_unsent = false;
            }
            const std::uint32_t r = g.rumble;
            if (r != g.rumble_sent || (r && now - g.rumble_renewed > 200)) {
                SDL_RumbleGamepad(g.pad, std::uint16_t(r >> 16), std::uint16_t(r), r ? 400 : 0);
                g.rumble_sent = r;
                g.rumble_renewed = now;
            }
        }
    }
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_QUIT || e.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
            if (!closed_) std::fprintf(stderr, "[window] closed (SDL event 0x%X)\n", unsigned(e.type));
            closed_ = true;
        }
        if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) stale_ = true;
        if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.scancode == SDL_SCANCODE_RETURN &&
            (e.key.mod & SDL_KMOD_ALT))
            SetFullscreen(!Fullscreen());
        if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.scancode == SDL_SCANCODE_F12) capture_ = true;
        if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.scancode == SDL_SCANCODE_TAB) texture_pack_ = true;
        if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat && e.key.scancode == SDL_SCANCODE_F1) fps_toggle_ = true;
        if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN &&
            (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK || e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK)) {
            SDL_Gamepad* pad = SDL_GetGamepadFromID(e.gbutton.which);
            if (pad && SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_LEFT_STICK) &&
                SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_RIGHT_STICK))
                capture_ = true;
        }
        if (e.type == SDL_EVENT_MOUSE_MOTION) MouseAt(e.motion.x, e.motion.y);
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            MouseAt(e.button.x, e.button.y);
            const int bit = e.button.button == SDL_BUTTON_LEFT ? 0 : e.button.button == SDL_BUTTON_MIDDLE ? 1
                          : e.button.button == SDL_BUTTON_RIGHT ? 2 : -1;
            std::lock_guard<std::mutex> hold(mouse_lock_);
            if (bit >= 0 && e.button.down) {
                mouse_.held |= 1u << bit;
                mouse_.pressed |= 1u << bit;
            } else if (bit >= 0) {
                mouse_.held &= ~(1u << bit);
            }
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            std::lock_guard<std::mutex> hold(mouse_lock_);
            mouse_.wheel += e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.y : e.wheel.y;
        }
        if (e.type == SDL_EVENT_WINDOW_MOUSE_LEAVE) {
            std::lock_guard<std::mutex> hold(mouse_lock_);
            mouse_.over = false;
            mouse_.held = 0;
        }
        if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            std::lock_guard<std::mutex> hold(mouse_lock_);
            mouse_.held = 0;
        }
        if (e.type == SDL_EVENT_GAMEPAD_ADDED) {
            std::lock_guard<std::mutex> hold(pads_lock_);
            if (!FindPad(e.gdevice.which)) {
                if (SDL_Gamepad* pad = SDL_OpenGamepad(e.gdevice.which)) {
                    Gamepad g;
                    g.pad = pad;
                    g.id = e.gdevice.which;
                    pads_.push_back(g);
                    std::fprintf(stderr, "[window] controller %u: %s\n", unsigned(g.id), SDL_GetGamepadName(pad));
                }
            }
        }
        if (e.type == SDL_EVENT_GAMEPAD_REMOVED) {
            std::lock_guard<std::mutex> hold(pads_lock_);
            for (auto g = pads_.begin(); g != pads_.end(); ++g)
                if (g->id == e.gdevice.which) {
                    std::fprintf(stderr, "[window] controller %u unplugged\n", unsigned(g->id));
                    SDL_CloseGamepad(g->pad);
                    pads_.erase(g);
                    break;
                }
        }
    }
    UpdateCursor();
    return !closed_;
}

/* The pointer at (x, y), in the window's own units (points): in pixels, then
   across the picture as shown. */
void SdlWindowDisplay::MouseAt(float x, float y) {
    const float density = window_ ? SDL_GetWindowPixelDensity(window_) : 1.0f;
    const float px = x * (density > 0.0f ? density : 1.0f), py = y * (density > 0.0f ? density : 1.0f);
    std::lock_guard<std::mutex> hold(mouse_lock_);
    if (px != mouse_px_ || py != mouse_py_ || !mouse_.over) mouse_.moved = true;
    mouse_px_ = px;
    mouse_py_ = py;
    mouse_.over = true;
    mouse_.x = (px - float(shown_x0_.load())) / float(shown_w_.load());
    mouse_.y = (py - float(shown_y0_.load())) / float(shown_h_.load());
}

/* The window's pointer: the game's, at a size that goes with the picture's
   (its art's height a twentieth of the picture's), while the game wants it;
   otherwise the system's - hidden in fullscreen. */
void SdlWindowDisplay::UpdateCursor() {
    if (!window_) return;
#ifdef __ANDROID__
    // A phone: no pointer to draw (a tap is the click) - and SDL's cursors
    // call into Java, which the game's fibers must never do.
    return;
#endif
    const bool want = cursor_wanted_.load();
    std::uint32_t size = std::uint32_t(float(shown_h_.load()) / 20.0f + 0.5f);
    size = std::clamp<std::uint32_t>(size, 20, 128);
    if (want) {
        std::lock_guard<std::mutex> hold(mouse_lock_);
        if (!cursor_art_.empty() && (!cursor_ || cursor_size_ != size)) {
            // The art resampled to `size` (bilinear, by premultiplied colour).
            const std::uint32_t w = size, h = std::uint32_t(float(size) * float(cursor_height_) /
                                                            float(cursor_width_) + 0.5f);
            SDL_Surface* surface = SDL_CreateSurface(int(w), int(h), SDL_PIXELFORMAT_RGBA32);
            if (surface) {
                for (std::uint32_t y = 0; y < h; ++y) {
                    auto* row = reinterpret_cast<std::uint8_t*>(surface->pixels) + std::size_t(y) * surface->pitch;
                    for (std::uint32_t x = 0; x < w; ++x) {
                        const float sx = (float(x) + 0.5f) * float(cursor_width_) / float(w) - 0.5f;
                        const float sy = (float(y) + 0.5f) * float(cursor_height_) / float(h) - 0.5f;
                        const int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
                        const float fx = sx - float(x0), fy = sy - float(y0);
                        float sum[4] = {};
                        for (int k = 0; k < 4; ++k) {
                            const int ix = std::clamp(x0 + (k & 1), 0, int(cursor_width_) - 1);
                            const int iy = std::clamp(y0 + (k >> 1), 0, int(cursor_height_) - 1);
                            const float weight = ((k & 1) ? fx : 1.0f - fx) * ((k >> 1) ? fy : 1.0f - fy);
                            const std::uint32_t p = cursor_art_[std::size_t(iy) * cursor_width_ + ix];
                            const float a = float(p >> 24) / 255.0f;
                            sum[0] += weight * float(p & 0xFF) * a;
                            sum[1] += weight * float((p >> 8) & 0xFF) * a;
                            sum[2] += weight * float((p >> 16) & 0xFF) * a;
                            sum[3] += weight * a;
                        }
                        const float a = sum[3];
                        row[4 * x + 0] = std::uint8_t(a > 0.0f ? std::min(255.0f, sum[0] / a + 0.5f) : 0.0f);
                        row[4 * x + 1] = std::uint8_t(a > 0.0f ? std::min(255.0f, sum[1] / a + 0.5f) : 0.0f);
                        row[4 * x + 2] = std::uint8_t(a > 0.0f ? std::min(255.0f, sum[2] / a + 0.5f) : 0.0f);
                        row[4 * x + 3] = std::uint8_t(std::min(255.0f, a * 255.0f + 0.5f));
                    }
                }
                SDL_Cursor* made = SDL_CreateColorCursor(surface, int(float(cursor_hot_x_) * float(w) / float(cursor_width_)),
                                                         int(float(cursor_hot_y_) * float(h) / float(cursor_height_)));
                SDL_DestroySurface(surface);
                if (made) {
                    if (cursor_showing_ == 1) SDL_SetCursor(made);
                    if (cursor_) SDL_DestroyCursor(cursor_);
                    cursor_ = made;
                    cursor_size_ = size;
                    std::fprintf(stderr, "[display] the game's pointer at %ux%u\n", w, h);
                } else {
                    std::fprintf(stderr, "[display] the game's pointer: %s\n", SDL_GetError());
                }
            }
        }
    }
    const int showing = want && cursor_ ? 1 : 0;
    if (showing == cursor_showing_ && !(showing == 0 && fullscreen_cursor_stale_)) return;
    cursor_showing_ = showing;
    fullscreen_cursor_stale_ = false;
    if (showing) {
        SDL_SetCursor(cursor_);
        SDL_ShowCursor();
    } else {
        SDL_SetCursor(SDL_GetDefaultCursor());
        if (Fullscreen()) SDL_HideCursor();
        else SDL_ShowCursor();
    }
}

namespace {
std::uint8_t FamilyOf(SDL_Gamepad* pad) {
    switch (SDL_GetGamepadType(pad)) {
        case SDL_GAMEPAD_TYPE_PS3:
        case SDL_GAMEPAD_TYPE_PS4:
        case SDL_GAMEPAD_TYPE_PS5:
            return Display::kPlayStationController;
        case SDL_GAMEPAD_TYPE_XBOX360:
        case SDL_GAMEPAD_TYPE_XBOXONE:
            return Display::kXboxController;
        default:
            return Display::kOtherController;
    }
}

/* One controller as a DualShock 2 reads. */
Display::PadState ReadPad(SDL_Gamepad* pad) {
    // SDL's buttons by position (south is the DualShock's cross) to
    // DualShock 2 inputs.
    static const std::pair<SDL_GamepadButton, int> kButtons[] = {
        {SDL_GAMEPAD_BUTTON_BACK, 0}, {SDL_GAMEPAD_BUTTON_LEFT_STICK, 1}, {SDL_GAMEPAD_BUTTON_RIGHT_STICK, 2},
        {SDL_GAMEPAD_BUTTON_START, 3}, {SDL_GAMEPAD_BUTTON_DPAD_UP, 4}, {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, 5},
        {SDL_GAMEPAD_BUTTON_DPAD_DOWN, 6}, {SDL_GAMEPAD_BUTTON_DPAD_LEFT, 7},
        {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, 10}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, 11},
        {SDL_GAMEPAD_BUTTON_NORTH, 12}, {SDL_GAMEPAD_BUTTON_EAST, 13}, {SDL_GAMEPAD_BUTTON_SOUTH, 14},
        {SDL_GAMEPAD_BUTTON_WEST, 15}};
    Display::PadState s;
    for (const auto& [button, input] : kButtons)
        if (SDL_GetGamepadButton(pad, button)) s.buttons |= std::uint16_t(1u << input);
    // Triggers (0-32767) are L2 and R2: pressed past a quarter of their travel.
    const int l2 = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
    const int r2 = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
    s.l2 = std::uint8_t(std::clamp(l2, 0, 32767) >> 7);
    s.r2 = std::uint8_t(std::clamp(r2, 0, 32767) >> 7);
    if (l2 > 8192) s.buttons |= 1u << 8;
    if (r2 > 8192) s.buttons |= 1u << 9;
    // Sticks: -32768-32767 to 0-255, down and right the larger, as the
    // DualShock's; a small dead zone reads as centred.
    static const SDL_GamepadAxis kAxes[4] = {SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY,
                                             SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY};
    for (int i = 0; i < 4; ++i) {
        const int v = SDL_GetGamepadAxis(pad, kAxes[i]);
        s.sticks[i] = (v > -4096 && v < 4096) ? 0x80 : std::uint8_t((v + 32768) >> 8);
    }
    s.family = FamilyOf(pad);
    return s;
}
}  // namespace

namespace {
std::mutex g_touch_lock;
Display::PadState g_touch;          // SetTouchPad's
bool g_touch_shown = false;         // ShowTouchPad's
// A tap shorter than a frame read as a press all the same: each button held
// at least this long from when it went down.
constexpr std::uint64_t kTapHoldNs = 60000000;
std::uint64_t g_touch_down[16] = {};

Display::PadState TouchPadNow() {
    Display::PadState s = g_touch;
    const std::uint64_t now = SDL_GetTicksNS();
    for (int b = 0; b < 16; ++b)
        if (g_touch_down[b] && now - g_touch_down[b] < kTapHoldNs) s.buttons |= std::uint16_t(1u << b);
    if (s.buttons & (1u << 8)) s.l2 = 255;
    if (s.buttons & (1u << 9)) s.r2 = 255;
    return s;
}
}  // namespace

void SetTouchPad(const Display::PadState& state) {
    std::lock_guard<std::mutex> hold(g_touch_lock);
    const std::uint16_t went_down = std::uint16_t(state.buttons & ~g_touch.buttons);
    const std::uint64_t now = SDL_GetTicksNS();
    for (int b = 0; b < 16; ++b)
        if (went_down & (1u << b)) g_touch_down[b] = now;
    g_touch = state;
    g_touch.family = Display::kPlayStationController;
}

void ShowTouchPad(bool shown) {
    std::lock_guard<std::mutex> hold(g_touch_lock);
    g_touch_shown = shown;
    if (!shown) g_touch = Display::PadState{};
}

bool TouchPadShown() {
    std::lock_guard<std::mutex> hold(g_touch_lock);
    return g_touch_shown;
}

bool SdlWindowDisplay::Pad(PadState* state) const {
    {
        std::lock_guard<std::mutex> hold(pads_lock_);
        if (!pads_.empty()) {
            *state = ReadPad(pads_.front().pad);
            return true;
        }
    }
    std::lock_guard<std::mutex> hold(g_touch_lock);
    if (!g_touch_shown) return false;
    *state = TouchPadNow();
    return true;
}

bool SdlWindowDisplay::PadOf(std::uint32_t id, PadState* state) const {
    if (id == kTouchPadId) {
        std::lock_guard<std::mutex> hold(g_touch_lock);
        if (!g_touch_shown) return false;
        *state = TouchPadNow();
        return true;
    }
    std::lock_guard<std::mutex> hold(pads_lock_);
    const Gamepad* g = FindPad(id);
    if (!g) return false;
    *state = ReadPad(g->pad);
    return true;
}

std::vector<Display::ControllerInfo> SdlWindowDisplay::Controllers() const {
    std::vector<ControllerInfo> list;
    {
        std::lock_guard<std::mutex> hold(pads_lock_);
        for (const Gamepad& g : pads_) {
            const char* name = SDL_GetGamepadName(g.pad);
            list.push_back({g.id, name ? name : "Controller", FamilyOf(g.pad)});
        }
    }
    std::lock_guard<std::mutex> hold(g_touch_lock);
    if (g_touch_shown) list.push_back({kTouchPadId, "Touch screen", kPlayStationController});
    return list;
}

/* A DualSense's effects packet (SDL's HIDAPI PS5 driver takes it as the
   controller's output report's effects part: SDL_SendGamepadEffect), the
   triggers' alone: their effects at 10 (right) and 21 (left), 11 bytes each,
   enabled by bits 2 and 3 of the first byte. An effect: its mode, then its
   parameters - 0x05 off; 0x01 resistance (start, force); 0x06 vibration
   (frequency, amplitude, start); and by zones (the travel in ten), 0x21
   resistance and 0x26 vibration: the zones on (a bit each, two bytes), then
   each one's strength less 1 (3 bits each, four bytes), and for 0x26 the
   frequency at +9. */
void SdlWindowDisplay::SendTriggers(SDL_Gamepad* pad, const TriggerEffect& left, const TriggerEffect& right) {
    if (!pad || SDL_GetGamepadType(pad) != SDL_GAMEPAD_TYPE_PS5) return;
    std::uint8_t packet[47] = {};
    packet[0] = 0x04 | 0x08;
    auto put = [](std::uint8_t* at, const TriggerEffect& t) {
        if (t.kind == TriggerEffect::kZones || t.kind == TriggerEffect::kZoneVibration) {
            std::uint32_t on = 0, strengths = 0;
            for (int i = 0; i < 10; ++i)
                if (t.zones[i]) {
                    on |= 1u << i;
                    strengths |= std::uint32_t((std::min<int>(t.zones[i], 8) - 1) & 7) << (3 * i);
                }
            if (!on || (t.kind == TriggerEffect::kZoneVibration && !t.frequency)) {
                at[0] = 0x05;
                return;
            }
            at[0] = t.kind == TriggerEffect::kZones ? 0x21 : 0x26;
            at[1] = std::uint8_t(on), at[2] = std::uint8_t(on >> 8);
            for (int i = 0; i < 4; ++i) at[3 + i] = std::uint8_t(strengths >> (8 * i));
            if (t.kind == TriggerEffect::kZoneVibration) at[9] = t.frequency;
            return;
        }
        switch (t.kind) {
            case TriggerEffect::kResistance:
                at[0] = 0x01, at[1] = t.start, at[2] = t.force;
                break;
            case TriggerEffect::kVibration:
                at[0] = 0x06, at[1] = t.frequency, at[2] = t.force, at[3] = t.start;
                break;
            default:
                at[0] = 0x05;
        }
    };
    put(packet + 10, right);
    put(packet + 21, left);
    if (!SDL_SendGamepadEffect(pad, packet, sizeof packet))
        std::fprintf(stderr, "[window] trigger effects: %s\n", SDL_GetError());
}

void SdlWindowDisplay::ReleaseController() {
    std::lock_guard<std::mutex> hold(pads_lock_);
    for (Gamepad& g : pads_) {
        SendTriggers(g.pad, TriggerEffect{}, TriggerEffect{});
        SDL_RumbleGamepad(g.pad, 0, 0, 0);
    }
}

bool SdlWindowDisplay::Focused() const {
    return window_ && (SDL_GetWindowFlags(window_) & SDL_WINDOW_INPUT_FOCUS) != 0;
}

bool SdlWindowDisplay::KeyDown(int scancode) const {
    int n = 0;
    const bool* keys = SDL_GetKeyboardState(&n);
#ifdef __ANDROID__
    if (scancode == SDL_SCANCODE_ESCAPE && keys && SDL_SCANCODE_AC_BACK < n && keys[SDL_SCANCODE_AC_BACK]) return true;
#endif
    return keys && scancode >= 0 && scancode < n && keys[scancode];
}

bool SdlWindowDisplay::RefreshTiming(double* period_ns, std::uint64_t* blank_ns) const {
    if (!vblank_watch_) return false;
    std::lock_guard<std::mutex> hold(vblank_watch_->lock);
    const VblankFit& fit = vblank_watch_->fit;
    // (Blanks no longer coming - the thread lost its screen: not known.)
    if (!fit.Valid() || SteadyNs() > fit.Last() + 250'000'000) return false;
    *period_ns = fit.Period();
    *blank_ns = fit.Anchor();
    return true;
}

}  // namespace ps2gpu
