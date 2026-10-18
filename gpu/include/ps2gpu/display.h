/*
 * The port's display: a window showing the console's picture, and the keyboard,
 * game controller and window events that come with it.
 *
 * Display is what the host talks to; VulkanDisplay (SDL3 + Vulkan) is the one
 * implementation so far. Stage one presents a finished picture - the software
 * GS's - scaled to the window; a GPU GS renderer will draw into the same
 * swapchain later.
 */

#ifndef PS2GPU_DISPLAY_H
#define PS2GPU_DISPLAY_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ps2gpu {

class Display {
public:
    virtual ~Display() = default;

    /* Shows a picture: 32-bit pixels, BGRA in memory (B first), rows top-down,
       scaled to the window. False when the window is gone. */
    virtual bool Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) = 0;

    /* Shows a picture the GPU already has: the top-left width x height of a
       VkImage (on the display's own device, as its 64-bit handle), in
       `layout` (a VkImageLayout) and left in it, copied to the window on the
       GPU - no read-back, no upload. Whatever drew into it must have been
       submitted to the queue before. False when the window is gone, or a
       display cannot. */
    virtual bool PresentImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) {
        (void)image, (void)layout, (void)width, (void)height;
        return false;
    }

    /* PresentImage, the picture shown at `due` (a steady_clock time, in
       nanoseconds since its epoch) rather than now: the image is copied at
       once (it may be drawn over next), and a thread of the display's shows
       it then - so pictures reach the screen evenly however unevenly they
       were finished. The newest picture wins: one not yet shown when the
       next comes is never shown. */
    virtual bool PresentImageAt(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                                std::uint64_t due_ns) {
        (void)due_ns;
        return PresentImage(image, layout, width, height);
    }

    /* Handles the window's events; false once it has been closed. Alt+Enter
       toggles fullscreen. */
    virtual bool Pump() = 0;

    /* How evenly pictures went to the screen since the last call (--frame-times):
       presents, and the time between them - mean, standard deviation and
       longest, in milliseconds. */
    struct Pacing { std::uint64_t presents = 0; double mean_ms = 0, jitter_ms = 0, longest_ms = 0; };
    virtual Pacing TakePacing() { return {}; }
    /* Pictures shown a second, over the last half second or so: the FPS
       counter's (F1), apart from TakePacing so it takes nothing from it. */
    virtual double PresentsPerSecond() const { return 0.0; }

    /* Vertical sync: each picture waits for the screen's refresh (no
       tearing, FIFO), or goes at once (IMMEDIATE, or MAILBOX where that is
       all there is). */
    virtual void SetVsync(bool on) { (void)on; }
    virtual bool Vsync() const { return false; }

    /* The window's height in the screen's pixels (0 when not known). */
    virtual std::uint32_t PixelHeight() const { return 0; }

    /* Fullscreen (borderless, the desktop's resolution) or a window. */
    virtual void SetFullscreen(bool on) { (void)on; }
    virtual bool Fullscreen() const { return false; }
    /* The picture's shape on the screen, width over height: 4:3 as a
       television showed it (the default), 16:9, or 0 - the window's own
       shape, filled. Letterboxed or pillarboxed in the window otherwise. */
    virtual void SetAspect(float aspect) { (void)aspect; }
    /* Anti-aliasing: the picture shown as drawn (the default); its edges
       smoothed (FXAA, or SMAA's three passes) before it is scaled to the
       window; or, drawn larger than shown (supersampling), averaged down to
       the shown size rather than scaled. */
    enum class AntiAliasing { kOff, kFxaa, kDownsample, kSmaa };
    virtual void SetAntiAliasing(AntiAliasing mode) { (void)mode; }
    /* The window's shape in the screen's pixels, width over height (0 when
       not known). */
    virtual float WindowAspect() const { return 0.0f; }
    /* A panel over the picture, at the window's top left: BGRA pixels
       (opaque), rows top-down; none when empty. */
    virtual void SetOverlay(std::vector<std::uint32_t> pixels, std::uint32_t width, std::uint32_t height) {
        (void)pixels, (void)width, (void)height;
    }
    /* The window's title (taken up by the window's own thread, in Pump). */
    virtual void SetTitle(const std::string& title) { (void)title; }
    /* F12, or L3 with R3 on a controller, since the last call: a capture asked for. */
    virtual bool TakeCapture() { return false; }
    /* Tab since the last call: the texture pack turned on or off. */
    virtual bool TakeTexturePackToggle() { return false; }
    /* F1 since the last call: the FPS counter turned on or off. */
    virtual bool TakeFpsToggle() { return false; }

    /* Whether the window has the keyboard, and whether a key is down - by SDL
       scancode (SDL_SCANCODE_*), which names keys by position. */
    virtual bool Focused() const = 0;
    virtual bool KeyDown(int scancode) const = 0;

    /* The first game controller, as a DualShock 2 reads: false when none is
       plugged in. */
    struct PadState {
        std::uint16_t buttons = 0;      // bit n: DualShock 2 input n held (libpad's order:
                                        // Select L3 R3 Start Up Right Down Left L2 R2 L1 R1
                                        // Triangle Circle Cross Square)
        std::uint8_t sticks[4] = {0x80, 0x80, 0x80, 0x80};   // right X, Y, left X, Y; 0x80 centred
        std::uint8_t l2 = 0, r2 = 0;    // the triggers' travel, 0-255
        std::uint8_t family = 0;        // whose it is (its buttons' names): ControllerFamily
    };
    enum ControllerFamily : std::uint8_t { kOtherController, kPlayStationController, kXboxController };
    virtual bool Pad(PadState* state) const { (void)state; return false; }
    /* Every controller plugged in, in the order they were (SDL's id, the
       name it gives, its family) - and one's own state by its id: false when
       it is gone. */
    struct ControllerInfo {
        std::uint32_t id = 0;
        std::string name;
        std::uint8_t family = 0;
    };
    virtual std::vector<ControllerInfo> Controllers() const { return {}; }
    virtual bool PadOf(std::uint32_t id, PadState* state) const { (void)id, (void)state; return false; }
    /* The controller's motors, 0-65535 each (the low-frequency, large one;
       the high-frequency, small one), until set again - 0, 0 stops them.
       Taken up by the window's own thread (Pump). SetRumble sets every
       controller's, SetRumbleOf one's. */
    virtual void SetRumble(std::uint16_t large, std::uint16_t small) { (void)large, (void)small; }
    virtual void SetRumbleOf(std::uint32_t id, std::uint16_t large, std::uint16_t small) {
        (void)id, (void)large, (void)small;
    }
    /* The controllers SetTriggerEffects is for: none named, every one. */
    virtual void SetTriggerTargets(std::vector<std::uint32_t> ids) { (void)ids; }
    /* A controller's adaptive triggers (a DualSense's; others have none):
       off, a resistance from `start` (0-255 of the travel) of `force`
       (0-255), or a vibration from `start` of `force` at `frequency` (Hz),
       until set again. Taken up by the window's own thread (Pump), and
       again when a controller is plugged in. */
    struct TriggerEffect {
        enum Kind : std::uint8_t { kOff, kResistance, kVibration } kind = kOff;
        std::uint8_t start = 0, force = 0, frequency = 0;
        bool operator==(const TriggerEffect& o) const {
            return kind == o.kind && start == o.start && force == o.force && frequency == o.frequency;
        }
    };
    virtual void SetTriggerEffects(TriggerEffect left, TriggerEffect right) { (void)left, (void)right; }
    /* The controller left as it was found, at once (the game is ending):
       its motors stopped, its triggers' effects off. */
    virtual void ReleaseController() {}

    /* The mouse over the window, in the picture's terms: x and y from 0 to 1
       across the game's picture as it is shown (inside its letterbox; outside
       it they go past 0 or 1). `held` the buttons down (bit 0 left, 1 middle,
       2 right), `pressed` those that went down since the last call, `wheel`
       the notches turned since (up positive), `moved` whether it moved. */
    struct MouseState {
        bool over = false;              // the pointer is in the window
        float x = 0.0f, y = 0.0f;
        std::uint32_t held = 0, pressed = 0;
        float wheel = 0.0f;
        bool moved = false;
    };
    virtual bool TakeMouse(MouseState* state) { (void)state; return false; }
    /* The game's own mouse pointer: RGBA pixels (rows top-down), the point
       that clicks at (hot_x, hot_y); scaled with the window. Shown while
       ShowGameCursor(true) - otherwise the system's (hidden in fullscreen). */
    virtual void SetCursorArt(std::vector<std::uint32_t> rgba, std::uint32_t width, std::uint32_t height,
                              std::uint32_t hot_x, std::uint32_t hot_y) {
        (void)rgba, (void)width, (void)height, (void)hot_x, (void)hot_y;
    }
    virtual void ShowGameCursor(bool on) { (void)on; }

    /* Told each picture PresentImageAt was given: shown (when it was due, when
       it was), or dropped unshown (shown_ns 0) - on the presenting thread. */
    using PresentObserver = std::function<void(std::uint64_t due_ns, std::uint64_t shown_ns, bool dropped)>;
    virtual void SetPresentObserver(PresentObserver observer) { (void)observer; }

    /* The GPU and driver, for the report. */
    virtual std::string Describe() const = 0;

    /* The window's device as a Gpu, for a GsRenderer to draw with: one
       Vulkan instance and device for both (owned by the display). */
    virtual class Gpu* SharedGpu() = 0;
};

/* A window with a Vulkan swapchain; null, with `error` saying why, when there is
   no Vulkan device or window to be had. */
std::unique_ptr<Display> CreateVulkanDisplay(const char* title, int width, int height, std::string* error);

}  // namespace ps2gpu

#endif  /* PS2GPU_DISPLAY_H */
