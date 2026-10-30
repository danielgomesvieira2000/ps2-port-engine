/* SdlWindowDisplay: what every display backend shares - the SDL3 window, its
   events, the keyboard, controllers (rumble, a DualSense's triggers), the
   mouse and the game's pointer, fullscreen, the screen's refresh and its
   measured blanks, the title and the overlay's pixels. A backend (Vulkan,
   Direct3D 11) adds the picture: its device, swapchain and presents. */

#ifndef PS2GPU_WINDOW_DISPLAY_H
#define PS2GPU_WINDOW_DISPLAY_H

#include "ps2gpu/display.h"
#include "ps2gpu/vblank_fit.h"

#include <SDL3/SDL.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ps2gpu {

/* The screen's vertical blanks (RefreshTiming): a thread waits on each of
   the window's screen's (DXGI's WaitForVBlank) and puts its time into the
   fit. Shared with the thread, which the display leaves to end by itself (a
   wait on a screen gone to sleep may not come back soon). */
struct VblankWatch {
    std::mutex lock;
    VblankFit fit;
    std::atomic<bool> stop{false};
    std::atomic<float> nominal_hz{0.0f};        // the mode's refresh (Pump's)
};

std::uint64_t SteadyNs();

class SdlWindowDisplay : public Display {
public:
    ~SdlWindowDisplay() override;
    bool Pump() override;
    void SetFullscreen(bool on) override;
    bool Fullscreen() const override;
    bool TakeCapture() override {
        const bool asked = capture_;
        capture_ = false;
        return asked;
    }
    bool TakeTexturePackToggle() override {
        const bool asked = texture_pack_;
        texture_pack_ = false;
        return asked;
    }
    bool TakeFpsToggle() override {
        const bool asked = fps_toggle_;
        fps_toggle_ = false;
        return asked;
    }
    void SetAspect(float aspect) override { aspect_.store(aspect); }
    void SetOverlay(std::vector<std::uint32_t> pixels, std::uint32_t width, std::uint32_t height) override {
        std::lock_guard<std::mutex> hold(overlay_lock_);
        overlay_ = std::move(pixels);
        overlay_width_ = width;
        overlay_height_ = height;
        ++overlay_generation_;
    }
    void SetTitle(const std::string& title) override {
        std::lock_guard<std::mutex> hold(title_lock_);
        title_ = title;
        title_changed_ = true;
    }
    float WindowAspect() const override {
        int w = 0, h = 0;
        if (!window_ || !SDL_GetWindowSizeInPixels(window_, &w, &h) || w <= 0 || h <= 0) return 0.0f;
        return float(w) / float(h);
    }
    std::uint32_t PixelHeight() const override {
        int w = 0, h = 0;
        if (!window_ || !SDL_GetWindowSizeInPixels(window_, &w, &h)) return 0;
        return std::uint32_t(h > 0 ? h : 0);
    }
    float RefreshRate() const override { return refresh_hz_.load(std::memory_order_relaxed); }
    bool RefreshTiming(double* period_ns, std::uint64_t* blank_ns) const override;
    void SetKeepRefresh(bool keep) override {
        if (keep_refresh_ == keep) return;
        keep_refresh_ = keep;
        if (Fullscreen()) SetFullscreen(true);
    }
    bool Focused() const override;
    bool KeyDown(int scancode) const override;
    bool Minimized() const override {
        return window_ && (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0;
    }
    bool Pad(PadState* state) const override;
    bool PadOf(std::uint32_t id, PadState* state) const override;
    std::vector<ControllerInfo> Controllers() const override;
    void SetRumble(std::uint16_t large, std::uint16_t small) override {
        std::lock_guard<std::mutex> hold(pads_lock_);
        for (Gamepad& g : pads_) g.rumble = std::uint32_t(large) << 16 | small;
    }
    void SetRumbleOf(std::uint32_t id, std::uint16_t large, std::uint16_t small) override {
        std::lock_guard<std::mutex> hold(pads_lock_);
        for (Gamepad& g : pads_)
            if (g.id == id) g.rumble = std::uint32_t(large) << 16 | small;
    }
    void SetTriggerEffects(TriggerEffect left, TriggerEffect right) override {
        std::lock_guard<std::mutex> hold(triggers_lock_);
        triggers_[0] = left;
        triggers_[1] = right;
    }
    void SetTriggerTargets(std::vector<std::uint32_t> ids) override {
        std::lock_guard<std::mutex> hold(triggers_lock_);
        trigger_targets_ = std::move(ids);
    }
    void ReleaseController() override;
    bool TakeMouse(MouseState* state) override {
        std::lock_guard<std::mutex> hold(mouse_lock_);
        *state = mouse_;
        mouse_.pressed = 0;
        mouse_.wheel = 0.0f;
        mouse_.moved = false;
        return true;
    }
    void SetCursorArt(std::vector<std::uint32_t> rgba, std::uint32_t width, std::uint32_t height,
                      std::uint32_t hot_x, std::uint32_t hot_y) override {
        std::lock_guard<std::mutex> hold(mouse_lock_);
        cursor_art_ = std::move(rgba);
        cursor_width_ = width;
        cursor_height_ = height;
        cursor_hot_x_ = hot_x;
        cursor_hot_y_ = hot_y;
        cursor_size_ = 0;                   // made again at the window's size
    }
    void ShowGameCursor(bool on) override { cursor_wanted_.store(on); }
    const std::string& error() const { return error_; }

protected:
    /* SDL, its controllers and the window (`graphics`: SDL_WINDOW_VULKAN, or
       none for a backend that takes the window's handle), and the thread
       watching the screen's blanks. False, with error_ set, when there is
       none. */
    bool OpenWindow(const char* title, int width, int height, SDL_WindowFlags graphics);
    /* The picture's place in the window, in pixels, as a present put it (the
       mouse's position over the picture, the pointer's size). */
    void SetShownRect(std::int32_t x0, std::int32_t y0, std::int32_t w, std::int32_t h) {
        shown_x0_.store(x0);
        shown_y0_.store(y0);
        shown_w_.store(w);
        shown_h_.store(h);
    }

    SDL_Window* window_ = nullptr;
    std::string describe_, error_;
    // Set by Pump (the main thread), read where pictures are presented (the
    // GS thread, when it presents): the window closed, or its swapchain to be
    // made again (its size, fullscreen, its present mode).
    std::atomic<bool> closed_{false}, stale_{false};
    bool keep_refresh_ = false;                 // SetKeepRefresh's
    std::atomic<float> aspect_{4.0f / 3.0f};
    // SetOverlay's panel, copied by the backend's present.
    std::mutex overlay_lock_;
    std::vector<std::uint32_t> overlay_;
    std::uint32_t overlay_width_ = 0, overlay_height_ = 0;
    std::uint64_t overlay_generation_ = 1;      // a SetOverlay's
    std::shared_ptr<VblankWatch> vblank_watch_;     // RefreshTiming's (null: no such thread)
    std::atomic<std::int32_t> shown_x0_{0}, shown_y0_{0}, shown_w_{1}, shown_h_{1};

private:
    bool capture_ = false, texture_pack_ = false, fps_toggle_ = false;
    std::mutex title_lock_;
    std::string title_;
    bool title_changed_ = false;
    std::atomic<float> refresh_hz_{0.0f};       // Pump's
    unsigned refresh_looks_ = 0;
    // Every controller plugged in, in the order they were: opened and closed
    // by the window's thread (Pump), read from the EE's - under pads_lock_.
    struct Gamepad {
        SDL_Gamepad* pad = nullptr;
        std::uint32_t id = 0;
        std::uint32_t rumble = 0;           // as wanted: large << 16 | small
        std::uint32_t rumble_sent = 0;
        std::uint64_t rumble_renewed = 0;   // when SDL was last told (ms): its rumble lasts a while
        bool triggers_unsent = true;        // (plugged in since they were sent)
        TriggerEffect triggers_sent[2];
    };
    mutable std::mutex pads_lock_;
    std::vector<Gamepad> pads_;
    std::vector<std::uint32_t> trigger_targets_;        // SetTriggerTargets: none, every controller
    const Gamepad* FindPad(std::uint32_t id) const {
        for (const Gamepad& g : pads_)
            if (g.id == id) return &g;
        return nullptr;
    }
    std::mutex triggers_lock_;
    TriggerEffect triggers_[2];                         // left, right: wanted
    void SendTriggers(SDL_Gamepad* pad, const TriggerEffect& left, const TriggerEffect& right);
    // The mouse (TakeMouse), from the window's events.
    std::mutex mouse_lock_;
    MouseState mouse_;
    float mouse_px_ = 0.0f, mouse_py_ = 0.0f;   // the pointer, in the window's pixels
    void MouseAt(float x, float y);
    // The game's pointer: its art, and the cursor made from it at a size.
    std::vector<std::uint32_t> cursor_art_;
    std::uint32_t cursor_width_ = 0, cursor_height_ = 0, cursor_hot_x_ = 0, cursor_hot_y_ = 0;
    std::uint32_t cursor_size_ = 0;
    SDL_Cursor* cursor_ = nullptr;
    std::atomic<bool> cursor_wanted_{false};
    int cursor_showing_ = -1;               // what the window shows: -1 not yet set, 0 the system's, 1 the game's
    bool fullscreen_cursor_stale_ = false;
    void UpdateCursor();
};

}  // namespace ps2gpu

#endif  /* PS2GPU_WINDOW_DISPLAY_H */
