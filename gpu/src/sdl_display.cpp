/*
 * SdlDisplay: the window's picture through SDL's own 2D renderer - the last
 * resort, for a PC with neither a Vulkan driver nor (off Windows) Direct3D 11:
 * an old GPU on Linux, a desktop without its driver installed. SDL picks what
 * it has (OpenGL, OpenGL ES, or its software renderer into the window).
 *
 * There is no GPU to draw with (SharedGpu is null), so the GS is the software
 * one and each picture comes here as pixels (Present): uploaded into a
 * streaming texture and drawn scaled into its place (4:3, 16:9 or the
 * window's shape: SetAspect), the overlay 1:1 at the top left. The window,
 * input and refresh are SdlWindowDisplay's, as for the GPU displays.
 *
 * Present is called on the thread that pumps the window (PresentFrame), the
 * one SDL's renderer wants.
 */

#include "window_display.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace ps2gpu {

namespace {

class SdlDisplay final : public SdlWindowDisplay {
public:
    ~SdlDisplay() override {
        if (overlay_texture_) SDL_DestroyTexture(overlay_texture_);
        if (picture_) SDL_DestroyTexture(picture_);
        if (renderer_) SDL_DestroyRenderer(renderer_);
    }
    bool Init(const char* title, int width, int height) {
        if (!OpenWindow(title, width, height, 0)) return false;
        renderer_ = SDL_CreateRenderer(window_, nullptr);
        if (!renderer_) {
            error_ = std::string("SDL_CreateRenderer: ") + SDL_GetError();
            return false;
        }
        const char* name = SDL_GetRendererName(renderer_);
        describe_ = std::string("SDL's ") + (name ? name : "?") + " renderer (no GPU display)";
        return true;
    }
    bool Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) override {
        if (closed_) return false;
        if (!width || !height || Minimized()) return true;
        if (vsync_ != vsync_set_) {
            SDL_SetRenderVSync(renderer_, vsync_ ? 1 : SDL_RENDERER_VSYNC_DISABLED);
            vsync_set_ = vsync_;
        }
        if (!picture_ || picture_width_ != width || picture_height_ != height) {
            if (picture_) SDL_DestroyTexture(picture_);
            // The pictures' words are BGRA in memory, as the GPU displays
            // upload them (B8G8R8A8): ARGB8888 to SDL.
            picture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
                                         int(width), int(height));
            if (!picture_) return true;
            SDL_SetTextureBlendMode(picture_, SDL_BLENDMODE_NONE);
            SDL_SetTextureScaleMode(picture_, SDL_SCALEMODE_LINEAR);
            picture_width_ = width;
            picture_height_ = height;
        }
        SDL_UpdateTexture(picture_, nullptr, pixels, int(width * 4));
        int w = 0, h = 0;
        SDL_GetCurrentRenderOutputSize(renderer_, &w, &h);
        // 4:3 as a television shows it, 16:9, or the window's own shape,
        // centred in the window.
        int dw = w, dh = h;
        const float aspect = aspect_.load();
        if (aspect > 0.0f) {
            if (float(w) > float(h) * aspect) dw = int(float(h) * aspect + 0.5f);
            else dh = int(float(w) / aspect + 0.5f);
        }
        const int x0 = (w - dw) / 2, y0 = (h - dh) / 2;
        SetShownRect(x0, y0, std::max(dw, 1), std::max(dh, 1));
        SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
        SDL_RenderClear(renderer_);
        const SDL_FRect place{float(x0), float(y0), float(dw), float(dh)};
        SDL_RenderTexture(renderer_, picture_, nullptr, &place);
        DrawOverlay();
        SDL_RenderPresent(renderer_);
        CountPresent();
        return !closed_;
    }
    void SetVsync(bool on) override { vsync_ = on; }
    bool Vsync() const override { return vsync_; }
    double PresentsPerSecond() const override { return fps_.load(); }
    std::string Describe() const override { return describe_; }
    Gpu* SharedGpu() override { return nullptr; }

private:
    void DrawOverlay() {
        std::lock_guard<std::mutex> hold(overlay_lock_);
        if (overlay_.empty() || !overlay_width_ || !overlay_height_) return;
        if (overlay_uploaded_ != overlay_generation_) {
            if (!overlay_texture_ || overlay_texture_width_ != overlay_width_ ||
                overlay_texture_height_ != overlay_height_) {
                if (overlay_texture_) SDL_DestroyTexture(overlay_texture_);
                // The overlay's words are BGRA in memory, as the GPU displays
                // upload them (B8G8R8A8): ARGB8888 to SDL.
                overlay_texture_ = SDL_CreateTexture(renderer_, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC,
                                                     int(overlay_width_), int(overlay_height_));
                if (!overlay_texture_) return;
                SDL_SetTextureScaleMode(overlay_texture_, SDL_SCALEMODE_NEAREST);
                overlay_texture_width_ = overlay_width_;
                overlay_texture_height_ = overlay_height_;
            }
            SDL_UpdateTexture(overlay_texture_, nullptr, overlay_.data(), int(overlay_width_ * 4));
            overlay_uploaded_ = overlay_generation_;
        }
        const SDL_FRect place{16.0f, 16.0f, float(overlay_width_), float(overlay_height_)};
        SDL_RenderTexture(renderer_, overlay_texture_, nullptr, &place);
    }
    void CountPresent() {
        const auto now = std::chrono::steady_clock::now();
        ++fps_presents_;
        if (!fps_since_.time_since_epoch().count()) fps_since_ = now;
        const double seconds = std::chrono::duration<double>(now - fps_since_).count();
        if (seconds >= 0.5) {
            fps_.store(double(fps_presents_) / seconds);
            fps_presents_ = 0;
            fps_since_ = now;
        }
    }

    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* picture_ = nullptr;
    std::uint32_t picture_width_ = 0, picture_height_ = 0;
    SDL_Texture* overlay_texture_ = nullptr;
    std::uint32_t overlay_texture_width_ = 0, overlay_texture_height_ = 0;
    std::uint64_t overlay_uploaded_ = 0;
    std::atomic<bool> vsync_{false};
    bool vsync_set_ = true;             // (SDL's renderer starts without: set at the first present)
    std::atomic<double> fps_{0.0};
    unsigned fps_presents_ = 0;
    std::chrono::steady_clock::time_point fps_since_{};
};

}  // namespace

std::unique_ptr<Display> CreateSdlDisplay(const char* title, int width, int height, std::string* error) {
    auto display = std::make_unique<SdlDisplay>();
    if (!display->Init(title, width, height)) {
        if (error) *error = display->error();
        return nullptr;
    }
    return display;
}

}  // namespace ps2gpu
