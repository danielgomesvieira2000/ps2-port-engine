/*
 * D3D11Display: the window's picture through Direct3D 11 - for a PC without a
 * Vulkan 1.2 driver (Intel HD 4000/4600 and Iris on Windows have none: issue
 * #24), or chosen (renderer=d3d11).
 *
 * The window, input and refresh are SdlWindowDisplay's. A flip-model
 * swapchain (FLIP_DISCARD, three buffers, BGRA8 as the pictures come) is
 * drawn into each present: cleared, the picture drawn scaled into its place
 * (4:3, 16:9 or the window's shape: SetAspect) with a linear sampler, the
 * overlay 1:1 at the top left. Without vertical sync it presents at once -
 * with tearing where DXGI allows it (a VRR screen's own pacing, as Vulkan's
 * IMMEDIATE), else at the next refresh without waiting for it.
 *
 * Shaders are compiled from the HLSL below when the display is made
 * (d3dcompiler_47.dll, part of Windows); ANTI-ALIASING's passes - FXAA, SMAA's
 * three, supersampling's average - are the Vulkan display's compute shaders,
 * their HLSL built with the GS renderer's (gpu/CMakeLists.txt).
 */

#ifdef _WIN32

#include "window_display.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_5.h>
#undef small
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include <algorithm>
#include <atomic>
#include <chrono>
#include <initializer_list>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "d3d11_gpu.h"
#ifdef PS2GPU_D3D11
#include "AreaTex.h"             // gpu/third_party/smaa
#include "SearchTex.h"
#include "d3d_downsample_dxbc.h"
#include "d3d_fxaa_dxbc.h"
#include "d3d_smaa_blend_dxbc.h"
#include "d3d_smaa_edges_dxbc.h"
#include "d3d_smaa_weights_dxbc.h"
#endif

namespace ps2gpu {

namespace {

template <typename T>
void Release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// A triangle over the viewport, sampling its texture across it.
const char kBlitHlsl[] = R"(
Texture2D picture : register(t0);
SamplerState sampled : register(s0);
struct Out { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Out vs_main(uint id : SV_VertexID) {
    Out o;
    o.uv = float2((id << 1) & 2, id & 2);
    o.position = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
float4 ps_main(Out i) : SV_Target { return picture.Sample(sampled, i.uv); }
)";

class D3D11Display final : public SdlWindowDisplay {
public:
    ~D3D11Display() override;
    bool Init(const char* title, int width, int height);
    bool Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) override;
    bool PresentImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) override;
    bool PresentImageAt(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                        std::uint64_t due_ns) override;
    int HoldImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) override;
    void ShowHeld(int slot, std::uint64_t due_ns) override;
    void SetPresentObserver(PresentObserver observer) override {
        std::lock_guard<std::mutex> hold(paced_lock_);
        observer_ = std::move(observer);
    }
    void SetVsync(bool on) override { vsync_ = on; }
    bool Vsync() const override { return vsync_; }
    void SetVrr(bool on) override { vrr_ = on; }
    bool Vrr() const override { return vrr_; }
    Pacing TakePacing() override;
    double PresentsPerSecond() const override { return fps_.load(); }
    void SetAntiAliasing(AntiAliasing mode) override { anti_aliasing_.store(int(mode)); }
    std::string Describe() const override { return describe_; }
    Gpu* SharedGpu() override { return gpu_.device ? &gpu_ : nullptr; }

private:
    bool Fail(HRESULT hr, const char* what);
    bool EnsureSwapchain();
    bool EnsurePicture(std::uint32_t width, std::uint32_t height);
    /* The picture (an SRV, width x height of it used) and the overlay into the
       back buffer, and presented. */
    bool Show(ID3D11ShaderResourceView* picture, std::uint32_t width, std::uint32_t height,
              std::uint32_t texture_width, std::uint32_t texture_height);
    void DrawOverlay(std::uint32_t target_width, std::uint32_t target_height);
    void CountPresent();
    /* ANTI-ALIASING: the picture's used part copied into post_in_, then
       FXAA or SMAA into post_out_ (its size), or supersampling's average
       down to the shown size; the picture shown is then post_out_. False
       (shown as drawn) where the passes cannot be made. */
    bool Smooth(ID3D11ShaderResourceView** picture, std::uint32_t* width, std::uint32_t* height,
                std::uint32_t* texture_width, std::uint32_t* texture_height, std::uint32_t shown_width,
                std::uint32_t shown_height);
    struct PostImage {
        ID3D11Texture2D* texture = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        ID3D11UnorderedAccessView* uav = nullptr;
        std::uint32_t width = 0, height = 0;
    };
    bool EnsurePostImage(PostImage& image, std::uint32_t width, std::uint32_t height, bool uav);
    void ReleasePostImage(PostImage& image);
    void Dispatch(ID3D11ComputeShader* shader, std::initializer_list<ID3D11ShaderResourceView*> srvs,
                  ID3D11UnorderedAccessView* uav, UINT uav_slot, const void* params, std::uint32_t width,
                  std::uint32_t height);
    std::atomic<int> anti_aliasing_{int(AntiAliasing::kOff)};
    bool post_made_ = false, post_failed_ = false;
    ID3D11ComputeShader* fxaa_ = nullptr;
    ID3D11ComputeShader* downsample_ = nullptr;
    ID3D11ComputeShader* smaa_[3] = {};
    ID3D11Buffer* post_params_ = nullptr;
    PostImage post_in_, post_out_, smaa_edges_, smaa_weights_;
    ID3D11Texture2D* smaa_tables_[2] = {};
    ID3D11ShaderResourceView* smaa_table_views_[2] = {};

    D3D11Gpu gpu_;                          // the device (SharedGpu: what a D3D11 GS renderer draws with)
    HWND hwnd_ = nullptr;
    IDXGISwapChain1* swapchain_ = nullptr;
    ID3D11RenderTargetView* back_ = nullptr;
    HANDLE waitable_ = nullptr;
    std::uint32_t back_width_ = 0, back_height_ = 0;
    bool tearing_ = false;                  // DXGI_FEATURE_PRESENT_ALLOW_TEARING
    std::atomic<bool> vsync_{false}, vrr_{false};
    ID3D11VertexShader* vs_ = nullptr;
    ID3D11PixelShader* ps_ = nullptr;
    ID3D11SamplerState* linear_ = nullptr;
    ID3D11SamplerState* point_ = nullptr;
    ID3D11Texture2D* picture_ = nullptr;
    ID3D11ShaderResourceView* picture_view_ = nullptr;
    std::uint32_t picture_width_ = 0, picture_height_ = 0;
    ID3D11Texture2D* overlay_texture_ = nullptr;
    ID3D11ShaderResourceView* overlay_view_ = nullptr;
    std::uint32_t overlay_texture_width_ = 0, overlay_texture_height_ = 0;
    std::uint64_t overlay_uploaded_ = 0;
    std::mutex present_lock_;
    /* PresentImageAt's pictures: copies of the renderer's images (it may draw
       over them next), shown by a thread of the display's at their time - the
       newest winning: one still waiting when the next comes is dropped. One
       held (HoldImage) until ShowHeld gives it its time. */
    static constexpr int kPaced = 4;
    struct Paced {
        ID3D11Texture2D* texture = nullptr;
        ID3D11ShaderResourceView* view = nullptr;
        std::uint32_t width = 0, height = 0, texture_width = 0, texture_height = 0;
        std::uint64_t due_ns = 0;
    };
    Paced paced_[kPaced];
    std::mutex paced_lock_;                 // ready_, held_, taken_, stop_, observer_
    std::condition_variable paced_wake_;
    int ready_ = -1, held_ = -1, taken_ = -1;
    bool stop_ = false;
    std::thread present_thread_;
    PresentObserver observer_;
    /* The image copied into a free slot (-1: none): the caller holds present_lock_. */
    int CopyPaced(std::uint64_t image, std::uint32_t width, std::uint32_t height);
    void Ready(int slot, std::uint64_t due_ns);
    void PresentLoop();
    // TakePacing's and PresentsPerSecond's.
    std::mutex pacing_lock_;
    std::chrono::steady_clock::time_point last_present_{};
    std::uint64_t presents_ = 0;
    double interval_sum_ = 0, interval_squares_ = 0, interval_longest_ = 0;
    std::atomic<double> fps_{0.0};
    std::chrono::steady_clock::time_point fps_since_{};
    std::uint64_t fps_presents_ = 0;
};

bool D3D11Display::Fail(HRESULT hr, const char* what) {
    char text[160];
    std::snprintf(text, sizeof text, "%s failed (HRESULT 0x%08lX)", what, static_cast<unsigned long>(hr));
    error_ = text;
    std::fprintf(stderr, "[display] %s\n", text);
    return false;
}

bool D3D11Display::Init(const char* title, int width, int height) {
    std::string why;
    if (!gpu_.Init(&why)) {
        error_ = why;
        return false;
    }
    if (!OpenWindow(title, width, height, 0)) return false;
    hwnd_ = static_cast<HWND>(
        SDL_GetPointerProperty(SDL_GetWindowProperties(window_), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    if (!hwnd_) {
        error_ = "no window handle";
        return false;
    }
    describe_ = gpu_.Describe();
    IDXGIFactory5* factory5 = nullptr;
    if (SUCCEEDED(gpu_.factory->QueryInterface(__uuidof(IDXGIFactory5), reinterpret_cast<void**>(&factory5)))) {
        BOOL allow = FALSE;
        tearing_ = SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof allow)) &&
                   allow;
        factory5->Release();
    }
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    HRESULT hr = D3DCompile(kBlitHlsl, sizeof kBlitHlsl - 1, "blit", nullptr, nullptr, "vs_main", "vs_4_0", 0, 0,
                            &code, &errors);
    if (FAILED(hr)) {
        if (errors) std::fprintf(stderr, "[display] %s\n", static_cast<const char*>(errors->GetBufferPointer()));
        Release(errors);
        return Fail(hr, "D3DCompile (the blit's vertex shader)");
    }
    hr = gpu_.device->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &vs_);
    Release(code);
    if (FAILED(hr)) return Fail(hr, "CreateVertexShader");
    hr = D3DCompile(kBlitHlsl, sizeof kBlitHlsl - 1, "blit", nullptr, nullptr, "ps_main", "ps_4_0", 0, 0, &code,
                    &errors);
    Release(errors);
    if (FAILED(hr)) return Fail(hr, "D3DCompile (the blit's pixel shader)");
    hr = gpu_.device->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &ps_);
    Release(code);
    if (FAILED(hr)) return Fail(hr, "CreatePixelShader");
    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(hr = gpu_.device->CreateSamplerState(&sd, &linear_))) return Fail(hr, "CreateSamplerState");
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    if (FAILED(hr = gpu_.device->CreateSamplerState(&sd, &point_))) return Fail(hr, "CreateSamplerState");
    std::fprintf(stderr, "[display] Direct3D 11 on %s%s\n", describe_.c_str(),
                 tearing_ ? " (tearing allowed: VRR, unsynced presents)" : "");
    return EnsureSwapchain();
}

bool D3D11Display::EnsureSwapchain() {
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    w = std::max(w, 1);
    h = std::max(h, 1);
    if (swapchain_ && !stale_ && std::uint32_t(w) == back_width_ && std::uint32_t(h) == back_height_) return true;
    stale_ = false;
    Release(back_);
    const UINT flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                       (tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u);
    HRESULT hr;
    if (!swapchain_) {
        DXGI_SWAP_CHAIN_DESC1 desc{};
        desc.Width = UINT(w);
        desc.Height = UINT(h);
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 3;
        desc.Scaling = DXGI_SCALING_STRETCH;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        desc.Flags = flags;
        hr = gpu_.factory->CreateSwapChainForHwnd(gpu_.device, hwnd_, &desc, nullptr, nullptr, &swapchain_);
        if (FAILED(hr)) return Fail(hr, "CreateSwapChainForHwnd");
        gpu_.factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);     // (SDL's Alt+Enter)
        IDXGISwapChain2* two = nullptr;
        if (SUCCEEDED(swapchain_->QueryInterface(__uuidof(IDXGISwapChain2), reinterpret_cast<void**>(&two)))) {
            two->SetMaximumFrameLatency(2);
            waitable_ = two->GetFrameLatencyWaitableObject();
            two->Release();
        }
    } else {
        gpu_.context->OMSetRenderTargets(0, nullptr, nullptr);
        gpu_.context->Flush();
        hr = swapchain_->ResizeBuffers(0, UINT(w), UINT(h), DXGI_FORMAT_UNKNOWN, flags);
        if (FAILED(hr)) return Fail(hr, "ResizeBuffers");
    }
    ID3D11Texture2D* buffer = nullptr;
    if (FAILED(hr = swapchain_->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&buffer))))
        return Fail(hr, "GetBuffer");
    hr = gpu_.device->CreateRenderTargetView(buffer, nullptr, &back_);
    buffer->Release();
    if (FAILED(hr)) return Fail(hr, "CreateRenderTargetView");
    back_width_ = std::uint32_t(w);
    back_height_ = std::uint32_t(h);
    std::fprintf(stderr, "[display] swapchain %ux%u (Direct3D 11, flip model)\n", back_width_, back_height_);
    return true;
}

bool D3D11Display::EnsurePicture(std::uint32_t width, std::uint32_t height) {
    if (picture_ && picture_width_ == width && picture_height_ == height) return true;
    Release(picture_view_);
    Release(picture_);
    D3D11_TEXTURE2D_DESC td{};
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    HRESULT hr = gpu_.device->CreateTexture2D(&td, nullptr, &picture_);
    if (FAILED(hr)) return Fail(hr, "CreateTexture2D (the picture)");
    if (FAILED(hr = gpu_.device->CreateShaderResourceView(picture_, nullptr, &picture_view_)))
        return Fail(hr, "CreateShaderResourceView (the picture)");
    picture_width_ = width;
    picture_height_ = height;
    return true;
}

bool D3D11Display::Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) {
    if (!width || !height) return !closed_;
    std::lock_guard<std::mutex> hold(present_lock_);
    std::lock_guard<std::mutex> context(gpu_.context_lock);
    if (!EnsurePicture(width, height)) return false;
    gpu_.context->UpdateSubresource(picture_, 0, nullptr, pixels, width * 4, 0);
    return Show(picture_view_, width, height, width, height);
}

bool D3D11Display::PresentImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) {
    (void)layout;
    if (!width || !height || !image) return !closed_;
    // The GS renderer's target: an image of the Vulkan layer's, on this device.
    std::uint32_t texture_width = 0, texture_height = 0;
    ID3D11ShaderResourceView* view = vk11::ImageSrv(image, &texture_width, &texture_height);
    if (!view) return !closed_;
    std::lock_guard<std::mutex> hold(present_lock_);
    std::lock_guard<std::mutex> context(gpu_.context_lock);
    return Show(view, width, height, texture_width, texture_height);
}

bool D3D11Display::Show(ID3D11ShaderResourceView* picture, std::uint32_t width, std::uint32_t height,
                        std::uint32_t texture_width, std::uint32_t texture_height) {
    if (closed_) return false;
    if (Minimized()) return true;
    if (!EnsureSwapchain()) return false;
    if (waitable_) WaitForSingleObjectEx(waitable_, 100, TRUE);
    ID3D11DeviceContext* c = gpu_.context;
    const float black[4] = {0, 0, 0, 1};
    c->ClearRenderTargetView(back_, black);
    // 4:3 as a television shows it, 16:9, or the window's own shape (SetAspect),
    // centred in the window.
    const std::int32_t w = std::int32_t(back_width_), h = std::int32_t(back_height_);
    std::int32_t dw = w, dh = h;
    const float aspect = aspect_.load();
    if (aspect > 0.0f) {
        if (float(w) > float(h) * aspect) dw = std::int32_t(float(h) * aspect + 0.5f);
        else dh = std::int32_t(float(w) / aspect + 0.5f);
    }
    const std::int32_t x0 = (w - dw) / 2, y0 = (h - dh) / 2;
    SetShownRect(x0, y0, std::max(dw, 1), std::max(dh, 1));
    Smooth(&picture, &width, &height, &texture_width, &texture_height, std::uint32_t(std::max(dw, 1)),
           std::uint32_t(std::max(dh, 1)));
    // The picture's used part stretched over the shown rectangle: the
    // viewport is as much larger as the texture is past the part.
    const float sx = float(dw) / float(width), sy = float(dh) / float(height);
    const D3D11_VIEWPORT vp{float(x0), float(y0), float(texture_width) * sx, float(texture_height) * sy, 0.0f, 1.0f};
    const D3D11_RECT scissor{x0, y0, x0 + dw, y0 + dh};
    c->OMSetRenderTargets(1, &back_, nullptr);
    c->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFFu);
    c->OMSetDepthStencilState(nullptr, 0);
    c->RSSetState(gpu_.scissored);
    c->RSSetViewports(1, &vp);
    c->RSSetScissorRects(1, &scissor);
    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(vs_, nullptr, 0);
    c->PSSetShader(ps_, nullptr, 0);
    c->PSSetShaderResources(0, 1, &picture);
    c->PSSetSamplers(0, 1, &linear_);
    c->Draw(3, 0);
    DrawOverlay(back_width_, back_height_);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    const bool sync = vsync_.load();
    const HRESULT hr = swapchain_->Present(sync ? 1 : 0, !sync && tearing_ && !Fullscreen() ? DXGI_PRESENT_ALLOW_TEARING
                                                                                              : 0);
    if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) return Fail(hr, "Present (the device was lost)");
    CountPresent();
    return !closed_;
}

void D3D11Display::DrawOverlay(std::uint32_t target_width, std::uint32_t target_height) {
    std::lock_guard<std::mutex> hold(overlay_lock_);
    if (overlay_.empty() || !overlay_width_ || !overlay_height_) return;
    if (overlay_uploaded_ != overlay_generation_) {
        if (!overlay_texture_ || overlay_texture_width_ != overlay_width_ || overlay_texture_height_ != overlay_height_) {
            Release(overlay_view_);
            Release(overlay_texture_);
            D3D11_TEXTURE2D_DESC td{};
            td.Width = overlay_width_;
            td.Height = overlay_height_;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(gpu_.device->CreateTexture2D(&td, nullptr, &overlay_texture_)) ||
                FAILED(gpu_.device->CreateShaderResourceView(overlay_texture_, nullptr, &overlay_view_))) {
                Release(overlay_texture_);
                return;
            }
            overlay_texture_width_ = overlay_width_;
            overlay_texture_height_ = overlay_height_;
        }
        gpu_.context->UpdateSubresource(overlay_texture_, 0, nullptr, overlay_.data(), overlay_width_ * 4, 0);
        overlay_uploaded_ = overlay_generation_;
    }
    std::uint32_t x = 16, y = 16;
    if (x + 16 >= target_width) x = 16;
    if (y + 16 >= target_height) y = 16;
    const D3D11_VIEWPORT vp{float(x), float(y), float(overlay_width_), float(overlay_height_), 0.0f, 1.0f};
    const D3D11_RECT scissor{LONG(x), LONG(y), LONG(std::min(target_width, x + overlay_width_)),
                             LONG(std::min(target_height, y + overlay_height_))};
    ID3D11DeviceContext* c = gpu_.context;
    c->RSSetViewports(1, &vp);
    c->RSSetScissorRects(1, &scissor);
    c->PSSetShaderResources(0, 1, &overlay_view_);
    c->PSSetSamplers(0, 1, &point_);
    c->Draw(3, 0);
}

bool D3D11Display::EnsurePostImage(PostImage& image, std::uint32_t width, std::uint32_t height, bool uav) {
    if (image.texture && image.width == width && image.height == height) return true;
    ReleasePostImage(image);
    D3D11_TEXTURE2D_DESC td{};
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0u);
    if (FAILED(gpu_.device->CreateTexture2D(&td, nullptr, &image.texture)) ||
        FAILED(gpu_.device->CreateShaderResourceView(image.texture, nullptr, &image.srv)) ||
        (uav && FAILED(gpu_.device->CreateUnorderedAccessView(image.texture, nullptr, &image.uav)))) {
        ReleasePostImage(image);
        return false;
    }
    image.width = width;
    image.height = height;
    return true;
}

void D3D11Display::ReleasePostImage(PostImage& image) {
    Release(image.uav);
    Release(image.srv);
    Release(image.texture);
    image.width = image.height = 0;
}

void D3D11Display::Dispatch(ID3D11ComputeShader* shader, std::initializer_list<ID3D11ShaderResourceView*> srvs,
                            ID3D11UnorderedAccessView* uav, UINT uav_slot, const void* params, std::uint32_t width,
                            std::uint32_t height) {
    ID3D11DeviceContext* c = gpu_.context;
    c->UpdateSubresource(post_params_, 0, nullptr, params, 0, 0);
    ID3D11SamplerState* samplers[3] = {linear_, linear_, point_};      // (SMAA's search table: point-sampled)
    c->CSSetShader(shader, nullptr, 0);
    c->CSSetConstantBuffers(0, 1, &post_params_);
    c->CSSetSamplers(0, 3, samplers);
    ID3D11ShaderResourceView* views[3] = {};
    UINT n = 0;
    for (ID3D11ShaderResourceView* v : srvs) views[n++] = v;
    c->CSSetShaderResources(0, n, views);
    c->CSSetUnorderedAccessViews(uav_slot, 1, &uav, nullptr);
    c->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    ID3D11ShaderResourceView* none[3] = {};
    ID3D11UnorderedAccessView* no_uav = nullptr;
    c->CSSetShaderResources(0, 3, none);
    c->CSSetUnorderedAccessViews(uav_slot, 1, &no_uav, nullptr);
}

bool D3D11Display::Smooth(ID3D11ShaderResourceView** picture, std::uint32_t* width, std::uint32_t* height,
                          std::uint32_t* texture_width, std::uint32_t* texture_height, std::uint32_t shown_width,
                          std::uint32_t shown_height) {
#ifdef PS2GPU_D3D11
    const AntiAliasing mode = AntiAliasing(anti_aliasing_.load());
    const bool larger = *width > shown_width || *height > shown_height;
    if (mode == AntiAliasing::kOff || (mode == AntiAliasing::kDownsample && !larger) || post_failed_) return false;
    ID3D11Device* dev = gpu_.device;
    if (!post_made_) {
        post_made_ = true;
        auto cs = [&](const void* code, std::size_t bytes, ID3D11ComputeShader** out) {
            return SUCCEEDED(dev->CreateComputeShader(code, bytes, nullptr, out));
        };
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = 16;
        bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        bool ok = cs(dxbc_d3d_fxaa, sizeof dxbc_d3d_fxaa, &fxaa_) &&
                  cs(dxbc_d3d_downsample, sizeof dxbc_d3d_downsample, &downsample_) &&
                  cs(dxbc_d3d_smaa_edges, sizeof dxbc_d3d_smaa_edges, &smaa_[0]) &&
                  cs(dxbc_d3d_smaa_weights, sizeof dxbc_d3d_smaa_weights, &smaa_[1]) &&
                  cs(dxbc_d3d_smaa_blend, sizeof dxbc_d3d_smaa_blend, &smaa_[2]) &&
                  SUCCEEDED(dev->CreateBuffer(&bd, nullptr, &post_params_));
        // SMAA's area (RG8) and search (R8) tables.
        const struct { DXGI_FORMAT format; UINT w, h, pitch; const void* bytes; } tables[2] = {
            {DXGI_FORMAT_R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, AREATEX_PITCH, areaTexBytes},
            {DXGI_FORMAT_R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, SEARCHTEX_PITCH, searchTexBytes}};
        for (int i = 0; i < 2 && ok; ++i) {
            D3D11_TEXTURE2D_DESC td{};
            td.Width = tables[i].w;
            td.Height = tables[i].h;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = tables[i].format;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_IMMUTABLE;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            const D3D11_SUBRESOURCE_DATA data{tables[i].bytes, tables[i].pitch, 0};
            ok = SUCCEEDED(dev->CreateTexture2D(&td, &data, &smaa_tables_[i])) &&
                 SUCCEEDED(dev->CreateShaderResourceView(smaa_tables_[i], nullptr, &smaa_table_views_[i]));
        }
        if (!ok) {
            post_failed_ = true;
            std::fprintf(stderr, "[display] anti-aliasing's passes could not be made: pictures shown as drawn\n");
            return false;
        }
    }
    const std::uint32_t w = *width, h = *height;
    // The picture's used part, alone, as the passes read it.
    if (!EnsurePostImage(post_in_, w, h, false)) return false;
    ID3D11Resource* source = nullptr;
    (*picture)->GetResource(&source);
    const D3D11_BOX box{0, 0, 0, w, h, 1};
    gpu_.context->CopySubresourceRegion(post_in_.texture, 0, 0, 0, 0, source, 0, &box);
    source->Release();
    if (mode == AntiAliasing::kDownsample) {
        if (!EnsurePostImage(post_out_, shown_width, shown_height, true)) return false;
        const struct { float texel[2]; std::int32_t size[2]; } params{{1.0f / float(w), 1.0f / float(h)},
                                                                       {std::int32_t(shown_width), std::int32_t(shown_height)}};
        Dispatch(downsample_, {post_in_.srv}, post_out_.uav, 1, &params, shown_width, shown_height);
        *width = *texture_width = shown_width;
        *height = *texture_height = shown_height;
    } else if (mode == AntiAliasing::kFxaa) {
        if (!EnsurePostImage(post_out_, w, h, true)) return false;
        const struct { float texel[2]; std::int32_t size[2]; } params{{1.0f / float(w), 1.0f / float(h)},
                                                                       {std::int32_t(w), std::int32_t(h)}};
        Dispatch(fxaa_, {post_in_.srv}, post_out_.uav, 1, &params, w, h);
        *texture_width = w;
        *texture_height = h;
    } else {
        if (!EnsurePostImage(post_out_, w, h, true) || !EnsurePostImage(smaa_edges_, w, h, true) ||
            !EnsurePostImage(smaa_weights_, w, h, true))
            return false;
        const float metrics[4] = {1.0f / float(w), 1.0f / float(h), float(w), float(h)};
        // (Pixels with no edge keep what was there: cleared first, as Vulkan's.)
        const float zero[4] = {};
        gpu_.context->ClearUnorderedAccessViewFloat(smaa_edges_.uav, zero);
        gpu_.context->ClearUnorderedAccessViewFloat(smaa_weights_.uav, zero);
        Dispatch(smaa_[0], {post_in_.srv}, smaa_edges_.uav, 3, metrics, w, h);
        Dispatch(smaa_[1], {smaa_edges_.srv, smaa_table_views_[0], smaa_table_views_[1]}, smaa_weights_.uav, 3, metrics,
                 w, h);
        Dispatch(smaa_[2], {post_in_.srv, smaa_weights_.srv}, post_out_.uav, 3, metrics, w, h);
        *texture_width = w;
        *texture_height = h;
    }
    *picture = post_out_.srv;
    return true;
#else
    (void)picture, (void)width, (void)height, (void)texture_width, (void)texture_height, (void)shown_width,
        (void)shown_height;
    return false;
#endif
}

int D3D11Display::CopyPaced(std::uint64_t image, std::uint32_t width, std::uint32_t height) {
    std::uint32_t texture_width = 0, texture_height = 0;
    ID3D11ShaderResourceView* view = vk11::ImageSrv(image, &texture_width, &texture_height);
    if (!view) return -1;
    int slot = -1;
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        for (int i = 0; i < kPaced && slot < 0; ++i)
            if (i != ready_ && i != held_ && i != taken_) slot = i;
    }
    if (slot < 0) return -1;
    Paced& p = paced_[slot];
    if (!p.texture || p.texture_width != width || p.texture_height != height) {
        Release(p.view);
        Release(p.texture);
        D3D11_TEXTURE2D_DESC td{};
        td.Width = width;
        td.Height = height;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(gpu_.device->CreateTexture2D(&td, nullptr, &p.texture)) ||
            FAILED(gpu_.device->CreateShaderResourceView(p.texture, nullptr, &p.view))) {
            Release(p.texture);
            return -1;
        }
        p.texture_width = width;
        p.texture_height = height;
    }
    ID3D11Resource* source = nullptr;
    view->GetResource(&source);
    const D3D11_BOX box{0, 0, 0, std::min(width, texture_width), std::min(height, texture_height), 1};
    {
        std::lock_guard<std::mutex> context(gpu_.context_lock);
        gpu_.context->CopySubresourceRegion(p.texture, 0, 0, 0, 0, source, 0, &box);
    }
    source->Release();
    p.width = width;
    p.height = height;
    if (!present_thread_.joinable()) present_thread_ = std::thread([this] { PresentLoop(); });
    return slot;
}

void D3D11Display::Ready(int slot, std::uint64_t due_ns) {
    PresentObserver observer;
    std::uint64_t dropped_due = 0;
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        if (ready_ >= 0) {                  // (the newest wins)
            dropped_due = paced_[ready_].due_ns;
            observer = observer_;
        }
        paced_[slot].due_ns = due_ns;
        ready_ = slot;
    }
    if (observer) observer(dropped_due, 0, true);
    paced_wake_.notify_all();
}

bool D3D11Display::PresentImageAt(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                                  std::uint64_t due_ns) {
    if (!width || !height || !image) return !closed_;
    const int slot = CopyPaced(image, width, height);
    if (slot < 0) return PresentImage(image, layout, width, height);
    Ready(slot, due_ns);
    return !closed_;
}

int D3D11Display::HoldImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) {
    (void)layout;
    if (!width || !height || !image) return -1;
    const int slot = CopyPaced(image, width, height);
    if (slot >= 0) {
        std::lock_guard<std::mutex> hold(paced_lock_);
        held_ = slot;
    }
    return slot;
}

void D3D11Display::ShowHeld(int slot, std::uint64_t due_ns) {
    if (slot < 0 || slot >= kPaced) return;
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        if (held_ == slot) held_ = -1;
    }
    Ready(slot, due_ns);
}

void D3D11Display::PresentLoop() {
    using Clock = std::chrono::steady_clock;
    // Each picture shown at its time: the highest priority (asleep almost always).
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    for (;;) {
        std::uint64_t due_ns;
        {
            std::unique_lock<std::mutex> hold(paced_lock_);
            paced_wake_.wait(hold, [&] { return stop_ || ready_ >= 0; });
            if (stop_) return;
            due_ns = paced_[ready_].due_ns;
        }
        // Asleep until just before it is due, the last moment spun (Windows'
        // sleeps end a millisecond or more late).
        const Clock::time_point due = std::min(Clock::time_point{std::chrono::nanoseconds(due_ns)},
                                               Clock::now() + std::chrono::milliseconds(50));
        if (Clock::now() + std::chrono::microseconds(1500) < due)
            std::this_thread::sleep_until(due - std::chrono::microseconds(1200));
        while (Clock::now() < due) std::this_thread::yield();
        int slot;
        PresentObserver observer;
        {
            std::lock_guard<std::mutex> hold(paced_lock_);
            if (stop_) return;
            if (ready_ < 0) continue;
            slot = ready_;
            ready_ = -1;
            taken_ = slot;
            observer = observer_;
        }
        {
            std::lock_guard<std::mutex> hold(present_lock_);
            std::lock_guard<std::mutex> context(gpu_.context_lock);
            const Paced& p = paced_[slot];
            Show(p.view, p.width, p.height, p.texture_width, p.texture_height);
        }
        if (observer)
            observer(paced_[slot].due_ns,
                     std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                                       .count()),
                     false);
        {
            std::lock_guard<std::mutex> hold(paced_lock_);
            taken_ = -1;
        }
    }
}

void D3D11Display::CountPresent() {
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> hold(pacing_lock_);
        if (last_present_.time_since_epoch().count()) {
            const double ms = std::chrono::duration<double, std::milli>(now - last_present_).count();
            interval_sum_ += ms;
            interval_squares_ += ms * ms;
            interval_longest_ = std::max(interval_longest_, ms);
            ++presents_;
        }
        last_present_ = now;
    }
    ++fps_presents_;
    if (!fps_since_.time_since_epoch().count()) fps_since_ = now;
    const double seconds = std::chrono::duration<double>(now - fps_since_).count();
    if (seconds >= 0.5) {
        fps_.store(double(fps_presents_) / seconds);
        fps_presents_ = 0;
        fps_since_ = now;
    }
}

Display::Pacing D3D11Display::TakePacing() {
    std::lock_guard<std::mutex> hold(pacing_lock_);
    Pacing p;
    p.presents = presents_;
    if (presents_) {
        p.mean_ms = interval_sum_ / double(presents_);
        p.jitter_ms = std::sqrt(std::max(0.0, interval_squares_ / double(presents_) - p.mean_ms * p.mean_ms));
        p.longest_ms = interval_longest_;
    }
    presents_ = 0;
    interval_sum_ = interval_squares_ = interval_longest_ = 0;
    return p;
}

D3D11Display::~D3D11Display() {
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        stop_ = true;
    }
    paced_wake_.notify_all();
    if (present_thread_.joinable()) present_thread_.join();
    for (Paced& p : paced_) {
        Release(p.view);
        Release(p.texture);
    }
    if (gpu_.context) {
        gpu_.context->ClearState();
        gpu_.context->Flush();
    }
    Release(overlay_view_);
    Release(overlay_texture_);
    ReleasePostImage(post_in_);
    ReleasePostImage(post_out_);
    ReleasePostImage(smaa_edges_);
    ReleasePostImage(smaa_weights_);
    for (int i = 0; i < 2; ++i) {
        Release(smaa_table_views_[i]);
        Release(smaa_tables_[i]);
    }
    for (ID3D11ComputeShader*& s : smaa_) Release(s);
    Release(fxaa_);
    Release(downsample_);
    Release(post_params_);
    Release(picture_view_);
    Release(picture_);
    Release(linear_);
    Release(point_);
    Release(ps_);
    Release(vs_);
    Release(back_);
    if (swapchain_) {
        swapchain_->SetFullscreenState(FALSE, nullptr);
        Release(swapchain_);
    }
    // (the window, after the swapchain: SdlWindowDisplay's destructor)
}

}  // namespace

std::unique_ptr<Display> CreateD3D11Display(const char* title, int width, int height, std::string* error) {
    auto display = std::make_unique<D3D11Display>();
    if (!display->Init(title, width, height)) {
        if (error) *error = display->error();
        return nullptr;
    }
    return display;
}

}  // namespace ps2gpu

#else  // !_WIN32

#include "ps2gpu/display.h"

namespace ps2gpu {
std::unique_ptr<Display> CreateD3D11Display(const char*, int, int, std::string* error) {
    if (error) *error = "Direct3D 11 is Windows's";
    return nullptr;
}
}  // namespace ps2gpu

#endif
