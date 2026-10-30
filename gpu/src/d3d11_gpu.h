/* D3D11Gpu: a Direct3D 11 device and its immediate context - the D3D11
   display's, and what a D3D11 GS renderer draws with (Gpu::Api::kD3D11).
   Windows only. */

#ifndef PS2GPU_D3D11_GPU_H
#define PS2GPU_D3D11_GPU_H

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#undef small

#include <mutex>
#include <string>

#include "ps2gpu/gs_renderer.h"

namespace ps2gpu {

class D3D11Gpu final : public Gpu {
public:
    ~D3D11Gpu() override;
    /* A hardware device at feature level 11_0 or above (11_1 where there is
       one), on PS2HOST_GPU's adapter if it names one, else the first. */
    bool Init(std::string* error);
    std::string Describe() const override { return describe_; }
    Api api() const override { return Api::kD3D11; }

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIFactory2* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    /* The immediate context is one thread's at a time: the display's
       presents and the renderer's work hold this. */
    std::mutex context_lock;
    ID3D11RasterizerState* scissored = nullptr;     // no culling, scissor on
    bool logic_op = false;          // D3D11.1's output-merger logic ops (UNORM targets: OutputMergerLogicOp)

private:
    std::string describe_;
};

}  // namespace ps2gpu

#endif  // _WIN32
#endif  // PS2GPU_D3D11_GPU_H
