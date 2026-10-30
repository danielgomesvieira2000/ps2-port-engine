/* D3D11Gpu (d3d11_gpu.h). */

#ifdef _WIN32

#include "d3d11_gpu.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace ps2gpu {

namespace {
std::string Narrow(const wchar_t* text) {
    std::string out;
    for (; *text; ++text) out += char(*text < 128 ? *text : '?');
    return out;
}

std::string Lower(std::string text) {
    for (char& ch : text) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    return text;
}
}  // namespace

bool D3D11Gpu::Init(std::string* error) {
    auto fail = [&](const char* what, HRESULT hr) {
        char text[160];
        std::snprintf(text, sizeof text, "%s failed (HRESULT 0x%08lX)", what, static_cast<unsigned long>(hr));
        if (error) *error = text;
        return false;
    };
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&factory));
    if (FAILED(hr)) return fail("CreateDXGIFactory1", hr);
    // PS2HOST_GPU=<part of its name> picks the adapter (as for Vulkan), else
    // the system's first - the one the desktop is on - leaving out
    // Microsoft's software adapter.
    const char* want = std::getenv("PS2HOST_GPU");
    IDXGIAdapter1* candidate = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &candidate) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        candidate->GetDesc1(&desc);
        const bool software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
        const bool named = want && *want && Lower(Narrow(desc.Description)).find(Lower(want)) != std::string::npos;
        if (!software && (!adapter || named)) {
            if (adapter) adapter->Release();
            adapter = candidate;
            if (named) break;
        } else {
            candidate->Release();
        }
    }
    if (!adapter) {
        if (error) *error = "no Direct3D adapter";
        return false;
    }
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if (std::getenv("PS2HOST_D3D11_DEBUG")) flags |= D3D11_CREATE_DEVICE_DEBUG;     // (the SDK layers installed)
    hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels, 2, D3D11_SDK_VERSION, &device,
                           &level, &context);
    if (FAILED(hr)) {
        // (An 11_0-only runtime refuses a list naming 11_1.)
        hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels + 1, 1, D3D11_SDK_VERSION,
                               &device, &level, &context);
    }
    if (FAILED(hr)) return fail("D3D11CreateDevice (feature level 11_0)", hr);
    D3D11_FEATURE_DATA_D3D11_OPTIONS options{};
    if (SUCCEEDED(device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof options)))
        logic_op = options.OutputMergerLogicOp != FALSE;
    D3D11_RASTERIZER_DESC rd{};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    rd.ScissorEnable = TRUE;
    if (FAILED(hr = device->CreateRasterizerState(&rd, &scissored))) return fail("CreateRasterizerState", hr);
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    char text[320];
    std::snprintf(text, sizeof text, "%s (Direct3D 11, feature level %s)", Narrow(desc.Description).c_str(),
                  level == D3D_FEATURE_LEVEL_11_1 ? "11_1" : "11_0");
    describe_ = text;
    return true;
}

D3D11Gpu::~D3D11Gpu() {
    if (scissored) scissored->Release();
    if (context) context->Release();
    if (device) device->Release();
    if (adapter) adapter->Release();
    if (factory) factory->Release();
}

}  // namespace ps2gpu

#endif  // _WIN32
