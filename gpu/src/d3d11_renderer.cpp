/*
 * The GS renderer on Direct3D 11 (BACKENDS_ROADMAP.md, X3): the Vulkan
 * renderer itself, on a VulkanGpu whose calls d3d11_vulkan.cpp's layer makes
 * with the D3D11 display's device. The same renderer draws on both, by the
 * same rules - what it cannot do on Direct3D 11 (a pipeline with no HLSL) it
 * goes without, as on a Vulkan device without that feature.
 */

#include "gpu_internal.h"

#ifdef _WIN32
#include "d3d11_gpu.h"
#endif

namespace ps2gpu {

std::unique_ptr<Gpu> CreateD3D11Gpu(std::string* error) {
#if defined(_WIN32) && defined(PS2GPU_D3D11)
    auto gpu = std::make_unique<D3D11Gpu>();
    if (!gpu->Init(error)) return nullptr;
    return gpu;
#else
    if (error) *error = "no Direct3D 11 in this build";
    return nullptr;
#endif
}

std::unique_ptr<GsRenderer> CreateD3D11GsRenderer(Gpu& gpu, std::string* error) {
#if defined(_WIN32) && defined(PS2GPU_D3D11)
    auto& d3d = static_cast<D3D11Gpu&>(gpu);
    if (!d3d.vulkan) {
        vk11::Install(d3d);
        // Handles of the layer's own: never dereferenced as Vulkan's.
        static int instance, physical, device, queue;
        auto vulkan = std::make_unique<VulkanGpu>();
        std::string why;
        if (!vulkan->InitShared(reinterpret_cast<VkInstance>(&instance), reinterpret_cast<VkPhysicalDevice>(&physical),
                                reinterpret_cast<VkDevice>(&device), 0, reinterpret_cast<VkQueue>(&queue), true,
                                d3d.Describe(), &why)) {
            if (error) *error = "the Vulkan layer over Direct3D 11: " + why;
            return nullptr;
        }
        d3d.vulkan = std::move(vulkan);
    }
    return CreateGsRenderer(*d3d.vulkan, error);
#else
    (void)gpu;
    if (error) *error = "no Direct3D 11 GS renderer in this build";
    return nullptr;
#endif
}

}  // namespace ps2gpu
