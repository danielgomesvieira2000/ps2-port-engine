/*
 * Which Vulkan a device draws the GS with, and what it is made with: shared by
 * the windowless device (gpu.cpp) and the Display's (vulkan_display.cpp).
 *
 * Vulkan 1.3 is the first choice. A 1.2 driver (NVIDIA Kepler, AMD GCN 1-3:
 * issue #24) draws too: with VK_KHR_dynamic_rendering where it is offered, else
 * with render passes (gs_renderer.cpp's BeginPass), and with
 * VK_EXT_extended_dynamic_state either way. The renderer calls the core 1.3
 * names; on a 1.2 device they are pointed at the extensions' own
 * (VkUseLegacyFunctions), whose structures are the same.
 */

#include "gpu_internal.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps2gpu {

const char* VkPathName(VkPath path) {
    switch (path) {
        case kVk13: return "Vulkan 1.3";
        case kVk12Dynamic: return "Vulkan 1.2 + dynamic rendering";
        case kVk12RenderPass: return "Vulkan 1.2, render passes";
    }
    return "?";
}

int VkLegacyForced() {
    const char* env = std::getenv("PS2HOST_VK_LEGACY");
    if (!env || (env[0] != '1' && env[0] != '2')) return 0;
    return env[0] - '0';
}

std::uint32_t VkInstanceVersion() { return VkLegacyForced() ? VK_API_VERSION_1_2 : VK_API_VERSION_1_3; }

int VkDeviceScore(const VkPhysicalDeviceProperties& p) {
    int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
    if (const char* want = std::getenv("PS2HOST_GPU")) {
        std::string name = p.deviceName, part = want;
        for (char& ch : name) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        for (char& ch : part) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        if (!part.empty() && name.find(part) != std::string::npos) score += 10;
    }
    return score;
}

namespace {
bool Offers(const std::vector<VkExtensionProperties>& offered, const char* name) {
    for (const VkExtensionProperties& e : offered)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

std::vector<VkExtensionProperties> Extensions(VkPhysicalDevice gpu) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> offered(count);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, offered.data());
    offered.resize(count);
    return offered;
}
}  // namespace

bool VkDevicePath(VkPhysicalDevice gpu, VkPath* path, std::string* why) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu, &p);
    const int forced = VkLegacyForced();
    if (p.apiVersion >= VK_API_VERSION_1_3 && !forced) {
        *path = kVk13;
        return true;
    }
    char text[160];
    if (p.apiVersion < VK_API_VERSION_1_2) {
        std::snprintf(text, sizeof text, "Vulkan %u.%u.%u, 1.2 needed", VK_API_VERSION_MAJOR(p.apiVersion),
                      VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (why) *why = text;
        return false;
    }
    const std::vector<VkExtensionProperties> offered = Extensions(gpu);
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    dynamic_rendering.pNext = &dynamic_state;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &dynamic_rendering;
    vkGetPhysicalDeviceFeatures2(gpu, &features);
    if (!Offers(offered, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME) || !dynamic_state.extendedDynamicState) {
        if (why) *why = "Vulkan 1.2 without VK_EXT_extended_dynamic_state";
        return false;
    }
    const bool has_dynamic_rendering =
        Offers(offered, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) && dynamic_rendering.dynamicRendering;
    *path = forced == 2 || !has_dynamic_rendering ? kVk12RenderPass : kVk12Dynamic;
    return true;
}

void VkDeviceSetup::Fill(VkPhysicalDevice gpu, VkPath path) {
    const std::vector<VkExtensionProperties> offered = Extensions(gpu);
    vkGetPhysicalDeviceFeatures(gpu, &supported);
    features.features.dualSrcBlend = supported.dualSrcBlend;    // the GS's blend factor as a second output
    features.features.logicOp = supported.logicOp;              // FBMSK's partly masked channels
    features.features.textureCompressionBC = supported.textureCompressionBC;     // texture packs' DDS
    features.features.samplerAnisotropy = supported.samplerAnisotropy;           // texture packs' samplers
    if (path == kVk13) {
        v13.dynamicRendering = VK_TRUE;
        features.pNext = &v13;
    } else {
        extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
        dynamic_state.extendedDynamicState = VK_TRUE;
        features.pNext = &dynamic_state;
        if (path == kVk12Dynamic) {
            extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            dynamic_rendering.dynamicRendering = VK_TRUE;
            dynamic_state.pNext = &dynamic_rendering;
        }
    }
    // A pass's attachment it never reads, not loaded (the GS renderer's deferred passes: a tiler's).
    if (Offers(offered, "VK_KHR_load_store_op_none")) load_none = "VK_KHR_load_store_op_none";
    else if (Offers(offered, "VK_EXT_load_store_op_none")) load_none = "VK_EXT_load_store_op_none";
    if (load_none) extensions.push_back(load_none);
}

void VkUseLegacyFunctions(VkPath path) {
    g_vk_path = path;
    // STORE_OP_NONE: core in 1.3, else the load_store_op_none extensions'
    // (g_load_op_none) or dynamic rendering's own.
    g_store_op_none = path != kVk12RenderPass || g_load_op_none;
    if (path == kVk13) return;
    // The functions volk fetched through the instance (the loader's, which
    // dispatch by device): the extensions' entries take the core names.
    vkCmdSetDepthTestEnable = vkCmdSetDepthTestEnableEXT;
    vkCmdSetDepthWriteEnable = vkCmdSetDepthWriteEnableEXT;
    vkCmdSetDepthCompareOp = vkCmdSetDepthCompareOpEXT;
    vkCmdSetStencilTestEnable = vkCmdSetStencilTestEnableEXT;
    vkCmdSetStencilOp = vkCmdSetStencilOpEXT;
    vkCmdBindVertexBuffers2 = vkCmdBindVertexBuffers2EXT;
    if (path == kVk12Dynamic) {
        vkCmdBeginRendering = vkCmdBeginRenderingKHR;
        vkCmdEndRendering = vkCmdEndRenderingKHR;
    } else {
        vkCmdBeginRendering = nullptr;          // (BeginPass begins a render pass instead)
        vkCmdEndRendering = nullptr;
    }
}

}  // namespace ps2gpu
