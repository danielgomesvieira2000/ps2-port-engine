/*
 * Which Vulkan a device draws the GS with, and what it is made with: shared by
 * the windowless device (gpu.cpp) and the Display's (vulkan_display.cpp).
 *
 * Vulkan 1.3 is the first choice. A 1.2 driver (NVIDIA Kepler, AMD GCN 1-3:
 * issue #24) draws too: with VK_KHR_dynamic_rendering where it is offered, else
 * with render passes (gs_renderer.cpp's BeginPass); a 1.1 one (older phones)
 * with render passes. With VK_EXT_extended_dynamic_state the renderer calls
 * the core 1.3 names, pointed at the extension's own (VkUseLegacyFunctions),
 * whose structures are the same; without it, those states are made into its
 * pipelines (g_vk_baked_states).
 */

#include "gpu_internal.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps2gpu {

const char* VkPathName(VkPath path, bool baked) {
    switch (path) {
        case kVk13: return baked ? "Vulkan 1.3, states in pipelines" : "Vulkan 1.3";
        case kVk12Dynamic:
            return baked ? "Vulkan 1.2 + dynamic rendering, states in pipelines" : "Vulkan 1.2 + dynamic rendering";
        case kVk12RenderPass: return baked ? "Vulkan 1.1+, render passes, states in pipelines" : "Vulkan 1.1+, render passes";
    }
    return "?";
}

namespace {
bool BakedForced() {
    const char* env = std::getenv("PS2HOST_VK_BAKED");
    return env && env[0] == '1';
}
}  // namespace

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

bool VkDevicePath(VkPhysicalDevice gpu, VkPath* path, bool* baked, std::string* why) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu, &p);
    const int forced = VkLegacyForced();
    *baked = BakedForced();
    const std::vector<VkExtensionProperties> offered = Extensions(gpu);
#ifdef _WIN32
    // A Vulkan that is a layer over Direct3D 12 (Microsoft's, beside the
    // GPU's own driver) says so with VK_KHR_portability_subset: passed over -
    // Direct3D 11 draws natively on that GPU, without what such a layer
    // lacks.
    if (Offers(offered, "VK_KHR_portability_subset")) {
        if (why) *why = "a Vulkan layered over another API";
        return false;
    }
#endif
    if (p.apiVersion >= VK_API_VERSION_1_3 && !forced) {
        *path = kVk13;
        return true;
    }
    char text[160];
    if (p.apiVersion < VK_API_VERSION_1_1) {
        std::snprintf(text, sizeof text, "Vulkan %u.%u.%u, 1.1 needed", VK_API_VERSION_MAJOR(p.apiVersion),
                      VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (why) *why = text;
        return false;
    }
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    dynamic_rendering.pNext = &dynamic_state;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &dynamic_rendering;
    vkGetPhysicalDeviceFeatures2(gpu, &features);
    if (!Offers(offered, VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME) || !dynamic_state.extendedDynamicState)
        *baked = true;
    // (On 1.1 dynamic rendering needs two more extensions enabled: render passes there.)
    const bool has_dynamic_rendering = p.apiVersion >= VK_API_VERSION_1_2 &&
                                       Offers(offered, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) &&
                                       dynamic_rendering.dynamicRendering;
    *path = forced == 2 || !has_dynamic_rendering ? kVk12RenderPass : kVk12Dynamic;
    return true;
}

void VkDeviceSetup::Fill(VkPhysicalDevice gpu, VkPath path, bool baked) {
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
        void** chain = &features.pNext;
        if (!baked) {
            extensions.push_back(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
            dynamic_state.extendedDynamicState = VK_TRUE;
            *chain = &dynamic_state;
            chain = &dynamic_state.pNext;
        }
        if (path == kVk12Dynamic) {
            extensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            dynamic_rendering.dynamicRendering = VK_TRUE;
            *chain = &dynamic_rendering;
        }
    }
    // An image's view formats named (the GS renderer's): core in 1.2, else its extension's.
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu, &p);
    g_image_format_list = p.apiVersion >= VK_API_VERSION_1_2;
    if (!g_image_format_list && Offers(offered, VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME);
        g_image_format_list = true;
    }
    // A device that is a layer over another API (on Metal: Windows passes
    // them over, VkDevicePath) says what it lacks this way, and must have it
    // enabled.
    if (Offers(offered, "VK_KHR_portability_subset")) extensions.push_back("VK_KHR_portability_subset");
    // A pass's attachment it never reads, not loaded (the GS renderer's deferred passes: a tiler's).
    if (Offers(offered, "VK_KHR_load_store_op_none")) load_none = "VK_KHR_load_store_op_none";
    else if (Offers(offered, "VK_EXT_load_store_op_none")) load_none = "VK_EXT_load_store_op_none";
    if (load_none) extensions.push_back(load_none);
}

namespace {
VkBool32 VKAPI_PTR DebugMessage(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    std::string where;
    for (std::uint32_t i = data->cmdBufLabelCount; i-- > 0;)
        where += std::string(" < ") + (data->pCmdBufLabels[i].pLabelName ? data->pCmdBufLabels[i].pLabelName : "?");
    std::fprintf(stderr, "[vulkan] %s: %s%s%s\n", severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning",
                 data->pMessage ? data->pMessage : "", where.empty() ? "" : "\n[vulkan]   in", where.c_str());
    return VK_FALSE;
}
}  // namespace

void VkDebugExtensions(std::vector<const char*>* extensions) {
    const char* want = std::getenv("PS2HOST_VK_LABELS");
    if (!want || want[0] != '1') return;
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> offered(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, offered.data());
    offered.resize(count);
    for (const char* e : *extensions)
        if (std::strcmp(e, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) return;
    if (!Offers(offered, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        std::fprintf(stderr, "[vulkan] PS2HOST_VK_LABELS: no VK_EXT_debug_utils\n");
        return;
    }
    extensions->push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    g_vk_labels = true;
}

void VkDebugStart(VkInstance instance) {
    if (!g_vk_labels || !vkCreateDebugUtilsMessengerEXT) return;
    static VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (messenger) return;
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = DebugMessage;
    vkCreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &messenger);
}

void VkUseLegacyFunctions(VkPath path, bool baked) {
    g_vk_path = path;
    g_vk_baked_states = baked;
    // STORE_OP_NONE: core in 1.3, else the load_store_op_none extensions'
    // (g_load_op_none) or dynamic rendering's own.
    g_store_op_none = path != kVk12RenderPass || g_load_op_none;
    if (path == kVk13) return;
    // The functions volk fetched through the instance (the loader's, which
    // dispatch by device): the extensions' entries take the core names -
    // none without the extension (a call left in would fail at once).
    vkCmdSetDepthTestEnable = baked ? nullptr : vkCmdSetDepthTestEnableEXT;
    vkCmdSetDepthWriteEnable = baked ? nullptr : vkCmdSetDepthWriteEnableEXT;
    vkCmdSetDepthCompareOp = baked ? nullptr : vkCmdSetDepthCompareOpEXT;
    vkCmdSetStencilTestEnable = baked ? nullptr : vkCmdSetStencilTestEnableEXT;
    vkCmdSetStencilOp = baked ? nullptr : vkCmdSetStencilOpEXT;
    vkCmdBindVertexBuffers2 = baked ? nullptr : vkCmdBindVertexBuffers2EXT;
    if (path == kVk12Dynamic) {
        vkCmdBeginRendering = vkCmdBeginRenderingKHR;
        vkCmdEndRendering = vkCmdEndRenderingKHR;
    } else {
        vkCmdBeginRendering = nullptr;          // (BeginPass begins a render pass instead)
        vkCmdEndRendering = nullptr;
    }
}

}  // namespace ps2gpu
