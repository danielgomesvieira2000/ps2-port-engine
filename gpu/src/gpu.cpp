/*
 * VulkanGpu: an instance and a device with one graphics queue, no window.
 * Vulkan 1.3, else 1.2 (vk_device.cpp).
 */

#include "gpu_internal.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps2gpu {

void SetGpuWaitObserver(GpuWaitObserver observer) { g_wait_observer = observer; }
void (*g_renderer_thread_start)() = nullptr;
void SetRendererThreadStart(void (*start)()) { g_renderer_thread_start = start; }

bool VulkanGpu::Init(std::string* error) {
    auto fail = [&](const char* what, VkResult r) {
        char text[160];
        std::snprintf(text, sizeof text, "%s failed (VkResult %d)", what, int(r));
        if (error) *error = text;
        return false;
    };
    VkResult r = volkInitialize();
    if (r != VK_SUCCESS) return fail("volkInitialize (is there a Vulkan driver?)", r);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ps2gpu";
    app.apiVersion = VkInstanceVersion();
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if ((r = vkCreateInstance(&ici, nullptr, &instance)) != VK_SUCCESS) return fail("vkCreateInstance", r);
    volkLoadInstance(instance);

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> list(n);
    vkEnumeratePhysicalDevices(instance, &n, list.data());
    int best = -1;
    VkPath path = kVk13;
    bool baked = false;
    std::string passed_over;
    for (VkPhysicalDevice candidate : list) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(candidate, &p);
        VkPath candidate_path;
        bool candidate_baked = false;
        std::string why;
        if (!VkDevicePath(candidate, &candidate_path, &candidate_baked, &why)) {
            passed_over += std::string(passed_over.empty() ? "" : ", ") + p.deviceName + " (" + why + ")";
            continue;
        }
        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, props.data());
        for (std::uint32_t f = 0; f < families; ++f) {
            if (!(props[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            const int score = VkDeviceScore(p);
            if (score > best) {
                best = score;
                gpu = candidate;
                family = f;
                path = candidate_path;
                baked = candidate_baked;
            }
            break;
        }
    }
    if (!gpu) {
        if (error) *error = "no Vulkan 1.1 device with a graphics queue" + (passed_over.empty() ? std::string()
                                                                         : "; passed over: " + passed_over);
        return false;
    }
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu, &p);
    describe_ = std::string(p.deviceName) + " (" + VkPathName(path, baked) + ")";

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceSetup setup;
    setup.Fill(gpu, path, baked);
    dual_source = setup.supported.dualSrcBlend == VK_TRUE;
    logic_op = setup.supported.logicOp == VK_TRUE;
    bc = setup.supported.textureCompressionBC == VK_TRUE;
    if (setup.supported.samplerAnisotropy) anisotropy = p.limits.maxSamplerAnisotropy;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &setup.features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = std::uint32_t(setup.extensions.size());
    dci.ppEnabledExtensionNames = setup.extensions.data();
    if ((r = vkCreateDevice(gpu, &dci, nullptr, &device)) != VK_SUCCESS) return fail("vkCreateDevice", r);
    g_load_op_none = setup.load_none != nullptr;
    VkUseLegacyFunctions(path, baked);
    // Device functions stay the loader's (volkLoadInstance): they dispatch by
    // handle, so the window's device and this one can live side by side. volk
    // keeps one global table; loading a device into it would point the other
    // device's calls at this one's driver entries (--window with --gpu-shadow
    // crashed so).
    vkGetDeviceQueue(device, family, 0, &queue);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    if ((r = vkCreateCommandPool(device, &pci, nullptr, &pool)) != VK_SUCCESS)
        return fail("vkCreateCommandPool", r);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = kSlots;
    if ((r = vkAllocateCommandBuffers(device, &cai, cmds)) != VK_SUCCESS)
        return fail("vkAllocateCommandBuffers", r);
    cmd = cmds[0];
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (VkFence& f : fences)
        if ((r = vkCreateFence(device, &fci, nullptr, &f)) != VK_SUCCESS) return fail("vkCreateFence", r);
    MakeTimestamps();
    return true;
}

void VulkanGpu::MakeTimestamps() {
    // Only where the queue keeps time (timestampValidBits) and says how fast.
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(gpu, &properties);
    if (family >= count || families[family].timestampValidBits == 0 || properties.limits.timestampPeriod <= 0)
        return;
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 2 * kSlots;
    if (vkCreateQueryPool(device, &qci, nullptr, &timestamps_) != VK_SUCCESS) {
        timestamps_ = VK_NULL_HANDLE;
        return;
    }
    timestamp_ns_ = properties.limits.timestampPeriod;
    if (std::getenv("PS2HOST_GPU_PROFILE")) {
        qci.queryCount = kMarks * kSlots;
        if (vkCreateQueryPool(device, &qci, nullptr, &profile_) != VK_SUCCESS) profile_ = VK_NULL_HANDLE;
    }
}

bool VulkanGpu::InitShared(VkInstance shared_instance, VkPhysicalDevice shared_gpu, VkDevice shared_device,
                           std::uint32_t shared_family, VkQueue shared_queue, bool shared_dual_source,
                           const std::string& describe, std::string* error) {
    owns_device = false;
    instance = shared_instance;
    gpu = shared_gpu;
    device = shared_device;
    family = shared_family;
    queue = shared_queue;
    dual_source = shared_dual_source;
    VkPhysicalDeviceFeatures supported;                 // the Display enables these where there are (VkDeviceSetup)
    vkGetPhysicalDeviceFeatures(shared_gpu, &supported);
    logic_op = supported.logicOp == VK_TRUE;
    bc = supported.textureCompressionBC == VK_TRUE;
    // Anisotropy for texture packs' and ANISOTROPIC's samplers, as the
    // headless device has it (a window's device kept 1: no anisotropy).
    if (supported.samplerAnisotropy) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(shared_gpu, &properties);
        anisotropy = properties.limits.maxSamplerAnisotropy;
    }
    describe_ = describe;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateCommandPool(device, &pci, nullptr, &pool) != VK_SUCCESS) {
        if (error) *error = "vkCreateCommandPool failed";
        return false;
    }
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = kSlots;
    bool made = vkAllocateCommandBuffers(device, &cai, cmds) == VK_SUCCESS;
    for (VkFence& f : fences) made = made && vkCreateFence(device, &fci, nullptr, &f) == VK_SUCCESS;
    if (!made) {
        if (error) *error = "no command buffer or fence";
        return false;
    }
    cmd = cmds[0];
    MakeTimestamps();
    return true;
}

std::uint32_t VulkanGpu::FindMemory(std::uint32_t bits, VkMemoryPropertyFlags wanted) const {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return ~0u;
}

VulkanGpu::~VulkanGpu() {
    if (device) {
        {
            std::lock_guard<std::mutex> hold(g_queue_lock);
            vkDeviceWaitIdle(device);
        }
        for (VkFence f : fences)
            if (f) vkDestroyFence(device, f, nullptr);
        if (timestamps_) vkDestroyQueryPool(device, timestamps_, nullptr);
        if (pool) vkDestroyCommandPool(device, pool, nullptr);
        if (owns_device) vkDestroyDevice(device, nullptr);
    }
    if (instance && owns_device) vkDestroyInstance(instance, nullptr);
}

std::unique_ptr<Gpu> CreateGpu(std::string* error) {
    auto gpu = std::make_unique<VulkanGpu>();
    if (!gpu->Init(error)) return nullptr;
    return gpu;
}

}  // namespace ps2gpu
