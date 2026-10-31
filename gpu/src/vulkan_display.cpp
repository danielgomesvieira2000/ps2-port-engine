/*
 * VulkanDisplay: an SDL3 window and a Vulkan swapchain presenting a picture.
 *
 * Each frame the picture goes into a host-visible staging buffer, is copied to
 * an image, and blitted - scaled, 4:3, letterboxed - onto the swapchain image,
 * which is cleared to black first. One frame is in flight: the fence is waited
 * before the staging buffer is written again. The swapchain presents in
 * MAILBOX mode where the driver has it (never blocking the emulation thread),
 * else IMMEDIATE, else FIFO (IMMEDIATE first on a VRR screen: SetVrr), and is
 * recreated when the window changes size.
 */

#include "ps2gpu/display.h"
#include "window_display.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <volk.h>

#include "gpu_internal.h"
#include "AreaTex.h"             // gpu/third_party/smaa
#include "SearchTex.h"
#include "downsample_comp.h"
#include "fxaa_comp.h"
#include "smaa_blend_comp.h"
#include "smaa_edges_comp.h"
#include "smaa_weights_comp.h"

#include <algorithm>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#undef small            // rpcndr.h's, through the COM headers; SetRumble names a parameter so
#pragma comment(lib, "dxgi.lib")
#endif
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ps2gpu {

namespace {

class VulkanDisplay final : public SdlWindowDisplay {
public:
    ~VulkanDisplay() override;
    bool Init(const char* title, int width, int height);
    bool Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) override;
    bool PresentImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) override;
    bool PresentImageAt(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                        std::uint64_t due_ns) override;
    int HoldImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) override;
    void ShowHeld(int slot, std::uint64_t due_ns) override;
    int CopyPaced(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height, std::uint64_t due_ns,
                  bool held);
    void SetVsync(bool on) override;
    double PresentsPerSecond() const override { return fps_.load(); }
    std::atomic<double> fps_{0.0};
    std::chrono::steady_clock::time_point fps_since_{};
    std::uint64_t fps_presents_ = 0;
    void SetAntiAliasing(AntiAliasing mode) override { anti_aliasing_.store(int(mode)); }
    /* The overlay into the swapchain image (TRANSFER_DST) being made. */
    void DrawOverlay(VkImage target);
    static constexpr VkDeviceSize kOverlayBytes = VkDeviceSize(1280) * 720 * 4;
    VkBuffer overlay_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory overlay_memory_ = VK_NULL_HANDLE;
    void* overlay_mapped_ = nullptr;
    bool Vsync() const override { return vsync_; }
    void SetVrr(bool on) override;
    bool Vrr() const override { return vrr_; }
    Pacing TakePacing() override;
    std::string Describe() const override { return describe_; }
    Gpu* SharedGpu() override;

private:
    bool Check(VkResult result, const char* what);
    bool CreateSwapchain();
    void DestroySwapchain();
    bool EnsurePicture(std::uint32_t width, std::uint32_t height);
    void DestroyPicture();
    std::uint32_t FindMemory(std::uint32_t bits, VkMemoryPropertyFlags wanted) const;
    // PresentImageAt's: pictures copied and waiting, and the thread that
    // shows them when they are due.
    bool EnsurePaced(std::uint32_t width, std::uint32_t height);
    void DestroyPaced();
    void PresentLoop();

    void SetPresentObserver(PresentObserver observer) override {
        std::lock_guard<std::mutex> hold(paced_lock_);
        observer_ = std::move(observer);
    }
    void SetShownObserver(ShownObserver observer) override {
        std::lock_guard<std::mutex> hold(pacing_lock_);
        shown_observer_ = std::move(observer);
    }
    ShownObserver shown_observer_;          // SetShownObserver (under pacing_lock_)
    void SetGpuDoneObserver(GpuDoneObserver observer) override;
    void SetPresentTag(std::uint64_t tag) override { present_tag_.store(tag); }
private:
    /* SetGpuDoneObserver's: after each tagged present an empty submit with
       a fence of its own - signalled once all before it on the queue is
       done - waited for by a thread that notes the time. Its fences its own
       (a present's are reset by the next present). */
    struct DoneWatch {
        std::mutex lock;
        std::condition_variable wake;
        std::vector<std::pair<VkFence, std::uint64_t>> waiting;     // (fence, tag), oldest first
        std::vector<VkFence> spare;
        bool stop = false;
        GpuDoneObserver observer;
        std::thread thread;
    };
    std::unique_ptr<DoneWatch> done_watch_;
    PresentObserver observer_;              // SetPresentObserver (under paced_lock_)
    std::atomic<std::uint64_t> present_tag_{0};     // SetPresentTag's, taken by the next Submit
    void WatchDone();
    void StopDoneWatch();
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    std::uint32_t family_ = 0;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkSemaphore> rendered_;     // one per swapchain image
    VkCommandPool pool_ = VK_NULL_HANDLE;
    // Presents in flight: each its command buffer, acquire semaphore and
    // fence, so a present waits for the one three before it, not the last
    // (with one, every present waited for all the GPU had been given - the
    // GS thread, which presents, in step with the GPU once a vertical blank).
    static constexpr int kInFlight = 3;
    VkCommandBuffer cmds_[kInFlight] = {};
    VkSemaphore acquireds_[kInFlight] = {};
    VkFence fences_[kInFlight] = {};
    /* Signalled as the acquired image is free (BeginPresent waits for it on
       the CPU): the present's submission never waits on the queue for the
       display - the queue the renderer shares, its work held behind each
       present's wait for a refresh (at 144 Hz the GS thread waited on its
       fences most of a loading screen, kept to 144 frames a second). */
    VkFence acquire_fences_[kInFlight] = {};
    /* The overlay's copy each slot's present reads (DrawOverlay): which of
       SetOverlay's panels it holds. One copy for all would be overwritten
       while an earlier present still read it - a panel changing twice a
       second (the FPS counter) would tear. */
    std::uint64_t overlay_slot_generation_[kInFlight] = {};
    int slot_ = 0;
    // The current present's (slot_'s).
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkSemaphore acquired_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkPresentModeKHR mode_ = VK_PRESENT_MODE_FIFO_KHR;
    std::atomic<bool> vsync_{false};        // FIFO when set (SetVsync)
    std::atomic<bool> vrr_{false};          // IMMEDIATE before MAILBOX when set (SetVrr)
    // SetGpuBoost: VK_NV_low_latency2 enabled on the device (the driver has
    // it), the boost wanted, and its call (the loader's, by name).
    bool low_latency_ = false;
    std::atomic<bool> boost_{true};
    PFN_vkSetLatencySleepModeNV set_sleep_mode_ = nullptr;
    void ApplyBoost();
    void SetGpuBoost(bool on) override;
    bool GpuBoosted() const override { return boosted_.load(std::memory_order_relaxed); }
    std::atomic<bool> boosted_{false};      // ApplyBoost's: the driver took the boost

    VkBuffer staging_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkImage picture_ = VK_NULL_HANDLE;
    VkDeviceMemory picture_memory_ = VK_NULL_HANDLE;
    std::uint32_t picture_width_ = 0, picture_height_ = 0;

    /* Acquires the next swapchain image and begins the command buffer:
       false (with *ok) when there is nothing to present into. */
    bool BeginPresent(std::uint32_t* index, bool* ok);
    /* Clears the swapchain image, blits `source` (TRANSFER_SRC) to it 4:3,
       submits and presents. */
    bool FinishPresent(std::uint32_t index, VkImage source, std::uint32_t width, std::uint32_t height);
    /* Ends the command buffer, submits it and presents the image. */
    bool Submit(std::uint32_t index);
    bool dual_source_ = false;
    // ANTI-ALIASING (SetAntiAliasing): the picture copied into post_in_, and
    // smoothed by fxaa.comp into post_out_ (its size), or brought down to the
    // shown size by downsample.comp (supersampling's) - which goes to the
    // window instead.
    std::atomic<int> anti_aliasing_{int(AntiAliasing::kOff)};
    VkImage post_in_ = VK_NULL_HANDLE, post_out_ = VK_NULL_HANDLE;
    VkDeviceMemory post_in_memory_ = VK_NULL_HANDLE, post_out_memory_ = VK_NULL_HANDLE;
    VkImageView post_in_view_ = VK_NULL_HANDLE, post_out_view_ = VK_NULL_HANDLE;
    VkSampler post_sampler_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout post_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool post_descriptors_ = VK_NULL_HANDLE;
    VkDescriptorSet post_set_ = VK_NULL_HANDLE;
    VkPipelineLayout post_layout_ = VK_NULL_HANDLE;
    VkPipeline fxaa_pipeline_ = VK_NULL_HANDLE, downsample_pipeline_ = VK_NULL_HANDLE;
    std::uint32_t post_in_width_ = 0, post_in_height_ = 0, post_out_width_ = 0, post_out_height_ = 0;
    bool post_failed_ = false;              // the passes could not be made: pictures shown as drawn
    /* The pipelines (once) and the images at these sizes; false when there are none. */
    bool EnsurePost(std::uint32_t in_width, std::uint32_t in_height, std::uint32_t out_width,
                    std::uint32_t out_height);
    void DestroyPostImages();
    /* `source` (TRANSFER_SRC, width x height) through `pipeline` into post_out_ (left
       TRANSFER_SRC), in cmd_. */
    void RecordPost(VkImage source, std::uint32_t width, std::uint32_t height, VkPipeline pipeline);
    // SMAA: post_in_ through its edges and blending weights (images of the
    // picture's size) into post_out_ - smaa_edges, smaa_weights and
    // smaa_blend.comp, each with a set of its own (three samplers, an
    // output image); its area and search tables uploaded once, from a
    // staging buffer kept until the end.
    enum { kSmaaEdges, kSmaaWeights, kSmaaArea, kSmaaSearch, kSmaaImages };
    VkImage smaa_image_[kSmaaImages] = {};
    VkDeviceMemory smaa_memory_[kSmaaImages] = {};
    VkImageView smaa_view_[kSmaaImages] = {};
    VkSampler smaa_point_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout smaa_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool smaa_descriptors_ = VK_NULL_HANDLE;
    VkDescriptorSet smaa_sets_[3] = {};
    VkPipelineLayout smaa_layout_ = VK_NULL_HANDLE;
    VkPipeline smaa_pipelines_[3] = {};
    VkBuffer smaa_staging_ = VK_NULL_HANDLE;
    VkDeviceMemory smaa_staging_memory_ = VK_NULL_HANDLE;
    bool smaa_tables_sent_ = false, smaa_failed_ = false;
    std::uint32_t smaa_width_ = 0, smaa_height_ = 0;
    VkImageView smaa_for_in_ = VK_NULL_HANDLE;      // post_in_'s view its sets were written with
    bool MakeSmaaImage(int which, VkFormat format, std::uint32_t width, std::uint32_t height, VkImageUsageFlags usage);
    /* The passes (once), their tables and images at this size; false when there are none. */
    bool EnsureSmaa(std::uint32_t width, std::uint32_t height);
    /* As RecordPost, through SMAA's three passes. */
    void RecordSmaa(VkImage source, std::uint32_t width, std::uint32_t height);
    void DestroySmaa();
    // PresentImageAt's pictures: copies of the frames' targets, three, each
    // its copy's command buffer and fence (the GS thread's own pool).
    static constexpr int kPaced = 4;          // (one of them held: HoldImage)
    struct PacedPicture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkCommandBuffer copy = VK_NULL_HANDLE;
        VkFence copied = VK_NULL_HANDLE;
        std::uint32_t width = 0, height = 0;
        std::uint64_t due_ns = 0;
    };
    PacedPicture paced_[kPaced];
    std::uint32_t paced_width_ = 0, paced_height_ = 0;
    VkCommandPool copy_pool_ = VK_NULL_HANDLE;
    std::mutex paced_lock_;                 // ready_, taken_, stop_
    std::condition_variable paced_wake_;
    std::deque<int> ready_;                 // pictures waiting to be shown, oldest first
    int taken_ = -1;                        // the one being shown
    int held_ = -1;                         // HoldImage's, until ShowHeld
    bool bounded_ = false;                  // BeginPresent's waits bounded (PresentImage's, under present_lock_)
    std::uint64_t skipped_ = 0;             // pictures not shown: no image within the bound
    bool stop_ = false;
    std::thread present_thread_;
    std::mutex present_lock_;               // the swapchain and cmds_: the present thread, or Present
    // Present times (TakePacing): intervals since the last take.
    std::mutex pacing_lock_;
    std::chrono::steady_clock::time_point last_present_{};
    std::uint64_t presents_ = 0;
    double interval_sum_ = 0, interval_squares_ = 0, interval_longest_ = 0;
    std::unique_ptr<VulkanGpu> shared_;
};

Gpu* VulkanDisplay::SharedGpu() {
    if (!shared_) {
        auto gpu = std::make_unique<VulkanGpu>();
        std::string error;
        if (!gpu->InitShared(instance_, gpu_, device_, family_, queue_, dual_source_, describe_, &error))
            return nullptr;
        shared_ = std::move(gpu);
    }
    return shared_.get();
}

bool VulkanDisplay::Check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    char text[160];
    std::snprintf(text, sizeof text, "%s failed (VkResult %d)", what, int(result));
    error_ = text;
    std::fprintf(stderr, "[display] %s\n", text);
    return false;
}

bool VulkanDisplay::Init(const char* title, int width, int height) {
    if (!Check(volkInitialize(), "volkInitialize (is there a Vulkan driver?)")) return false;
    if (!OpenWindow(title, width, height, SDL_WINDOW_VULKAN)) return false;

    // The instance, with the extensions SDL needs for its surface.
    Uint32 count = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = title;
    app.apiVersion = VkInstanceVersion();     // the GS renderer draws on this device too
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = count;
    ici.ppEnabledExtensionNames = extensions;
    if (!Check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance")) return false;
    volkLoadInstance(instance_);
    if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
        error_ = std::string("SDL_Vulkan_CreateSurface: ") + SDL_GetError();
        return false;
    }

    // A GPU with a queue that draws and presents; a discrete one first.
    std::uint32_t gpus = 0;
    vkEnumeratePhysicalDevices(instance_, &gpus, nullptr);
    std::vector<VkPhysicalDevice> list(gpus);
    vkEnumeratePhysicalDevices(instance_, &gpus, list.data());
    int best = -1;
    VkPath path = kVk13;
    bool baked = false;
    std::string too_old;     // GPUs whose driver is older than Vulkan 1.2 (issue #24): named in the error
    for (VkPhysicalDevice candidate : list) {
        // The GS renderer draws with Vulkan 1.3, 1.2 or 1.1 (vk_device.cpp):
        // an older driver's device is passed over and the window draws with
        // GDI.
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(candidate, &p);
        VkPath candidate_path;
        bool candidate_baked = false;
        std::string why;
        if (!VkDevicePath(candidate, &candidate_path, &candidate_baked, &why)) {
            too_old += std::string(too_old.empty() ? "" : ", ") + p.deviceName + " (" + why + ")";
            continue;
        }
        std::uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
        std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, props.data());
        for (std::uint32_t f = 0; f < families; ++f) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, f, surface_, &present);
            if (!(props[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
            const int score = VkDeviceScore(p);
            if (score > best) {
                best = score;
                gpu_ = candidate;
                family_ = f;
                path = candidate_path;
                baked = candidate_baked;
            }
            break;
        }
    }
    if (!gpu_) {
        error_ = too_old.empty() ? "no Vulkan device can draw and present to the window"
                                 : "no Vulkan 1.1 GPU; passed over: " + too_old;
        return false;
    }
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(gpu_, &p);
    char text[320];
    std::snprintf(text, sizeof text, "%s (Vulkan %u.%u.%u, driver %u; drawn with %s)", p.deviceName,
                  VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                  VK_API_VERSION_PATCH(p.apiVersion), p.driverVersion, VkPathName(path, baked));
    describe_ = text;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    // What the GS renderer needs, for when it draws on this device
    // (SharedGpu), and the window's own extensions.
    VkDeviceSetup setup;
    setup.Fill(gpu_, path, baked);
    setup.extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    dual_source_ = setup.supported.dualSrcBlend == VK_TRUE;
    if (setup.load_none) {
        g_load_op_none = true;
        std::fprintf(stderr, "[display] %s: a pass's unread attachments not loaded\n", setup.load_none);
    }
    {
        // NVIDIA's low latency (its boost: SetGpuBoost), where the driver has it.
        std::uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(gpu_, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> offered(count);
        vkEnumerateDeviceExtensionProperties(gpu_, nullptr, &count, offered.data());
        for (const VkExtensionProperties& e : offered)
            if (std::strcmp(e.extensionName, VK_NV_LOW_LATENCY_2_EXTENSION_NAME) == 0) low_latency_ = true;
        if (const char* env = std::getenv("PS2HOST_GPU_BOOST"); env && env[0] == '0') low_latency_ = false;
        if (low_latency_) setup.extensions.push_back(VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
    }
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &setup.features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = std::uint32_t(setup.extensions.size());
    dci.ppEnabledExtensionNames = setup.extensions.data();
    if (!Check(vkCreateDevice(gpu_, &dci, nullptr, &device_), "vkCreateDevice")) return false;
    VkUseLegacyFunctions(path, baked);
    if (low_latency_) {
        set_sleep_mode_ = reinterpret_cast<PFN_vkSetLatencySleepModeNV>(
            vkGetDeviceProcAddr(device_, "vkSetLatencySleepModeNV"));
        low_latency_ = set_sleep_mode_ != nullptr;
    }
    // Device functions stay the loader's: see gpu.cpp.
    vkGetDeviceQueue(device_, family_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family_;
    if (!Check(vkCreateCommandPool(device_, &pci, nullptr, &pool_), "vkCreateCommandPool")) return false;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = kInFlight;
    if (!Check(vkAllocateCommandBuffers(device_, &cai, cmds_), "vkAllocateCommandBuffers")) return false;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < kInFlight; ++i) {
        if (!Check(vkCreateSemaphore(device_, &sci, nullptr, &acquireds_[i]), "vkCreateSemaphore")) return false;
        if (!Check(vkCreateFence(device_, &fci, nullptr, &fences_[i]), "vkCreateFence")) return false;
        VkFenceCreateInfo unsignalled{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (!Check(vkCreateFence(device_, &unsignalled, nullptr, &acquire_fences_[i]), "vkCreateFence")) return false;
    }
    cmd_ = cmds_[0];
    acquired_ = acquireds_[0];
    fence_ = fences_[0];
    return CreateSwapchain();
}

bool VulkanDisplay::CreateSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surface_, &caps);
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    extent_ = caps.currentExtent.width != 0xFFFFFFFFu ? caps.currentExtent
              : VkExtent2D{std::uint32_t(w), std::uint32_t(h)};
    // A screen turned from its natural way (a phone on its side): the
    // surface's size is the natural way's. The pictures are drawn as the
    // window stands (no pre-transform: the compositor turns them), so the
    // swapchain's images take the window's way round.
    VkSurfaceTransformFlagBitsKHR transform = caps.currentTransform;
    if (transform & (VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR | VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)) {
        std::swap(extent_.width, extent_.height);
    }
    if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    else
        std::swap(extent_.width, extent_.height);    // (the turn can only be the app's: as it was)
#ifdef __ANDROID__
    // The window's way round, whatever the surface said: an app started in
    // portrait and turned to landscape (sensorLandscape) can be asked before
    // the surface has caught up - a picture squeezed into a strip otherwise.
    if (w > 0 && h > 0 && (w > h) != (extent_.width > extent_.height)) std::swap(extent_.width, extent_.height);
    std::fprintf(stderr, "[display] swapchain %ux%u (window %dx%d, surface %ux%u, transform 0x%X)\n", extent_.width,
                 extent_.height, w, h, caps.currentExtent.width, caps.currentExtent.height,
                 unsigned(caps.currentTransform));
#endif
    if (extent_.width == 0 || extent_.height == 0) return true;      // minimised: nothing to show
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        error_ = "the swapchain cannot be a transfer destination";
        return false;
    }

    std::uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surface_, &n, formats.data());
    VkSurfaceFormatKHR chosen = formats.empty() ? VkSurfaceFormatKHR{} : formats[0];
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) chosen = f;
    format_ = chosen.format;

    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu_, surface_, &n, modes.data());
    mode_ = VK_PRESENT_MODE_FIFO_KHR;            // every device has it
    // The last of these the driver has: MAILBOX first (no tearing on a
    // fixed refresh), IMMEDIATE first on a VRR screen (SetVrr).
    if (!vsync_)
        for (VkPresentModeKHR m : vrr_ ? std::initializer_list<VkPresentModeKHR>{VK_PRESENT_MODE_MAILBOX_KHR,
                                                                                VK_PRESENT_MODE_IMMEDIATE_KHR}
                                       : std::initializer_list<VkPresentModeKHR>{VK_PRESENT_MODE_IMMEDIATE_KHR,
                                                                                VK_PRESENT_MODE_MAILBOX_KHR})
            if (std::find(modes.begin(), modes.end(), m) != modes.end()) mode_ = m;
    // (In a player's log when it changes: what a VRR screen was given.)
    static VkPresentModeKHR logged = VK_PRESENT_MODE_MAX_ENUM_KHR;
    if (mode_ != logged) {
        logged = mode_;
        std::fprintf(stderr, "[display] presenting %s (vsync %s, vrr %s)\n",
                     mode_ == VK_PRESENT_MODE_FIFO_KHR      ? "FIFO"
                     : mode_ == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX"
                                                            : "IMMEDIATE",
                     vsync_ ? "on" : "off", vrr_ ? "on" : "off");
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface_;
    sci.minImageCount = std::max(caps.minImageCount, 2u);
    if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount + 1, caps.maxImageCount);
    sci.imageFormat = format_;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent_;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = transform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode_;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain_;
    VkSwapchainLatencyCreateInfoNV latency{VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV};
    latency.latencyModeEnable = VK_TRUE;
    if (low_latency_) sci.pNext = &latency;
    VkSwapchainKHR created = VK_NULL_HANDLE;
    if (!Check(vkCreateSwapchainKHR(device_, &sci, nullptr, &created), "vkCreateSwapchainKHR")) return false;
    DestroySwapchain();
    swapchain_ = created;
    ApplyBoost();
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, images_.data());
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    rendered_.resize(n);
    for (auto& s : rendered_)
        if (!Check(vkCreateSemaphore(device_, &semaphore, nullptr, &s), "vkCreateSemaphore")) return false;
    stale_ = false;
    return true;
}

void VulkanDisplay::DestroySwapchain() {
    if (!device_) return;
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(device_);
    }
    for (VkSemaphore s : rendered_) vkDestroySemaphore(device_, s, nullptr);
    rendered_.clear();
    images_.clear();
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

std::uint32_t VulkanDisplay::FindMemory(std::uint32_t bits, VkMemoryPropertyFlags wanted) const {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_, &mp);
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
    return ~0u;
}

void VulkanDisplay::DestroyPicture() {
    if (!device_) return;
    if (mapped_) vkUnmapMemory(device_, staging_memory_);
    mapped_ = nullptr;
    if (staging_) vkDestroyBuffer(device_, staging_, nullptr);
    if (staging_memory_) vkFreeMemory(device_, staging_memory_, nullptr);
    if (picture_) vkDestroyImage(device_, picture_, nullptr);
    if (picture_memory_) vkFreeMemory(device_, picture_memory_, nullptr);
    staging_ = VK_NULL_HANDLE;
    staging_memory_ = picture_memory_ = VK_NULL_HANDLE;
    picture_ = VK_NULL_HANDLE;
    picture_width_ = picture_height_ = 0;
}

bool VulkanDisplay::EnsurePicture(std::uint32_t width, std::uint32_t height) {
    if (width == picture_width_ && height == picture_height_ && picture_) return true;
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(device_);
    }
    DestroyPicture();
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!Check(vkCreateBuffer(device_, &bci, nullptr, &staging_), "vkCreateBuffer")) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, staging_, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!Check(vkAllocateMemory(device_, &mai, nullptr, &staging_memory_), "vkAllocateMemory (staging)"))
        return false;
    vkBindBufferMemory(device_, staging_, staging_memory_, 0);
    vkMapMemory(device_, staging_memory_, 0, bytes, 0, &mapped_);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!Check(vkCreateImage(device_, &ici, nullptr, &picture_), "vkCreateImage")) return false;
    vkGetImageMemoryRequirements(device_, picture_, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!Check(vkAllocateMemory(device_, &mai, nullptr, &picture_memory_), "vkAllocateMemory (picture)"))
        return false;
    vkBindImageMemory(device_, picture_, picture_memory_, 0);
    picture_width_ = width;
    picture_height_ = height;
    return true;
}

void Barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
             VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags src_stage,
             VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

bool VulkanDisplay::BeginPresent(std::uint32_t* index, bool* ok) {
    *ok = true;
    if (closed_) {
        *ok = false;
        return false;
    }
    // PS2HOST_PRESENT_WAIT_LOG=1: each wait here over 100 ms, which.
    static const bool wait_log = std::getenv("PS2HOST_PRESENT_WAIT_LOG") != nullptr;
    auto stamp = [] { return std::chrono::steady_clock::now(); };
    auto over = [&](const char* what, std::chrono::steady_clock::time_point from) {
        if (!wait_log) return;
        const double ms = std::chrono::duration<double, std::milli>(stamp() - from).count();
        if (ms > 100) std::fprintf(stderr, "[present] %s: %.0f ms%s\n", what, ms, bounded_ ? " (bounded)" : "");
    };
    auto t0 = stamp();
    if (stale_ && !CreateSwapchain()) {
        *ok = false;
        return false;
    }
    if (!swapchain_ || extent_.width == 0 || extent_.height == 0) return false;
    // Presenting from the caller's thread (PresentImage: the GS thread's,
    // unpaced), the waits for a slot and an image are bounded (bounded_):
    // past them the picture is not shown and the caller goes on. Unbounded,
    // a compositor's or a driver's hiccup held the GS thread - the whole
    // game - for 300 ms to seconds (two cores, the integrated GPU; caught
    // by PS2HOST_STALL_WATCH in vkAcquireNextImageKHR's wait).
    const std::uint64_t limit = bounded_ ? 25'000'000ull : UINT64_MAX;
    const int next = (slot_ + 1) % kInFlight;
    over("swapchain made", t0);
    t0 = stamp();
    const bool slot_free = vkWaitForFences(device_, 1, &fences_[next], VK_TRUE, limit) != VK_TIMEOUT;
    over("the slot's last submission", t0);
    if (!slot_free) return false;
    t0 = stamp();
    VkFence acquire_fence = acquire_fences_[next];
    vkResetFences(device_, 1, &acquire_fence);
    VkResult r = vkAcquireNextImageKHR(device_, swapchain_, limit, acquireds_[next], acquire_fence, index);
    over("vkAcquireNextImageKHR", t0);
    if (r == VK_TIMEOUT || r == VK_NOT_READY) {
        ++skipped_;
        return false;               // (nothing acquired: the semaphore and the fence untouched)
    }
    slot_ = next;
    cmd_ = cmds_[slot_];
    acquired_ = acquireds_[slot_];
    fence_ = fences_[slot_];
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        stale_ = true;
        return false;
    }
    if (r != VK_SUBOPTIMAL_KHR && !Check(r, "vkAcquireNextImageKHR")) {
        *ok = false;
        return false;
    }
    t0 = stamp();
    vkWaitForFences(device_, 1, &acquire_fence, VK_TRUE, UINT64_MAX);
    over("the acquired image's fence", t0);
    vkResetFences(device_, 1, &fence_);
    vkResetCommandBuffer(cmd_, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_, &begin);
    return true;
}

void VulkanDisplay::DrawOverlay(VkImage target) {
    std::lock_guard<std::mutex> hold(overlay_lock_);
    if (overlay_.empty() || !overlay_width_ || !overlay_height_) return;
    if (!overlay_buffer_) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = kOverlayBytes * kInFlight;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(device_, &bci, nullptr, &overlay_buffer_) != VK_SUCCESS) return;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, overlay_buffer_, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device_, &mai, nullptr, &overlay_memory_) != VK_SUCCESS) return;
        vkBindBufferMemory(device_, overlay_buffer_, overlay_memory_, 0);
        vkMapMemory(device_, overlay_memory_, 0, kOverlayBytes * kInFlight, 0, &overlay_mapped_);
    }
    // At the top left - a phone's at the picture's own corner, clear of the
    // touch controls' shoulder buttons beside it.
    // (And below the touch controls' button at the top's middle.)
    std::uint32_t x = 16, y = 16;
#ifdef __ANDROID__
    x = std::uint32_t(std::max(0, shown_x0_.load())) + 16;
    y = extent_.height / 9;
#endif
    if (x + 16 >= extent_.width) x = 16;
    if (y + 16 >= extent_.height) y = 16;
    const std::uint32_t w = std::min(overlay_width_, extent_.width > x ? extent_.width - x : 0u);
    const std::uint32_t h = std::min(overlay_height_, extent_.height > y ? extent_.height - y : 0u);
    if (!w || !h || VkDeviceSize(overlay_width_) * overlay_height_ * 4 > kOverlayBytes) return;
    // This slot's copy (BeginPresent waited for its fence: no present reads it).
    const VkDeviceSize offset = kOverlayBytes * VkDeviceSize(slot_);
    if (overlay_slot_generation_[slot_] != overlay_generation_) {
        std::memcpy(static_cast<char*>(overlay_mapped_) + offset, overlay_.data(),
                    std::size_t(overlay_width_) * overlay_height_ * 4);
        overlay_slot_generation_[slot_] = overlay_generation_;
    }
    // After the picture's blit to the same image (its write first).
    Barrier(cmd_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.bufferOffset = offset;
    copy.bufferRowLength = overlay_width_;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {std::int32_t(x), std::int32_t(y), 0};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cmd_, overlay_buffer_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
}

void VulkanDisplay::DestroyPostImages() {
    if (!device_) return;
    if (post_in_view_) vkDestroyImageView(device_, post_in_view_, nullptr);
    if (post_out_view_) vkDestroyImageView(device_, post_out_view_, nullptr);
    if (post_in_) vkDestroyImage(device_, post_in_, nullptr);
    if (post_out_) vkDestroyImage(device_, post_out_, nullptr);
    if (post_in_memory_) vkFreeMemory(device_, post_in_memory_, nullptr);
    if (post_out_memory_) vkFreeMemory(device_, post_out_memory_, nullptr);
    post_in_view_ = post_out_view_ = VK_NULL_HANDLE;
    post_in_ = post_out_ = VK_NULL_HANDLE;
    post_in_memory_ = post_out_memory_ = VK_NULL_HANDLE;
    post_in_width_ = post_in_height_ = post_out_width_ = post_out_height_ = 0;
}

bool VulkanDisplay::EnsurePost(std::uint32_t width, std::uint32_t height, std::uint32_t out_width,
                               std::uint32_t out_height) {
    if (post_failed_) return false;
    if (width == post_in_width_ && height == post_in_height_ && out_width == post_out_width_ &&
        out_height == post_out_height_ && post_out_)
        return true;
    post_failed_ = true;                    // until it is all made
    if (!fxaa_pipeline_) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(vkCreateSampler(device_, &sci, nullptr, &post_sampler_), "vkCreateSampler (FXAA)")) return false;
        VkDescriptorSetLayoutBinding bindings[2] = {};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 2;
        lci.pBindings = bindings;
        if (!Check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &post_set_layout_), "FXAA set layout"))
            return false;
        const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                               {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = sizes;
        if (!Check(vkCreateDescriptorPool(device_, &dpi, nullptr, &post_descriptors_), "FXAA descriptor pool"))
            return false;
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = post_descriptors_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &post_set_layout_;
        if (!Check(vkAllocateDescriptorSets(device_, &dai, &post_set_), "FXAA descriptor set")) return false;
        const VkPushConstantRange size{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &post_set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &size;
        if (!Check(vkCreatePipelineLayout(device_, &pli, nullptr, &post_layout_), "FXAA pipeline layout"))
            return false;
        const struct { const std::uint32_t* code; std::size_t bytes; VkPipeline* pipeline; } shaders[2] = {
            {spirv_fxaa_comp, sizeof spirv_fxaa_comp, &fxaa_pipeline_},
            {spirv_downsample_comp, sizeof spirv_downsample_comp, &downsample_pipeline_},
        };
        for (const auto& shader : shaders) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = shader.bytes;
            smi.pCode = shader.code;
            VkShaderModule module = VK_NULL_HANDLE;
            if (!Check(vkCreateShaderModule(device_, &smi, nullptr, &module), "anti-aliasing shader")) return false;
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = module;
            cpi.stage.pName = "main";
            cpi.layout = post_layout_;
            const VkResult made = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, shader.pipeline);
            vkDestroyShaderModule(device_, module, nullptr);
            if (!Check(made, "anti-aliasing pipeline")) return false;
        }
    }
    // Earlier presents may still read the old images.
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(device_);
    }
    DestroyPostImages();
    struct Made { VkImage* image; VkDeviceMemory* memory; VkImageView* view; VkImageUsageFlags usage;
                  std::uint32_t width, height; };
    const Made made[2] = {
        {&post_in_, &post_in_memory_, &post_in_view_, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
         width, height},
        {&post_out_, &post_out_memory_, &post_out_view_, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         out_width, out_height},
    };
    for (const Made& m : made) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {m.width, m.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = m.usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!Check(vkCreateImage(device_, &ici, nullptr, m.image), "vkCreateImage (FXAA)")) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, *m.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!Check(vkAllocateMemory(device_, &mai, nullptr, m.memory), "vkAllocateMemory (FXAA)")) return false;
        vkBindImageMemory(device_, *m.image, *m.memory, 0);
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = *m.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!Check(vkCreateImageView(device_, &vci, nullptr, m.view), "vkCreateImageView (FXAA)")) return false;
    }
    const VkDescriptorImageInfo in{post_sampler_, post_in_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo out{VK_NULL_HANDLE, post_out_view_, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    writes[0].dstSet = writes[1].dstSet = post_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &in;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &out;
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    post_in_width_ = width;
    post_in_height_ = height;
    post_out_width_ = out_width;
    post_out_height_ = out_height;
    post_failed_ = false;
    return true;
}

void VulkanDisplay::RecordPost(VkImage source, std::uint32_t width, std::uint32_t height, VkPipeline pipeline) {
    // The picture as RGBA (a blit converts the display's BGRA), to be sampled;
    // the last present's reads of both images done before they are written.
    Barrier(cmd_, post_in_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
    region.dstOffsets[1] = region.srcOffsets[1];
    vkCmdBlitImage(cmd_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, post_in_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_NEAREST);
    Barrier(cmd_, post_in_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    Barrier(cmd_, post_out_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, post_layout_, 0, 1, &post_set_, 0, nullptr);
    struct { float texel[2]; std::int32_t size[2]; } size{{1.0f / float(width), 1.0f / float(height)},
                                                         {std::int32_t(post_out_width_), std::int32_t(post_out_height_)}};
    vkCmdPushConstants(cmd_, post_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof size, &size);
    vkCmdDispatch(cmd_, (post_out_width_ + 7) / 8, (post_out_height_ + 7) / 8, 1);
    Barrier(cmd_, post_out_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

bool VulkanDisplay::MakeSmaaImage(int which, VkFormat format, std::uint32_t width, std::uint32_t height,
                                  VkImageUsageFlags usage) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!Check(vkCreateImage(device_, &ici, nullptr, &smaa_image_[which]), "vkCreateImage (SMAA)")) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device_, smaa_image_[which], &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!Check(vkAllocateMemory(device_, &mai, nullptr, &smaa_memory_[which]), "vkAllocateMemory (SMAA)")) return false;
    vkBindImageMemory(device_, smaa_image_[which], smaa_memory_[which], 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = smaa_image_[which];
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return Check(vkCreateImageView(device_, &vci, nullptr, &smaa_view_[which]), "vkCreateImageView (SMAA)");
}

void VulkanDisplay::DestroySmaa() {
    if (!device_) return;
    for (int i = 0; i < kSmaaImages; ++i) {
        if (smaa_view_[i]) vkDestroyImageView(device_, smaa_view_[i], nullptr);
        if (smaa_image_[i]) vkDestroyImage(device_, smaa_image_[i], nullptr);
        if (smaa_memory_[i]) vkFreeMemory(device_, smaa_memory_[i], nullptr);
        smaa_view_[i] = VK_NULL_HANDLE;
        smaa_image_[i] = VK_NULL_HANDLE;
        smaa_memory_[i] = VK_NULL_HANDLE;
    }
    for (VkPipeline& p : smaa_pipelines_)
        if (p) vkDestroyPipeline(device_, p, nullptr), p = VK_NULL_HANDLE;
    if (smaa_layout_) vkDestroyPipelineLayout(device_, smaa_layout_, nullptr);
    if (smaa_descriptors_) vkDestroyDescriptorPool(device_, smaa_descriptors_, nullptr);
    if (smaa_set_layout_) vkDestroyDescriptorSetLayout(device_, smaa_set_layout_, nullptr);
    if (smaa_point_) vkDestroySampler(device_, smaa_point_, nullptr);
    if (smaa_staging_) vkDestroyBuffer(device_, smaa_staging_, nullptr);
    if (smaa_staging_memory_) vkFreeMemory(device_, smaa_staging_memory_, nullptr);
    smaa_layout_ = VK_NULL_HANDLE;
    smaa_descriptors_ = VK_NULL_HANDLE;
    smaa_set_layout_ = VK_NULL_HANDLE;
    smaa_point_ = VK_NULL_HANDLE;
    smaa_staging_ = VK_NULL_HANDLE;
    smaa_staging_memory_ = VK_NULL_HANDLE;
}

bool VulkanDisplay::EnsureSmaa(std::uint32_t width, std::uint32_t height) {
    if (smaa_failed_) return false;
    if (width == smaa_width_ && height == smaa_height_ && smaa_for_in_ == post_in_view_ && smaa_image_[kSmaaEdges])
        return true;
    smaa_failed_ = true;                    // until it is all made
    if (!smaa_pipelines_[0]) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (!Check(vkCreateSampler(device_, &sci, nullptr, &smaa_point_), "vkCreateSampler (SMAA)")) return false;
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (std::uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = 4;
        lci.pBindings = bindings;
        if (!Check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &smaa_set_layout_), "SMAA set layout"))
            return false;
        const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9},
                                               {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3}};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 3;
        dpi.poolSizeCount = 2;
        dpi.pPoolSizes = sizes;
        if (!Check(vkCreateDescriptorPool(device_, &dpi, nullptr, &smaa_descriptors_), "SMAA descriptor pool"))
            return false;
        const VkDescriptorSetLayout layouts[3] = {smaa_set_layout_, smaa_set_layout_, smaa_set_layout_};
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = smaa_descriptors_;
        dai.descriptorSetCount = 3;
        dai.pSetLayouts = layouts;
        if (!Check(vkAllocateDescriptorSets(device_, &dai, smaa_sets_), "SMAA descriptor sets")) return false;
        const VkPushConstantRange metrics{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &smaa_set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &metrics;
        if (!Check(vkCreatePipelineLayout(device_, &pli, nullptr, &smaa_layout_), "SMAA pipeline layout")) return false;
        const struct { const std::uint32_t* code; std::size_t bytes; } shaders[3] = {
            {spirv_smaa_edges_comp, sizeof spirv_smaa_edges_comp},
            {spirv_smaa_weights_comp, sizeof spirv_smaa_weights_comp},
            {spirv_smaa_blend_comp, sizeof spirv_smaa_blend_comp},
        };
        for (int i = 0; i < 3; ++i) {
            VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smi.codeSize = shaders[i].bytes;
            smi.pCode = shaders[i].code;
            VkShaderModule module = VK_NULL_HANDLE;
            if (!Check(vkCreateShaderModule(device_, &smi, nullptr, &module), "SMAA shader")) return false;
            VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpi.stage.module = module;
            cpi.stage.pName = "main";
            cpi.layout = smaa_layout_;
            const VkResult made = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &smaa_pipelines_[i]);
            vkDestroyShaderModule(device_, module, nullptr);
            if (!Check(made, "SMAA pipeline")) return false;
        }
        // The tables: their images, and their bytes in a staging buffer
        // (RecordSmaa copies them over the first time).
        const VkImageUsageFlags table = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (!MakeSmaaImage(kSmaaArea, VK_FORMAT_R8G8_UNORM, AREATEX_WIDTH, AREATEX_HEIGHT, table) ||
            !MakeSmaaImage(kSmaaSearch, VK_FORMAT_R8_UNORM, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, table))
            return false;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sizeof areaTexBytes + sizeof searchTexBytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!Check(vkCreateBuffer(device_, &bci, nullptr, &smaa_staging_), "SMAA staging buffer")) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device_, smaa_staging_, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!Check(vkAllocateMemory(device_, &mai, nullptr, &smaa_staging_memory_), "SMAA staging memory")) return false;
        vkBindBufferMemory(device_, smaa_staging_, smaa_staging_memory_, 0);
        void* mapped = nullptr;
        if (!Check(vkMapMemory(device_, smaa_staging_memory_, 0, bci.size, 0, &mapped), "SMAA staging map")) return false;
        std::memcpy(mapped, areaTexBytes, sizeof areaTexBytes);
        std::memcpy(static_cast<char*>(mapped) + sizeof areaTexBytes, searchTexBytes, sizeof searchTexBytes);
        vkUnmapMemory(device_, smaa_staging_memory_);
    }
    // Earlier presents may still read the old images.
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(device_);
    }
    for (int which : {int(kSmaaEdges), int(kSmaaWeights)}) {
        if (smaa_view_[which]) vkDestroyImageView(device_, smaa_view_[which], nullptr);
        if (smaa_image_[which]) vkDestroyImage(device_, smaa_image_[which], nullptr);
        if (smaa_memory_[which]) vkFreeMemory(device_, smaa_memory_[which], nullptr);
        smaa_view_[which] = VK_NULL_HANDLE;
        smaa_image_[which] = VK_NULL_HANDLE;
        smaa_memory_[which] = VK_NULL_HANDLE;
        if (!MakeSmaaImage(which, VK_FORMAT_R8G8B8A8_UNORM, width, height,
                           VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
            return false;
    }
    // Each pass's set: its inputs (sampled; one unused stands for the
    // others) and its output.
    const VkImageLayout read = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkDescriptorImageInfo colour{post_sampler_, post_in_view_, read};
    const VkDescriptorImageInfo edges{post_sampler_, smaa_view_[kSmaaEdges], read};
    const VkDescriptorImageInfo weights{post_sampler_, smaa_view_[kSmaaWeights], read};
    const VkDescriptorImageInfo area{post_sampler_, smaa_view_[kSmaaArea], read};
    const VkDescriptorImageInfo search{smaa_point_, smaa_view_[kSmaaSearch], read};
    const VkDescriptorImageInfo out_edges{VK_NULL_HANDLE, smaa_view_[kSmaaEdges], VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo out_weights{VK_NULL_HANDLE, smaa_view_[kSmaaWeights], VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo out_picture{VK_NULL_HANDLE, post_out_view_, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo* inputs[3][4] = {{&colour, &colour, &colour, &out_edges},
                                                 {&edges, &area, &search, &out_weights},
                                                 {&colour, &weights, &colour, &out_picture}};
    VkWriteDescriptorSet writes[12] = {};
    for (int set = 0; set < 3; ++set)
        for (std::uint32_t b = 0; b < 4; ++b) {
            VkWriteDescriptorSet& w = writes[set * 4 + b];
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = smaa_sets_[set];
            w.dstBinding = b;
            w.descriptorCount = 1;
            w.descriptorType = b < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w.pImageInfo = inputs[set][b];
        }
    vkUpdateDescriptorSets(device_, 12, writes, 0, nullptr);
    smaa_width_ = width;
    smaa_height_ = height;
    smaa_for_in_ = post_in_view_;
    smaa_failed_ = false;
    return true;
}

void VulkanDisplay::RecordSmaa(VkImage source, std::uint32_t width, std::uint32_t height) {
    // The picture as RGBA, to be sampled (as RecordPost).
    Barrier(cmd_, post_in_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
    region.dstOffsets[1] = region.srcOffsets[1];
    vkCmdBlitImage(cmd_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, post_in_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_NEAREST);
    Barrier(cmd_, post_in_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!smaa_tables_sent_) {
        const struct { int which; VkDeviceSize offset; std::uint32_t w, h; } tables[2] = {
            {kSmaaArea, 0, AREATEX_WIDTH, AREATEX_HEIGHT},
            {kSmaaSearch, sizeof areaTexBytes, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT}};
        for (const auto& t : tables) {
            Barrier(cmd_, smaa_image_[t.which], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy copy{};
            copy.bufferOffset = t.offset;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {t.w, t.h, 1};
            vkCmdCopyBufferToImage(cmd_, smaa_staging_, smaa_image_[t.which], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &copy);
            Barrier(cmd_, smaa_image_[t.which], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }
        smaa_tables_sent_ = true;
    }
    const float metrics[4] = {1.0f / float(width), 1.0f / float(height), float(width), float(height)};
    const VkImage outputs[3] = {smaa_image_[kSmaaEdges], smaa_image_[kSmaaWeights], post_out_};
    for (int pass = 0; pass < 3; ++pass) {
        Barrier(cmd_, outputs[pass], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, smaa_pipelines_[pass]);
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, smaa_layout_, 0, 1, &smaa_sets_[pass], 0, nullptr);
        vkCmdPushConstants(cmd_, smaa_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof metrics, metrics);
        vkCmdDispatch(cmd_, (width + 7) / 8, (height + 7) / 8, 1);
        if (pass < 2)                       // read by the next pass
            Barrier(cmd_, outputs[pass], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    Barrier(cmd_, post_out_, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
}

bool VulkanDisplay::FinishPresent(std::uint32_t index, VkImage source, std::uint32_t width, std::uint32_t height) {
    const VkImage target = images_[index];
    Barrier(cmd_, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    const VkClearColorValue black{};
    const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    // The clear done before the picture is blitted over it: two writes to the
    // same image, unordered without this (AMD's RDNA 3 finished the clear
    // late, over the picture in a staircase of black tiles - issue #28;
    // anti-aliasing's passes between happened to order them).
    Barrier(cmd_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    // 4:3 as a television shows it, 16:9, or the window's own shape (SetAspect),
    // centred in the window.
    std::int32_t w = std::int32_t(extent_.width), h = std::int32_t(extent_.height);
    std::int32_t dw = w, dh = h;
    const float aspect = aspect_.load();
    if (aspect > 0.0f) {
        if (float(w) > float(h) * aspect) dw = std::int32_t(float(h) * aspect + 0.5f);
        else dh = std::int32_t(float(w) / aspect + 0.5f);
    }
    const std::int32_t x0 = (w - dw) / 2, y0 = (h - dh) / 2;
    shown_x0_.store(x0);
    shown_y0_.store(y0);
    shown_w_.store(std::max(dw, 1));
    shown_h_.store(std::max(dh, 1));
    // Anti-aliasing: FXAA at the picture's size; supersampling's picture,
    // larger than shown, averaged down to the shown size (then copied 1:1).
    const AntiAliasing mode = AntiAliasing(anti_aliasing_.load());
    const bool larger = width > std::uint32_t(dw) || height > std::uint32_t(dh);
    if (mode == AntiAliasing::kFxaa && EnsurePost(width, height, width, height)) {
        RecordPost(source, width, height, fxaa_pipeline_);
        source = post_out_;
    } else if (mode == AntiAliasing::kSmaa && EnsurePost(width, height, width, height) && EnsureSmaa(width, height)) {
        RecordSmaa(source, width, height);
        source = post_out_;
    } else if (mode == AntiAliasing::kDownsample && larger && dw > 0 && dh > 0 &&
               EnsurePost(width, height, std::uint32_t(dw), std::uint32_t(dh))) {
        RecordPost(source, width, height, downsample_pipeline_);
        source = post_out_;
        width = std::uint32_t(dw);
        height = std::uint32_t(dh);
    }
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {x0, y0, 0};
    blit.dstOffsets[1] = {x0 + dw, y0 + dh, 1};
    vkCmdBlitImage(cmd_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    DrawOverlay(target);
    Barrier(cmd_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    return true;
}

bool VulkanDisplay::Present(const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) {
    if (!width || !height) return !closed_;
    std::lock_guard<std::mutex> hold(present_lock_);
    std::uint32_t index = 0;
    bool ok = true;
    if (!BeginPresent(&index, &ok)) return ok;
    if (!EnsurePicture(width, height)) {
        vkEndCommandBuffer(cmd_);
        return false;
    }
    // One staging buffer and picture for all: what an earlier present still
    // copies from goes first.
    for (int i = 0; i < kInFlight; ++i)
        if (i != slot_) vkWaitForFences(device_, 1, &fences_[i], VK_TRUE, UINT64_MAX);
    std::memcpy(mapped_, pixels, std::size_t(width) * height * 4);
    Barrier(cmd_, picture_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd_, staging_, picture_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    Barrier(cmd_, picture_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    FinishPresent(index, picture_, width, height);
    return Submit(index);
}

bool VulkanDisplay::PresentImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) {
    if (!width || !height || !image) return !closed_;
    std::lock_guard<std::mutex> hold(present_lock_);
    std::uint32_t index = 0;
    bool ok = true;
    bounded_ = true;
    const bool began = BeginPresent(&index, &ok);
    bounded_ = false;
    if (!began) return ok;
    const VkImage source = reinterpret_cast<VkImage>(image);
    const VkImageLayout kept = VkImageLayout(layout);
    // Whatever drew it was submitted before on this queue: a barrier from
    // all of it, and back to the layout its renderer has it in.
    Barrier(cmd_, source, kept, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    FinishPresent(index, source, width, height);
    Barrier(cmd_, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, kept, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    return Submit(index);
}

bool VulkanDisplay::EnsurePaced(std::uint32_t width, std::uint32_t height) {
    if (width == paced_width_ && height == paced_height_ && paced_[0].image) return true;
    {
        std::unique_lock<std::mutex> hold(paced_lock_);
        ready_.clear();
        paced_wake_.wait(hold, [&] { return taken_ < 0; });   // none being shown
    }
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(device_);
    }
    DestroyPaced();
    if (!copy_pool_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family_;
        if (!Check(vkCreateCommandPool(device_, &pci, nullptr, &copy_pool_), "vkCreateCommandPool")) return false;
    }
    for (PacedPicture& p : paced_) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_B8G8R8A8_UNORM;
        ici.extent = {width, height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!Check(vkCreateImage(device_, &ici, nullptr, &p.image), "vkCreateImage (paced)")) return false;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(device_, p.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!Check(vkAllocateMemory(device_, &mai, nullptr, &p.memory), "vkAllocateMemory (paced)")) return false;
        vkBindImageMemory(device_, p.image, p.memory, 0);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = copy_pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (!Check(vkAllocateCommandBuffers(device_, &cai, &p.copy), "vkAllocateCommandBuffers (paced)")) return false;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (!Check(vkCreateFence(device_, &fci, nullptr, &p.copied), "vkCreateFence (paced)")) return false;
    }
    paced_width_ = width;
    paced_height_ = height;
    if (!present_thread_.joinable()) present_thread_ = std::thread([this] { PresentLoop(); });
    return true;
}

void VulkanDisplay::DestroyPaced() {
    if (!device_) return;
    for (PacedPicture& p : paced_) {
        if (p.copied) vkDestroyFence(device_, p.copied, nullptr);
        if (p.copy) vkFreeCommandBuffers(device_, copy_pool_, 1, &p.copy);
        if (p.image) vkDestroyImage(device_, p.image, nullptr);
        if (p.memory) vkFreeMemory(device_, p.memory, nullptr);
        p = PacedPicture{};
    }
    paced_width_ = paced_height_ = 0;
}

/* PresentImageAt and HoldImage: the image into a slot (-1: none to copy,
   -2: failed or closed); queued with its time, or held. */
int VulkanDisplay::CopyPaced(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                             std::uint64_t due_ns, bool held) {
    if (!width || !height || !image) return closed_ ? -2 : -1;
    if (closed_) return -2;
    if (!EnsurePaced(width, height)) return -2;
    // A picture neither waiting to be shown nor being shown - or, with all
    // of them waiting, the oldest waiting (its turn is lost: the shown
    // picture keeps no more than two frames behind). What the GPU still
    // reads of it (an earlier present's blit, submitted before) goes first,
    // by the queue's order and the copy's barrier.
    int slot = -1;
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        for (int i = 0; i < kPaced && slot < 0; ++i)
            if (i != taken_ && i != held_ && std::find(ready_.begin(), ready_.end(), i) == ready_.end()) slot = i;
        if (slot < 0) {
            slot = ready_.front();
            ready_.pop_front();
            if (observer_) observer_(paced_[slot].due_ns, 0, true);     // its turn lost
        }
    }
    PacedPicture& p = paced_[slot];
    vkWaitForFences(device_, 1, &p.copied, VK_TRUE, UINT64_MAX);    // its command buffer, from last time
    vkResetFences(device_, 1, &p.copied);
    vkResetCommandBuffer(p.copy, 0);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(p.copy, &begin);
    const VkImage source = reinterpret_cast<VkImage>(image);
    const VkImageLayout kept = VkImageLayout(layout);
    Barrier(p.copy, source, kept, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    Barrier(p.copy, p.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    // A blit, not a copy: the renderer's targets are RGBA and these BGRA (a
    // copy takes the bytes as they are - red and blue swapped).
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = region.srcSubresource;
    region.srcOffsets[1] = {std::int32_t(width), std::int32_t(height), 1};
    region.dstOffsets[1] = region.srcOffsets[1];
    vkCmdBlitImage(p.copy, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, p.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &region, VK_FILTER_NEAREST);
    Barrier(p.copy, p.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    Barrier(p.copy, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, kept, VK_ACCESS_TRANSFER_READ_BIT,
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    vkEndCommandBuffer(p.copy);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &p.copy;
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        if (!Check(vkQueueSubmit(queue_, 1, &submit, p.copied), "vkQueueSubmit (paced)")) return -2;
    }
    p.width = width;
    p.height = height;
    p.due_ns = due_ns;
    if (held) {
        std::lock_guard<std::mutex> hold(paced_lock_);
        held_ = slot;
        return slot;
    }
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        ready_.push_back(slot);
    }
    paced_wake_.notify_all();
    return closed_ ? -2 : slot;
}

bool VulkanDisplay::PresentImageAt(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height,
                                   std::uint64_t due_ns) {
    return CopyPaced(image, layout, width, height, due_ns, false) != -2;
}

int VulkanDisplay::HoldImage(std::uint64_t image, int layout, std::uint32_t width, std::uint32_t height) {
    const int slot = CopyPaced(image, layout, width, height, 0, true);
    return slot >= 0 ? slot : -1;
}

void VulkanDisplay::ShowHeld(int slot, std::uint64_t due_ns) {
    if (slot < 0 || slot >= kPaced) return;
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        if (held_ != slot) return;
        held_ = -1;
        paced_[slot].due_ns = due_ns;
        ready_.push_back(slot);
    }
    paced_wake_.notify_all();
}

namespace {
/* Asleep until `when` to a tenth of a millisecond (a high-resolution
   waitable timer), sleep_until where there is none. */
void SleepUntil(std::chrono::steady_clock::time_point when) {
#ifdef _WIN32
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                                       TIMER_ALL_ACCESS);
    const auto left = when - std::chrono::steady_clock::now();
    if (left <= std::chrono::steady_clock::duration::zero()) return;
    if (timer) {
        LARGE_INTEGER due;
        due.QuadPart = -std::max<long long>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() / 100);
        if (SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0)) {
            WaitForSingleObject(timer, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_until(when);
}

/* Whether a deadline may be waited for by spinning its last stretch (a
   sleep's wake-up can be late by a little): only with cores to spare - more
   than three for the process. With fewer, a time-critical thread spinning
   takes a core the game's threads need, the game falls behind, and the
   timer, retrying the blank it has not taken, spins the more: on two cores
   the race lost a third of its blanks and stopped for seconds at a time.
   There the high-resolution timer's sleep is the whole wait.
   PS2HOST_SPIN=0 or 1 decides instead. Looked at again now and then (the
   process's cores can change). */
bool SpareCores() {
    static const char* forced = std::getenv("PS2HOST_SPIN");
    if (forced) return forced[0] == '1';
    static std::atomic<int> looks{0};
    static std::atomic<bool> spare{true};
    if (looks.fetch_add(1, std::memory_order_relaxed) % 64 == 0) {
        int cores = int(std::thread::hardware_concurrency());
#ifdef _WIN32
        DWORD_PTR process = 0, system = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process) {
            cores = 0;
            for (DWORD_PTR m = process; m; m &= m - 1) ++cores;
        }
#endif
        spare.store(cores >= 4, std::memory_order_relaxed);
    }
    return spare.load(std::memory_order_relaxed);
}
}  // namespace

void VulkanDisplay::PresentLoop() {
    using Clock = std::chrono::steady_clock;
#ifdef _WIN32
    // Each picture shown at its time with every core busy: the highest priority
    // (asleep almost always).
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
    for (;;) {
        int slot;
        {
            std::unique_lock<std::mutex> hold(paced_lock_);
            paced_wake_.wait(hold, [&] { return stop_ || !ready_.empty(); });
            if (stop_) return;
            slot = ready_.front();
        }
        // The swapchain's next image, and the last present's command buffer
        // done, ahead of the picture's time: what may wait (the compositor
        // holding the images, a 144 Hz desktop's) waits now, not when it is
        // due - then only the copy and the present are left.
        std::unique_lock<std::mutex> presenting(present_lock_);
        std::uint32_t index = 0;
        bool ok = true;
        const bool began = BeginPresent(&index, &ok);
        // In order, each when it is due.
        const Clock::time_point due = std::min(Clock::time_point{std::chrono::nanoseconds(paced_[slot].due_ns)},
                                               Clock::now() + std::chrono::milliseconds(50));
        if (!SpareCores()) {
            SleepUntil(due);
        } else {
            if (Clock::now() + std::chrono::microseconds(1500) < due) SleepUntil(due - std::chrono::microseconds(700));
            while (Clock::now() < due) std::this_thread::yield();
        }
        {
            std::lock_guard<std::mutex> hold(paced_lock_);
            if (stop_) {
                if (began) vkEndCommandBuffer(cmd_);
                return;
            }
            // The oldest waiting now (a newer one may have taken its turn
            // meanwhile): into the image already taken.
            if (ready_.empty()) {
                if (began) vkEndCommandBuffer(cmd_);
                continue;
            }
            slot = ready_.front();
            ready_.pop_front();
            taken_ = slot;
        }
        {
            if (began) {
                FinishPresent(index, paced_[slot].image, paced_[slot].width, paced_[slot].height);
                Submit(index);
                PresentObserver observer;
                {
                    std::lock_guard<std::mutex> hold_paced(paced_lock_);
                    observer = observer_;
                }
                if (observer)
                    observer(paced_[slot].due_ns, std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                   Clock::now().time_since_epoch()).count()), false);
            }
        }
        presenting.unlock();
        {
            std::lock_guard<std::mutex> hold(paced_lock_);
            taken_ = -1;
        }
        paced_wake_.notify_all();
    }
}

bool VulkanDisplay::Submit(std::uint32_t index) {
    vkEndCommandBuffer(cmd_);
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquired_;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &rendered_[index];
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &rendered_[index];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain_;
    present.pImageIndices = &index;
    VkResult r;
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        if (!Check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit")) return false;
        r = vkQueuePresentKHR(queue_, &present);
        // The GPU done with it: an empty submit's fence (after the copy, and
        // all drawn before on this queue), for SetGpuDoneObserver.
        const std::uint64_t tag = present_tag_.exchange(0);
        if (done_watch_ && tag) {
            VkFence watch = VK_NULL_HANDLE;
            bool room = false;
            {
                std::lock_guard<std::mutex> hold(done_watch_->lock);
                if (!done_watch_->spare.empty()) {
                    watch = done_watch_->spare.back();
                    done_watch_->spare.pop_back();
                }
                room = done_watch_->waiting.size() < 16;
            }
            if (!watch && room) {
                VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                vkCreateFence(device_, &fci, nullptr, &watch);
            }
            if (watch) {
                if (vkQueueSubmit(queue_, 0, nullptr, watch) == VK_SUCCESS) {
                    std::lock_guard<std::mutex> hold(done_watch_->lock);
                    done_watch_->waiting.emplace_back(watch, tag);
                    done_watch_->wake.notify_one();
                } else {
                    vkDestroyFence(device_, watch, nullptr);
                }
            }
        }
    }
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> hold(pacing_lock_);
        if (last_present_.time_since_epoch().count()) {
            const double ms = std::chrono::duration<double, std::milli>(now - last_present_).count();
            ++presents_;
            interval_sum_ += ms;
            interval_squares_ += ms * ms;
            interval_longest_ = std::max(interval_longest_, ms);
            static const bool hitch = std::getenv("PS2HOST_HITCH") != nullptr;
            if (hitch && ms > 25.0)
                std::fprintf(stderr, "[hitch] present: %.1f ms since the last\n", ms);
        }
        last_present_ = now;
        // The FPS counter's: presents counted over half a second at a time.
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) ++fps_presents_;
        if (shown_observer_ && (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR))
            shown_observer_(std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              now.time_since_epoch()).count()));
        const double counted = std::chrono::duration<double>(now - fps_since_).count();
        if (counted >= 0.5) {
            fps_.store(fps_since_.time_since_epoch().count() ? fps_presents_ / counted : 0.0);
            fps_presents_ = 0;
            fps_since_ = now;
        }
    }
#ifdef __ANDROID__
    // Android says SUBOPTIMAL for every picture of a turned screen not
    // pre-rotated (the compositor turns it): shown, and not a reason to make
    // the swapchain again.
    const bool remake = r == VK_ERROR_OUT_OF_DATE_KHR;
    if (r == VK_SUBOPTIMAL_KHR) return true;
#else
    const bool remake = r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR;
#endif
    if (remake) stale_ = true;
    else if (!Check(r, "vkQueuePresentKHR")) return false;
    return true;
}

Display::Pacing VulkanDisplay::TakePacing() {
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

void VulkanDisplay::SetVsync(bool on) {
    if (vsync_ == on) return;
    vsync_ = on;
    stale_ = true;                              // the swapchain again, with the other mode
}

/* The boost as wanted, on the swapchain (made again with each). Low latency
   mode itself on with it - the boost asks for it - but its waits
   (vkLatencySleepNV) never made: the pictures go as before. */
void VulkanDisplay::ApplyBoost() {
    if (!low_latency_ || !swapchain_) return;
    VkLatencySleepModeInfoNV mode{VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV};
    mode.lowLatencyMode = boost_ ? VK_TRUE : VK_FALSE;
    mode.lowLatencyBoost = boost_ ? VK_TRUE : VK_FALSE;
    mode.minimumIntervalUs = 0;
    const VkResult r = set_sleep_mode_(device_, swapchain_, &mode);
    boosted_ = boost_ && r == VK_SUCCESS;
    static int said = -1;
    if (said != int(boost_.load())) {
        said = int(boost_.load());
        std::fprintf(stderr, "[display] the GPU's low-latency boost %s%s\n", boost_ ? "on" : "off",
                     r == VK_SUCCESS ? "" : " (the driver refused)");
    }
}

void VulkanDisplay::SetGpuBoost(bool on) {
    if (boost_ == on) return;
    boost_ = on;
    std::lock_guard<std::mutex> hold(present_lock_);
    ApplyBoost();
}

void VulkanDisplay::SetVrr(bool on) {
    if (vrr_ == on) return;
    vrr_ = on;
    if (!vsync_) stale_ = true;                 // (vertical sync's FIFO either way)
}

void VulkanDisplay::SetGpuDoneObserver(GpuDoneObserver observer) {
    if (done_watch_ || !observer) return;
    done_watch_ = std::make_unique<DoneWatch>();
    done_watch_->observer = std::move(observer);
    done_watch_->thread = std::thread([this] { WatchDone(); });
}

void VulkanDisplay::WatchDone() {
    DoneWatch& w = *done_watch_;
    while (true) {
        std::pair<VkFence, std::uint64_t> next;
        {
            std::unique_lock<std::mutex> hold(w.lock);
            w.wake.wait(hold, [&] { return w.stop || !w.waiting.empty(); });
            if (w.waiting.empty()) return;
            next = w.waiting.front();
        }
        // (A tenth of a second at a time: stopped meanwhile, it goes.)
        VkResult r = VK_TIMEOUT;
        while ((r = vkWaitForFences(device_, 1, &next.first, VK_TRUE, 100'000'000)) == VK_TIMEOUT) {
            std::lock_guard<std::mutex> hold(w.lock);
            if (w.stop) return;
        }
        const std::uint64_t done = std::uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        vkResetFences(device_, 1, &next.first);
        {
            std::lock_guard<std::mutex> hold(w.lock);
            w.waiting.erase(w.waiting.begin());
            w.spare.push_back(next.first);
        }
        if (r == VK_SUCCESS) w.observer(next.second, done);
    }
}

void VulkanDisplay::StopDoneWatch() {
    if (!done_watch_) return;
    {
        std::lock_guard<std::mutex> hold(done_watch_->lock);
        done_watch_->stop = true;
    }
    done_watch_->wake.notify_all();
    if (done_watch_->thread.joinable()) done_watch_->thread.join();
    for (auto& [fence, tag] : done_watch_->waiting) vkDestroyFence(device_, fence, nullptr);
    for (VkFence fence : done_watch_->spare) vkDestroyFence(device_, fence, nullptr);
    done_watch_.reset();
}

VulkanDisplay::~VulkanDisplay() {
    {
        std::lock_guard<std::mutex> hold(paced_lock_);
        stop_ = true;
    }
    paced_wake_.notify_all();
    if (present_thread_.joinable()) present_thread_.join();
    shared_.reset();
    if (device_) vkDeviceWaitIdle(device_);
    StopDoneWatch();
    DestroyPaced();
    if (copy_pool_) vkDestroyCommandPool(device_, copy_pool_, nullptr);
    DestroyPicture();
    DestroyPostImages();
    DestroySmaa();
    if (fxaa_pipeline_) vkDestroyPipeline(device_, fxaa_pipeline_, nullptr);
    if (downsample_pipeline_) vkDestroyPipeline(device_, downsample_pipeline_, nullptr);
    if (post_layout_) vkDestroyPipelineLayout(device_, post_layout_, nullptr);
    if (post_descriptors_) vkDestroyDescriptorPool(device_, post_descriptors_, nullptr);
    if (post_set_layout_) vkDestroyDescriptorSetLayout(device_, post_set_layout_, nullptr);
    if (post_sampler_) vkDestroySampler(device_, post_sampler_, nullptr);
    if (overlay_buffer_) vkDestroyBuffer(device_, overlay_buffer_, nullptr);
    if (overlay_memory_) vkFreeMemory(device_, overlay_memory_, nullptr);
    DestroySwapchain();
    if (device_) {
        for (int i = 0; i < kInFlight; ++i) {
            if (fences_[i]) vkDestroyFence(device_, fences_[i], nullptr);
            if (acquire_fences_[i]) vkDestroyFence(device_, acquire_fences_[i], nullptr);
            if (acquireds_[i]) vkDestroySemaphore(device_, acquireds_[i], nullptr);
        }
        if (pool_) vkDestroyCommandPool(device_, pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

}  // namespace

std::unique_ptr<Display> CreateVulkanDisplay(const char* title, int width, int height, std::string* error) {
    auto display = std::make_unique<VulkanDisplay>();
    if (!display->Init(title, width, height)) {
        if (error) *error = display->error();
        return nullptr;
    }
    return display;
}

}  // namespace ps2gpu
