/* The Vulkan objects behind ps2gpu::Gpu, for the library's own sources. */

#ifndef PS2GPU_GPU_INTERNAL_H
#define PS2GPU_GPU_INTERNAL_H

#include <volk.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include "ps2gpu/gs_renderer.h"

namespace ps2gpu {

/* SubmitStats: submissions, the time spent waiting for the GPU, and what
   waited (a read-back of which kind, a resize, a full ring). */
inline std::uint64_t g_submits = 0;
/* The one queue the renderer and the display share: Vulkan wants its
   submissions, presents and the device's wait-idle from one thread at a
   time, and the display presents from a thread of its own. */
inline std::mutex g_queue_lock;
inline double g_wait_seconds = 0;
inline GpuWaitObserver g_wait_observer = nullptr;      // SetGpuWaitObserver
/* The GPU's time on the command buffers handed over, in all (timestamps at
   each one's start and end; GpuBusyNanoseconds). */
inline std::atomic<std::uint64_t> g_gpu_ns{0};
/* PS2HOST_GPU_PROFILE: the GPU's time by what it did (GpuProfileReport). */
enum ProfileKind { kProfOther, kProfDraw, kProfSwizzle, kProfLinear, kProfMemory, kProfFeedback, kProfDecode,
                   kProfUpload, kProfClear, kProfDate, kProfReadback, kProfEffects, kProfKinds };
inline std::atomic<std::uint64_t> g_profile_ns[kProfKinds] = {};
/* PS2HOST_GPU_PROFILE: the draws' time by their tag too (MarkDraw: the
   renderer's - its target and what kind of draw), for the report's list of
   the dearest. Only where the slots are retired, under the lock. */
inline std::mutex g_draw_ns_lock;
inline std::unordered_map<std::uint32_t, std::uint64_t> g_draw_ns;
struct WaitReason { const char* why; std::uint64_t count; double seconds; };
inline WaitReason g_waits[16] = {};
/* The device takes VK_ATTACHMENT_LOAD_OP_NONE (VK_EXT/KHR_load_store_op_none,
   enabled by the Display where offered): an attachment a pass never reads. */
inline bool g_load_op_none = false;

/* Which Vulkan the device draws with (vk_device.cpp): 1.3; 1.2 with
   VK_KHR_dynamic_rendering; 1.2 with render passes (the renderer's BeginPass).
   PS2HOST_VK_LEGACY=1|2 takes the second or third on a 1.3 GPU. */
enum VkPath { kVk13, kVk12Dynamic, kVk12RenderPass };
inline VkPath g_vk_path = kVk13;
/* VK_ATTACHMENT_STORE_OP_NONE can be used (1.3, dynamic rendering, or the
   load_store_op_none extensions). */
inline bool g_store_op_none = true;
const char* VkPathName(VkPath path);
int VkLegacyForced();                       // PS2HOST_VK_LEGACY: 0, 1 or 2
std::uint32_t VkInstanceVersion();          // 1.2 when the legacy path is forced, else 1.3
int VkDeviceScore(const VkPhysicalDeviceProperties& p);      // PS2HOST_GPU's, then a discrete GPU
/* Whether `gpu` can draw the GS, and on which path; why not, when not. */
bool VkDevicePath(VkPhysicalDevice gpu, VkPath* path, std::string* why);
/* What a device on a path is made with: its features' chain (features, for
   VkDeviceCreateInfo::pNext) and its extensions, to which a caller adds its
   own. Not to be copied (the chain points into it). */
struct VkDeviceSetup {
    VkDeviceSetup() = default;
    VkDeviceSetup(const VkDeviceSetup&) = delete;
    VkDeviceSetup& operator=(const VkDeviceSetup&) = delete;
    void Fill(VkPhysicalDevice gpu, VkPath path);
    VkPhysicalDeviceFeatures supported{};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    std::vector<const char*> extensions;
    const char* load_none = nullptr;        // a load_store_op_none extension, if offered
};
/* After the device is made: the path taken, and on 1.2 the core names the
   renderer calls pointed at the extensions' entries. */
void VkUseLegacyFunctions(VkPath path);

inline void CountWait(const char* why, double seconds) {
    for (WaitReason& w : g_waits) {
        if (!w.why) w.why = why;
        if (w.why == why || std::string(w.why) == why) {
            ++w.count;
            w.seconds += seconds;
            return;
        }
    }
}

class VulkanGpu final : public Gpu {
public:
    ~VulkanGpu() override;
    bool Init(std::string* error);
    /* On a device someone else made and keeps (a Display's): only the command
       pool, command buffers and fences are this one's own. */
    bool InitShared(VkInstance instance, VkPhysicalDevice gpu, VkDevice device, std::uint32_t family, VkQueue queue,
                    bool dual_source, const std::string& describe, std::string* error);
    std::string Describe() const override { return describe_; }

    std::uint32_t FindMemory(std::uint32_t bits, VkMemoryPropertyFlags wanted) const;
    /* Records into the open command buffer (opening one if none is), to run
       at the next Submit: a frame's work goes to the GPU in one submission.
       A rendering pass left open (RecordInPass) is ended first. */
    template <typename F>
    void Record(F&& record);
    /* The same, leaving a rendering pass a previous call left open: for draws
       that continue it (end_pass says how to end one). */
    template <typename F>
    void RecordInPass(F&& record);
    std::function<void(VkCommandBuffer)> end_pass;
    /* Runs what has been recorded and waits for it, and for everything
       Flush handed over before; then what was waiting for that (AfterSubmit:
       freeing staging buffers, say) runs. */
    bool Submit(const char* why = "other");
    /* Hands what has been recorded to the GPU without waiting for it, so that
       the GPU works while the CPU goes on: the next recording goes into the
       next of kSlots command buffers (waiting only if that one is still
       running). What waited on the work handed over runs once it has run. */
    bool Flush();
    void AfterSubmit(std::function<void()> done) { after_submit_.push_back(std::move(done)); }
    bool recording() const { return open_; }
    /* PS2HOST_GPU_PROFILE: the GPU's time since the last mark (or the command
       buffer's start) counted as `kind` - a timestamp as the work before it
       ends. */
    void Mark(VkCommandBuffer c, int kind) {
        if (!profile_ || marks_[current_].size() >= kMarks) return;
        vkCmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profile_,
                            std::uint32_t(current_) * kMarks + std::uint32_t(marks_[current_].size()));
        marks_[current_].push_back(kind);
        tags_[current_].push_back(0);
    }
    /* A draw's mark with its tag. */
    void MarkDraw(VkCommandBuffer c, std::uint32_t tag) {
        if (!profile_ || marks_[current_].size() >= kMarks) return;
        Mark(c, kProfDraw);
        tags_[current_].back() = tag;
    }
    /* A secondary command buffer of this slot's, begun to continue a
       rendering pass into an RGBA8 colour and the GS's depth-stencil (the
       renderer's deferred passes): free again once the slot's primary is.
       With `pass` (a compatible render pass: Vulkan 1.2 without dynamic
       rendering), its subpass is what it continues. */
    VkCommandBuffer BeginSecondary(VkFormat colour, VkFormat depth, VkRenderPass pass = VK_NULL_HANDLE) {
        std::vector<VkCommandBuffer>& list = secondaries_[current_];
        if (secondary_used_[current_] == list.size()) {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer made = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cai, &made) != VK_SUCCESS) return VK_NULL_HANDLE;
            list.push_back(made);
        }
        VkCommandBuffer c = list[secondary_used_[current_]++];
        vkResetCommandBuffer(c, 0);
        VkCommandBufferInheritanceRenderingInfo rendering{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &colour;
        rendering.depthAttachmentFormat = depth;
        rendering.stencilAttachmentFormat = depth;
        rendering.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        if (pass) inherit.renderPass = pass;
        else inherit.pNext = &rendering;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        begin.pInheritanceInfo = &inherit;
        vkBeginCommandBuffer(c, &begin);
        return c;
    }
    /* Counts command buffers begun: what was set in one is not in the next. */
    std::uint64_t recording_id() const { return recordings_; }
    static constexpr int kSlots = 3;
    /* The command buffer recording goes into (or will). */
    int slot() const { return current_; }
    /* Records with `record`, submits, and waits for it to finish. */
    template <typename F>
    bool Run(F&& record, const char* why = "other") {
        Record(std::forward<F>(record));
        return Submit(why);
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;           /* the current slot's */
    VkCommandBuffer cmds[kSlots] = {};
    VkFence fences[kSlots] = {};
    bool dual_source = false;       /* dualSrcBlend: the GS's blending */
    bool logic_op = false;          /* logicOp: FBMSK's partly masked channels */
    bool bc = false;                /* textureCompressionBC: texture packs' DDS uploaded as they are */
    float anisotropy = 1.0f;        /* samplerAnisotropy's most (1: none): texture packs' samplers */
    bool owns_device = true;

private:
    std::string describe_;
    bool open_ = false;
    std::uint64_t recordings_ = 0;
    std::vector<VkCommandBuffer> secondaries_[kSlots];     // BeginSecondary's
    std::size_t secondary_used_[kSlots] = {};
    // Two timestamps a slot: as its command buffer starts and as it ends.
    VkQueryPool timestamps_ = VK_NULL_HANDLE;
    double timestamp_ns_ = 0;               // a tick, in nanoseconds
    bool stamped_[kSlots] = {};
    void MakeTimestamps();
    static constexpr std::uint32_t kMarks = 8192;
    VkQueryPool profile_ = VK_NULL_HANDLE;
    std::vector<int> marks_[kSlots];
    std::vector<std::uint32_t> tags_[kSlots];
    std::vector<std::function<void()>> after_submit_;
    int current_ = 0;
    bool in_flight_[kSlots] = {};
    std::vector<std::function<void()>> slot_done_[kSlots];
    /* Waits for a slot's work, if handed over, and runs what waited on it;
       the time it waited goes to `why`. */
    bool Retire(int slot, const char* why);
};

template <typename F>
void VulkanGpu::RecordInPass(F&& record) {
    if (!open_) {
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        open_ = true;
        ++recordings_;
        secondary_used_[current_] = 0;
        if (timestamps_) {
            vkCmdResetQueryPool(cmd, timestamps_, 2 * std::uint32_t(current_), 2);
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestamps_, 2 * std::uint32_t(current_));
        }
        if (profile_) {
            vkCmdResetQueryPool(cmd, profile_, std::uint32_t(current_) * kMarks, kMarks);
            marks_[current_].clear();
            tags_[current_].clear();
            Mark(cmd, kProfOther);
        }
    }
    record(cmd);
}

template <typename F>
void VulkanGpu::Record(F&& record) {
    RecordInPass([&](VkCommandBuffer c) {
        if (end_pass) end_pass(c);
        record(c);
    });
}

inline bool VulkanGpu::Retire(int slot, const char* why) {
    if (!in_flight_[slot]) return true;
    const auto start = std::chrono::steady_clock::now();
    const bool ok = vkWaitForFences(device, 1, &fences[slot], VK_TRUE, UINT64_MAX) == VK_SUCCESS;
    const auto end = std::chrono::steady_clock::now();
    const double waited = std::chrono::duration<double>(end - start).count();
    g_wait_seconds += waited;
    if (g_wait_observer && waited > 0.0001)
        g_wait_observer(why, std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()),
                        std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()));
    CountWait(why, waited);
    in_flight_[slot] = false;
    if (stamped_[slot]) {
        std::uint64_t ticks[2] = {};
        if (vkGetQueryPoolResults(device, timestamps_, 2 * std::uint32_t(slot), 2, sizeof ticks, ticks,
                                  sizeof ticks[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ticks[1] > ticks[0])
            g_gpu_ns += std::uint64_t(double(ticks[1] - ticks[0]) * timestamp_ns_);
        stamped_[slot] = false;
    }
    if (profile_ && marks_[slot].size() > 1) {
        std::vector<std::uint64_t> ticks(marks_[slot].size());
        if (vkGetQueryPoolResults(device, profile_, std::uint32_t(slot) * kMarks, std::uint32_t(ticks.size()),
                                  ticks.size() * sizeof ticks[0], ticks.data(), sizeof ticks[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (std::size_t i = 1; i < ticks.size(); ++i)
                if (ticks[i] > ticks[i - 1]) {
                    const std::uint64_t ns = std::uint64_t(double(ticks[i] - ticks[i - 1]) * timestamp_ns_);
                    g_profile_ns[marks_[slot][i]] += ns;
                    if (marks_[slot][i] == kProfDraw) {
                        std::lock_guard<std::mutex> hold(g_draw_ns_lock);
                        g_draw_ns[tags_[slot][i]] += ns;
                    }
                }
        marks_[slot].clear();
        tags_[slot].clear();
    }
    std::vector<std::function<void()>> done;
    done.swap(slot_done_[slot]);
    for (auto& f : done) f();
    return ok;
}

inline bool VulkanGpu::Flush() {
    if (!open_) return true;
    if (end_pass) end_pass(cmd);
    Mark(cmd, kProfOther);
    open_ = false;
    if (timestamps_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestamps_, 2 * std::uint32_t(current_) + 1);
        stamped_[current_] = true;
    }
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkResetFences(device, 1, &fences[current_]);
    bool ok;
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        ok = vkQueueSubmit(queue, 1, &submit, fences[current_]) == VK_SUCCESS;
    }
    ++g_submits;
    in_flight_[current_] = ok;
    slot_done_[current_].swap(after_submit_);
    if (!ok) {                                      // nothing will run: nothing to wait for
        for (auto& f : slot_done_[current_]) f();
        slot_done_[current_].clear();
    }
    current_ = (current_ + 1) % kSlots;
    cmd = cmds[current_];
    return Retire(current_, "a full ring") && ok;
}

inline bool VulkanGpu::Submit(const char* why) {
    bool ok = Flush();
    // Oldest first: current_ is free, the one after it was handed over first.
    for (int i = 1; i <= kSlots; ++i) ok = Retire((current_ + i) % kSlots, why) && ok;
    std::vector<std::function<void()>> done;        // waited on nothing recorded
    done.swap(after_submit_);
    for (auto& f : done) f();
    return ok;
}

}  // namespace ps2gpu

#endif  /* PS2GPU_GPU_INTERNAL_H */
