/* The PC's effects on a game's picture (GsRenderer::RunEffect): compute
   passes (shaders/effects.comp) on a target and its depth buffer, with
   scratch images of their own - ambient occlusion, a camera's motion blur, a
   glow, a grade. For gs_renderer.cpp, which hands over the images in the
   layouts named. */
#ifndef PS2GPU_EFFECTS_H
#define PS2GPU_EFFECTS_H

#include <array>
#include <cstdint>
#include <map>
#include <vector>

#include "gpu_internal.h"

namespace ps2gpu {

class GsEffects {
public:
    explicit GsEffects(VulkanGpu& gpu) : g_(gpu) {}
    ~GsEffects();
    bool Init(VkPipelineCache cache);
    bool ready() const { return ready_; }
    /* The target (`view` a storage-capable RGBA8 view, in GENERAL) and its
       depth buffer (`depth_view` its depth alone, in
       DEPTH_STENCIL_READ_ONLY_OPTIMAL; null for none), each `scale` image
       pixels to a GS pixel; the effect's rectangle in GS pixels. */
    struct Target {
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t pw = 0, ph = 0, scale = 1;
        VkImageView depth_view = VK_NULL_HANDLE;
    };
    void Record(VkCommandBuffer cmd, const GsEffect& effect, const Target& target);
    /* Drops every scratch image and descriptor set (the device idle: a new
       internal resolution). */
    void Forget();
    /* An image view going: the descriptor sets naming it with it. */
    void ForgetView(VkImageView view);
    /* The scene's depth the last kSceneDepth saved (GENERAL, the target's
       size), or null. */
    VkImageView SceneDepth() const { return scene_.view; }

private:
    struct Scratch {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        std::uint32_t w = 0, h = 0;
    };
    /* A view's scratch images, by its size in image pixels. */
    struct Set {
        Scratch full, half_a, half_b;      // half_*: the occlusion's, 1/ao_div of the view
        std::uint32_t ao_div = 2;
        std::vector<Scratch> down, up;      // the glow's chain
        bool made = false;
    };
    enum Pass { kTint, kAo, kAoBlurX, kAoBlurY, kAoApply, kBlur, kCopy, kBloomDown, kBloomUp, kBloomApply, kGrade,
                kSceneDepth, kDof, kPasses };
    struct Push {
        std::int32_t rect[4];
        std::int32_t size[4];
        float p[22];
    };
    static_assert(sizeof(Push) <= 128, "push constants");

    bool MakeScratch(Scratch& s, std::uint32_t w, std::uint32_t h, VkCommandBuffer cmd);
    void DestroyScratch(Scratch& s);
    Set& SetFor(std::uint32_t w, std::uint32_t h, VkCommandBuffer cmd);
    VkDescriptorSet Descriptors(const Target& t, VkImageView dst, VkImageView a, VkImageView b);
    void Dispatch(VkCommandBuffer cmd, Pass pass, VkDescriptorSet set, const Push& push, std::uint32_t w,
                  std::uint32_t h);
    static void Barrier(VkCommandBuffer cmd);

    VulkanGpu& g_;
    bool ready_ = false;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipelines_[kPasses] = {};
    VkShaderModule modules_[kPasses] = {};
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkSampler linear_ = VK_NULL_HANDLE, nearest_ = VK_NULL_HANDLE;
    Scratch dummy_;                         // what a pass does not use is bound to
    Scratch scene_;                         // the scene's depth (SceneDepth)
    std::map<std::uint64_t, Set> sets_;     // by w << 32 | h
    std::map<std::array<VkImageView, 5>, VkDescriptorSet> descriptors_;
    std::uint32_t frame_ = 0;
};

}  // namespace ps2gpu

#endif  /* PS2GPU_EFFECTS_H */
