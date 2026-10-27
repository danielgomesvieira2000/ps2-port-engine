/* The PC's effects on a game's picture: effects.h. */

#include "effects.h"

#include <algorithm>
#include <cstring>

#include "effects_0.h"
#include "effects_1.h"
#include "effects_2.h"
#include "effects_3.h"
#include "effects_4.h"
#include "effects_5.h"
#include "effects_6.h"
#include "effects_7.h"
#include "effects_8.h"
#include "effects_9.h"
#include "effects_10.h"
#include "effects_11.h"
#include "effects_12.h"
#include "effects_13.h"
#include "effects_14.h"
#include "effects_15.h"

namespace ps2gpu {

namespace {

struct Code {
    const std::uint32_t* words;
    std::size_t bytes;
};

constexpr VkFormat kScratchFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr int kBloomLevels = 6;

}  // namespace

GsEffects::~GsEffects() {
    Forget();
    DestroyScratch(dummy_);
    DestroyScratch(dummy_mask_);
    DestroyScratch(scene_);
    for (int i = 0; i < kPasses; ++i) {
        if (pipelines_[i]) vkDestroyPipeline(g_.device, pipelines_[i], nullptr);
        if (modules_[i]) vkDestroyShaderModule(g_.device, modules_[i], nullptr);
    }
    if (pipeline_layout_) vkDestroyPipelineLayout(g_.device, pipeline_layout_, nullptr);
    if (layout_) vkDestroyDescriptorSetLayout(g_.device, layout_, nullptr);
    if (pool_) vkDestroyDescriptorPool(g_.device, pool_, nullptr);
    if (linear_) vkDestroySampler(g_.device, linear_, nullptr);
    if (nearest_) vkDestroySampler(g_.device, nearest_, nullptr);
}

bool GsEffects::Init(VkPipelineCache cache) {
    VkDescriptorSetLayoutBinding b[7] = {};
    for (std::uint32_t i = 0; i < 7; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = 1;
        b[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = 7;
    dl.pBindings = b;
    if (vkCreateDescriptorSetLayout(g_.device, &dl, nullptr, &layout_) != VK_SUCCESS) return false;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &layout_;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(g_.device, &pl, nullptr, &pipeline_layout_) != VK_SUCCESS) return false;
    const Code code[kPasses] = {
        {spirv_effects_0, sizeof spirv_effects_0}, {spirv_effects_1, sizeof spirv_effects_1},
        {spirv_effects_2, sizeof spirv_effects_2}, {spirv_effects_3, sizeof spirv_effects_3},
        {spirv_effects_4, sizeof spirv_effects_4}, {spirv_effects_5, sizeof spirv_effects_5},
        {spirv_effects_6, sizeof spirv_effects_6}, {spirv_effects_7, sizeof spirv_effects_7},
        {spirv_effects_8, sizeof spirv_effects_8}, {spirv_effects_9, sizeof spirv_effects_9},
        {spirv_effects_10, sizeof spirv_effects_10}, {spirv_effects_11, sizeof spirv_effects_11},
        {spirv_effects_12, sizeof spirv_effects_12}, {spirv_effects_13, sizeof spirv_effects_13},
        {spirv_effects_14, sizeof spirv_effects_14}, {spirv_effects_15, sizeof spirv_effects_15},
    };
    for (int i = 0; i < kPasses; ++i) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = code[i].bytes;
        ci.pCode = code[i].words;
        if (vkCreateShaderModule(g_.device, &ci, nullptr, &modules_[i]) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpi.stage.module = modules_[i];
        cpi.stage.pName = "main";
        cpi.layout = pipeline_layout_;
        if (vkCreateComputePipelines(g_.device, cache, 1, &cpi, nullptr, &pipelines_[i]) != VK_SUCCESS) return false;
    }
    const VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * 1024},
                                           {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5 * 1024}};
    VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp.maxSets = 1024;
    dp.poolSizeCount = 2;
    dp.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(g_.device, &dp, nullptr, &pool_) != VK_SUCCESS) return false;
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(g_.device, &sci, nullptr, &linear_) != VK_SUCCESS) return false;
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    if (vkCreateSampler(g_.device, &sci, nullptr, &nearest_) != VK_SUCCESS) return false;
    ready_ = true;
    return true;
}

bool GsEffects::MakeScratch(Scratch& s, std::uint32_t w, std::uint32_t h, VkCommandBuffer cmd, VkFormat format) {
    s.w = std::max(1u, w);
    s.h = std::max(1u, h);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {s.w, s.h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = format == kScratchFormat ? VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                         : VK_IMAGE_USAGE_SAMPLED_BIT;
    if (vkCreateImage(g_.device, &ici, nullptr, &s.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_.device, s.image, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g_.device, &ai, nullptr, &s.memory) != VK_SUCCESS) return false;
    vkBindImageMemory(g_.device, s.image, s.memory, 0);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = s.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(g_.device, &vci, nullptr, &s.view) != VK_SUCCESS) return false;
    // GENERAL from the start, for good: written and read by the passes alike.
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
    return true;
}

void GsEffects::DestroyScratch(Scratch& s) {
    if (s.view) vkDestroyImageView(g_.device, s.view, nullptr);
    if (s.image) vkDestroyImage(g_.device, s.image, nullptr);
    if (s.memory) vkFreeMemory(g_.device, s.memory, nullptr);
    s = Scratch{};
}

void GsEffects::Forget() {
    for (auto& [key, set] : sets_) {
        DestroyScratch(set.full);
        DestroyScratch(set.half_a);
        DestroyScratch(set.half_b);
        for (Scratch& s : set.down) DestroyScratch(s);
        for (Scratch& s : set.up) DestroyScratch(s);
    }
    sets_.clear();
    DestroyScratch(scene_);
    descriptors_.clear();
    if (pool_) vkResetDescriptorPool(g_.device, pool_, 0);
}

void GsEffects::ForgetView(VkImageView view) {
    if (!view || descriptors_.empty()) return;
    for (auto at = descriptors_.begin(); at != descriptors_.end();) {
        if (std::find(at->first.begin(), at->first.end(), view) != at->first.end()) at = descriptors_.erase(at);
        else ++at;
    }
}

GsEffects::Set& GsEffects::SetFor(std::uint32_t w, std::uint32_t h, VkCommandBuffer cmd) {
    Set& set = sets_[std::uint64_t(w) << 32 | h];
    if (set.made) return set;
    set.made = true;
    MakeScratch(set.full, w, h, cmd);
    // The occlusion at half the view's pixels, a third from 4x up: about
    // the PS2's own count or more - it is soft, and at 4x it was the
    // effects' dearest (2 ms of 4.5 on an RTX 4070 laptop).
    set.ao_div = w >= 4u * 640u ? 3u : 2u;
    const std::uint32_t d = set.ao_div;
    MakeScratch(set.half_a, (w + d - 1) / d, (h + d - 1) / d, cmd);
    MakeScratch(set.half_b, (w + d - 1) / d, (h + d - 1) / d, cmd);
    std::uint32_t lw = w, lh = h;
    for (int i = 0; i < kBloomLevels && lw >= 8 && lh >= 8; ++i) {
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
        set.down.emplace_back();
        MakeScratch(set.down.back(), lw, lh, cmd);
        set.up.emplace_back();
        MakeScratch(set.up.back(), lw, lh, cmd);
    }
    return set;
}

VkDescriptorSet GsEffects::Descriptors(const Target& t, VkImageView dst, VkImageView a, VkImageView b) {
    if (!dst) dst = dummy_.view;
    if (!a) a = dummy_.view;
    if (!b) b = dummy_.view;
    const VkImageView depth = t.depth_view ? t.depth_view : dummy_.view;
    const VkImageView mask = t.stencil_view ? t.stencil_view : dummy_mask_.view;
    const std::array<VkImageView, 6> key{t.view, dst, depth, a, b, mask};
    if (auto found = descriptors_.find(key); found != descriptors_.end()) return found->second;
    if (descriptors_.size() >= 1000) {          // (images come and go with the scale alone)
        descriptors_.clear();
        vkResetDescriptorPool(g_.device, pool_, 0);
    }
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout_;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(g_.device, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    const VkDescriptorImageInfo info[7] = {
        {VK_NULL_HANDLE, t.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL},
        {linear_, t.view, VK_IMAGE_LAYOUT_GENERAL},
        {nearest_, depth,
         t.depth_view ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL},
        {linear_, a, VK_IMAGE_LAYOUT_GENERAL},
        {linear_, b, VK_IMAGE_LAYOUT_GENERAL},
        {nearest_, mask,
         t.stencil_view ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL},
    };
    VkWriteDescriptorSet writes[7] = {};
    for (std::uint32_t i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &info[i];
    }
    vkUpdateDescriptorSets(g_.device, 7, writes, 0, nullptr);
    descriptors_.emplace(key, set);
    return set;
}

void GsEffects::Barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

void GsEffects::Dispatch(VkCommandBuffer cmd, Pass pass, VkDescriptorSet set, const Push& push, std::uint32_t w,
                         std::uint32_t h) {
    if (!set || !w || !h) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[pass]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}

void GsEffects::Record(VkCommandBuffer cmd, const GsEffect& e, const Target& t) {
    if (!ready_ || !t.view) return;
    if (!dummy_.image) MakeScratch(dummy_, 1, 1, cmd);
    if (!dummy_mask_.image) MakeScratch(dummy_mask_, 1, 1, cmd, VK_FORMAT_R8_UINT);
    // (each effect a frame - the noise turned for each - but the reflections:
    // turned on, they would change the blur's and the occlusion's noise)
    if (e.id != GsEffectId::kReflections) ++frame_;
    const std::uint32_t k = t.scale;
    const std::uint32_t x0 = std::min(std::uint32_t(std::max(e.x0, 0)) * k, t.pw);
    const std::uint32_t y0 = std::min(std::uint32_t(std::max(e.y0, 0)) * k, t.ph);
    const std::uint32_t x1 = std::min(std::uint32_t(std::max(e.x1, 0)) * k, t.pw);
    const std::uint32_t y1 = std::min(std::uint32_t(std::max(e.y1, 0)) * k, t.ph);
    if (x1 <= x0 || y1 <= y0) return;
    const std::uint32_t w = x1 - x0, h = y1 - y0;
    Push push{};
    push.rect[0] = std::int32_t(x0);
    push.rect[1] = std::int32_t(y0);
    push.rect[2] = std::int32_t(x1);
    push.rect[3] = std::int32_t(y1);
    push.size[0] = std::int32_t(w);
    push.size[1] = std::int32_t(h);
    push.size[2] = std::int32_t(frame_);
    push.size[3] = std::int32_t(k);
    std::memcpy(push.p, e.params, sizeof push.p);
    // What the game drew before, finished (the target was handed over in
    // GENERAL by a barrier of its own).
    switch (e.id) {
        case GsEffectId::kTint:
            Dispatch(cmd, kTint, Descriptors(t, nullptr, nullptr, nullptr), push, w, h);
            break;
        case GsEffectId::kOcclusion: {
            if (!t.depth_view) return;
            Set& set = SetFor(w, h, cmd);
            const std::uint32_t hw = set.half_a.w, hh = set.half_a.h;
            push.size[0] = std::int32_t(hw);
            push.size[1] = std::int32_t(hh);
            push.size[3] = std::int32_t(set.ao_div);       // (its passes: the occlusion's divisor)
            // Its distances first, one a pixel of its own (half_b), read by
            // its samples: the full depth buffer's scattered reads were a
            // fifth of its time at 3x (0.9 ms a picture -> 0.75 in the
            // explosion; the rest its own arithmetic, the sun's 0.17).
            Dispatch(cmd, kAoDepth, Descriptors(t, set.half_b.view, nullptr, nullptr), push, hw, hh);
            Barrier(cmd);
            Dispatch(cmd, kAo, Descriptors(t, set.half_a.view, set.half_b.view, nullptr), push, hw, hh);
            Barrier(cmd);
            Dispatch(cmd, kAoBlurX, Descriptors(t, set.half_b.view, set.half_a.view, nullptr), push, hw, hh);
            Barrier(cmd);
            Dispatch(cmd, kAoBlurY, Descriptors(t, set.half_a.view, set.half_b.view, nullptr), push, hw, hh);
            Barrier(cmd);
            push.size[0] = std::int32_t(w);
            push.size[1] = std::int32_t(h);
            Dispatch(cmd, kAoApply, Descriptors(t, nullptr, set.half_a.view, nullptr), push, w, h);
            break;
        }
        case GsEffectId::kMotionBlur: {
            if (!t.depth_view) return;
            Set& set = SetFor(w, h, cmd);
            Dispatch(cmd, kBlur, Descriptors(t, set.full.view, nullptr, nullptr), push, w, h);
            Barrier(cmd);
            Dispatch(cmd, kCopy, Descriptors(t, nullptr, set.full.view, nullptr), push, w, h);
            break;
        }
        case GsEffectId::kBloom: {
            Set& set = SetFor(w, h, cmd);
            const std::size_t n = set.down.size();
            if (!n) return;
            for (std::size_t i = 0; i < n; ++i) {
                push.size[0] = std::int32_t(set.down[i].w);
                push.size[1] = std::int32_t(set.down[i].h);
                push.p[10] = i == 0 ? 1.0f : 0.0f;
                Dispatch(cmd, kBloomDown, Descriptors(t, set.down[i].view, i ? set.down[i - 1].view : nullptr, nullptr),
                         push, set.down[i].w, set.down[i].h);
                Barrier(cmd);
            }
            push.p[11] = 1.0f;
            for (std::size_t i = n - 1; i-- > 0;) {
                const VkImageView below = i == n - 2 ? set.down[n - 1].view : set.up[i + 1].view;
                push.size[0] = std::int32_t(set.up[i].w);
                push.size[1] = std::int32_t(set.up[i].h);
                Dispatch(cmd, kBloomUp, Descriptors(t, set.up[i].view, below, set.down[i].view), push, set.up[i].w,
                         set.up[i].h);
                Barrier(cmd);
            }
            push.size[0] = std::int32_t(w);
            push.size[1] = std::int32_t(h);
            const VkImageView top = n > 1 ? set.up[0].view : set.down[0].view;
            // (sharpening, p17: into the scratch, then copied back)
            const bool sharpen = push.p[17] > 0.0f;
            Dispatch(cmd, kBloomApply, Descriptors(t, sharpen ? set.full.view : nullptr, top, nullptr), push, w, h);
            if (sharpen) {
                Barrier(cmd);
                Dispatch(cmd, kCopy, Descriptors(t, nullptr, set.full.view, nullptr), push, w, h);
            }
            break;
        }
        case GsEffectId::kGrade:
            if (push.p[17] > 0.0f) {
                Set& set = SetFor(w, h, cmd);
                Dispatch(cmd, kGrade, Descriptors(t, set.full.view, nullptr, nullptr), push, w, h);
                Barrier(cmd);
                Dispatch(cmd, kCopy, Descriptors(t, nullptr, set.full.view, nullptr), push, w, h);
            } else {
                Dispatch(cmd, kGrade, Descriptors(t, nullptr, nullptr, nullptr), push, w, h);
            }
            break;
        case GsEffectId::kDepthOfField: {
            if (!t.depth_view) return;
            Set& set = SetFor(w, h, cmd);
            Dispatch(cmd, kDof, Descriptors(t, set.full.view, nullptr, nullptr), push, w, h);
            Barrier(cmd);
            Dispatch(cmd, kCopy, Descriptors(t, nullptr, set.full.view, nullptr), push, w, h);
            break;
        }
        case GsEffectId::kReflections: {
            if (!t.depth_view) return;
            Set& set = SetFor(w, h, cmd);
            Barrier(cmd);       // (the occlusion's images, used just before)
            push.p[15] = float(k);      // (the target's scale: the normal's footprint)
            // p16: traced at the full resolution (HIGH), else the occlusion's (LOW)
            const bool full = push.p[16] > 0.0f;
            const Scratch& into = full ? set.full : set.half_a;
            push.size[0] = std::int32_t(into.w);
            push.size[1] = std::int32_t(into.h);
            push.size[3] = full ? 1 : std::int32_t(set.ao_div);
            Dispatch(cmd, kSsr, Descriptors(t, into.view, nullptr, nullptr), push, into.w, into.h);
            Barrier(cmd);
            push.size[0] = std::int32_t(w);
            push.size[1] = std::int32_t(h);
            Dispatch(cmd, kSsrApply, Descriptors(t, nullptr, into.view, nullptr), push, w, h);
            break;
        }
        case GsEffectId::kSceneDepth:
            if (!t.depth_view) return;
            if (scene_.w != t.pw || scene_.h != t.ph) {
                DestroyScratch(scene_);
                MakeScratch(scene_, t.pw, t.ph, cmd);
            }
            Dispatch(cmd, kSceneDepth, Descriptors(t, scene_.view, nullptr, nullptr), push, w, h);
            break;
    }
}

}  // namespace ps2gpu
