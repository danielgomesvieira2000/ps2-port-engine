/*
 * GsRenderer on Vulkan (gs_renderer.h): triangles batched by state and drawn
 * with dynamic rendering into R8G8B8A8 targets, one per (FBP, FBW, PSM).
 *
 * One vertex and one fragment shader (shaders/gs.vert, gs.frag) do the GS's
 * per-pixel work: the texture function, the alpha test. Blending is the
 * fixed-function blender's, with the GS's factor - source alpha or FIX, where
 * 0x80 is 1.0 - as the fragment's second output (dual-source blending): the
 * GS's Cv = (A - B) * C + D becomes a source and a destination factor and an
 * operation, one pipeline per ALPHA setting, made when first used.
 *
 * Everything is synchronous for now - a flush records, submits and waits - which
 * keeps the first steps simple to check; batching across frames and keeping the
 * GPU busy come once the output matches the software GS.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "gpu_internal.h"
#include "effects.h"
#include "gs_frag.h"
#include "gs_frag_single.h"
#include "gs_frag_blend.h"
#include "gs_vert.h"
#include "date_frag.h"
#include "date_vert.h"
#include "swizzle_comp.h"
#include "swizzle_image_comp.h"
#include "swizzle_depth_comp.h"
#include "depth_load_frag.h"
#include "decode_comp.h"
#include "decode_image_comp.h"
#include "patch_comp.h"
#include "shuffle_comp.h"

namespace ps2gpu {

namespace {

struct PackedVertex {
    float x, y;
    std::uint8_t r, g, b, a;
    float s, t, q;
    std::uint32_t z;
    std::uint8_t fog, pad[3];
};
// A sprite's S, T at its first and last pixels (gs.vert's `range`), in a
// stream of its own: only a sprite or a flat quad on a target drawn larger
// reads it (gs.frag's snap), so every other batch binds one value for all
// its vertices (stride 0) - the vertex half the size it was with it inside,
// half the bytes written, copied to the GPU and fetched there.
struct VertexRange {
    float range[4];
};
constexpr VertexRange kNoRange{{-1e9f, -1e9f, 1e9f, 1e9f}};
// The vertex ring hands out 256-byte aligned runs, a batch's first vertex
// found by its offset: whole vertices to 256 bytes.
static_assert(256 % sizeof(PackedVertex) == 0, "a vertex must divide the ring's alignment");

/* A batch's vertices so far: room handed out a primitive at a time (Grow)
   and its fields written in place - no vertex built and then copied, nor
   its padding cleared (the vertex shader reads none of it). A splitscreen
   race appends ~400,000 a picture: push_back's temporary and copy were a
   twelfth of the GS thread. */
class PendingVertices {
public:
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void clear() { size_ = 0; }
    PackedVertex* data() { return data_.get(); }
    PackedVertex& operator[](std::size_t i) { return data_[i]; }
    PackedVertex* Grow(std::size_t n) {
        if (size_ + n > capacity_) {
            const std::size_t capacity = std::max<std::size_t>(capacity_ * 2, std::max<std::size_t>(size_ + n, 4096));
            std::unique_ptr<PackedVertex[]> data(new PackedVertex[capacity]);
            if (size_) std::memcpy(data.get(), data_.get(), size_ * sizeof(PackedVertex));
            data_ = std::move(data);
            capacity_ = capacity;
        }
        PackedVertex* at = data_.get() + size_;
        size_ += n;
        return at;
    }

private:
    std::unique_ptr<PackedVertex[]> data_;
    std::size_t size_ = 0, capacity_ = 0;
};

/* A batch's ranges so far, in step with its vertices - kept only for a
   batch that can read them (VulkanGsRenderer::RangesWanted). */
class PendingRanges {
public:
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    void clear() { size_ = 0; }
    VertexRange* data() { return data_.get(); }
    VertexRange& operator[](std::size_t i) { return data_[i]; }
    VertexRange* Grow(std::size_t n) {
        if (size_ + n > capacity_) {
            const std::size_t capacity = std::max<std::size_t>(capacity_ * 2, std::max<std::size_t>(size_ + n, 4096));
            std::unique_ptr<VertexRange[]> data(new VertexRange[capacity]);
            if (size_) std::memcpy(data.get(), data_.get(), size_ * sizeof(VertexRange));
            data_ = std::move(data);
            capacity_ = capacity;
        }
        VertexRange* at = data_.get() + size_;
        size_ += n;
        return at;
    }

private:
    std::unique_ptr<VertexRange[]> data_;
    std::size_t size_ = 0, capacity_ = 0;
};

/* What the fragment shader is told, after the vertex shader's 16 bytes. */
struct FragmentParams {
    std::uint32_t textured, tfx, tcc, atst, aref, fix_factor, fix, afail, fge, fogcol, pabe;
    std::uint32_t bilinear, wms, wmt, minu, maxu, minv, maxv;
    std::uint32_t tw, th, tex_format;
    std::uint32_t texa;         // TA0 in bits 0-7, TA1 8-15, AEM 16
    std::uint32_t image_size;   // the bound image's pixels: width in bits 0-15, height 16-31
    std::uint32_t logic, keep;
    std::uint32_t scale;        // the texture's pixels to a GS texel (a target drawn larger: the internal resolution)
    std::uint32_t blend_ops;    // bits 0-7: blending in the shader, ALPHA's A, B, C, D, two bits each;
                                // 8-15: a sprite on a target drawn larger, the target's scale (0: none)
};
constexpr std::uint32_t kFragmentOffset = 16;

/* gs.frag's kFeatures: what a draw with these parameters may use (each bit
   as the shader has it), for a pipeline made without the rest. Kept in a
   pipeline key's bits 16-27, inverted: a key without them is the whole
   shader. */
constexpr int kFeatureShift = 16;
constexpr std::uint32_t kFeatureBits = 0xFFFu;
std::uint32_t Features(const FragmentParams& p) {
    std::uint32_t f = 0;
    if (p.textured) {
        f |= 1u;                                                    // kTextured
        if (p.tex_format & 0x100u) f |= 2u;                         // kReplaced
        if (p.bilinear & 2u) f |= 4u | 8u;                          // kMip (its levels may be bilinear)
        if (p.bilinear & 1u) f |= 8u;                               // kLinear
        if (p.tex_format & 0xFFu) f |= 16u;                         // kExpand
        if (p.scale > 1u) f |= 32u;                                 // kFine
        if (((p.blend_ops >> 8) & 255u) > 1u) f |= 64u;             // kSnap
    }
    if (p.fge) f |= 128u;                                           // kFog
    if (p.atst != 1u || p.afail == 1u) f |= 256u;                   // kAlphaTest: anything discarded by it
    if (p.pabe) f |= 512u;                                          // kPabe
    if (p.logic) f |= 1024u;                                        // kLogic
    if ((p.blend_ops >> 16) & 1u) f |= 2048u;                       // kSixteen
    return f;
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // Where in `memory` (ImageMemory: a piece of a block, of a size class -
    // class 0 its own allocation), and of which memory type.
    VkDeviceSize offset = 0;
    std::uint32_t piece_class = 0, memory_type = 0;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView words = VK_NULL_HANDLE;     /* an RGBA8 image's texels as 32-bit words (gs.frag's gather) */
    std::uint32_t width = 0, height = 0;    // in GS pixels
    std::uint32_t scale = 1;                // the image's pixels to a GS pixel, each way (targets, depth)
    std::uint32_t levels = 1;               // mipmap levels (a texture pack's replacements)
    std::uint32_t pw() const { return width * scale; }
    std::uint32_t ph() const { return height * scale; }
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageUsageFlags usage = 0;                 /* as made (MakeImage) */
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkDescriptorSet sampled = VK_NULL_HANDLE;   /* a target read by the DATE setup pass */
    VkDescriptorSet storage = VK_NULL_HANDLE;   /* a target as the swizzle's storage image */
    VkImageView depth_view = VK_NULL_HANDLE;    /* a depth buffer's depth alone, to sample */
    VkImageView stencil_view = VK_NULL_HANDLE;  /* ...and its stencil alone (bit 2: a car's paint) */
    VkDescriptorSet depth_sampled = VK_NULL_HANDLE;     /* ...for the swizzle */
};

// Depth with stencil: the stencil carries the destination alpha test.
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

struct Texture {
    Image image;
    VkDescriptorSet sets[28] = {};      /* by sampler (samplers_, then replacement_samplers_) */
    /* A texture drawn as a texture pack's replacement (UseReplacement): its
       id, 0 for none. A replacement itself (replacements_): gs.frag's
       tex_format for it - bit 8, levels - 1 in bits 9-11 - when it was last
       drawn (frames, Kick's), its size in memory. */
    std::uint64_t replacement = 0;
    std::uint32_t replaced = 0;
    std::uint64_t last_used = 0;
    VkDeviceSize bytes = 0;
};

// What the frames' command buffers hold (RendererCounters, PS2HOST_GPU_LOG).
std::uint64_t g_barriers = 0, g_transitions = 0, g_dispatches = 0, g_draws = 0, g_passes = 0, g_feedback = 0;
std::uint64_t g_feedback_blend = 0;     // of them, for the shader's blending alone (its own pixel: no texture of itself)
// Why each pass ended (RendererCounters): a draw's destination alpha test,
// its copy of its own target, a source to make readable, another target,
// another depth buffer; anything else recorded (VulkanGpu::end_pass).
enum PassEnd { kEndDate, kEndFeedback, kEndSource, kEndTarget, kEndDepth, kEndOther, kEndWhys };
std::uint64_t g_pass_ends[kEndWhys] = {};
// Deferred passes (PS2HOST_GPU_COUNT): the pixels their render areas took, of
// their images' whole; the depth-stencils neither stored nor loaded.
std::uint64_t g_area_used = 0, g_area_whole = 0, g_depth_unstored = 0, g_depth_unloaded = 0;
std::uint64_t g_beside = 0;     // work recorded beside an open deferred pass (RecordBeside)
std::uint64_t g_renamed = 0;    // textures decoded into another image for it (TextureFromMemory)

void Transition(VkCommandBuffer cmd, Image& t, VkImageLayout to, VkAccessFlags dst_access,
                VkPipelineStageFlags dst_stage) {
    if (t.layout == to) return;
    ++g_transitions;
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = t.layout;
    b.newLayout = to;
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = dst_access;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {t.aspect, 0, t.levels, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
    t.layout = to;
}

/* Everything written so far, visible to everything after. */
void MemoryBarrier(VkCommandBuffer cmd) {
    ++g_barriers;
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

/* decode.comp's push constants: GsTextureDecode without the CLUT. */
struct DecodeParams {
    std::uint32_t tbp, tbw, psm, tw, th, wms, wmt, minu, maxu, minv, maxv, texa_lo, ta1, csa;
    std::uint32_t mode = 0;     // 0 a texture, 1 a CLUT load
    std::uint32_t row0 = 0;     // into the image (decode_image_pipeline_): the level's first row
};

/* swizzle.comp's push constants. */
struct SwizzleParams {
    std::uint32_t bp, bw, psm, cols, rows, stride, to_memory, depth, row0;
    std::uint32_t scale = 1;    // the image's pixels to a GS pixel: one taken to memory, a square filled from it
    std::uint32_t keep_colour = 0;  // from memory: the square's colour kept where memory differs in alpha alone
    std::uint32_t col0 = 0;         // the first column moved (`cols` the end): a write-back of the pages drawn
};

/* shuffle.comp's push constants (ChannelShuffle). */
struct ShuffleParams {
    std::uint32_t dst_bp, dst_bw, dst_view;
    std::uint32_t view_bp;
    std::uint32_t rx0, ry0, cols, rows;
    std::int32_t qx0, qy0, qx1, qy1;
    std::int32_t period, run;
    std::int32_t du, dv;
    std::uint32_t src_bp, src_bw, src_view;
    std::uint32_t img_bp, img_w, img_h;
    std::uint32_t write16;
    std::uint32_t ta0, ta1, flags;
    std::uint32_t scale;
};

constexpr std::uint32_t kDepthRows = 512;   // a depth buffer's rows (DrawTriangles; the bridge's)
constexpr VkDeviceSize kLocalMemory = 4u << 20;
constexpr VkDeviceSize kLinear = VkDeviceSize(2048) * 1024 * 4;    // the largest image, a value a pixel (at scale 1)
constexpr std::uint32_t kMaxScale = 8;      // 4 the resolution's, twice that supersampled

/* FBMSK as the bits of the target's RGBA bytes it keeps. FBMSK is in RGBA32
   bit positions whatever the format (the GS masks the colour before it is
   made 16-bit, as the software GS and PCSX2 do): a 16-bit format stores the
   top five bits of each colour byte and alpha's top bit (0x80F8F8F8), the
   rest stay 0 either way, so count as kept; a 24-bit format keeps the alpha
   it does not store. (Read as a 16-bit pixel's bits, Burnout 3's FBMSK
   0x3FFF wrote alpha's top two bits alone, not its byte: its depth into the
   picture's alpha - the haze, the sun's halo - came out wrong.) */
std::uint32_t KeptBits(std::uint32_t fbmsk, std::uint32_t psm) {
    if (psm == 0x02 || psm == 0x0A || psm == 0x32 || psm == 0x3A) return fbmsk | ~0x80F8F8F8u;
    if (psm == 0x01 || psm == 0x31) return fbmsk | 0xFF000000u;
    return fbmsk;
}

/* The channels written whole or in part (Vulkan's write mask: R, G, B, A in
   bits 0-3). */
std::uint32_t WriteMask(std::uint32_t kept) {
    std::uint32_t mask = 0;
    for (int channel = 0; channel < 4; ++channel)
        if (((kept >> (8 * channel)) & 0xFF) != 0xFF) mask |= 1u << channel;
    return mask;
}

/* Whether some channel is written only in part. */
bool PartlyMasked(std::uint32_t kept) {
    for (int channel = 0; channel < 4; ++channel) {
        const std::uint32_t byte = (kept >> (8 * channel)) & 0xFF;
        if (byte != 0 && byte != 0xFF) return true;
    }
    return false;
}

/* A blend the blender cannot do: FIX over 0x80 is a factor over 1.0, which
   a fixed-point target clamps to 1 (Burnout 3's glow: (Cs - Cd) * 0xFF/128,
   then Cs * 0xFF/128 - at half its strength). The shader blends instead,
   from a copy of the target. */
bool Is16Bit(std::uint32_t psm) { return psm == 0x02 || psm == 0x0A || psm == 0x32 || psm == 0x3A; }

/* Dual-source blending (the factor a second output): Blender16 needs it. */
bool g_dual_source_blend = false;

/* How the blender can make a 16-bit target's blend exactly, the GS's five
   bits a channel, without the shader reading a copy of the target: where
   the target counts once (Cd's coefficient 1 - as D, or as A with FIX 0x80,
   a factor of 1) or not at all. The shader works out the rest, S (gs.frag:
   the blend with each Cd as 0), and the blender
     1  writes S (Cd not in it, or it cancels: (Cs - Cd) * 1 + Cd);
     2  adds S cut to five bits to the target;
     3  takes away -S, cut up to five bits (D - Cs, as Cd - Cs).
   The target's values are whole fives, so cutting the sum cuts only what is
   added; a sum past 255 keeps 255, which every reader of a 16-bit target
   cuts to the GS's 248. 0: the shader blends. Below Freezing Point's
   shadows, its characters added into and taken out of a 16-bit buffer: 900
   copies of it a picture, three quarters of the GPU's time at 3x. */
int Blender16(const GsDrawState& s) {
    if (!g_dual_source_blend || !s.blend || !Is16Bit(s.psm) || s.c == BlendAlpha::Dest) return 0;
    using C = BlendColour;
    const bool one = s.c == BlendAlpha::Fix && s.fix == 0x80;
    auto sign = [](C plus, C minus) { return plus == C::Zero && minus == C::Source ? 3 : 2; };
    if (s.a != C::Dest && s.b != C::Dest) return s.d == C::Dest ? sign(s.a, s.b) : 1;
    if (!one) return 0;
    if (s.a == C::Dest && s.b != C::Dest && s.d != C::Dest) return sign(s.d, s.b);     // Cd + D - B
    if (s.b == C::Dest && s.a != C::Dest && s.d == C::Dest) return 1;                  // A
    return 0;
}

bool ShaderBlend(const GsDrawState& s) {
    if (!s.blend) return false;
    // Into a 16-bit target: the blend's result cut to five bits a channel,
    // as the GS stores it (the blender would keep eight: the next blend over
    // it would read more than the GS has - Burnout 3's picture copied into
    // its 16-bit display buffer, blended over the last, came out 9 brighter
    // in the dark) - unless the blender can (Blender16).
    if (Is16Bit(s.psm)) return Blender16(s) == 0;
    if (s.a == s.b) return false;
    // The destination's alpha as the factor (C = Ad): over 128 on the GS,
    // over 255 as the blender reads a byte (DST_ALPHA) - at half its strength.
    // In the shader for a small target (Burnout 3's sky: its 64x32 gradient
    // blended by the alpha laid down first) and for sprites (its full-screen
    // passes - Cs * Ad + Cd over the picture, after a crash: strips side by
    // side, one copy of the picture for all), not for the race's scene: it
    // blends by Ad too, thousands of overlapping triangles, each needing a
    // fresh copy of the picture - 60 frames a second at 3x became 47. There
    // the frame's alpha is small (under 16) while the scene is drawn: half
    // strength of almost nothing.
    if (s.c == BlendAlpha::Dest) return s.fbw <= 2 || s.sprite_prim;
    return s.c == BlendAlpha::Fix && s.fix > 0x80;
}
constexpr std::uint32_t kShaderBlend = 1u << 15;

/* The pipeline a state needs: blending and its operands, and the write mask
   (bits 9-12, inverted so that 0 is the plain pipeline); bit 15 the blend
   in the shader. */
std::uint32_t PipelineKey(const GsDrawState& s) {
    const std::uint32_t masked = (~WriteMask(KeptBits(s.fbmsk, s.psm)) & 0xF) << 9;
    if (!s.blend) return masked;
    if (const int way = Blender16(s)) {
        // S written, or Cd + S, Cd - |S| (the shader's S, its factor 1): as
        // the blend Cs * FIX + Cd, or (0 - Cs) * FIX + Cd.
        if (way == 1) return masked;
        const BlendColour a = way == 3 ? BlendColour::Zero : BlendColour::Source;
        const BlendColour b = way == 3 ? BlendColour::Source : BlendColour::Zero;
        return masked | 1u | std::uint32_t(a) << 1 | std::uint32_t(b) << 3 | std::uint32_t(BlendAlpha::Fix) << 5 |
               std::uint32_t(BlendColour::Dest) << 7;
    }
    return masked | 1u | std::uint32_t(s.a) << 1 | std::uint32_t(s.b) << 3 | std::uint32_t(s.c) << 5 |
           std::uint32_t(s.d) << 7 | (ShaderBlend(s) ? kShaderBlend : 0u);
}

/* Cv = (A - B) * C + D as the blender's src * sf (op) dst * df. Each of Cs and
   Cd ends up with a coefficient (A is it) - (B is it), times C, plus (D is it):
   0, 1, C, -C, 1 - C, or 1 + C (more than the blender's factors reach; taken
   as 1). */
VkPipelineColorBlendAttachmentState Blending(std::uint32_t key, bool dual_source) {
    VkPipelineColorBlendAttachmentState b{};
    b.colorWriteMask = ~(key >> 9) & 0xF;           // R, G, B, A are bits 0-3, as Vulkan has them
    if (!(key & 1) || (key >> 13) & 3) return b;    // (a logic op replaces blending)
    if (key & kShaderBlend) return b;               // (the shader blends)
    const auto a = BlendColour((key >> 1) & 3), bb = BlendColour((key >> 3) & 3), d = BlendColour((key >> 7) & 3);
    const auto c = BlendAlpha((key >> 5) & 3);
    const bool dest_alpha = c == BlendAlpha::Dest;
    const VkBlendFactor kC = dest_alpha ? VK_BLEND_FACTOR_DST_ALPHA
                             : dual_source ? VK_BLEND_FACTOR_SRC1_ALPHA : VK_BLEND_FACTOR_SRC_ALPHA;
    const VkBlendFactor kOneMinusC = dest_alpha ? VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA
                                     : dual_source ? VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA
                                                   : VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    auto coefficient = [&](BlendColour who, bool* negative) {
        const int m = int(a == who) - int(bb == who);
        const bool plus_one = d == who;
        *negative = false;
        if (m == 0) return plus_one ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_ZERO;
        if (m == 1) return plus_one ? VK_BLEND_FACTOR_ONE : kC;
        if (plus_one) return kOneMinusC;
        *negative = true;
        return kC;
    };
    bool src_negative, dst_negative;
    b.srcColorBlendFactor = coefficient(BlendColour::Source, &src_negative);
    b.dstColorBlendFactor = coefficient(BlendColour::Dest, &dst_negative);
    b.colorBlendOp = VK_BLEND_OP_ADD;
    if (!src_negative && dst_negative) b.colorBlendOp = VK_BLEND_OP_SUBTRACT;
    else if (src_negative && !dst_negative) b.colorBlendOp = VK_BLEND_OP_REVERSE_SUBTRACT;
    else if (src_negative && dst_negative) b.srcColorBlendFactor = b.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    // The GS writes the source alpha as it is; blending is for colour only.
    b.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    b.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    b.alphaBlendOp = VK_BLEND_OP_ADD;
    b.blendEnable = VK_TRUE;
    return b;
}

bool SameBatch(const GsDrawState& x, const GsDrawState& y) {
    // Most primitives come with the very state of the one before (the
    // bridge's fast path): the same bytes are the same fields, at once.
    if (std::memcmp(&x, &y, sizeof x) == 0) return true;
    return x.tag == y.tag && x.fbp == y.fbp && x.fbw == y.fbw && x.psm == y.psm && x.fbmsk == y.fbmsk && x.scissor_x0 == y.scissor_x0 &&
           x.scissor_y0 == y.scissor_y0 && x.scissor_x1 == y.scissor_x1 && x.scissor_y1 == y.scissor_y1 &&
           x.textured == y.textured && (!x.textured || (x.texture == y.texture && x.tfx == y.tfx &&
           x.tcc == y.tcc && x.bilinear == y.bilinear && x.mip == y.mip && x.wms == y.wms && x.wmt == y.wmt && x.minu == y.minu &&
           x.maxu == y.maxu && x.minv == y.minv && x.maxv == y.maxv && x.tw == y.tw && x.th == y.th &&
           x.texture_target == y.texture_target && x.tex_fbp == y.tex_fbp && x.tex_fbw == y.tex_fbw &&
           x.tex_target_psm == y.tex_target_psm && x.tex_format == y.tex_format && x.ta0 == y.ta0 &&
           x.ta1 == y.ta1 && x.aem == y.aem)) &&
           x.alpha_test == y.alpha_test && (!x.alpha_test || (x.atst == y.atst && x.aref == y.aref && x.afail == y.afail)) &&
           // PipelineKey's inputs compared rather than two keys built a
           // primitive: the format, mask, width and sprite kind are compared
           // already, and the blend's operands and FIX are here.
           x.blend == y.blend && (!x.blend || (x.a == y.a && x.b == y.b && x.c == y.c && x.d == y.d &&
           x.fix == y.fix && x.pabe == y.pabe)) && x.depth == y.depth &&
           (!x.depth || (x.zbp == y.zbp && x.ztst == y.ztst && x.depth_write == y.depth_write)) &&
           x.fog == y.fog && (!x.fog || x.fogcol == y.fogcol) && x.date == y.date && (!x.date || x.datm == y.datm) &&
           x.sprite == y.sprite && x.sprite_prim == y.sprite_prim;
}

class VulkanGsRenderer final : public GsRenderer {
public:
    explicit VulkanGsRenderer(VulkanGpu& gpu) : g_(gpu) {}
    ~VulkanGsRenderer() override;
    bool Init(std::string* error);
    void UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                       std::uint32_t height) override;
    bool UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width, std::uint32_t height,
                           std::uint32_t levels, const std::uint8_t* data, std::size_t bytes) override;
    // (any thread: the threaded renderer asks it each draw without waiting
    // for its queue - replacements_ is the renderer thread's)
    bool HasReplacement(std::uint64_t id) const override {
        std::lock_guard<std::mutex> hold(resident_lock_);
        return resident_.count(id) != 0;
    }
    void UseReplacement(std::uint64_t key, std::uint64_t id) override { textures_[key].replacement = id; }
    bool BlockCompression() const override { return g_.bc; }
    void UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, const std::uint32_t* z) override;
    void UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, const std::uint32_t* pixels,
                      std::uint32_t width, std::uint32_t height) override;
    void ReleaseTexture(std::uint64_t key) override;
    bool HasTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm) const override {
        return targets_.count(Key{fbp, fbw, psm}) != 0;
    }
    void DrawTriangles(const GsDrawState& state, const GsVertex* vertices, std::size_t count) override;
    void DrawTriangleRun(const GsDrawState& state, const GsVertex* vertices, std::size_t per,
                         std::size_t prims) override;
    /* The batch's vertices (and ranges, where it keeps them) for `count`
       vertices, three a triangle, each with `range`. */
    void Append(const GsDrawState& state, const GsVertex* v, std::size_t count, const float* range);
    void Flush() override;
    void Kick() override {
        Flush();
        g_.Flush();
        ++frame_;
        EvictReplacements();
        SavePipelineCache();
    }
    void TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                              std::uint32_t height, std::uint32_t first_row, std::uint32_t rows, std::uint32_t first_col,
                              std::uint32_t end_col) override;
    void ReloadKeepsColour(bool keep) override { keep_colour_ = keep; }
    void DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                             std::uint32_t zpsm, std::uint32_t first_row, std::uint32_t rows) override;
    bool DisplayTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint64_t* image, int* layout,
                       std::uint32_t* width, std::uint32_t* height) override {
        auto found = targets_.find(Key{fbp, fbw, psm});
        if (found == targets_.end() || !found->second.image || found->second.layout == VK_IMAGE_LAYOUT_UNDEFINED)
            return false;
        Kick();                         // the drawing, on the queue before the picture's copy
        const Image& t = found->second;
        *image = reinterpret_cast<std::uint64_t>(t.image);
        *layout = int(t.layout);
        *width = t.pw();
        *height = t.ph();
        return true;
    }
    bool ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::vector<std::uint32_t>* z) override;
    std::uint32_t Scale() const override { return scale_; }
    void SetScale(std::uint32_t scale) override;
    void SetScaleRule(std::uint32_t fbp_begin, std::uint32_t fbp_end, std::uint32_t max_fbw) override;
    void MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) override;
    void MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) override;
    void MemoryPatch(const std::uint32_t* words, std::size_t count) override;
    bool TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                        std::uint32_t rows, std::uint32_t first_row, std::uint32_t first_col) override;
    void TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t width,
                          std::uint32_t height) override;
    bool DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                       std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows, std::uint32_t first_row,
                       std::uint32_t first_col) override;
    void DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                         std::uint32_t zpsm) override;
    void TextureFromMemory(std::uint64_t key, const GsTextureDecode& decode) override;
    void ClutFromMemory(const GsClutLoad& load) override;
    void ClutRows(const std::uint32_t* clut, std::uint32_t rows) override;
    bool CanShuffle() const override { return shuffle_pipeline_ != VK_NULL_HANDLE; }
    void ChannelShuffle(const GsShuffle& shuffle) override;
    bool CanRunEffects() const override { return effects_.ready(); }
    void RunEffect(const GsEffect& effect) override;
    bool ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                    std::uint32_t* width, std::uint32_t* height) override;
    bool ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::vector<std::uint32_t>* pixels,
                        std::uint32_t* width, std::uint32_t* height) override;
    bool ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                     std::uint32_t* height) override;
    bool ReadImage(Image& image, std::vector<std::uint32_t>* pixels);

private:
    using Key = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    Image* TargetFor(const GsDrawState& state);
    /* The depth buffer a draw uses: ZBP's, or the target's own when it tests
       nothing (dynamic rendering wants one attachment layout throughout). */
    Image* DepthFor(std::uint32_t key, std::uint32_t width, std::uint32_t height);
    bool MakeImage(Image& image, std::uint32_t width, std::uint32_t height, VkImageUsageFlags usage,
                   VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, std::uint32_t scale = 1, std::uint32_t levels = 1);
    void DestroyImage(Image& image);
    /* Device-local memory for images: pieces of 64 MB blocks, a power of two
       (64 KB to 16 MB) each, kept for reuse once freed - the textures that
       come and go a frame (a full cache, a new area of a track) made no
       allocation of the driver's each (vkAllocateMemory goes to the system's
       kernel). Larger, an allocation of its own. */
    bool ImageMemory(const VkMemoryRequirements& req, std::uint32_t type, Image& image);
    void FreeImageMemory(VkDeviceMemory memory, VkDeviceSize offset, std::uint32_t piece_class, std::uint32_t type);
    std::map<std::uint64_t, std::vector<std::pair<VkDeviceMemory, VkDeviceSize>>> image_pieces_;   // (type, class) -> free
    std::vector<VkDeviceMemory> image_blocks_;
    void Fill(Texture& texture, const std::uint32_t* pixels);
    VkPipeline PipelineFor(std::uint32_t key);
    /* The pipeline cache (SetPipelineCacheDir): every pipeline is made
       through it; the keys PipelineFor has made, kept to make again at the
       next start (WarmPipelines); written back now and then as they grow
       (SavePipelineCache: on a thread of its own, the frame not waiting). */
    VkPipelineCache cache_ = VK_NULL_HANDLE;
    std::set<std::uint32_t> known_keys_;
    bool cache_dirty_ = false;
    std::chrono::steady_clock::time_point cache_saved_{};
    std::atomic<bool> cache_saving_{false};
    void OpenPipelineCache();
    void WarmPipelines();
    void SavePipelineCache();
    /* The destination alpha test's setup pass: stencil 1 where the target's
       alpha bit equals DATM. */
    bool MakeDatePipeline();
    void DateSetup(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm);
    /* A target's descriptor for sampling it. */
    VkDescriptorSet SampledSet(Image& image);
    /* Copies of the target being drawn, when it is also the texture: one a
       size, kept (one image resized in turn - the race alternates targets of
       several sizes - waited for the GPU each time: 32 s of a race script). */
    std::map<std::pair<std::uint32_t, std::uint32_t>, Image> feedbacks_;
    /* A game has tagged its paint to be marked in the stencil (tag bit 3). */
    bool paint_mask_ = false;
    Image* feedback_ = nullptr;     /* the one the batch being recorded copies into */
    /* A batch that samples its own target reads one copy, made as it begins;
       the GS reads each primitive's texels as the ones before it left them.
       What the batch has drawn so far (GS pixels), so a primitive that reads
       there starts a batch of its own, with a new copy (Burnout 3's speed
       blur: layers each drawn over the last, as one batch read all from the
       picture before them - a fainter blur). */
    bool drawn_any_ = false;
    int drawn_x0_ = 0, drawn_y0_ = 0, drawn_x1_ = 0, drawn_y1_ = 0;     /* pixels, inclusive */
    /* The batch's sprites (DrawTriangles: those meeting another edge to edge
       reach that edge): where on the screen and in the texture, and their
       first vertex in pending_. */
    struct SpriteSpan {
        float x0, x1, y0, y1, s0, s1, t0, t1;
        std::size_t vertex;
    };
    std::vector<SpriteSpan> sprites_;
    /* A flat triangle (one Z, one Q: text, a HUD) on a target drawn larger,
       the last one (a quad is two, sharing a diagonal), and the batch's
       quads: each kept half a texel inside its texels (a letter of a font
       page off its neighbour's: dark lines along italic letters' edges),
       but for an edge it shares with another (a mesh of one picture). */
    struct FlatTriangle {
        GsVertex v[3];
        std::size_t vertex;
    };
    FlatTriangle last_flat_{};
    bool last_flat_valid_ = false;
    struct FlatQuad {
        GsVertex v[4];
        std::size_t vertex[2];          // its triangles' first vertices in pending_
        float lo[2], hi[2];             // S, T: its texels' extent
        float range[4];                 // ...and what it keeps to
    };
    std::vector<FlatQuad> quads_;
    void FlatQuadRange(FlatQuad& q, const GsDrawState& state);
    /* ...and what it reads of its target (texels; for blending in the shader
       the pixels it draws), so the copy is of that alone - a whole target a
       copy was 60 a race frame, ten megabytes each at 3x. read_all_: what it
       reads is not a rectangle (REPEAT past the texture, REGION_REPEAT). */
    bool read_any_ = false, read_all_ = false;
    int read_x0_ = 0, read_y0_ = 0, read_x1_ = 0, read_y1_ = 0;
    void Reads(int x0, int y0, int x1, int y1) {
        if (!read_any_) {
            read_x0_ = x0;
            read_y0_ = y0;
            read_x1_ = x1;
            read_y1_ = y1;
            read_any_ = true;
            return;
        }
        read_x0_ = std::min(read_x0_, x0);
        read_y0_ = std::min(read_y0_, y0);
        read_x1_ = std::max(read_x1_, x1);
        read_y1_ = std::max(read_y1_, y1);
    }
    /* The rendering pass the last batch left open, so the next one into the
       same target and depth buffer goes on in it (thousands of batches a
       frame, each beginning and ending its own, cost the GPU most of its
       time); anything else recorded ends it (VulkanGpu::end_pass). */
    std::uint32_t scale_ = 1;        // the internal resolution: targets' pixels to a GS pixel
    /* The scale an image of this width is drawn at (targets and depth
       buffers alike: one width, one scale, so the images a key names agree):
       the internal resolution, but a small one's - 320 wide or less: a
       game's scratch targets for its glow, blur and reflections, which it
       only reads blurred - at the PS2's own. Burnout 3's glow bounces
       through 64-320-wide views of one memory some fifty times a frame,
       each brought back from memory: at 4x those were a third of the GPU's
       time, and the upscaled glow ghosted (a second, offset outline around
       its words). 1024 wide or more (FBW 16 and up) too: wider than any
       picture the GS shows, a view of memory as texture pages - Burnout 3's
       1024x256 over its glow's memory, read as PSMT8H, brought back from
       memory 18 times a splitscreen frame and never drawn into: at 3x its
       reloads were a third of the swizzles. */
    std::uint32_t ScaleOf(std::uint32_t width, std::uint32_t fbp) const {
        if (fbp >= rule_begin_ && fbp < rule_end_ && width <= rule_max_fbw_ * 64) return scale_;
        return width <= 320 || width >= 1024 ? 1u : scale_;
    }
    std::uint32_t rule_begin_ = 0, rule_end_ = 0, rule_max_fbw_ = 0;       // SetScaleRule
    void DropImages();
    Image* open_target_ = nullptr;
    Image* open_depth_ = nullptr;
    void EndPass(VkCommandBuffer cmd, PassEnd why = kEndOther) {
        if (!open_target_) return;
        ++g_pass_ends[why];
        if (pass_cmd_) {
            BeginDeferredPass(cmd);
        } else {
            EndRendering(cmd);
        }
        open_target_ = open_depth_ = nullptr;
    }
    /* A rendering pass begun and ended: dynamic rendering (Vulkan 1.3, or
       1.2's extension), else (kVk12RenderPass) the same as a render pass with
       its framebuffer, whose size is `whole` (the attachments' common size,
       which the render area lies in). */
    void BeginRendering(VkCommandBuffer cmd, const VkRenderingInfo& ri, VkExtent2D whole);
    void EndRendering(VkCommandBuffer cmd) {
        if (g_vk_path == kVk12RenderPass) vkCmdEndRenderPass(cmd);
        else vkCmdEndRendering(cmd);
    }
    /* Render passes (kVk12RenderPass): a compatible one for each attachment
       set - [0] depth-stencil alone, [1] colour and depth-stencil - that
       pipelines and deferred secondaries are made for; the ones begun, by
       their attachments' load and store ops; framebuffers by their views and
       size, dropped with their images (DestroyImage). */
    VkRenderPass compatible_passes_[2] = {};
    std::unordered_map<std::uint32_t, VkRenderPass> render_passes_;
    struct FramebufferKey {
        VkImageView colour, depth;
        std::uint32_t width, height;
        bool operator==(const FramebufferKey& o) const {
            return colour == o.colour && depth == o.depth && width == o.width && height == o.height;
        }
    };
    struct FramebufferKeyHash {
        std::size_t operator()(const FramebufferKey& k) const {
            return std::hash<const void*>()(reinterpret_cast<const void*>(k.colour)) * 31 ^
                   std::hash<const void*>()(reinterpret_cast<const void*>(k.depth)) ^
                   (std::size_t(k.width) << 20 | k.height);
        }
    };
    std::unordered_map<FramebufferKey, VkFramebuffer, FramebufferKeyHash> framebuffers_;
    VkRenderPass RenderPass(bool colour, std::uint32_t ops);
    bool MakeCompatiblePasses();
    /* What a pipeline or a deferred secondary drawing into colour and depth
       (or depth alone) is made for: the compatible render pass on
       kVk12RenderPass, else none (dynamic rendering). */
    VkRenderPass CompatiblePass(bool colour) const {
        return g_vk_path == kVk12RenderPass ? compatible_passes_[colour ? 1 : 0] : VK_NULL_HANDLE;
    }
    /* Deferred passes (a tiler's - Android's - or PS2HOST_GPU_DEFER_PASSES=1):
       a pass's draws recorded into a secondary command buffer, the pass
       itself begun only as it ends - its render area then the draws' own
       (tiles outside it neither loaded nor stored), the depth-stencil
       stored only if a draw wrote it, loaded only if one read it. */
    bool defer_passes_ = false;
    VkCommandBuffer pass_cmd_ = VK_NULL_HANDLE;
    VkRenderingAttachmentInfo pass_colour_{}, pass_z_{};
    VkRect2D pass_area_{};                  // the image's whole, at the pass's start
    std::int32_t pass_x0_ = 0, pass_y0_ = 0, pass_x1_ = 0, pass_y1_ = 0;    // the draws' bounds (x1, y1 past)
    bool pass_drew_ = false, pass_depth_read_ = false, pass_depth_written_ = false;
    std::vector<VkImage> pass_sampled_;     // the images the open deferred pass's draws read
    /* Work recorded beside the open deferred pass - ahead of it, the pass
       left open - where it writes nothing the pass's draws read: no image of
       theirs (`image`, or none: VK_NULL_HANDLE) and not the CLUT buffer
       (`clut`), which every draw reads (gs.frag's set 1). Else the pass ends
       first, as always. */
    template <typename F>
    void RecordBeside(VkImage image, bool clut, F&& record) {
        if (pass_cmd_ && !clut &&
            (!image || std::find(pass_sampled_.begin(), pass_sampled_.end(), image) == pass_sampled_.end())) {
            ++g_beside;
            g_.RecordInPass(std::forward<F>(record));
        } else {
            g_.Record(std::forward<F>(record));
        }
    }
    void BeginDeferredPass(VkCommandBuffer cmd) {
        vkEndCommandBuffer(pass_cmd_);
        const VkCommandBuffer drawn = pass_cmd_;
        pass_cmd_ = VK_NULL_HANDLE;
        if (!pass_drew_) return;                // (nothing drawn: no pass at all)
        const std::int32_t right = std::int32_t(pass_area_.extent.width), bottom = std::int32_t(pass_area_.extent.height);
        const std::int32_t x0 = std::clamp(pass_x0_, 0, right), y0 = std::clamp(pass_y0_, 0, bottom);
        const std::int32_t x1 = std::clamp(pass_x1_, x0, right), y1 = std::clamp(pass_y1_, y0, bottom);
        if (x1 <= x0 || y1 <= y0) return;
        VkRenderingAttachmentInfo colour = pass_colour_, z = pass_z_;
        if (!pass_depth_written_ && g_store_op_none) {
            z.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
            ++g_depth_unstored;
        }
        if (!pass_depth_read_ && !pass_depth_written_ && g_load_op_none) {
            z.loadOp = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
            ++g_depth_unloaded;
        }
        g_area_used += std::uint64_t(x1 - x0) * std::uint64_t(y1 - y0);
        g_area_whole += std::uint64_t(right) * std::uint64_t(bottom);
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
        ri.renderArea = {{x0, y0}, {std::uint32_t(x1 - x0), std::uint32_t(y1 - y0)}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &colour;
        ri.pDepthAttachment = &z;
        ri.pStencilAttachment = &z;
        BeginRendering(cmd, ri, pass_area_.extent);
        vkCmdExecuteCommands(cmd, 1, &drawn);
        EndRendering(cmd);
    }
    VkPipeline date_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet SetFor(Texture& texture, int sampler);
    void RetireTexture(Texture texture);
    void WriteTextureSet(VkDescriptorSet set, VkSampler sampler, const Image& image);
    bool Buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer, VkDeviceMemory* memory, void** mapped);

    VulkanGpu& g_;
    VkShaderModule vert_ = VK_NULL_HANDLE, frag_ = VK_NULL_HANDLE, frag_blend_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    bool keep_colour_ = false;      /* ReloadKeepsColour */
    VkSampler samplers_[28] = {};   /* bilinear << 2 | clamp_s << 1 | clamp_t; from 8 a texture pack's
                                       (ReplacementSampler) */
    /* A texture pack's replacement, filtered by the GPU (gs.frag's
       ReplacedSample): linear in and between its levels, CLAMP's REPEAT or
       CLAMP each way (clamp_s << 1 | clamp_t) and 1, 2, 4, 8 or 16 taps of
       anisotropy (the player's ANISOTROPIC: GsDrawState::tag bits 8-12) -
       one of samplers_' from 8. */
    static int ReplacementSampler(const GsDrawState& s) {
        const int clamps = int((s.wms & 1u) << 1 | (s.wmt & 1u));
        const int taps = ((s.tag >> 11) & 3u) == 3u ? int(std::min((s.tag >> 8) & 7u, 4u)) : 0;
        return 8 + clamps * 5 + taps;
    }
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    std::map<std::uint32_t, VkPipeline> pipelines_;
    std::uint32_t last_pipeline_key_ = ~0u;
    VkPipeline last_pipeline_ = VK_NULL_HANDLE;
    /* A pipeline for a kind of draw (a key with features, Features) made by
       a thread of its own while playing: until it is ready the batch draws
       with the whole shader (the same pictures) - none made in the middle of
       a frame. Scripted runs (no pipeline cache) make them at once. */
    bool async_pipelines_ = false;
    std::vector<std::thread> pipeline_workers_;
    std::mutex pipeline_lock_;
    std::condition_variable pipeline_wake_;
    std::deque<std::uint32_t> pipeline_queue_;
    std::vector<std::pair<std::uint32_t, VkPipeline>> pipelines_made_;
    std::set<std::uint32_t> pipelines_asked_;
    std::atomic<bool> pipelines_ready_{false};
    bool pipeline_stop_ = false;
    VkPipeline MakePipeline(std::uint32_t key);
    void AskWorker(std::uint32_t key, bool soon);
    std::size_t warm_left_ = 0;         // WarmPipelines' still to come
    std::chrono::steady_clock::time_point warm_began_{};
    void KeepPipeline(std::uint32_t key, VkPipeline pipeline, double ms);
    void TakeMadePipelines();
    std::map<Key, Image> targets_;
    std::map<Key, Image> depths_;   /* by (ZBP or a target's own, width, height) */
    std::map<std::uint64_t, Texture> textures_;
    /* A texture pack's replacements, by id (UploadReplacement); those not
       drawn for the longest go when they come to over kReplacementBudget. */
    std::unordered_map<std::uint64_t, Texture> replacements_;
    VkDeviceSize replacement_bytes_ = 0;
    // Their ids, for HasReplacement from another thread.
    mutable std::mutex resident_lock_;
    std::unordered_set<std::uint64_t> resident_;
    void Resident(std::uint64_t id, bool in) {
        std::lock_guard<std::mutex> hold(resident_lock_);
        if (in) resident_.insert(id);
        else resident_.erase(id);
    }
    std::uint64_t frame_ = 0;
    static constexpr VkDeviceSize kReplacementBudget = VkDeviceSize(1536) << 20;
    void EvictReplacements();
    void DestroyTexture(Texture texture);
    Texture blank_;                 /* bound when nothing is textured */
    GsDrawState state_;
    PendingVertices pending_;
    PendingRanges ranges_;          // empty, or as many as pending_ (RangesWanted)
    /* Whether a batch can read its ranges: textured on a target drawn larger,
       a sprite or a flat quad (DrawTriangles) - at the PS2's own resolution
       none. */
    bool RangesWanted(const GsDrawState& s) const {
        return scale_ > 1 && s.textured && (s.sprite || !s.texture_target);
    }
    /* Vertex data for what is recorded: appended to chunks of host-visible
       memory, all reusable once the submission that reads them has run. */
    struct VertexChunk {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped;
        VkDeviceSize size;
    };
    // One ring a command buffer slot: a slot's vertices are free again once
    // its submission has run, whatever the others are doing.
    struct VertexRing {
        std::vector<VertexChunk> chunks;
        std::size_t chunk = 0;
        VkDeviceSize used = 0;
        bool reset_pending = false;
    };
    VertexRing vertex_rings_[VulkanGpu::kSlots];
    void* Vertices(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset);
    /* Upload data, the same way: a slot's chunks, free again once its
       submission has run. A staging buffer an upload, freed (or kept, 64 at
       most, of one size) after the submission, was the driver allocating and
       freeing memory hundreds of times a frame in a race. */
    VertexRing staging_rings_[VulkanGpu::kSlots];
    /* Released textures by size (width << 32 | height), image, memory, view and
       descriptor sets whole, for the next texture of that size: a race releases
       and makes textures by the hundred a frame, and each made one was a driver
       allocation and each released one a free. At most kPooledTextures kept. */
    std::unordered_map<std::uint64_t, std::vector<Texture>> texture_pool_;
    std::size_t pooled_textures_ = 0;
    static constexpr std::size_t kPooledTextures = 1024;
    void* Staging(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset);
    void* RingAlloc(VertexRing* rings, VkBufferUsageFlags usage, VkDeviceSize bytes, VkBuffer* buffer,
                    VkDeviceSize* offset);
    /* A staging buffer, freed once the submission that reads it has run. */
    void FreeAfterSubmit(VkBuffer buffer, VkDeviceMemory memory);
    /* Staging buffers for uploads, kept mapped and used again once the GPU
       is done with them: allocating and freeing one an upload (the race
       uploads to GPU memory tens of thousands of times) was 3% of the run
       in the driver. By size class (a power of two, 64 KB at least). */
    struct Staged {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };
    std::vector<Staged> staging_free_;
    std::unordered_map<VkBuffer, Staged> staging_live_;

    /* GS local memory on the GPU (swizzle.comp): the memory itself, and the
       linear buffer texture decodes go through (images go to and from
       memory straight: SwizzleImage, DepthOut, DepthIn). */
    bool InitMemory(std::string* error);
    bool DeviceBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer, VkDeviceMemory* memory);
    /* A texture image of this size under `key`, made (or made again) if need be. */
    Texture* TextureImage(std::uint64_t key, std::uint32_t width, std::uint32_t height);
    /* The state a command buffer's draws last set: set again only when it
       changes (a race records ~2,000 batches a picture, and most set what
       the one before did - a dozen calls into the driver each). */
    struct DrawState {
        std::uint64_t recording = ~0ull;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet sets[2] = {};
        VkDescriptorSet set2 = VK_NULL_HANDLE;     // gs.frag's set 2
        int depth_test = -1, stencil_test = -1, compare = -1, depth_write = -1;
        VkViewport viewport{};
        VkRect2D scissor{};
        float size[2] = {-1, -1};
        VkBuffer vertices = VK_NULL_HANDLE;
        VkBuffer ranges = VK_NULL_HANDLE;   // bound for all the vertices (stride 0), with `vertices`
    } drawn_;
    // kNoRange, for a batch whose ranges are read but keep nothing in.
    VkBuffer no_range_ = VK_NULL_HANDLE;
    VkDeviceMemory no_range_memory_ = VK_NULL_HANDLE;
    // The CLUT for the draws (gs.frag's set 1): a target read as PSMT8H.
    VkDescriptorSetLayout clut_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet clut_set_ = VK_NULL_HANDLE;
    VkBuffer gsmem_ = VK_NULL_HANDLE, linear_ = VK_NULL_HANDLE, clut_ = VK_NULL_HANDLE, tables_ = VK_NULL_HANDLE;
    VkDeviceMemory gsmem_memory_ = VK_NULL_HANDLE, linear_memory_ = VK_NULL_HANDLE, clut_memory_ = VK_NULL_HANDLE,
                   tables_memory_ = VK_NULL_HANDLE;     // decode.comp's layout tables
    VkShaderModule decode_ = VK_NULL_HANDLE;
    /* The decode into a texture's own image (storage image, set 1 - the
       swizzle's image layout), each texture's set from texture_storage_pool_;
       null where it would not build: the staging buffer and a copy then. */
    VkShaderModule decode_image_ = VK_NULL_HANDLE;
    VkPipeline decode_image_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool texture_storage_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet TextureStorageSet(Image& image);
    VkPipeline decode_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule swizzle_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout swizzle_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool swizzle_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet swizzle_set_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_layout_ = VK_NULL_HANDLE;
    VkPipeline swizzle_pipeline_ = VK_NULL_HANDLE;
    /* The swizzle straight to and from a colour target (StorageSet). */
    VkShaderModule swizzle_image_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout image_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool storage_pool_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_image_layout_ = VK_NULL_HANDLE;
    // MemoryPatch's: memory's set (the swizzle's), eight words in the push constants.
    VkShaderModule patch_ = VK_NULL_HANDLE;
    VkPipelineLayout patch_layout_ = VK_NULL_HANDLE;
    VkPipeline patch_pipeline_ = VK_NULL_HANDLE;
    VkPipeline swizzle_image_pipeline_ = VK_NULL_HANDLE;
    /* The same for a colour format, direction and scale (InitMemory): by
       kSpecialisedFormats' index, to memory or not, scale - 1. */
    static constexpr std::uint32_t kSpecialisedFormats[4] = {0x00, 0x01, 0x02, 0x0A};    // CT32, CT24, CT16, CT16S
    VkPipeline swizzle_image_specialised_[4][2][4] = {};
    VkDescriptorSet StorageSet(Image& image);
    void SwizzleImage(VkCommandBuffer cmd, const SwizzleParams& params, Image& image);
    /* Depth buffers: to memory sampled (the swizzle's DEPTH_IMAGE), back drawn
       (depth_load.frag) - no copy through a linear buffer either way. */
    VkShaderModule swizzle_depth_ = VK_NULL_HANDLE, depth_load_vert_ = VK_NULL_HANDLE,
                   depth_load_frag_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout depth_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout swizzle_depth_layout_ = VK_NULL_HANDLE, depth_load_layout_ = VK_NULL_HANDLE;
    VkPipeline swizzle_depth_pipeline_ = VK_NULL_HANDLE, depth_load_pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet DepthSampledSet(Image& image);
    void DepthOut(VkCommandBuffer cmd, const SwizzleParams& params, Image& depth);
    void DepthIn(VkCommandBuffer cmd, const SwizzleParams& params, Image& depth);
    bool MakeDepthPipelines();
    /* The channel shuffle (shuffle.comp): the target as storage (set 0), the
       depth buffer sampled (set 1). None if it did not build: shuffles are
       then drawn as they are. */
    VkShaderModule shuffle_ = VK_NULL_HANDLE;
    VkPipelineLayout shuffle_layout_ = VK_NULL_HANDLE;
    VkPipeline shuffle_pipeline_ = VK_NULL_HANDLE;
    void MakeShufflePipeline();
    /* Shuffles one after another, alike but for their columns - a comb of
       sprites, `run` wide every `period` (Burnout 3's fog: 40, 8 wide every
       16) - wait to be one dispatch: recorded by the next anything else
       (Flush, DrawTriangles). */
    bool shuffle_pending_ = false;
    GsShuffle shuffle_wait_;
    std::int32_t shuffle_period_ = 0, shuffle_count_ = 0;
    void RecordShuffle();
    /* The PC's effects (RunEffect, effects.cpp). */
    GsEffects effects_{g_};
    /* The scene's depth GsEffects saved, as gs.frag's set 2 (soft
       particles): made again when its view changes; null for none. */
    VkDescriptorSet scene_set_ = VK_NULL_HANDLE;
    VkImageView scene_view_ = VK_NULL_HANDLE;
    VkDescriptorSet SceneDepthSet() {
        const VkImageView view = effects_.SceneDepth();
        if (!view) return VK_NULL_HANDLE;
        if (view == scene_view_ && scene_set_) return scene_set_;
        if (!scene_set_) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &set_layout_;
            if (vkAllocateDescriptorSets(g_.device, &ai, &scene_set_) != VK_SUCCESS) return VK_NULL_HANDLE;
        }
        // (once the GPU is done with the set as it was: a scale's change, idle)
        const VkDescriptorImageInfo info[2] = {
            {samplers_[3], view, VK_IMAGE_LAYOUT_GENERAL},
            {samplers_[3], blank_.image.words, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        for (int i = 0; i < 2; ++i) {
            writes[i].dstSet = scene_set_;
            writes[i].dstBinding = std::uint32_t(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &info[i];
        }
        vkUpdateDescriptorSets(g_.device, blank_.image.words ? 2 : 1, writes, 0, nullptr);
        scene_view_ = view;
        return scene_set_;
    }
};

void* VulkanGsRenderer::Vertices(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset) {
    return RingAlloc(vertex_rings_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, bytes, buffer, offset);
}

void* VulkanGsRenderer::Staging(VkDeviceSize bytes, VkBuffer* buffer, VkDeviceSize* offset) {
    // (Not Buffer's pooled staging path: that is for TRANSFER_SRC alone.)
    return RingAlloc(staging_rings_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, bytes,
                     buffer, offset);
}

void* VulkanGsRenderer::RingAlloc(VertexRing* rings, VkBufferUsageFlags usage, VkDeviceSize bytes,
                                  VkBuffer* buffer, VkDeviceSize* offset) {
    // The slot the commands using these are recorded into.
    VertexRing& ring = rings[g_.slot()];
    if (!ring.reset_pending) {
        ring.reset_pending = true;
        g_.AfterSubmit([&ring] {
            ring.chunk = 0;
            ring.used = 0;
            ring.reset_pending = false;
        });
    }
    const VkDeviceSize aligned = (bytes + 255) & ~VkDeviceSize(255);
    while (ring.chunk < ring.chunks.size() && ring.used + aligned > ring.chunks[ring.chunk].size) {
        ++ring.chunk;
        ring.used = 0;
    }
    if (ring.chunk == ring.chunks.size()) {
        VertexChunk chunk{};
        chunk.size = std::max<VkDeviceSize>(VkDeviceSize(8) << 20, aligned);
        if (!Buffer(chunk.size, usage, &chunk.buffer, &chunk.memory, &chunk.mapped))
            return nullptr;
        ring.chunks.push_back(chunk);
        ring.used = 0;
    }
    VertexChunk& chunk = ring.chunks[ring.chunk];
    *buffer = chunk.buffer;
    *offset = ring.used;
    void* at = static_cast<std::uint8_t*>(chunk.mapped) + ring.used;
    ring.used += aligned;
    return at;
}

void VulkanGsRenderer::FreeAfterSubmit(VkBuffer buffer, VkDeviceMemory memory) {
    auto live = staging_live_.find(buffer);
    if (live != staging_live_.end()) {
        const Staged staged = live->second;
        staging_live_.erase(live);
        g_.AfterSubmit([this, staged] {
            if (staging_free_.size() < 64) {
                staging_free_.push_back(staged);
                return;
            }
            vkUnmapMemory(g_.device, staged.memory);
            vkDestroyBuffer(g_.device, staged.buffer, nullptr);
            vkFreeMemory(g_.device, staged.memory, nullptr);
        });
        return;
    }
    VkDevice device = g_.device;
    g_.AfterSubmit([device, buffer, memory] {
        vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
    });
}

bool VulkanGsRenderer::Buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer,
                              VkDeviceMemory* memory, void** mapped) {
    if (usage == VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
        VkDeviceSize size = 64 * 1024;
        while (size < bytes) size *= 2;
        for (std::size_t i = 0; i < staging_free_.size(); ++i)
            if (staging_free_[i].size == size) {
                const Staged staged = staging_free_[i];
                staging_free_[i] = staging_free_.back();
                staging_free_.pop_back();
                staging_live_[staged.buffer] = staged;
                *buffer = staged.buffer;
                *memory = staged.memory;
                *mapped = staged.mapped;
                return true;
            }
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
        vkBindBufferMemory(g_.device, *buffer, *memory, 0);
        if (vkMapMemory(g_.device, *memory, 0, size, 0, mapped) != VK_SUCCESS) return false;
        staging_live_[*buffer] = Staged{*buffer, *memory, *mapped, size};
        return true;
    }
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
    vkBindBufferMemory(g_.device, *buffer, *memory, 0);
    return vkMapMemory(g_.device, *memory, 0, bytes, 0, mapped) == VK_SUCCESS;
}

bool VulkanGsRenderer::MakeImage(Image& t, std::uint32_t width, std::uint32_t height, VkImageUsageFlags usage,
                                 VkFormat format, std::uint32_t scale, std::uint32_t levels) {
    t.width = width;
    t.height = height;
    t.scale = std::max(1u, scale);
    t.levels = std::max(1u, levels);
    t.aspect = format == kDepthFormat ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                      : VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {t.pw(), t.ph(), 1};
    ici.mipLevels = t.levels;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    t.usage = usage;
    // An RGBA8 image a draw reads is read as 32-bit words as well: four
    // texels in one gather (gs.frag's Sample). The two formats named, so a
    // driver can keep its compression.
    const bool words = format == VK_FORMAT_R8G8B8A8_UNORM && (usage & VK_IMAGE_USAGE_SAMPLED_BIT);
    const VkFormat both[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_UINT};
    VkImageFormatListCreateInfo formats{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    formats.viewFormatCount = 2;
    formats.pViewFormats = both;
    if (words) {
        ici.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        ici.pNext = &formats;
    }
    if (vkCreateImage(g_.device, &ici, nullptr, &t.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(g_.device, t.image, &req);
    if (!ImageMemory(req, g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), t)) return false;
    vkBindImageMemory(g_.device, t.image, t.memory, t.offset);
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {t.aspect, 0, t.levels, 0, 1};
    if (vkCreateImageView(g_.device, &vci, nullptr, &t.view) != VK_SUCCESS) return false;
    if (words) {
        vci.format = VK_FORMAT_R32_UINT;
        if (vkCreateImageView(g_.device, &vci, nullptr, &t.words) != VK_SUCCESS) return false;
        vci.format = format;
    }
    if (format == kDepthFormat && (usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(g_.device, &vci, nullptr, &t.depth_view) != VK_SUCCESS) return false;
        vci.subresourceRange = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(g_.device, &vci, nullptr, &t.stencil_view) != VK_SUCCESS) return false;
    }
    return true;
}

void VulkanGsRenderer::DestroyImage(Image& t) {
    if (!framebuffers_.empty() && t.view) {
        for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
            if (it->first.colour == t.view || it->first.depth == t.view) {
                vkDestroyFramebuffer(g_.device, it->second, nullptr);
                it = framebuffers_.erase(it);
            } else {
                ++it;
            }
        }
    }
    effects_.ForgetView(t.view);
    if (t.view) vkDestroyImageView(g_.device, t.view, nullptr);
    if (t.words) vkDestroyImageView(g_.device, t.words, nullptr);
    if (t.depth_view) vkDestroyImageView(g_.device, t.depth_view, nullptr);
    if (t.stencil_view) {
        effects_.ForgetView(t.stencil_view);
        vkDestroyImageView(g_.device, t.stencil_view, nullptr);
    }
    if (t.image) vkDestroyImage(g_.device, t.image, nullptr);
    if (t.memory) FreeImageMemory(t.memory, t.offset, t.piece_class, t.memory_type);
    t = Image{};
}

bool VulkanGsRenderer::ImageMemory(const VkMemoryRequirements& req, std::uint32_t type, Image& t) {
    constexpr VkDeviceSize kBlock = VkDeviceSize(64) << 20, kLargest = VkDeviceSize(16) << 20;
    std::uint32_t piece_class = 16;                 // 64 KB
    while ((VkDeviceSize(1) << piece_class) < std::max(req.size, req.alignment)) ++piece_class;
    t.memory_type = type;
    if ((VkDeviceSize(1) << piece_class) > kLargest) {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        t.piece_class = 0;
        t.offset = 0;
        return vkAllocateMemory(g_.device, &mai, nullptr, &t.memory) == VK_SUCCESS;
    }
    auto& free = image_pieces_[std::uint64_t(type) << 32 | piece_class];
    if (free.empty()) {
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = kBlock;
        mai.memoryTypeIndex = type;
        VkDeviceMemory block = VK_NULL_HANDLE;
        if (vkAllocateMemory(g_.device, &mai, nullptr, &block) != VK_SUCCESS) return false;
        image_blocks_.push_back(block);
        const VkDeviceSize piece = VkDeviceSize(1) << piece_class;
        for (VkDeviceSize at = kBlock; at >= piece; at -= piece) free.emplace_back(block, at - piece);
    }
    t.memory = free.back().first;
    t.offset = free.back().second;
    t.piece_class = piece_class;
    free.pop_back();
    return true;
}

void VulkanGsRenderer::FreeImageMemory(VkDeviceMemory memory, VkDeviceSize offset, std::uint32_t piece_class,
                                       std::uint32_t type) {
    if (!piece_class) {
        vkFreeMemory(g_.device, memory, nullptr);
        return;
    }
    image_pieces_[std::uint64_t(type) << 32 | piece_class].emplace_back(memory, offset);
}

bool VulkanGsRenderer::Init(std::string* error) {
    // Deferred passes: a tiler's (a phone's); PS2HOST_GPU_DEFER_PASSES=1 or 0 decides.
    if (const char* defer = std::getenv("PS2HOST_GPU_DEFER_PASSES")) defer_passes_ = defer[0] == '1';
#ifdef __ANDROID__
    else defer_passes_ = true;
#endif
    OpenPipelineCache();
    if (g_vk_path == kVk12RenderPass && !MakeCompatiblePasses()) {
        if (error) *error = "vkCreateRenderPass failed";
        return false;
    }
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    vert_ = module(spirv_gs_vert, sizeof spirv_gs_vert);
    frag_ = g_.dual_source ? module(spirv_gs_frag, sizeof spirv_gs_frag)
                           : module(spirv_gs_frag_single, sizeof spirv_gs_frag_single);
    g_dual_source_blend = g_.dual_source;
    frag_blend_ = module(spirv_gs_frag_blend, sizeof spirv_gs_frag_blend);
    if (!vert_ || !frag_ || !frag_blend_) {
        if (error) *error = "the GS shaders did not load";
        return false;
    }
    // The image a draw reads, and the same as 32-bit words (an image that
    // has none: the blank texture's).
    VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2;
    dli.pBindings = bindings;
    vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &set_layout_);
    VkDescriptorSetLayoutBinding clut_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                              nullptr};
    VkDescriptorSetLayoutCreateInfo cli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    cli.bindingCount = 1;
    cli.pBindings = &clut_binding;
    vkCreateDescriptorSetLayout(g_.device, &cli, nullptr, &clut_set_layout_);
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * 8192},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 8192 + 4;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes = sizes;
    vkCreateDescriptorPool(g_.device, &dpi, nullptr, &pool_);
    for (int i = 0; i < 8; ++i) {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = (i & 4) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = (i & 2) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = (i & 1) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 0.0f;
        vkCreateSampler(g_.device, &sci, nullptr, &samplers_[i]);
    }
    for (int i = 0; i < 20; ++i) {              // ReplacementSampler's
        const int clamps = i / 5, taps = 1 << (i % 5);
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = (clamps & 2) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = (clamps & 1) ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod = 16.0f;
        if (taps > 1 && g_.anisotropy > 1.0f) {
            sci.anisotropyEnable = VK_TRUE;
            sci.maxAnisotropy = std::min(float(taps), g_.anisotropy);
        }
        vkCreateSampler(g_.device, &sci, nullptr, &samplers_[8 + i]);
    }
    VkPushConstantRange push[2] = {{VK_SHADER_STAGE_VERTEX_BIT, 0, 8},
                                   {VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, sizeof(FragmentParams)}};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    // Set 2: the target's copy that blending in the shader reads (a texture's
    // layout, so a target's sampled set binds there too).
    const VkDescriptorSetLayout set_layouts[3] = {set_layout_, clut_set_layout_, set_layout_};
    lci.setLayoutCount = 3;
    lci.pSetLayouts = set_layouts;
    lci.pushConstantRangeCount = 2;
    lci.pPushConstantRanges = push;
    vkCreatePipelineLayout(g_.device, &lci, nullptr, &layout_);
    if (!PipelineFor(0) || !MakeDatePipeline()) {
        if (error) *error = "vkCreateGraphicsPipelines failed";
        return false;
    }
    const std::uint32_t white = 0xFFFFFFFFu;
    if (!MakeImage(blank_.image, 1, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        if (error) *error = "no image for the blank texture";
        return false;
    }
    Fill(blank_, &white);
    g_.end_pass = [this](VkCommandBuffer cmd) { EndPass(cmd); };
    if (!InitMemory(error)) return false;
    WarmPipelines();
    return true;
}

bool VulkanGsRenderer::DeviceBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer,
                                    VkDeviceMemory* memory) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = usage;
    if (vkCreateBuffer(g_.device, &bci, nullptr, buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_.device, *buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = g_.FindMemory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(g_.device, &mai, nullptr, memory) != VK_SUCCESS) return false;
    return vkBindBufferMemory(g_.device, *buffer, *memory, 0) == VK_SUCCESS;
}

bool VulkanGsRenderer::InitMemory(std::string* error) {
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!DeviceBuffer(kLocalMemory, usage, &gsmem_, &gsmem_memory_) ||
        !DeviceBuffer(kLinear, usage, &linear_, &linear_memory_) ||
        !DeviceBuffer(1024, usage, &clut_, &clut_memory_) ||
        !DeviceBuffer(4096, usage, &tables_, &tables_memory_)) {
        if (error) *error = "no GPU memory for GS local memory";
        return false;
    }
    {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &clut_set_layout_;
        if (vkAllocateDescriptorSets(g_.device, &ai, &clut_set_) != VK_SUCCESS) {
            if (error) *error = "no descriptor set for the CLUT";
            return false;
        }
        const VkDescriptorBufferInfo info{clut_, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = clut_set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    }
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sizeof spirv_swizzle_comp;
    ci.pCode = spirv_swizzle_comp;
    vkCreateShaderModule(g_.device, &ci, nullptr, &swizzle_);
    const VkShaderStageFlags memory_stages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding bindings[4] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, memory_stages, nullptr}};     // decode.comp's tables
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 4;
    dli.pBindings = bindings;
    vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &swizzle_set_layout_);
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &size;
    vkCreateDescriptorPool(g_.device, &dpi, nullptr, &swizzle_pool_);
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = swizzle_pool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &swizzle_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &dai, &swizzle_set_) != VK_SUCCESS) {
        if (error) *error = "no descriptor set for GS local memory";
        return false;
    }
    VkDescriptorBufferInfo infos[4] = {{gsmem_, 0, VK_WHOLE_SIZE}, {linear_, 0, VK_WHOLE_SIZE},
                                       {clut_, 0, VK_WHOLE_SIZE}, {tables_, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[4] = {};
    for (int i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = swizzle_set_;
        writes[i].dstBinding = std::uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(g_.device, 4, writes, 0, nullptr);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             std::uint32_t(std::max(sizeof(SwizzleParams), sizeof(DecodeParams)))};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &swizzle_set_layout_;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    vkCreatePipelineLayout(g_.device, &lci, nullptr, &swizzle_layout_);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = swizzle_;
    cpi.stage.pName = "main";
    cpi.layout = swizzle_layout_;
    if (!swizzle_ ||
        vkCreateComputePipelines(g_.device, cache_, 1, &cpi, nullptr, &swizzle_pipeline_) != VK_SUCCESS) {
        if (error) *error = "the swizzle shader did not load";
        return false;
    }
    {
        // The swizzle on a target's image: set 1 the image.
        VkShaderModuleCreateInfo ii{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ii.codeSize = sizeof spirv_swizzle_image_comp;
        ii.pCode = spirv_swizzle_image_comp;
        vkCreateShaderModule(g_.device, &ii, nullptr, &swizzle_image_);
        VkDescriptorSetLayoutBinding image_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo il{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        il.bindingCount = 1;
        il.pBindings = &image_binding;
        vkCreateDescriptorSetLayout(g_.device, &il, nullptr, &image_set_layout_);
        VkDescriptorPoolSize image_size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024};
        VkDescriptorPoolCreateInfo ip{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        ip.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ip.maxSets = 1024;
        ip.poolSizeCount = 1;
        ip.pPoolSizes = &image_size;
        vkCreateDescriptorPool(g_.device, &ip, nullptr, &storage_pool_);
        const VkDescriptorSetLayout layouts[2] = {swizzle_set_layout_, image_set_layout_};
        VkPipelineLayoutCreateInfo pl = lci;
        pl.setLayoutCount = 2;
        pl.pSetLayouts = layouts;
        vkCreatePipelineLayout(g_.device, &pl, nullptr, &swizzle_image_layout_);
        VkComputePipelineCreateInfo pi = cpi;
        pi.stage.module = swizzle_image_;
        pi.layout = swizzle_image_layout_;
        if (!swizzle_image_ ||
            vkCreateComputePipelines(g_.device, cache_, 1, &pi, nullptr, &swizzle_image_pipeline_) !=
                VK_SUCCESS) {
            if (error) *error = "the swizzle shader (images) did not load";
            return false;
        }
        // The common colour transfers (CT32, CT24, CT16, CT16S; each way; at
        // 1x-4x) made at start as pipelines of their own, the format,
        // direction and scale constants (swizzle.comp's specialisation) so
        // the driver drops the branches the rest take. One that fails to
        // build leaves that transfer to the general pipeline.
        const VkSpecializationMapEntry entries[] = {{0, 0, 4}, {1, 4, 4}, {2, 8, 4}, {3, 12, 4}};
        for (std::uint32_t f = 0; f < 4; ++f)
            for (std::uint32_t direction = 0; direction < 2; ++direction)
                for (std::uint32_t scale = 1; scale <= 4; ++scale) {
                    const std::uint32_t values[] = {kSpecialisedFormats[f], direction, scale, 0};
                    const VkSpecializationInfo info{4, entries, sizeof values, values};
                    pi.stage.pSpecializationInfo = &info;
                    vkCreateComputePipelines(g_.device, cache_, 1, &pi, nullptr,
                                             &swizzle_image_specialised_[f][direction][scale - 1]);
                }
    }
    if (!MakeDepthPipelines()) {
        if (error) *error = "the depth buffer shaders did not load";
        return false;
    }
    MakeShufflePipeline();
    effects_.Init(cache_);
    {
        VkShaderModuleCreateInfo pi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        pi.codeSize = sizeof spirv_patch_comp;
        pi.pCode = spirv_patch_comp;
        vkCreateShaderModule(g_.device, &pi, nullptr, &patch_);
        VkPushConstantRange words{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &swizzle_set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &words;
        vkCreatePipelineLayout(g_.device, &pl, nullptr, &patch_layout_);
        VkComputePipelineCreateInfo pp = cpi;
        pp.stage.module = patch_;
        pp.layout = patch_layout_;
        if (!patch_ || vkCreateComputePipelines(g_.device, cache_, 1, &pp, nullptr, &patch_pipeline_) != VK_SUCCESS) {
            if (error) *error = "the memory patch shader did not load";
            return false;
        }
    }
    ci.codeSize = sizeof spirv_decode_comp;
    ci.pCode = spirv_decode_comp;
    vkCreateShaderModule(g_.device, &ci, nullptr, &decode_);
    cpi.stage.module = decode_;
    if (!decode_ ||
        vkCreateComputePipelines(g_.device, cache_, 1, &cpi, nullptr, &decode_pipeline_) != VK_SUCCESS) {
        if (error) *error = "the texture decode shader did not load";
        return false;
    }
    if (swizzle_image_layout_) {
        VkShaderModuleCreateInfo di{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        di.codeSize = sizeof spirv_decode_image_comp;
        di.pCode = spirv_decode_image_comp;
        vkCreateShaderModule(g_.device, &di, nullptr, &decode_image_);
        VkComputePipelineCreateInfo dp = cpi;
        dp.stage.module = decode_image_;
        dp.layout = swizzle_image_layout_;
        if (!decode_image_ ||
            vkCreateComputePipelines(g_.device, cache_, 1, &dp, nullptr, &decode_image_pipeline_) != VK_SUCCESS)
            decode_image_pipeline_ = VK_NULL_HANDLE;
        // A set for every texture there may be: the bridge's cache, the pool of
        // spare images, and room.
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8192};
        VkDescriptorPoolCreateInfo tp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        tp.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        tp.maxSets = 8192;
        tp.poolSizeCount = 1;
        tp.pPoolSizes = &size;
        if (vkCreateDescriptorPool(g_.device, &tp, nullptr, &texture_storage_pool_) != VK_SUCCESS)
            decode_image_pipeline_ = VK_NULL_HANDLE;
    }
    // Local memory starts as zeros, the CPU's copy too; decode.comp's
    // tables copied to their buffer (mode 2), once.
    g_.Record([&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, gsmem_, 0, kLocalMemory, 0);
        DecodeParams params{};
        params.mode = 2;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_layout_, 0, 1, &swizzle_set_, 0, nullptr);
        vkCmdPushConstants(cmd, swizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
        vkCmdDispatch(cmd, 768 / 8, 1, 1);
        MemoryBarrier(cmd);
    });
    return true;
}

void VulkanGsRenderer::MakeShufflePipeline() {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = sizeof spirv_shuffle_comp;
    ci.pCode = spirv_shuffle_comp;
    if (vkCreateShaderModule(g_.device, &ci, nullptr, &shuffle_) != VK_SUCCESS) return;
    const VkDescriptorSetLayout layouts[2] = {image_set_layout_, depth_set_layout_};
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShuffleParams)};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 2;
    lci.pSetLayouts = layouts;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(g_.device, &lci, nullptr, &shuffle_layout_) != VK_SUCCESS) return;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = shuffle_;
    cpi.stage.pName = "main";
    cpi.layout = shuffle_layout_;
    if (vkCreateComputePipelines(g_.device, cache_, 1, &cpi, nullptr, &shuffle_pipeline_) != VK_SUCCESS)
        shuffle_pipeline_ = VK_NULL_HANDLE;
}

bool VulkanGsRenderer::MakeDepthPipelines() {
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    swizzle_depth_ = module(spirv_swizzle_depth_comp, sizeof spirv_swizzle_depth_comp);
    depth_load_vert_ = module(spirv_date_vert, sizeof spirv_date_vert);
    depth_load_frag_ = module(spirv_depth_load_frag, sizeof spirv_depth_load_frag);
    if (!swizzle_depth_ || !depth_load_vert_ || !depth_load_frag_) return false;
    // To memory: the swizzle with the depth buffer sampled as set 1.
    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 1;
    dli.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(g_.device, &dli, nullptr, &depth_set_layout_) != VK_SUCCESS) return false;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   std::uint32_t(std::max(sizeof(SwizzleParams), sizeof(DecodeParams)))};
    const VkDescriptorSetLayout layouts[2] = {swizzle_set_layout_, depth_set_layout_};
    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 2;
    lci.pSetLayouts = layouts;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(g_.device, &lci, nullptr, &swizzle_depth_layout_) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = swizzle_depth_;
    cpi.stage.pName = "main";
    cpi.layout = swizzle_depth_layout_;
    if (vkCreateComputePipelines(g_.device, cache_, 1, &cpi, nullptr, &swizzle_depth_pipeline_) !=
        VK_SUCCESS)
        return false;
    // Back: drawn, Z from memory as the depth, every pixel (ALWAYS) - the
    // stencil as it was.
    const VkPushConstantRange fragment_push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SwizzleParams)};
    VkPipelineLayoutCreateInfo fli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    fli.setLayoutCount = 1;
    fli.pSetLayouts = &swizzle_set_layout_;
    fli.pushConstantRangeCount = 1;
    fli.pPushConstantRanges = &fragment_push;
    if (vkCreatePipelineLayout(g_.device, &fli, nullptr, &depth_load_layout_) != VK_SUCCESS) return false;
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = depth_load_vert_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = depth_load_frag_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.renderPass = CompatiblePass(false);
    if (!pci.renderPass) pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pColorBlendState = &blending;
    pci.pDepthStencilState = &depth;
    pci.pDynamicState = &dynamic;
    pci.layout = depth_load_layout_;
    return vkCreateGraphicsPipelines(g_.device, cache_, 1, &pci, nullptr, &depth_load_pipeline_) ==
           VK_SUCCESS;
}

namespace {
/* Load and store ops in two bits each (RenderPass's key; 0 is LOAD, STORE). */
std::uint32_t LoadBits(VkAttachmentLoadOp op) {
    return op == VK_ATTACHMENT_LOAD_OP_LOAD ? 0u : op == VK_ATTACHMENT_LOAD_OP_CLEAR ? 1u
           : op == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2u : 3u;
}
std::uint32_t StoreBits(VkAttachmentStoreOp op) {
    return op == VK_ATTACHMENT_STORE_OP_STORE ? 0u : op == VK_ATTACHMENT_STORE_OP_DONT_CARE ? 1u : 2u;
}
VkAttachmentLoadOp LoadOp(std::uint32_t bits) {
    static const VkAttachmentLoadOp kOps[4] = {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                               VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_LOAD_OP_NONE_EXT};
    return kOps[bits & 3];
}
VkAttachmentStoreOp StoreOp(std::uint32_t bits) {
    static const VkAttachmentStoreOp kOps[3] = {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                                VK_ATTACHMENT_STORE_OP_NONE};
    return kOps[std::min(bits & 3, 2u)];
}
}  // namespace

/* A render pass of one subpass drawing into colour (RGBA8, if `colour`) and
   the GS's depth-stencil. `ops`: colour's load and store in bits 0-3,
   depth's in 4-7, stencil's in 8-11 (LoadBits, StoreBits). Every attachment
   starts and ends in its attachment layout: the renderer moves images
   between layouts itself, outside passes. */
VkRenderPass VulkanGsRenderer::RenderPass(bool colour, std::uint32_t ops) {
    const std::uint32_t key = ops << 1 | (colour ? 1u : 0u);
    if (auto found = render_passes_.find(key); found != render_passes_.end()) return found->second;
    VkAttachmentDescription attachments[2] = {};
    const VkAttachmentReference colour_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth_ref{colour ? 1u : 0u, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    std::uint32_t count = 0;
    if (colour) {
        VkAttachmentDescription& c = attachments[count++];
        c.format = VK_FORMAT_R8G8B8A8_UNORM;
        c.samples = VK_SAMPLE_COUNT_1_BIT;
        c.loadOp = LoadOp(ops);
        c.storeOp = StoreOp(ops >> 2);
        c.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        c.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        c.initialLayout = c.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    VkAttachmentDescription& z = attachments[count++];
    z.format = kDepthFormat;
    z.samples = VK_SAMPLE_COUNT_1_BIT;
    z.loadOp = LoadOp(ops >> 4);
    z.storeOp = StoreOp(ops >> 6);
    z.stencilLoadOp = LoadOp(ops >> 8);
    z.stencilStoreOp = StoreOp(ops >> 10);
    z.initialLayout = z.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = colour ? 1 : 0;
    subpass.pColorAttachments = colour ? &colour_ref : nullptr;
    subpass.pDepthStencilAttachment = &depth_ref;
    VkRenderPassCreateInfo rci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rci.attachmentCount = count;
    rci.pAttachments = attachments;
    rci.subpassCount = 1;
    rci.pSubpasses = &subpass;
    VkRenderPass pass = VK_NULL_HANDLE;
    if (vkCreateRenderPass(g_.device, &rci, nullptr, &pass) != VK_SUCCESS) return VK_NULL_HANDLE;
    render_passes_.emplace(key, pass);
    return pass;
}

bool VulkanGsRenderer::MakeCompatiblePasses() {
    // (Load and store ops do not make passes incompatible: the all-LOAD,
    // all-STORE ones serve, kept in render_passes_ like the rest.)
    compatible_passes_[0] = RenderPass(false, 0);
    compatible_passes_[1] = RenderPass(true, 0);
    std::fprintf(stderr, "[ps2gpu] drawing with render passes (Vulkan 1.2 without dynamic rendering)\n");
    return compatible_passes_[0] && compatible_passes_[1];
}

void VulkanGsRenderer::BeginRendering(VkCommandBuffer cmd, const VkRenderingInfo& ri, VkExtent2D whole) {
    if (g_vk_path != kVk12RenderPass) {
        vkCmdBeginRendering(cmd, &ri);
        return;
    }
    const VkRenderingAttachmentInfo* colour = ri.colorAttachmentCount ? ri.pColorAttachments : nullptr;
    const VkRenderingAttachmentInfo* depth = ri.pDepthAttachment;
    const VkRenderingAttachmentInfo* stencil = ri.pStencilAttachment ? ri.pStencilAttachment : depth;
    std::uint32_t ops = 0;
    if (colour) ops |= LoadBits(colour->loadOp) | StoreBits(colour->storeOp) << 2;
    ops |= LoadBits(depth->loadOp) << 4 | StoreBits(depth->storeOp) << 6;
    ops |= LoadBits(stencil->loadOp) << 8 | StoreBits(stencil->storeOp) << 10;
    const VkRenderPass pass = RenderPass(colour != nullptr, ops);
    const FramebufferKey key{colour ? colour->imageView : VK_NULL_HANDLE, depth->imageView, whole.width,
                             whole.height};
    VkFramebuffer& framebuffer = framebuffers_[key];
    if (!framebuffer) {
        const VkImageView views[2] = {colour ? colour->imageView : VK_NULL_HANDLE, depth->imageView};
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass = CompatiblePass(colour != nullptr);
        fci.attachmentCount = colour ? 2 : 1;
        fci.pAttachments = colour ? views : views + 1;
        fci.width = whole.width;
        fci.height = whole.height;
        fci.layers = 1;
        vkCreateFramebuffer(g_.device, &fci, nullptr, &framebuffer);
    }
    VkClearValue clears[2] = {};
    if (colour) clears[0] = colour->clearValue;
    clears[colour ? 1 : 0].depthStencil = {depth->clearValue.depthStencil.depth,
                                            stencil->clearValue.depthStencil.stencil};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = pass;
    rbi.framebuffer = framebuffer;
    rbi.renderArea = ri.renderArea;
    rbi.clearValueCount = colour ? 2 : 1;
    rbi.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rbi, (ri.flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)
                                        ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
                                        : VK_SUBPASS_CONTENTS_INLINE);
}

VkDescriptorSet VulkanGsRenderer::DepthSampledSet(Image& image) {
    if (image.depth_sampled) return image.depth_sampled;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &depth_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.depth_sampled) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{samplers_[3], image.depth_view,
                               VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.depth_sampled;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.depth_sampled;
}

void VulkanGsRenderer::DepthOut(VkCommandBuffer cmd, const SwizzleParams& params, Image& d) {
    drawn_ = DrawState{};
    Transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorSet sets[2] = {swizzle_set_, DepthSampledSet(d)};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_depth_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_depth_layout_, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, swizzle_depth_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    ++g_dispatches;
    vkCmdDispatch(cmd, (params.cols - params.col0 + 7) / 8, (params.rows + 7) / 8, 1);
    g_.Mark(cmd, kProfSwizzle);
}

void VulkanGsRenderer::DepthIn(VkCommandBuffer cmd, const SwizzleParams& params, Image& d) {
    drawn_ = DrawState{};
    Transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    z.imageView = d.view;
    z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {d.pw(), d.ph()}};
    ri.layerCount = 1;
    ri.pDepthAttachment = &z;
    ri.pStencilAttachment = &z;
    BeginRendering(cmd, ri, ri.renderArea.extent);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_load_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_load_layout_, 0, 1, &swizzle_set_, 0,
                            nullptr);
    vkCmdPushConstants(cmd, depth_load_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof params, &params);
    const VkViewport vp{0, 0, float(d.pw()), float(d.ph()), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    const std::uint32_t k = d.scale;
    const std::uint32_t y0 = std::min(params.row0, d.height) * k;
    const std::uint32_t y1 = std::min(params.row0 + params.rows, d.height) * k;
    const VkRect2D sc{{0, std::int32_t(y0)}, {std::min(params.cols, d.width) * k, y1 - std::min(y0, y1)}};
    vkCmdSetScissor(cmd, 0, 1, &sc);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    EndRendering(cmd);
    g_.Mark(cmd, kProfSwizzle);
}

VkDescriptorSet VulkanGsRenderer::StorageSet(Image& image) {
    if (image.storage) return image.storage;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = storage_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &image_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.storage) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkDescriptorImageInfo info{VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.storage;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.storage;
}

/* A texture's image as the decode's storage image (TextureFromMemory). */
VkDescriptorSet VulkanGsRenderer::TextureStorageSet(Image& image) {
    if (image.storage) return image.storage;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = texture_storage_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &image_set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.storage) != VK_SUCCESS) return image.storage = VK_NULL_HANDLE;
    VkDescriptorImageInfo info{VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = image.storage;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(g_.device, 1, &write, 0, nullptr);
    return image.storage;
}

/* The swizzle on a colour target's image itself (GENERAL, as storage). */
void VulkanGsRenderer::SwizzleImage(VkCommandBuffer cmd, const SwizzleParams& params, Image& image) {
    drawn_ = DrawState{};
    Transition(cmd, image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const VkDescriptorSet sets[2] = {swizzle_set_, StorageSet(image)};
    VkPipeline pipeline = swizzle_image_pipeline_;
    if (params.depth == 0 && params.to_memory <= 1 && params.scale >= 1 && params.scale <= 4)
        for (std::uint32_t f = 0; f < 4; ++f)
            if (params.psm == kSpecialisedFormats[f] && swizzle_image_specialised_[f][params.to_memory][params.scale - 1])
                pipeline = swizzle_image_specialised_[f][params.to_memory][params.scale - 1];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_image_layout_, 0, 2, sets, 0, nullptr);
    vkCmdPushConstants(cmd, swizzle_image_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    ++g_dispatches;
    vkCmdDispatch(cmd, (params.cols - params.col0 + 7) / 8, (params.rows + 7) / 8, 1);
    g_.Mark(cmd, kProfSwizzle);
}

void VulkanGsRenderer::MemoryWrite(std::uint32_t begin, std::uint32_t end, const std::uint8_t* bytes) {
    if (end <= begin) return;
    Flush();
    const VkDeviceSize size = VkDeviceSize(end - begin) * 256;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(size, &buffer, &offset);
    if (!mapped) return;
    std::memcpy(mapped, bytes, std::size_t(size));
    // (Local memory: read by decodes, swizzles and reloads, never by a draw.)
    RecordBeside(VK_NULL_HANDLE, false, [&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        const VkBufferCopy copy{offset, VkDeviceSize(begin) * 256, size};
        vkCmdCopyBuffer(cmd, buffer, gsmem_, 1, &copy);
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::MemoryPatch(const std::uint32_t* words, std::size_t count) {
    if (!count || !patch_pipeline_) return;
    Flush();
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, patch_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, patch_layout_, 0, 1, &swizzle_set_, 0, nullptr);
        for (std::size_t i = 0; i < count; i += 8) {
            std::uint32_t push[32] = {};
            for (std::size_t k = 0; k < 8 && i + k < count; ++k) {
                push[4 * k + 0] = words[3 * (i + k) + 0];
                push[4 * k + 1] = words[3 * (i + k) + 1];
                push[4 * k + 2] = words[3 * (i + k) + 2];
            }
            vkCmdPushConstants(cmd, patch_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
            ++g_dispatches;
            vkCmdDispatch(cmd, 1, 1, 1);
        }
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::MemoryRead(std::uint32_t begin, std::uint32_t end, std::uint8_t* bytes) {
    if (end <= begin) return;
    Flush();
    const VkDeviceSize size = VkDeviceSize(end - begin) * 256;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return;
    g_.Run([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        const VkBufferCopy copy{VkDeviceSize(begin) * 256, 0, size};
        vkCmdCopyBuffer(cmd, gsmem_, buffer, 1, &copy);
        g_.Mark(cmd, kProfMemory);
        MemoryBarrier(cmd);
    }, "a memory read-back");
    std::memcpy(bytes, mapped, std::size_t(size));
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
}

bool VulkanGsRenderer::TargetToMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm, std::uint32_t cols,
                                      std::uint32_t rows, std::uint32_t first_row, std::uint32_t first_col) {
    // The batch pending first: a target it draws into for the first time is
    // made as it goes (and the answer is the same however far behind a
    // threaded caller's queue is - MakeThreadedGsRenderer).
    Flush();
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    Image& t = found->second;
    cols = std::min(cols, t.width);
    rows = std::min(rows, t.height);
    if (first_row >= rows) return true;
    rows -= first_row;
    if (first_col >= cols) return true;
    g_.Record([&](VkCommandBuffer cmd) {
        SwizzleParams params{fbp * 32, fbw, psm, cols, rows, t.pw(), 1, 0, first_row, t.scale};
        params.col0 = first_col;
        SwizzleImage(cmd, params, t);
        MemoryBarrier(cmd);
    });
    return true;
}

void VulkanGsRenderer::TargetFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                        std::uint32_t width, std::uint32_t height) {
    Flush();
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    const bool existed = targets_.count(Key{fbp, fbw, psm}) != 0;
    Image* t = TargetFor(state);
    SwizzleParams params{fbp * 32, fbw, psm, t->width, t->height, t->pw(), 0, 0, 0, t->scale};
    params.keep_colour = existed && keep_colour_ && t->scale > 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        SwizzleImage(cmd, params, *t);
        MemoryBarrier(cmd);
    });
}

bool VulkanGsRenderer::DepthToMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height, std::uint32_t fbw,
                                     std::uint32_t zpsm, std::uint32_t cols, std::uint32_t rows,
                                     std::uint32_t first_row, std::uint32_t first_col) {
    Flush();                        // (as TargetToMemory)
    auto found = depths_.find(Key{zbp, width, height});
    if (found == depths_.end()) return false;
    Image& d = found->second;
    cols = std::min(cols, d.width);
    rows = std::min(rows, d.height);
    if (first_row >= rows) return true;
    rows -= first_row;
    if (first_col >= cols) return true;
    g_.Record([&](VkCommandBuffer cmd) {
        SwizzleParams params{zbp * 32, fbw, zpsm, cols, rows, d.pw(), 1, 1, first_row, d.scale};
        params.col0 = first_col;
        DepthOut(cmd, params, d);
        MemoryBarrier(cmd);
    });
    return true;
}

void VulkanGsRenderer::ChannelShuffle(const GsShuffle& s) {
    if (shuffle_pending_) {
        const GsShuffle& p = shuffle_wait_;
        const bool alike = s.fbp == p.fbp && s.fbw == p.fbw && s.view_fbp == p.view_fbp &&
                           s.view_psm == p.view_psm && s.y0 == p.y0 &&
                           s.y1 == p.y1 && s.du == p.du && s.dv == p.dv && s.tbp == p.tbp && s.tbw == p.tbw &&
                           s.tex_psm == p.tex_psm && s.zbp == p.zbp && s.zwidth == p.zwidth &&
                           s.zheight == p.zheight && s.write16 == p.write16 && s.ta0 == p.ta0 && s.ta1 == p.ta1 &&
                           s.aem == p.aem && s.fba == p.fba;
        const std::int32_t run = shuffle_wait_.x1 - shuffle_wait_.x0;
        const std::int32_t at = shuffle_wait_.x0 + shuffle_count_ * shuffle_period_;
        if (alike && s.x1 - s.x0 == run &&
            (shuffle_count_ == 1 ? s.x0 >= shuffle_wait_.x0 + run : s.x0 == at)) {
            if (shuffle_count_ == 1) shuffle_period_ = s.x0 - shuffle_wait_.x0;
            ++shuffle_count_;
            return;
        }
    }
    Flush();                        // (the one pending first)
    shuffle_wait_ = s;
    shuffle_period_ = s.x1 - s.x0;
    shuffle_count_ = 1;
    shuffle_pending_ = true;
}

void VulkanGsRenderer::RecordShuffle() {
    if (!shuffle_pending_) return;
    shuffle_pending_ = false;
    GsShuffle s = shuffle_wait_;
    const std::int32_t run = s.x1 - s.x0, period = shuffle_period_;
    s.x1 = s.x0 + (shuffle_count_ - 1) * period + run;
    auto target = targets_.find(Key{s.fbp, s.fbw, 0x00});
    auto depth = depths_.find(Key{s.zbp, s.zwidth, s.zheight});
    if (!shuffle_pipeline_ || target == targets_.end() || depth == depths_.end()) return;
    Image& t = target->second;
    Image& d = depth->second;
    // The pixels to visit: the 32-bit pages the sprite's 16-bit pages are
    // (a page is 64 by 64 16-bit pixels, 64 by 32 32-bit ones, the same
    // memory) - its pixels lie among them; the view's rows into the target.
    const std::uint32_t down = s.view_fbp > s.fbp ? (s.view_fbp - s.fbp) / std::max(1u, s.fbw) * 32 : 0;
    const std::uint32_t px0 = std::uint32_t(std::max(s.x0, 0)) / 64, px1 = (std::uint32_t(std::max(s.x1, 0)) + 63) / 64;
    const std::uint32_t py0 = std::uint32_t(std::max(s.y0, 0)) / 64, py1 = (std::uint32_t(std::max(s.y1, 0)) + 63) / 64;
    const std::uint32_t rx0 = std::min(px0 * 64, t.width), rx1 = std::min(px1 * 64, t.width);
    const std::uint32_t ry0 = std::min(down + py0 * 32, t.height), ry1 = std::min(down + py1 * 32, t.height);
    if (rx1 <= rx0 || ry1 <= ry0) return;
    const ShuffleParams params{s.fbp * 32, s.fbw, s.view_psm, s.view_fbp * 32, rx0, ry0, rx1 - rx0, ry1 - ry0,
                               s.x0, s.y0, s.x1, s.y1, std::max(period, 1), run, s.du, s.dv, s.tbp, s.tbw, s.tex_psm,
                               s.zbp * 32, d.width, d.height, s.write16, s.ta0, s.ta1,
                               (s.aem ? 1u : 0u) | (s.fba ? 2u : 0u), t.scale};
    g_.Record([&](VkCommandBuffer cmd) {
        drawn_ = DrawState{};
        // One after another on the same image (two sprites can write the two
        // halves of a pixel): what the one before wrote, first.
        if (t.layout == VK_IMAGE_LAYOUT_GENERAL) {
            ++g_barriers;
            VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &b, 0, nullptr, 0, nullptr);
        }
        Transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        const VkDescriptorSet sets[2] = {StorageSet(t), DepthSampledSet(d)};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shuffle_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shuffle_layout_, 0, 2, sets, 0, nullptr);
        vkCmdPushConstants(cmd, shuffle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
        ++g_dispatches;
        vkCmdDispatch(cmd, (params.cols * t.scale + 7) / 8, (params.rows * t.scale + 7) / 8, 1);
        g_.Mark(cmd, kProfSwizzle);
    });
}

void VulkanGsRenderer::RunEffect(const GsEffect& e) {
    if (!effects_.ready()) return;
    Flush();                                    // the draws before it, recorded
    auto target = targets_.find(Key{e.fbp, e.fbw, e.psm});
    if (target == targets_.end() || !target->second.image) return;
    Image& t = target->second;
    Image* d = nullptr;
    if (e.depth) {
        auto depth = depths_.find(Key{e.zbp, e.zwidth, e.zheight});
        if (depth != depths_.end() && depth->second.depth_view) d = &depth->second;
    }
    GsEffects::Target view;
    view.view = t.view;
    view.pw = t.pw();
    view.ph = t.ph();
    view.scale = t.scale;
    g_.Record([&](VkCommandBuffer cmd) {
        drawn_ = DrawState{};
        if (d) {
            Transition(cmd, *d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            view.depth_view = d->depth_view;
            view.stencil_view = d->stencil_view;
        }
        if (t.layout == VK_IMAGE_LAYOUT_GENERAL) {
            ++g_barriers;
            VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                 &b, 0, nullptr, 0, nullptr);
        }
        Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        effects_.Record(cmd, e, view);
        MemoryBarrier(cmd);                     // (what it wrote, to the draws after: the scene's depth)
        ++g_dispatches;
        g_.Mark(cmd, kProfEffects);
    });
}

void VulkanGsRenderer::TargetRowsFromMemory(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                            std::uint32_t width, std::uint32_t height, std::uint32_t first_row,
                                            std::uint32_t rows, std::uint32_t first_col, std::uint32_t end_col) {
    // An image not made yet holds nothing: all of it, then.
    if (!targets_.count(Key{fbp, fbw, psm})) return TargetFromMemory(fbp, fbw, psm, width, height);
    Flush();
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    Image* t = TargetFor(state);
    if (first_row >= t->height) return;
    rows = std::min(rows, t->height - first_row);
    SwizzleParams params{fbp * 32, fbw, psm, std::min(end_col, t->width), rows, t->pw(), 0, 0, first_row, t->scale};
    params.col0 = first_col;
    if (params.col0 >= params.cols) return;
    params.keep_colour = keep_colour_ && t->scale > 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        SwizzleImage(cmd, params, *t);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::DepthRowsFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                           std::uint32_t fbw, std::uint32_t zpsm, std::uint32_t first_row,
                                           std::uint32_t rows) {
    // The depth images are kept by size as well: one not made yet for this
    // size holds nothing, and comes whole.
    if (!depths_.count(Key{zbp, width, height})) return DepthFromMemory(zbp, width, height, fbw, zpsm);
    Flush();
    Image* d = DepthFor(zbp, width, height);
    if (first_row >= d->height) return;
    rows = std::min(rows, d->height - first_row);
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        DepthIn(cmd, SwizzleParams{zbp * 32, fbw, zpsm, d->width, rows, d->pw(), 0, 1, first_row, d->scale}, *d);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::DepthFromMemory(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                       std::uint32_t fbw, std::uint32_t zpsm) {
    Flush();
    Image* d = DepthFor(zbp, width, height);
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        DepthIn(cmd, SwizzleParams{zbp * 32, fbw, zpsm, d->width, d->height, d->pw(), 0, 1, 0, d->scale}, *d);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::SetScaleRule(std::uint32_t fbp_begin, std::uint32_t fbp_end, std::uint32_t max_fbw) {
    if (fbp_begin == rule_begin_ && fbp_end == rule_end_ && max_fbw == rule_max_fbw_) return;
    DropImages();
    rule_begin_ = fbp_begin;
    rule_end_ = fbp_end;
    rule_max_fbw_ = max_fbw;
}

void VulkanGsRenderer::SetScale(std::uint32_t scale) {
    scale = std::min(std::max(1u, scale), kMaxScale);
    if (scale == scale_) return;
    DropImages();
    scale_ = scale;
}

void VulkanGsRenderer::DropImages() {
    // Every target, depth buffer and feedback copy goes (the caller has had
    // what the GPU drew written to its local memory first); they come back
    // at the new size, from memory, as they are next asked for. The linear
    // buffer holds an image's own pixels: it grows with the scale.
    Flush();
    g_.Submit();
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(g_.device);
    }
    drawn_ = DrawState{};
    open_target_ = open_depth_ = nullptr;
    feedback_ = nullptr;
    auto drop = [&](Image& image) {
        if (image.sampled) vkFreeDescriptorSets(g_.device, pool_, 1, &image.sampled);
        if (image.storage) vkFreeDescriptorSets(g_.device, storage_pool_, 1, &image.storage);
        if (image.depth_sampled) vkFreeDescriptorSets(g_.device, pool_, 1, &image.depth_sampled);
        DestroyImage(image);
    };
    for (auto& [key, t] : targets_) drop(t);
    for (auto& [key, d] : depths_) drop(d);
    for (auto& [size, copy] : feedbacks_) drop(copy);
    targets_.clear();
    depths_.clear();
    feedbacks_.clear();
    effects_.Forget();
}

namespace {
std::string g_pipeline_cache_dir;       // SetPipelineCacheDir
std::vector<std::uint32_t> g_pipeline_hints;    // SetPipelineHints
// WarmPipelines' progress (PipelineWarmProgress): asked, and made by the workers.
std::atomic<std::size_t> g_warm_total{0}, g_warm_made{0};
std::string CacheFile(const char* name) { return g_pipeline_cache_dir + "/" + name; }
}  // namespace

void VulkanGsRenderer::OpenPipelineCache() {
    // What a last run left (the driver checks it is its own, and ignores it
    // if not - another GPU, another driver).
    std::vector<char> data;
    if (!g_pipeline_cache_dir.empty()) {
        std::ifstream in(CacheFile("gpu_pipeline_cache.bin"), std::ios::binary);
        if (in) data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vkCreatePipelineCache(g_.device, &ci, nullptr, &cache_) != VK_SUCCESS) {
        ci.initialDataSize = 0;
        ci.pInitialData = nullptr;
        if (vkCreatePipelineCache(g_.device, &ci, nullptr, &cache_) != VK_SUCCESS) cache_ = VK_NULL_HANDLE;
    }
    cache_saved_ = std::chrono::steady_clock::now();
}

/* The pipelines the game drew with before, made by the worker from the
   start (from the cache: a fraction of a millisecond each; after a new
   driver or shader, tens of milliseconds - seconds in all, which the start
   does not wait for) rather than when a race first needs one. */
void VulkanGsRenderer::WarmPipelines() {
    if (g_pipeline_cache_dir.empty()) return;
    async_pipelines_ = true;            // (a run that keeps a cache is a player's)
    std::ifstream in(CacheFile("gpu_pipelines.txt"));
    std::string word;
    std::vector<std::uint32_t> keys;
    while (in && in >> word) {
        const std::uint32_t key = std::uint32_t(std::strtoul(word.c_str(), nullptr, 16));
        if (pipelines_.count(key) || !pipelines_asked_.insert(key).second) continue;
        known_keys_.insert(key);
        keys.push_back(key);
    }
    // The game's own list after (SetPipelineHints): kept in the file only
    // once drawn with.
    for (std::uint32_t key : g_pipeline_hints)
        if (!pipelines_.count(key) && pipelines_asked_.insert(key).second) keys.push_back(key);
    if (keys.empty()) return;
    warm_began_ = std::chrono::steady_clock::now();
    warm_left_ = keys.size();
    g_warm_made = 0;
    g_warm_total = keys.size();
    for (std::uint32_t key : keys) AskWorker(key, false);
}

/* A pipeline for the worker to make: a player's draw first (`soon`), the
   start's list after. */
void VulkanGsRenderer::AskWorker(std::uint32_t key, bool soon) {
    // Several workers: a first start (no cache: a driver compiling each
    // shader - a phone's tens of milliseconds and more) made in a fraction
    // of the time, the processors otherwise idle while it waits.
    if (pipeline_workers_.empty()) {
        const unsigned processors = std::max(1u, std::thread::hardware_concurrency());
        const unsigned workers = std::clamp(processors / 2, 1u, 4u);
        for (unsigned i = 0; i < workers; ++i)
            pipeline_workers_.emplace_back([this] {
                std::unique_lock<std::mutex> hold(pipeline_lock_);
                for (;;) {
                    pipeline_wake_.wait(hold, [this] { return pipeline_stop_ || !pipeline_queue_.empty(); });
                    if (pipeline_stop_) return;
                    const std::uint32_t next = pipeline_queue_.front();
                    pipeline_queue_.pop_front();
                    hold.unlock();
                    // PS2HOST_PIPELINE_SLOW_MS: each made that much slower (a cold driver, to test).
                    static const int slow_ms = std::getenv("PS2HOST_PIPELINE_SLOW_MS") ? std::atoi(std::getenv("PS2HOST_PIPELINE_SLOW_MS")) : 0;
                    if (slow_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms));
                    const VkPipeline made = MakePipeline(next);
                    hold.lock();
                    pipelines_made_.emplace_back(next, made);
                    pipelines_ready_.store(true, std::memory_order_release);
                    g_warm_made.fetch_add(1, std::memory_order_relaxed);
                }
            });
    }
    {
        std::lock_guard<std::mutex> hold(pipeline_lock_);
        if (soon) pipeline_queue_.push_front(key);
        else pipeline_queue_.push_back(key);
    }
    pipeline_wake_.notify_one();
}

/* The cache and the keys, written when new pipelines came, at most every
   five seconds, by a thread of their own; a run ends by being ended
   (TerminateProcess), not by a destructor, so this is where they are kept. */
void VulkanGsRenderer::SavePipelineCache() {
    if (!cache_dirty_ || !cache_ || g_pipeline_cache_dir.empty() || cache_saving_.load()) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - cache_saved_ < std::chrono::seconds(5)) return;
    cache_saved_ = now;
    cache_dirty_ = false;
    std::string keys;
    for (std::uint32_t key : known_keys_) {
        char text[12];
        std::snprintf(text, sizeof text, "%08X\n", unsigned(key));
        keys += text;
    }
    cache_saving_ = true;
    std::thread([this, device = g_.device, cache = cache_, keys = std::move(keys)] {
        std::size_t size = 0;
        std::vector<char> data;
        if (vkGetPipelineCacheData(device, cache, &size, nullptr) == VK_SUCCESS && size) {
            data.resize(size);
            if (vkGetPipelineCacheData(device, cache, &size, data.data()) != VK_SUCCESS) data.clear();
            data.resize(size);
        }
        // Each written beside, then put in place: a run ended halfway leaves
        // the last whole one.
        auto write = [](const std::string& path, const char* bytes, std::size_t count) {
            {
                std::ofstream out(path + ".new", std::ios::binary | std::ios::trunc);
                if (!out) return;
                out.write(bytes, std::streamsize(count));
                if (!out) return;
            }
            std::remove(path.c_str());
            std::rename((path + ".new").c_str(), path.c_str());
        };
        if (!data.empty()) write(CacheFile("gpu_pipeline_cache.bin"), data.data(), data.size());
        write(CacheFile("gpu_pipelines.txt"), keys.data(), keys.size());
        cache_saving_ = false;
    }).detach();
}

VkPipeline VulkanGsRenderer::PipelineFor(std::uint32_t key) {
    // Batch after batch asks for the same one.
    if (key == last_pipeline_key_ && last_pipeline_) return last_pipeline_;
    if (pipelines_ready_.load(std::memory_order_acquire)) TakeMadePipelines();
    auto found = pipelines_.find(key);
    if (found != pipelines_.end()) {
        last_pipeline_key_ = key;
        last_pipeline_ = found->second;
        return found->second;
    }
    // A kind of draw's own, while playing: asked of the worker, the whole
    // shader's meanwhile (made now if it is not yet: one for many kinds - a
    // kind's own made here instead, the first time a player meets each, took
    // up to 0.7 s on an integrated GPU's cold driver).
    if (async_pipelines_ && (key >> kFeatureShift)) {
        const VkPipeline stand_in = PipelineFor(key & ((1u << kFeatureShift) - 1));
        if (stand_in) {
            if (pipelines_asked_.insert(key).second) AskWorker(key, true);
            return stand_in;
        }
    }
    const auto began = std::chrono::steady_clock::now();
    const VkPipeline pipeline = MakePipeline(key);
    if (!pipeline) return VK_NULL_HANDLE;
    KeepPipeline(key, pipeline, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
    return pipeline;
}

/* The worker's pipelines into the renderer's (the GS thread's) own. */
void VulkanGsRenderer::TakeMadePipelines() {
    std::vector<std::pair<std::uint32_t, VkPipeline>> made;
    {
        std::lock_guard<std::mutex> hold(pipeline_lock_);
        made.swap(pipelines_made_);
        pipelines_ready_.store(false, std::memory_order_relaxed);
    }
    for (const auto& [key, pipeline] : made) {
        pipelines_asked_.erase(key);
        if (warm_left_ && --warm_left_ == 0)
            std::fprintf(stderr, "[gpu] the pipelines known from the start (the last runs', the game's) made ready in %.0f ms\n",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_began_).count());
        if (!pipeline) continue;
        if (pipelines_.count(key)) vkDestroyPipeline(g_.device, pipeline, nullptr);
        else KeepPipeline(key, pipeline, -1.0);
    }
    last_pipeline_key_ = ~0u;           // (it may have been the whole shader's, standing in)
}

void VulkanGsRenderer::KeepPipeline(std::uint32_t key, VkPipeline pipeline, double ms) {
    pipelines_[key] = pipeline;
    // PS2HOST_PIPELINE_LOG=1: each pipeline made, and how long it took.
    static const bool log = std::getenv("PS2HOST_PIPELINE_LOG") != nullptr;
    if (log) {
        if (ms >= 0.0)
            std::fprintf(stderr, "[gpu] pipeline %08X made in %.2f ms (%zu)\n", unsigned(key), ms, pipelines_.size());
        else
            std::fprintf(stderr, "[gpu] pipeline %08X made by the worker (%zu)\n", unsigned(key), pipelines_.size());
    }
    if (known_keys_.insert(key).second) cache_dirty_ = true;
}

/* A pipeline for a key: blending and the write mask (PipelineKey), the
   shader's features (Features) as its specialisation. Any thread. */
VkPipeline VulkanGsRenderer::MakePipeline(std::uint32_t key) {
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = key & kShaderBlend ? frag_blend_ : frag_;
    stages[1].pName = "main";
    const std::uint32_t features = ~(key >> kFeatureShift);     // all of them for a key without
    const VkSpecializationMapEntry feature_entry{0, 0, sizeof features};
    VkSpecializationInfo specialisation{1, &feature_entry, sizeof features, &features};
    stages[1].pSpecializationInfo = &specialisation;
    const VkVertexInputBindingDescription bindings[2] = {{0, sizeof(PackedVertex), VK_VERTEX_INPUT_RATE_VERTEX},
                                                         {1, sizeof(VertexRange), VK_VERTEX_INPUT_RATE_VERTEX}};
    VkVertexInputAttributeDescription attributes[6] = {
        {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(PackedVertex, x)},
        {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(PackedVertex, r)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PackedVertex, s)},
        {3, 0, VK_FORMAT_R32_UINT, offsetof(PackedVertex, z)},
        {4, 0, VK_FORMAT_R8_UNORM, offsetof(PackedVertex, fog)},
        {5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    input.vertexBindingDescriptionCount = 2;
    input.pVertexBindingDescriptions = bindings;
    input.vertexAttributeDescriptionCount = 6;
    input.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    const VkPipelineColorBlendAttachmentState blend = Blending(key, g_.dual_source);
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    if ((key >> 13) & 3) {                          // bits 13-14: the logic op of a partly masked draw
        blending.logicOpEnable = VK_TRUE;
        blending.logicOp = ((key >> 13) & 3) == 1 ? VK_LOGIC_OP_AND : VK_LOGIC_OP_OR;
    }
    blending.attachmentCount = 1;
    blending.pAttachments = &blend;
    // Depth's test, write and comparison are set per batch (Vulkan 1.3).
    // The destination alpha test: drawn only where the setup pass set stencil 1.
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL, 1, 0, 1};
    depth.back = depth.front;
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
                                       VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
                                       VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_STENCIL_OP,
                                       VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                       VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 11;
    dynamic.pDynamicStates = dynamic_states;
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &format;
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.renderPass = CompatiblePass(true);
    if (!pci.renderPass) pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pColorBlendState = &blending;
    pci.pDepthStencilState = &depth;
    pci.pDynamicState = &dynamic;
    pci.layout = layout_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_.device, cache_, 1, &pci, nullptr, &pipeline) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return pipeline;
}

bool VulkanGsRenderer::MakeDatePipeline() {
    auto module = [&](const std::uint32_t* code, std::size_t bytes) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(g_.device, &ci, nullptr, &m);
        return m;
    };
    VkShaderModule vert = module(spirv_date_vert, sizeof spirv_date_vert);
    VkShaderModule frag = module(spirv_date_frag, sizeof spirv_date_frag);
    VkPipelineShaderStageCreateInfo stages[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.stencilTestEnable = VK_TRUE;
    depth.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 1, 1, 1};
    depth.back = depth.front;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.depthAttachmentFormat = kDepthFormat;
    rendering.stencilAttachmentFormat = kDepthFormat;
    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.renderPass = CompatiblePass(false);
    if (!pci.renderPass) pci.pNext = &rendering;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &input;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blending;
    pci.pDynamicState = &dynamic;
    pci.layout = layout_;
    const VkResult r = vkCreateGraphicsPipelines(g_.device, cache_, 1, &pci, nullptr, &date_pipeline_);
    vkDestroyShaderModule(g_.device, vert, nullptr);
    vkDestroyShaderModule(g_.device, frag, nullptr);
    return r == VK_SUCCESS;
}

VkDescriptorSet VulkanGsRenderer::SampledSet(Image& image) {
    if (image.sampled) return image.sampled;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &image.sampled) != VK_SUCCESS) return VK_NULL_HANDLE;
    WriteTextureSet(image.sampled, samplers_[3], image);
    return image.sampled;
}

/* A set of set_layout_: the image with its sampler, and its words (the
   blank texture's for an image without: a texture pack's compressed one). */
void VulkanGsRenderer::WriteTextureSet(VkDescriptorSet set, VkSampler sampler, const Image& image) {
    const VkImageView words = image.words ? image.words : blank_.image.words;
    VkDescriptorImageInfo info[2] = {{sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                     {samplers_[3], words, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkWriteDescriptorSet writes[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (int i = 0; i < 2; ++i) {
        writes[i].dstSet = set;
        writes[i].dstBinding = std::uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &info[i];
    }
    vkUpdateDescriptorSets(g_.device, words ? 2 : 1, writes, 0, nullptr);
}

void VulkanGsRenderer::DateSetup(VkCommandBuffer cmd, Image& target, Image& depth, std::uint32_t datm) {
    if (!SampledSet(target)) return;
    Transition(cmd, target, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    Transition(cmd, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    z.imageView = depth.view;
    z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingAttachmentInfo stencil = z;
    stencil.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;   // 0 everywhere first
    stencil.clearValue.depthStencil = {0.0f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {target.pw(), target.ph()}};
    ri.layerCount = 1;
    ri.pDepthAttachment = &z;
    ri.pStencilAttachment = &stencil;
    BeginRendering(cmd, ri, ri.renderArea.extent);
    drawn_ = DrawState{};           // its own pipeline, sets, viewport and scissor
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, date_pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &target.sampled, 0, nullptr);
    VkViewport vp{0, 0, float(target.pw()), float(target.ph()), 0, 1};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D all{{0, 0}, {target.pw(), target.ph()}};
    vkCmdSetScissor(cmd, 0, 1, &all);
    FragmentParams params{};
    params.textured = datm;                         // the setup shader's first word is DATM
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset, sizeof params, &params);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    g_.Mark(cmd, kProfDate);
    EndRendering(cmd);
}

VkDescriptorSet VulkanGsRenderer::SetFor(Texture& texture, int sampler) {
    VkDescriptorSet& set = texture.sets[sampler];
    if (set) return set;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set_layout_;
    if (vkAllocateDescriptorSets(g_.device, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    WriteTextureSet(set, samplers_[sampler], texture.image);
    return set;
}

void VulkanGsRenderer::Fill(Texture& texture, const std::uint32_t* pixels) {
    Image& t = texture.image;
    const VkDeviceSize bytes = VkDeviceSize(t.width) * t.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    std::memcpy(mapped, pixels, bytes);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.width, t.height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
}

void VulkanGsRenderer::UploadTexture(std::uint64_t key, const std::uint32_t* pixels, std::uint32_t width,
                                     std::uint32_t height) {
    Flush();                        // what is pending may sample the old contents
    Texture* texture = TextureImage(key, width, height);
    if (!texture) return;
    texture->replacement = 0;
    Fill(*texture, pixels);
}

bool VulkanGsRenderer::UploadReplacement(std::uint64_t id, std::uint32_t format, std::uint32_t width,
                                         std::uint32_t height, std::uint32_t levels, const std::uint8_t* data,
                                         std::size_t bytes) {
    static const VkFormat kFormats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                        VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK,
                                        VK_FORMAT_BC7_UNORM_BLOCK};
    if (format > 4 || (format != 0 && !g_.bc) || width == 0 || height == 0) return false;
    levels = std::min(8u, std::max(1u, levels));
    // Each level's place in `data`.
    std::vector<VkBufferImageCopy> copies;
    VkDeviceSize at = 0;
    for (std::uint32_t n = 0; n < levels; ++n) {
        const std::uint32_t w = std::max(1u, width >> n), h = std::max(1u, height >> n);
        const VkDeviceSize size = format == 0 ? VkDeviceSize(w) * h * 4
                                              : VkDeviceSize((w + 3) / 4) * ((h + 3) / 4) * (format == 1 ? 8 : 16);
        if (at + size > bytes) break;
        VkBufferImageCopy copy{};
        copy.bufferOffset = at;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, n, 0, 1};
        copy.imageExtent = {w, h, 1};
        copies.push_back(copy);
        at += size;
    }
    if (copies.empty()) return false;
    Flush();
    auto old = replacements_.find(id);
    if (old != replacements_.end()) {
        replacement_bytes_ -= old->second.bytes;
        DestroyTexture(old->second);
        replacements_.erase(old);
        Resident(id, false);
    }
    Texture texture;
    if (!MakeImage(texture.image, width, height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   kFormats[format], 1, std::uint32_t(copies.size()))) {
        DestroyTexture(texture);
        return false;
    }
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(at, &buffer, &offset);
    if (!mapped) {
        DestroyTexture(texture);
        return false;
    }
    std::memcpy(mapped, data, std::size_t(at));
    for (VkBufferImageCopy& copy : copies) copy.bufferOffset += offset;
    Image& t = texture.image;
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdCopyBufferToImage(cmd, buffer, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               std::uint32_t(copies.size()), copies.data());
        g_.Mark(cmd, kProfUpload);
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    });
    texture.replaced = 0x100u | (std::uint32_t(copies.size()) - 1) << 9;
    texture.last_used = frame_;
    texture.bytes = at;
    replacement_bytes_ += at;
    replacements_[id] = texture;
    Resident(id, true);
    return true;
}

void VulkanGsRenderer::EvictReplacements() {
    if (replacement_bytes_ <= kReplacementBudget) return;
    // The longest undrawn first, down to three quarters of the budget; none
    // drawn this frame or the one before.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> by_age;     // last_used, id
    for (const auto& [id, texture] : replacements_) by_age.emplace_back(texture.last_used, id);
    std::sort(by_age.begin(), by_age.end());
    for (const auto& [used, id] : by_age) {
        if (replacement_bytes_ <= kReplacementBudget / 4 * 3 || used + 2 > frame_) break;
        auto found = replacements_.find(id);
        replacement_bytes_ -= found->second.bytes;
        DestroyTexture(found->second);
        replacements_.erase(found);
        Resident(id, false);
    }
}

void VulkanGsRenderer::DestroyTexture(Texture texture) {
    // Recorded draws may still sample it: it goes after they have run.
    VkDevice device = g_.device;
    VkDescriptorPool pool = pool_;
    g_.AfterSubmit([this, device, pool, texture]() mutable {
        for (VkDescriptorSet& set : texture.sets)
            if (set) vkFreeDescriptorSets(device, pool, 1, &set);
        if (texture.image.storage) vkFreeDescriptorSets(device, texture_storage_pool_, 1, &texture.image.storage);
        if (texture.image.view) vkDestroyImageView(device, texture.image.view, nullptr);
        if (texture.image.image) vkDestroyImage(device, texture.image.image, nullptr);
        if (texture.image.memory)
            FreeImageMemory(texture.image.memory, texture.image.offset, texture.image.piece_class,
                            texture.image.memory_type);
    });
}

Texture* VulkanGsRenderer::TextureImage(std::uint64_t key, std::uint32_t width, std::uint32_t height) {
    Texture& texture = textures_[key];
    if (texture.image.width != width || texture.image.height != height) {
        if (texture.image.image) g_.Submit("a texture resized");   // recorded draws may still sample it
        for (VkDescriptorSet& set : texture.sets)
            if (set) {
                vkFreeDescriptorSets(g_.device, pool_, 1, &set);
                set = VK_NULL_HANDLE;
            }
        if (texture.image.storage) vkFreeDescriptorSets(g_.device, texture_storage_pool_, 1, &texture.image.storage);
        texture.image.storage = VK_NULL_HANDLE;
        DestroyImage(texture.image);
        auto pooled = texture_pool_.find(std::uint64_t(width) << 32 | height);
        if (pooled != texture_pool_.end() && !pooled->second.empty()) {
            texture = pooled->second.back();
            pooled->second.pop_back();
            --pooled_textures_;
            return &texture;
        }
        if (!MakeImage(texture.image, width, height,
                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           (decode_image_pipeline_ ? VK_IMAGE_USAGE_STORAGE_BIT : 0)))
            return nullptr;
    }
    return &texture;
}

void VulkanGsRenderer::TextureFromMemory(std::uint64_t key, const GsTextureDecode& d) {
    Flush();                        // what is pending may sample the old contents
    // Mipmap levels below level 0, each at the left edge (gs.frag's Texel).
    const std::uint32_t levels = std::min(7u, std::max(1u, d.levels));
    std::uint32_t height = 0;
    for (std::uint32_t n = 0; n < levels; ++n) height += std::max(1u, d.th >> n);
    Texture* texture = TextureImage(key, d.tw, height);
    if (!texture) return;
    // A deferred pass's draws sampled it already: they keep its image (and
    // their sets), retired once they have run; the new contents go into
    // another - the pass left open, not ended for the decode.
    if (pass_cmd_ && texture->image.image &&
        std::find(pass_sampled_.begin(), pass_sampled_.end(), texture->image.image) != pass_sampled_.end()) {
        const Texture old = *texture;
        *texture = Texture{};
        texture = TextureImage(key, d.tw, height);
        if (!texture || !texture->image.image) {
            if (texture) *texture = old;        // (no image to be had: the pass ends after all)
            else textures_[key] = old;
            texture = &textures_[key];
        } else {
            texture->replaced = old.replaced;
            texture->last_used = old.last_used;
            RetireTexture(old);
            ++g_renamed;
        }
    }
    texture->replacement = 0;
    Image& t = texture->image;
    // Straight into the image where it can be (see decode_image_pipeline_):
    // one barrier for what wrote local memory and the CLUT before, the
    // image's layout changes around the levels' dispatches - not a staging
    // buffer, a copy and two full barriers a level, which on an integrated
    // GPU were more than half its frame.
    const VkDescriptorSet direct =
        decode_image_pipeline_ && (t.usage & VK_IMAGE_USAGE_STORAGE_BIT) ? TextureStorageSet(t) : VK_NULL_HANDLE;
    if (direct) {
        RecordBeside(t.image, d.clut != nullptr, [&](VkCommandBuffer cmd) {
            MemoryBarrier(cmd);
            if (d.clut) {
                vkCmdUpdateBuffer(cmd, clut_, 0, 1024, d.clut);
                MemoryBarrier(cmd);
            }
            drawn_ = DrawState{};
            Transition(cmd, t, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            const VkDescriptorSet sets[2] = {swizzle_set_, direct};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_image_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_image_layout_, 0, 2, sets, 0, nullptr);
            std::uint32_t y = 0;
            for (std::uint32_t n = 0; n < levels; ++n) {
                const std::uint32_t w = std::max(1u, d.tw >> n), h = std::max(1u, d.th >> n);
                DecodeParams params{n ? d.level_tbp[n] : d.tbp, n ? std::max(1u, d.level_tbw[n]) : d.tbw, d.psm,
                                    w, h, d.wms, d.wmt, d.minu >> n, d.maxu >> n, d.minv >> n, d.maxv >> n,
                                    d.texa_lo, d.ta1, d.csa};
                params.row0 = y;
                vkCmdPushConstants(cmd, swizzle_image_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
                ++g_dispatches;
                vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
                g_.Mark(cmd, kProfDecode);
                y += h;
            }
            Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        });
        return;
    }
    RecordBeside(t.image, d.clut != nullptr, [&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        if (d.clut) vkCmdUpdateBuffer(cmd, clut_, 0, 1024, d.clut);
        MemoryBarrier(cmd);
        drawn_ = DrawState{};
        std::uint32_t y = 0;
        for (std::uint32_t n = 0; n < levels; ++n) {
            const std::uint32_t w = std::max(1u, d.tw >> n), h = std::max(1u, d.th >> n);
            const DecodeParams params{n ? d.level_tbp[n] : d.tbp, n ? std::max(1u, d.level_tbw[n]) : d.tbw, d.psm,
                                      w, h, d.wms, d.wmt, d.minu >> n, d.maxu >> n, d.minv >> n, d.maxv >> n,
                                      d.texa_lo, d.ta1, d.csa};
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_pipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_layout_, 0, 1, &swizzle_set_, 0,
                                    nullptr);
            vkCmdPushConstants(cmd, swizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
            ++g_dispatches;
            vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
            g_.Mark(cmd, kProfDecode);
            MemoryBarrier(cmd);
            Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkBufferImageCopy copy{};
            copy.bufferRowLength = w;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageOffset = {0, std::int32_t(y), 0};
            copy.imageExtent = {std::min(w, t.width), std::min(h, t.height - y), 1};
            vkCmdCopyBufferToImage(cmd, linear_, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            g_.Mark(cmd, kProfDecode);
            y += h;
            // The next level's decode writes where this copy reads.
            if (n + 1 < levels) MemoryBarrier(cmd);
        }
        Transition(cmd, t, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::ClutFromMemory(const GsClutLoad& load) {
    Flush();                        // what is pending may draw the CLUT, or sample through the old one
    DecodeParams params{load.cbp, 1, load.cpsm, load.eight ? 256u : 16u, 1, 0, 0, 0, 0, 0, 0,
                        load.texa_lo, load.ta1, load.csa};
    params.mode = 1;
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        drawn_ = DrawState{};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, decode_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, swizzle_layout_, 0, 1, &swizzle_set_, 0,
                                nullptr);
        vkCmdPushConstants(cmd, swizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
        vkCmdDispatch(cmd, (params.tw + 7) / 8, 1, 1);
        g_.Mark(cmd, kProfDecode);
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::ClutRows(const std::uint32_t* clut, std::uint32_t rows) {
    rows &= 0xFFFF;
    if (!rows) return;
    Flush();                        // what is pending samples through the old rows
    g_.Record([&](VkCommandBuffer cmd) {
        MemoryBarrier(cmd);
        for (std::uint32_t r = 0; r < 16;) {
            if (!(rows >> r & 1)) {
                ++r;
                continue;
            }
            std::uint32_t e = r;
            while (e < 16 && (rows >> e & 1)) ++e;
            vkCmdUpdateBuffer(cmd, clut_, VkDeviceSize(r) * 64, VkDeviceSize(e - r) * 64, clut + r * 16);
            r = e;
        }
        MemoryBarrier(cmd);
    });
}

void VulkanGsRenderer::UploadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                    const std::uint32_t* pixels, std::uint32_t width, std::uint32_t height) {
    Flush();                        // what is pending draws into the old contents first
    GsDrawState state;
    state.fbp = fbp;
    state.fbw = fbw;
    state.psm = psm;
    state.width = width;
    state.height = height;
    Image* t = TargetFor(state);
    const std::uint32_t w = std::min(width, t->width), h = std::min(height, t->height);
    const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    for (std::uint32_t y = 0; y < h; ++y)
        std::memcpy(static_cast<std::uint8_t*>(mapped) + std::size_t(y) * w * 4, pixels + std::size_t(y) * width,
                    std::size_t(w) * 4);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, *t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
    });
}

void VulkanGsRenderer::ReleaseTexture(std::uint64_t key) {
    auto found = textures_.find(key);
    if (found == textures_.end()) return;
    Flush();
    Texture texture = found->second;
    textures_.erase(found);
    RetireTexture(texture);
}

void VulkanGsRenderer::RetireTexture(Texture texture) {
    texture.replacement = 0;
    // Recorded draws may still sample it: it goes after they have run - into
    // the pool, or away.
    if (texture.image.image && pooled_textures_ < kPooledTextures) {
        ++pooled_textures_;
        g_.AfterSubmit([this, texture] {
            texture_pool_[std::uint64_t(texture.image.width) << 32 | texture.image.height].push_back(texture);
        });
        return;
    }
    VkDevice device = g_.device;
    VkDescriptorPool pool = pool_;
    g_.AfterSubmit([this, device, pool, texture]() mutable {
        for (VkDescriptorSet& set : texture.sets)
            if (set) vkFreeDescriptorSets(device, pool, 1, &set);
        if (texture.image.storage) vkFreeDescriptorSets(device, texture_storage_pool_, 1, &texture.image.storage);
        if (texture.image.view) vkDestroyImageView(device, texture.image.view, nullptr);
        if (texture.image.image) vkDestroyImage(device, texture.image.image, nullptr);
        if (texture.image.memory)
            FreeImageMemory(texture.image.memory, texture.image.offset, texture.image.piece_class,
                            texture.image.memory_type);
    });
}

Image* VulkanGsRenderer::TargetFor(const GsDrawState& state) {
    Image& t = targets_[Key{state.fbp, state.fbw, state.psm}];
    if (t.image) return &t;
    MakeImage(t, state.width, state.height,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
              VK_FORMAT_R8G8B8A8_UNORM, ScaleOf(state.width, state.fbp));
    // Local memory starts as zeros.
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearColorValue zero{};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        g_.Mark(cmd, kProfClear);
    });
    return &t;
}

Image* VulkanGsRenderer::DepthFor(std::uint32_t key, std::uint32_t width, std::uint32_t height) {
    Image& d = depths_[Key{key, width, height}];
    if (d.image) return &d;
    MakeImage(d, width, height,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              kDepthFormat, ScaleOf(width, key & 0xFFFFu));
    // Local memory starts as zeros: the farthest Z.
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkClearDepthStencilValue zero{0.0f, 0};
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, d.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
        g_.Mark(cmd, kProfClear);
    });
    return &d;
}

void VulkanGsRenderer::UploadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                   const std::uint32_t* z) {
    Flush();
    Image* d = DepthFor(zbp, width, height);
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    void* mapped = Staging(bytes, &buffer, &offset);
    if (!mapped) return;
    float* out = static_cast<float*>(mapped);
    for (std::size_t i = 0; i < std::size_t(width) * height; ++i) out[i] = float(double(z[i]) / 4294967296.0);
    g_.Record([&](VkCommandBuffer cmd) {
        Transition(cmd, *d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, d->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        g_.Mark(cmd, kProfUpload);
    });
}

void VulkanGsRenderer::DrawTriangles(const GsDrawState& state, const GsVertex* v, std::size_t count) {
    RecordShuffle();
    if (!SameBatch(state, state_)) Flush();
    // A primitive sampling the target it draws into, where the batch has
    // drawn already: the batch so far first (the next takes a new copy).
    const bool feedback = state.textured && state.texture_target && state.tex_fbp == state.fbp &&
                          state.tex_fbw == state.fbw && count >= 3;
    const bool shader_blend = ShaderBlend(state) && count >= 3;
    // An alpha test whose failing pixels still write (AFAIL FB_ONLY or
    // ZB_ONLY with depth written, RGB_ONLY) is drawn in two passes over the
    // batch - the passing pixels of all its primitives, then the failing
    // ones: a primitive over one drawn before it in the batch begins a batch
    // of its own, or the earlier one's failing pixels land on top of it
    // (Burnout 3's crash titles: the black outline drawn four times, the
    // letters' fill over them - the outline's soft edges came out on top, as
    // dark streaks through the letters).
    const bool depth_written = state.depth && state.depth_write;
    const bool two_pass = state.alpha_test && state.atst != 1 && count >= 3 &&
                          (((state.afail == 1 || state.afail == 2) && depth_written) || state.afail == 3);
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    if (feedback || shader_blend || two_pass) {
        float fx0 = 1e30f, fy0 = 1e30f, fx1 = -1e30f, fy1 = -1e30f;
        float u0 = 1e30f, v0 = 1e30f, u1 = -1e30f, v1 = -1e30f;
        const float tw = float(state.tw ? state.tw : 2048), th = float(state.th ? state.th : 2048);
        for (std::size_t i = 0; i < count; ++i) {
            fx0 = std::min(fx0, v[i].x);
            fy0 = std::min(fy0, v[i].y);
            fx1 = std::max(fx1, v[i].x);
            fy1 = std::max(fy1, v[i].y);
            const float q = v[i].q != 0.0f ? v[i].q : 1.0f;
            const float u = v[i].s / q * tw, w = v[i].t / q * th;
            u0 = std::min(u0, u);
            v0 = std::min(v0, w);
            u1 = std::max(u1, u);
            v1 = std::max(v1, w);
        }
        // The pixels it covers, and the texels it reads (two by two around
        // the coordinate when bilinear).
        x0 = int(std::floor(fx0));
        y0 = int(std::floor(fy0));
        x1 = int(std::ceil(fx1)) - 1;
        y1 = int(std::ceil(fy1)) - 1;
        const float b = state.bilinear ? 0.5f : 0.0f;
        const int tu0 = int(std::floor(u0 - b)), tv0 = int(std::floor(v0 - b));
        const int tu1 = int(std::ceil(u1 + b)) - 1, tv1 = int(std::ceil(v1 + b)) - 1;
        const auto overlaps = [&](int l, int t, int r, int bottom) {
            return drawn_any_ && r >= drawn_x0_ && l <= drawn_x1_ && bottom >= drawn_y0_ && t <= drawn_y1_;
        };
        if ((feedback && overlaps(tu0, tv0, tu1, tv1)) || ((shader_blend || two_pass) && overlaps(x0, y0, x1, y1)))
            Flush();
        // What the copy must hold: the texels as CLAMP leaves them.
        if (feedback) {
            const int w = int(tw), h = int(th);
            auto axis = [&](int lo, int hi, std::uint32_t mode, std::uint32_t min, std::uint32_t max, int size,
                            int* a, int* b) {
                if (mode == 1) lo = std::clamp(lo, 0, size - 1), hi = std::clamp(hi, 0, size - 1);
                else if (mode == 2) lo = std::clamp(lo, int(min), int(max)), hi = std::clamp(hi, int(min), int(max));
                else if (mode == 3 || lo < 0 || hi >= size) return false;
                *a = lo;
                *b = hi;
                return true;
            };
            int ra = 0, rb = 0, rc = 0, rd = 0;
            if (axis(tu0, tu1, state.wms, state.minu, state.maxu, w, &ra, &rb) &&
                axis(tv0, tv1, state.wmt, state.minv, state.maxv, h, &rc, &rd))
                Reads(ra, rc, rb, rd);
            else
                read_all_ = true;
        }
        if (shader_blend) Reads(x0, y0, x1, y1);
    }
    state_ = state;
    if (feedback || shader_blend || two_pass) {
        if (!drawn_any_) {
            drawn_x0_ = x0;
            drawn_y0_ = y0;
            drawn_x1_ = x1;
            drawn_y1_ = y1;
        } else {
            drawn_x0_ = std::min(drawn_x0_, x0);
            drawn_y0_ = std::min(drawn_y0_, y0);
            drawn_x1_ = std::max(drawn_x1_, x1);
            drawn_y1_ = std::max(drawn_y1_, y1);
        }
        drawn_any_ = true;
    }
    // A sprite's S and T at its first and last pixels (the GS samples a
    // pixel at its whole coordinates: x from ceil(x0) to ceil(x1) - 1): no
    // further than the GS reads - a glyph of a font's page stays off its
    // neighbour's texels (a dark line between letters). Where a sprite meets
    // another of the batch edge to edge, on the screen and in the texture
    // (strips of a movie, a picture), both reach the outer edge of their
    // pixels there, half a pixel on, and meet as one: no seam either.
    // (Not kept: a range no coordinate reaches.)
    float range[4] = {-1e9f, -1e9f, 1e9f, 1e9f};
    if (state.sprite && count >= 3) {
        SpriteSpan sp{};
        sp.x0 = sp.x1 = v[0].x;
        sp.y0 = sp.y1 = v[0].y;
        for (std::size_t i = 0; i < count; ++i) {
            const float q = v[i].q != 0.0f ? v[i].q : 1.0f;
            if (v[i].x <= sp.x0) sp.x0 = v[i].x, sp.s0 = v[i].s / q;
            if (v[i].x >= sp.x1) sp.x1 = v[i].x, sp.s1 = v[i].s / q;
            if (v[i].y <= sp.y0) sp.y0 = v[i].y, sp.t0 = v[i].t / q;
            if (v[i].y >= sp.y1) sp.y1 = v[i].y, sp.t1 = v[i].t / q;
        }
        // The coordinate at a's position along an axis: its first or last
        // pixel's, `out` pixels further out.
        auto at = [](float a0, float a1, float c0, float c1, bool last_end, float out) {
            const float first = std::ceil(a0), last = std::max(first, std::ceil(a1) - 1.0f);
            const float per = a1 > a0 ? (c1 - c0) / (a1 - a0) : 0.0f;
            return last_end ? c0 + (last + out - a0) * per : c0 + (first - out - a0) * per;
        };
        range[0] = at(sp.x0, sp.x1, sp.s0, sp.s1, false, 0.0f);
        range[2] = at(sp.x0, sp.x1, sp.s0, sp.s1, true, 0.0f);
        range[1] = at(sp.y0, sp.y1, sp.t0, sp.t1, false, 0.0f);
        range[3] = at(sp.y0, sp.y1, sp.t0, sp.t1, true, 0.0f);
        {
            // (A target's strips too: the frame's copy to the display buffer in
            // 32-pixel strips - a faint line through text at each. The race's
            // glow, over strips of a buffer cut up in memory, comes a strip a
            // batch: those keep to the GS's reach.)
            auto same = [](float a, float b) { return std::fabs(a - b) <= 1e-4f * std::max(1.0f, std::fabs(a)); };
            const std::size_t first_vertex = pending_.size();
            for (std::size_t n = sprites_.size(); n-- > 0 && sprites_.size() - n <= 256;) {
                SpriteSpan& o = sprites_[n];
                auto widen = [&](int mine, int theirs, float value_mine, float value_theirs) {
                    range[mine] = value_mine;
                    for (int k = 0; k < 6 && o.vertex + k < ranges_.size(); ++k)
                        ranges_[o.vertex + k].range[theirs] = value_theirs;
                };
                if (same(o.y0, sp.y0) && same(o.y1, sp.y1) && same(o.t0, sp.t0) && same(o.t1, sp.t1)) {
                    if (same(o.x1, sp.x0) && same(o.s1, sp.s0))         // it, then this one to its right
                        widen(0, 2, at(sp.x0, sp.x1, sp.s0, sp.s1, false, 0.5f),
                              at(o.x0, o.x1, o.s0, o.s1, true, 0.5f));
                    else if (same(o.x0, sp.x1) && same(o.s0, sp.s1))    // this one to its left
                        widen(2, 0, at(sp.x0, sp.x1, sp.s0, sp.s1, true, 0.5f),
                              at(o.x0, o.x1, o.s0, o.s1, false, 0.5f));
                }
                if (same(o.x0, sp.x0) && same(o.x1, sp.x1) && same(o.s0, sp.s0) && same(o.s1, sp.s1)) {
                    if (same(o.y1, sp.y0) && same(o.t1, sp.t0))         // it, then this one below
                        widen(1, 3, at(sp.y0, sp.y1, sp.t0, sp.t1, false, 0.5f),
                              at(o.y0, o.y1, o.t0, o.t1, true, 0.5f));
                    else if (same(o.y0, sp.y1) && same(o.t0, sp.t1))    // this one above
                        widen(3, 1, at(sp.y0, sp.y1, sp.t0, sp.t1, true, 0.5f),
                              at(o.y0, o.y1, o.t0, o.t1, false, 0.5f));
                }
            }
            sp.vertex = first_vertex;
            sprites_.push_back(sp);
        }
    }
    const bool flat = scale_ > 1 && state.textured && !state.sprite && !state.texture_target && count == 3 &&
                      v[0].z == v[1].z && v[1].z == v[2].z && v[0].q == v[1].q && v[1].q == v[2].q;
    if (!flat) last_flat_valid_ = false;
    if (flat) {
        auto same = [](const GsVertex& a, const GsVertex& b) {
            return a.x == b.x && a.y == b.y && a.s == b.s && a.t == b.t;
        };
        int shared = 0, lone = -1;
        if (last_flat_valid_) {
            for (int i = 0; i < 3; ++i) {
                bool found = false;
                for (int j = 0; j < 3; ++j) found = found || same(v[i], last_flat_.v[j]);
                if (found) ++shared;
                else lone = i;
            }
        }
        if (shared == 2 && lone >= 0) {
            // The second of a quad: the two kept to the quad's texels.
            FlatQuad q{};
            q.v[0] = last_flat_.v[0];
            q.v[1] = last_flat_.v[1];
            q.v[2] = last_flat_.v[2];
            q.v[3] = v[lone];
            q.vertex[0] = last_flat_.vertex;
            q.vertex[1] = pending_.size();
            FlatQuadRange(q, state);
            for (int n = 0; n < 4; ++n) range[n] = q.range[n];
            for (int k = 0; k < 3; ++k)
                for (int n = 0; n < 4 && q.vertex[0] + k < ranges_.size(); ++n) ranges_[q.vertex[0] + k].range[n] = q.range[n];
            quads_.push_back(q);
            last_flat_valid_ = false;
        } else {
            last_flat_.v[0] = v[0];
            last_flat_.v[1] = v[1];
            last_flat_.v[2] = v[2];
            last_flat_.vertex = pending_.size();
            last_flat_valid_ = true;
        }
    }
    Append(state, v, count, range);
}

/* A run of primitives in one state (the bridge's fast path, handed over as
   one): appended at once where none of what DrawTriangles does a primitive
   at a time applies - a target sampled as it is drawn, a blend in the
   shader, an alpha test in two passes, a sprite's or flat quad's range on a
   target drawn larger. Else each as DrawTriangles. */
void VulkanGsRenderer::DrawTriangleRun(const GsDrawState& state, const GsVertex* v, std::size_t per,
                                       std::size_t prims) {
    const bool feedback = state.textured && state.texture_target && state.tex_fbp == state.fbp &&
                          state.tex_fbw == state.fbw;
    const bool depth_written = state.depth && state.depth_write;
    const bool two_pass = state.alpha_test && state.atst != 1 &&
                          (((state.afail == 1 || state.afail == 2) && depth_written) || state.afail == 3);
    if (per < 3 || per % 3 || feedback || ShaderBlend(state) || two_pass || state.sprite ||
        (scale_ > 1 && state.textured && !state.texture_target)) {
        for (std::size_t i = 0; i < prims; ++i) DrawTriangles(state, v + i * per, per);
        return;
    }
    RecordShuffle();
    if (!SameBatch(state, state_)) Flush();
    state_ = state;
    last_flat_valid_ = false;
    Append(state, v, per * prims, kNoRange.range);
}

void VulkanGsRenderer::Append(const GsDrawState& state, const GsVertex* v, std::size_t count, const float* range) {
    // The ranges kept in step with the vertices where this batch can read
    // them (the earlier vertices' given none, should the batch start without).
    VertexRange* range_out = nullptr;
    if (RangesWanted(state) || !ranges_.empty()) {
        const std::size_t behind = pending_.size() - ranges_.size();
        VertexRange* fill = ranges_.Grow(behind + count / 3 * 3);
        for (std::size_t i = 0; i < behind; ++i) fill[i] = kNoRange;
        range_out = fill + behind;
        for (std::size_t i = 0; i < count / 3 * 3; ++i)
            for (int n = 0; n < 4; ++n) range_out[i].range[n] = range[n];
    }
    PackedVertex* out = pending_.Grow(count / 3 * 3);
    for (std::size_t i = 0; i + 2 < count; i += 3) {
        for (int k = 0; k < 3; ++k, ++out) {
            // Flat shading takes the colour of the triangle's last vertex.
            const GsVertex& c = state.gouraud ? v[i + k] : v[i + 2];
            const GsVertex& p = v[i + k];
            out->x = p.x;
            out->y = p.y;
            out->r = c.r;
            out->g = c.g;
            out->b = c.b;
            out->a = c.a;
            out->s = p.s;
            out->t = p.t;
            out->q = p.q;
            out->z = p.z;
            out->fog = p.fog;
        }
    }
}

void VulkanGsRenderer::FlatQuadRange(FlatQuad& q, const GsDrawState& state) {
    const float tw = float(state.tw ? state.tw : 2048), th = float(state.th ? state.th : 2048);
    for (int a = 0; a < 2; ++a) {
        q.lo[a] = 1e30f;
        q.hi[a] = -1e30f;
    }
    for (const GsVertex& p : q.v) {
        const float w = p.q != 0.0f ? p.q : 1.0f;
        q.lo[0] = std::min(q.lo[0], p.s / w);
        q.hi[0] = std::max(q.hi[0], p.s / w);
        q.lo[1] = std::min(q.lo[1], p.t / w);
        q.hi[1] = std::max(q.hi[1], p.t / w);
    }
    // Half a texel in (bilinear reads a texel either side of the coordinate).
    const float in[2] = {(state.bilinear ? 0.5f : 0.01f) / tw, (state.bilinear ? 0.5f : 0.01f) / th};
    bool open_lo[2] = {false, false}, open_hi[2] = {false, false};
    // An edge shared with another quad of the batch (both its corners, in
    // place and texture): that side not kept, on either.
    auto corner_on = [](const FlatQuad& f, const GsVertex& p, int axis, bool high) {
        const float w = p.q != 0.0f ? p.q : 1.0f;
        const float c = axis == 0 ? p.s / w : p.t / w;
        return std::fabs(c - (high ? f.hi[axis] : f.lo[axis])) <= 1e-6f;
    };
    for (std::size_t n = quads_.size(); n-- > 0 && quads_.size() - n <= 256;) {
        FlatQuad& o = quads_[n];
        int shared[2], found = 0;
        for (int i = 0; i < 4 && found < 2; ++i)
            for (int j = 0; j < 4; ++j)
                if (q.v[i].x == o.v[j].x && q.v[i].y == o.v[j].y && q.v[i].s == o.v[j].s && q.v[i].t == o.v[j].t) {
                    shared[found++] = i;
                    break;
                }
        if (found < 2) continue;
        bool changed = false;
        for (int axis = 0; axis < 2; ++axis)
            for (int high = 0; high < 2; ++high) {
                if (!corner_on(q, q.v[shared[0]], axis, high) || !corner_on(q, q.v[shared[1]], axis, high)) continue;
                (high ? open_hi : open_lo)[axis] = true;
                // ...and the other quad's side through the same corners.
                for (int other = 0; other < 2; ++other)
                    if (corner_on(o, q.v[shared[0]], axis, other) && corner_on(o, q.v[shared[1]], axis, other)) {
                        o.range[axis + (other ? 2 : 0)] = other ? o.hi[axis] : o.lo[axis];
                        changed = true;
                    }
            }
        if (changed)
            for (int t = 0; t < 2; ++t)
                for (int k = 0; k < 3; ++k)
                    for (int c = 0; c < 4 && o.vertex[t] + k < ranges_.size(); ++c) ranges_[o.vertex[t] + k].range[c] = o.range[c];
    }
    for (int axis = 0; axis < 2; ++axis) {
        q.range[axis] = open_lo[axis] ? q.lo[axis] : std::min(q.lo[axis] + in[axis], 0.5f * (q.lo[axis] + q.hi[axis]));
        q.range[axis + 2] = open_hi[axis] ? q.hi[axis] : std::max(q.hi[axis] - in[axis], 0.5f * (q.lo[axis] + q.hi[axis]));
    }
}

void VulkanGsRenderer::Flush() {
    RecordShuffle();
    drawn_any_ = false;
    sprites_.clear();
    quads_.clear();
    last_flat_valid_ = false;
    // What the batch reads of its own target (DrawTriangles), for its copy.
    const bool read_all = read_all_ || !read_any_;
    const int read_x0 = read_x0_, read_y0 = read_y0_, read_x1 = read_x1_, read_y1 = read_y1_;
    read_any_ = read_all_ = false;
    if (pending_.empty()) return;
    const VkDeviceSize bytes = pending_.size() * sizeof(PackedVertex);
    VkBuffer vertex_buffer = VK_NULL_HANDLE;
    VkDeviceSize vertex_offset = 0;
    void* vertex_data = Vertices(bytes, &vertex_buffer, &vertex_offset);
    if (!vertex_data) {
        pending_.clear();
        ranges_.clear();
        return;
    }
    std::memcpy(vertex_data, pending_.data(), bytes);
    Image* t = TargetFor(state_);
    // The ranges (gs.vert): read only on a target drawn larger (gs.frag's
    // snap) - there the batch's own, or the one that keeps nothing in for
    // all its vertices (stride 0). Elsewhere nothing reads them: the
    // vertices' own buffer stands in (stride 0), bound with them once a
    // chunk of the ring as before.
    VkBuffer range_buffer = vertex_buffer;      // (stride 0: one for all)
    VkDeviceSize range_offset = 0;
    const bool ranges_read = state_.textured && t->scale > 1;
    const bool own_ranges = ranges_read && !ranges_.empty() && ranges_.size() == pending_.size();
    if (own_ranges) {
        const VkDeviceSize range_bytes = ranges_.size() * sizeof(VertexRange);
        void* range_data = Vertices(range_bytes, &range_buffer, &range_offset);
        if (!range_data) {
            pending_.clear();
            ranges_.clear();
            return;
        }
        std::memcpy(range_data, ranges_.data(), range_bytes);
    } else if (ranges_read) {
        if (!no_range_) {
            void* mapped = nullptr;
            if (Buffer(sizeof kNoRange, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &no_range_, &no_range_memory_, &mapped))
                std::memcpy(mapped, &kNoRange, sizeof kNoRange);
        }
        range_buffer = no_range_;
    }
    // A target's own depth buffer is keyed past any ZBP (which has 9 bits).
    // A depth buffer is kDepthRows tall whatever draws with it (a 16-bit
    // target is twice as tall: the same memory), so it is one image.
    Image* depth = DepthFor(state_.depth ? state_.zbp : 0x10000u + state_.fbp, t->width,
                            state_.depth ? std::min(t->height, kDepthRows) : t->height);
    Texture* texture = &blank_;
    if (state_.textured) {
        auto found = textures_.find(state_.texture);
        if (found != textures_.end()) texture = &found->second;
        // Drawn as a texture pack's replacement.
        if (texture->replacement) {
            auto replacement = replacements_.find(texture->replacement);
            if (replacement != replacements_.end()) {
                texture = &replacement->second;
                texture->last_used = frame_;
            }
        }
    }
    // The shader addresses and filters texels itself (texelFetch), as the GS
    // does - but a texture pack's replacement in REPEAT or CLAMP, which the
    // GPU filters (ReplacementSampler), and MODERN filtering's levels (tag
    // bit 1: gs.frag's Bilinear, linear within a level, its texelFetch the same).
    const int sampler = (texture->replaced & 0x100u) && state_.wms < 2 && state_.wmt < 2
                            ? ReplacementSampler(state_)
                            : (state_.tag & 2u) ? 7 : 3;
    VkDescriptorSet set = SetFor(*texture, sampler);
    // Render to texture: the target itself, or a copy when it is also drawn into.
    Image* source = nullptr;
    const Image* bound = &texture->image;   // what the shader reads (its size, gs.frag's ImageSize)
    bool copy_first = false;
    const bool shader_blend = ShaderBlend(state_);
    if (state_.textured && state_.texture_target) {
        auto found = targets_.find(Key{state_.tex_fbp, state_.tex_fbw, state_.tex_target_psm});
        if (found != targets_.end()) {
            source = &found->second;
            if (source == t) {
                Image& copy = feedbacks_[{t->width, t->height}];
                if (!copy.image)
                    MakeImage(copy, t->width, t->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              VK_FORMAT_R8G8B8A8_UNORM, t->scale);
                feedback_ = &copy;
                copy_first = true;
                set = SampledSet(copy);
                bound = &copy;
            } else {
                set = SampledSet(*source);
                bound = source;
            }
        }
    }
    // Blending in the shader reads the target as the batch began: the copy.
    VkDescriptorSet destination_set = VK_NULL_HANDLE;
    if (shader_blend) {
        if (!copy_first) {
            Image& copy = feedbacks_[{t->width, t->height}];
            if (!copy.image)
                MakeImage(copy, t->width, t->height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                          VK_FORMAT_R8G8B8A8_UNORM, t->scale);
            feedback_ = &copy;
            copy_first = true;
        }
        destination_set = SampledSet(*feedback_);
    }
    // The profile's tag (MarkDraw): the target, and its texture's kind -
    // none, a target (its FBP), memory - with the shader's own blending.
    const std::uint32_t tag = (state_.fbp & 0x1FFu) << 20 | (state_.psm & 0x3Fu) << 14 |
                              (shader_blend ? 1u << 13 : 0u) |
                              (!state_.textured ? 0u : state_.texture_target ? (1u << 12) | (state_.tex_fbp & 0x1FFu) : 0x800u);
    FragmentParams params{};
    params.textured = state_.textured && (texture != &blank_ || source);
    params.tfx = std::uint32_t(state_.tfx);
    params.tcc = state_.tcc;
    params.atst = state_.alpha_test ? state_.atst : 1;
    params.aref = state_.aref;
    // AFAIL: what a pixel failing the alpha test still writes. Drawn as up to
    // two passes - the passing pixels with everything, then the failing ones
    // with what AFAIL keeps (the shader's mode 0 drops failing pixels, 1
    // passing ones, 2 none).
    struct Pass { std::uint32_t mode; bool depth_write; std::uint32_t write_mask; };
    const bool depth_write = state_.depth && state_.depth_write;
    const std::uint32_t kept = KeptBits(state_.fbmsk, state_.psm);
    const std::uint32_t mask = WriteMask(kept);
    // A channel FBMSK masks only in part is drawn exactly with two logic ops -
    // which replace blending, so only where the blend leaves the source as it
    // is (Burnout 3's race writes two bits of each half of its frame's pixels
    // through a 16-bit view). Otherwise the channel is written whole.
    const bool source_only = !state_.blend || (state_.a == state_.b && state_.d == BlendColour::Source) ||
                             (state_.a == BlendColour::Source && state_.b == BlendColour::Zero &&
                              state_.c == BlendAlpha::Fix && state_.fix == 0x80 && state_.d == BlendColour::Zero);
    const bool logic = g_.logic_op && PartlyMasked(kept) && source_only;
    std::vector<Pass> passes;
    if (!state_.alpha_test || state_.atst == 1) {
        passes = {{2, depth_write, mask}};
    } else {
        switch (state_.afail) {
            case 0: passes = {{0, depth_write, mask}}; break;                                   // KEEP
            case 1:                                                                              // FB_ONLY
                if (depth_write) passes = {{0, true, mask}, {1, false, mask}};
                else passes = {{2, false, mask}};
                break;
            case 2:                                                                              // ZB_ONLY
                passes = {{0, depth_write, mask}};
                if (depth_write) passes.push_back({1, true, 0});
                break;
            default: passes = {{0, depth_write, mask}, {1, false, mask & 0x7u}}; break;          // RGB_ONLY
        }
    }
    params.fge = state_.fog;
    params.fogcol = state_.fogcol;
    params.fix_factor = state_.c == BlendAlpha::Fix;
    params.fix = state_.fix;
    params.bilinear = std::uint32_t(state_.bilinear) | (source ? 0u : state_.mip);   // gs.frag's Mip
    params.wms = state_.wms;
    params.wmt = state_.wmt;
    params.minu = state_.minu;
    params.maxu = state_.maxu;
    params.minv = state_.minv;
    params.maxv = state_.maxv;
    params.tw = state_.tw;
    params.th = state_.th;
    params.tex_format = source ? state_.tex_format : texture->replaced;     // gs.frag's Replaced
    params.texa = (state_.ta0 & 0xFFu) | (state_.ta1 & 0xFFu) << 8 | (state_.aem ? 1u : 0u) << 16;
    params.image_size = (bound->pw() & 0xFFFFu) | bound->ph() << 16;
    params.scale = source ? source->scale : 1;
    params.blend_ops = std::uint32_t(state_.a) | std::uint32_t(state_.b) << 2 | std::uint32_t(state_.c) << 4 |
                       std::uint32_t(state_.d) << 6;
    if (state_.textured && t->scale > 1) params.blend_ops |= (t->scale & 0xFFu) << 8;
    if (Is16Bit(state_.psm)) params.blend_ops |= 1u << 16;    // gs.frag: colour cut to five bits a channel
    if (state_.tag & 1u) params.blend_ops |= 1u << 19;          // gs.frag: the paint's Fresnel term
    if (state_.tag & 2u) params.blend_ops |= 1u << 20;          // gs.frag: modern texture filtering
    params.blend_ops |= ((state_.tag >> 8) & 7u) << 22 |       // ...its taps, and the player's filter
                        ((state_.tag >> 11) & 3u) << 25;
    if ((state_.tag & 4u) && !ShaderBlend(state_) && SceneDepthSet())
        params.blend_ops |= 1u << 21;                           // gs.frag: a soft particle
    params.blend_ops |= std::uint32_t(Blender16(state_)) << 17;     // gs.frag: S for the blender
    const std::uint32_t count = std::uint32_t(pending_.size());
    static int complaints = 0;
    if (!set && complaints < 5) {
        ++complaints;
        std::fprintf(stderr, "[ps2gpu] a batch was dropped: no descriptor set\n");
    }
    if (set) {
        g_.RecordInPass([&](VkCommandBuffer cmd) {
            // Barriers and copies cannot go inside a rendering pass: a batch
            // needing one, or drawing elsewhere, ends the open pass first.
            const bool outside = state_.date || copy_first ||
                                 (source && source->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (outside || open_target_ != t || open_depth_ != depth)
                EndPass(cmd, state_.date ? kEndDate : copy_first ? kEndFeedback : outside ? kEndSource
                             : open_target_ != t ? kEndTarget : kEndDepth);
            if (state_.date) DateSetup(cmd, *t, *depth, state_.datm);
            if (copy_first) {
                ++g_feedback;
                if (shader_blend && source != t) ++g_feedback_blend;
                Transition(cmd, *t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
                Transition(cmd, *feedback_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);
                VkImageCopy region{};
                region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                region.dstSubresource = region.srcSubresource;
                region.extent = {t->pw(), t->ph(), 1};
                if (!read_all) {
                    // The texels it reads and one around (rounding), in the image's pixels.
                    const int k = int(t->scale);
                    const int x0 = std::clamp(read_x0 - 1, 0, int(t->width) - 1) * k;
                    const int y0 = std::clamp(read_y0 - 1, 0, int(t->height) - 1) * k;
                    const int x1 = (std::clamp(read_x1 + 1, 0, int(t->width) - 1) + 1) * k;
                    const int y1 = (std::clamp(read_y1 + 1, 0, int(t->height) - 1) + 1) * k;
                    region.srcOffset = {x0, y0, 0};
                    region.dstOffset = region.srcOffset;
                    region.extent = {std::uint32_t(x1 - x0), std::uint32_t(y1 - y0), 1};
                }
                vkCmdCopyImage(cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, feedback_->image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                g_.Mark(cmd, kProfFeedback);
                Transition(cmd, *feedback_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            } else if (source) {
                Transition(cmd, *source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            }
            if (!open_target_) {
            Transition(cmd, *t, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
            VkRenderingAttachmentInfo colour{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colour.imageView = t->view;
            colour.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colour.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            colour.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            Transition(cmd, *depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
            VkRenderingAttachmentInfo z{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            z.imageView = depth->view;
            z.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            z.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            z.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {t->pw(), std::min(t->ph(), depth->ph())}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &colour;
            ri.pDepthAttachment = &z;
            ri.pStencilAttachment = &z;
            ++g_passes;
            if (defer_passes_)
                pass_cmd_ = g_.BeginSecondary(VK_FORMAT_R8G8B8A8_UNORM, kDepthFormat, CompatiblePass(true));
            if (pass_cmd_) {
                pass_colour_ = colour;
                pass_z_ = z;
                pass_area_ = ri.renderArea;
                pass_x0_ = pass_y0_ = INT32_MAX;
                pass_x1_ = pass_y1_ = INT32_MIN;
                pass_drew_ = pass_depth_read_ = pass_depth_written_ = false;
                pass_sampled_.clear();
                drawn_ = DrawState{};           // (a secondary inherits no state)
            } else {
                BeginRendering(cmd, ri, ri.renderArea.extent);
            }
            open_target_ = t;
            open_depth_ = depth;
            }
            // The pass's draws into its own command buffer, when deferred.
            if (pass_cmd_) cmd = pass_cmd_;
            // ZTST: NEVER, ALWAYS, GEQUAL, GREATER - the larger Z is nearer.
            static const VkCompareOp kCompare[4] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS,
                                                    VK_COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER};
            if (!pass_cmd_ && drawn_.recording != g_.recording_id()) drawn_ = DrawState{g_.recording_id()};
            DrawState& d = drawn_;
            // The stencil: the destination alpha test (bit 1, the setup
            // pass's), else - once a game tags paint to be marked (tag bit 3,
            // its effects' screen-space reflections) - bit 2 set where the
            // paint is drawn and cleared where anything else is, so it holds
            // the paint left on top.
            if (state_.tag & 8u) paint_mask_ = true;
            const int stencil_test = state_.date ? 1 : paint_mask_ ? ((state_.tag & 1u) ? 2 : 3) : 0;
            const int depth_test = state_.depth ? 1 : 0;
            const int compare = int(state_.ztst & 3);
            if (d.depth_test != depth_test) vkCmdSetDepthTestEnable(cmd, VkBool32(d.depth_test = depth_test));
            if (d.stencil_test != stencil_test) {
                d.stencil_test = stencil_test;
                vkCmdSetStencilTestEnable(cmd, VkBool32(stencil_test != 0));
                const VkStencilFaceFlags both = VK_STENCIL_FACE_FRONT_AND_BACK;
                if (stencil_test == 1) {
                    vkCmdSetStencilOp(cmd, both, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                                      VK_COMPARE_OP_EQUAL);
                    vkCmdSetStencilCompareMask(cmd, both, 1);
                    vkCmdSetStencilWriteMask(cmd, both, 0);
                    vkCmdSetStencilReference(cmd, both, 1);
                } else {
                    // (off too: a pipeline's dynamic state is set before it draws)
                    vkCmdSetStencilOp(cmd, both, VK_STENCIL_OP_KEEP,
                                      stencil_test >= 2 ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP,
                                      VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
                    vkCmdSetStencilCompareMask(cmd, both, 0);
                    vkCmdSetStencilWriteMask(cmd, both, stencil_test >= 2 ? 2 : 0);
                    vkCmdSetStencilReference(cmd, both, stencil_test == 2 ? 2 : 0);
                }
            }
            if (d.compare != compare) vkCmdSetDepthCompareOp(cmd, kCompare[d.compare = compare]);
            if (d.sets[0] != set || d.sets[1] != clut_set_) {
                d.sets[0] = set;
                d.sets[1] = clut_set_;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 2, d.sets, 0, nullptr);
            }
            // Set 2: the copy blending in the shader reads, the scene's depth
            // for a soft particle, or the blank texture (bound all the same).
            VkDescriptorSet set2 = destination_set;
            if (!set2) set2 = (state_.tag & 4u) && SceneDepthSet() ? SceneDepthSet() : SampledSet(blank_.image);
            if (set2 && d.set2 != set2) {
                d.set2 = set2;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 2, 1, &set2, 0, nullptr);
            }
            const VkViewport vp{0, 0, float(t->pw()), float(t->ph()), 0, 1};
            if (std::memcmp(&vp, &d.viewport, sizeof vp) != 0) {
                d.viewport = vp;
                vkCmdSetViewport(cmd, 0, 1, &vp);
            }
            const std::int32_t x0 = std::max(0, state_.scissor_x0), y0 = std::max(0, state_.scissor_y0);
            const std::int32_t x1 = std::min(std::int32_t(t->width) - 1, state_.scissor_x1);
            const std::int32_t y1 = std::min(std::int32_t(std::min(t->height, depth->height)) - 1, state_.scissor_y1);
            const std::int32_t k = std::int32_t(t->scale);
            const VkRect2D sc{{x0 * k, y0 * k},
                              {std::uint32_t(std::max(0, x1 - x0 + 1) * k), std::uint32_t(std::max(0, y1 - y0 + 1) * k)}};
            if (std::memcmp(&sc, &d.scissor, sizeof sc) != 0) {
                d.scissor = sc;
                vkCmdSetScissor(cmd, 0, 1, &sc);
            }
            if (pass_cmd_ && sc.extent.width && sc.extent.height) {
                pass_drew_ = true;
                if (std::find(pass_sampled_.begin(), pass_sampled_.end(), bound->image) == pass_sampled_.end())
                    pass_sampled_.push_back(bound->image);
                pass_x0_ = std::min(pass_x0_, sc.offset.x);
                pass_y0_ = std::min(pass_y0_, sc.offset.y);
                pass_x1_ = std::max(pass_x1_, sc.offset.x + std::int32_t(sc.extent.width));
                pass_y1_ = std::max(pass_y1_, sc.offset.y + std::int32_t(sc.extent.height));
                // (ZTST ALWAYS - the GS's way of not testing - or NEVER reads no depth.)
                pass_depth_read_ = pass_depth_read_ || (depth_test && compare >= 2) || stencil_test == 1;
                pass_depth_written_ = pass_depth_written_ || stencil_test >= 2;
            }
            const float size[2] = {float(t->width), float(t->height)};
            if (size[0] != d.size[0] || size[1] != d.size[1]) {
                d.size[0] = size[0];
                d.size[1] = size[1];
                vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, size);
            }
            // Without ranges read: the ring's chunk bound once, at its start,
            // the batch by its first vertex. With them: the batch's vertices
            // and ranges each where they are (the ranges' stride dynamic).
            std::uint32_t first = 0;
            if (!own_ranges) {
                if (d.vertices != vertex_buffer || d.ranges != range_buffer) {
                    d.vertices = vertex_buffer;
                    d.ranges = range_buffer;
                    const VkBuffer buffers[2] = {vertex_buffer, range_buffer};
                    const VkDeviceSize offsets[2] = {0, 0};
                    const VkDeviceSize strides[2] = {sizeof(PackedVertex), 0};
                    vkCmdBindVertexBuffers2(cmd, 0, 2, buffers, offsets, nullptr, strides);
                }
                first = std::uint32_t(vertex_offset / sizeof(PackedVertex));
            } else {
                d.vertices = d.ranges = VK_NULL_HANDLE;
                const VkBuffer buffers[2] = {vertex_buffer, range_buffer};
                const VkDeviceSize offsets[2] = {vertex_offset, range_offset};
                const VkDeviceSize strides[2] = {sizeof(PackedVertex), sizeof(VertexRange)};
                vkCmdBindVertexBuffers2(cmd, 0, 2, buffers, offsets, nullptr, strides);
            }
            // PABE splits each pass again: blended where As >= 0x80 (the shader's
            // pabe 1), not blended below (pabe 2, the pipeline without blending).
            const bool split = state_.blend && state_.pabe;
            for (const Pass& pass : passes) {
                for (std::uint32_t half = split ? 1 : 0; half <= (split ? 2u : 0u); ++half) {
                    std::uint32_t key = (PipelineKey(state_) & ~(0xFu << 9)) | ((~pass.write_mask & 0xFu) << 9);
                    if (half == 2) key &= 0xFu << 9;                // blending off
                    // Partly masked: AND with the kept bits, then OR the rest
                    // in (a channel this pass leaves alone keeps every bit).
                    std::uint32_t keep = kept;
                    for (int channel = 0; channel < 4; ++channel)
                        if (!(pass.write_mask >> channel & 1)) keep |= 0xFFu << (8 * channel);
                    for (std::uint32_t op = logic ? 1 : 0; op <= (logic ? 2u : 0u); ++op) {
                        const std::uint32_t op_key = op ? ((key & (0xFu << 9)) | op << 13) : key;
                        params.afail = pass.mode;
                        params.pabe = half;
                        params.logic = op;
                        // PS2HOST_GPU_UBERSHADER=1: every draw with the whole shader.
                        static const bool whole = std::getenv("PS2HOST_GPU_UBERSHADER") != nullptr;
                        const std::uint32_t full_key =
                            whole ? op_key : op_key | ((~Features(params) & kFeatureBits) << kFeatureShift);
                        const VkPipeline chosen = PipelineFor(full_key);
                        if (!chosen) {
                            if (complaints < 5) {
                                ++complaints;
                                std::fprintf(stderr, "[ps2gpu] a batch was dropped: no pipeline\n");
                            }
                            continue;
                        }
                        if (d.pipeline != chosen) {
                            d.pipeline = chosen;
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, chosen);
                        }
                        const int depth_write = pass.depth_write && op != 1 ? 1 : 0;
                        if (depth_write) pass_depth_written_ = true;
                        if (d.depth_write != depth_write)
                            vkCmdSetDepthWriteEnable(cmd, VkBool32(d.depth_write = depth_write));
                        params.keep = keep;
                        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, kFragmentOffset,
                                           sizeof params, &params);
                        ++g_draws;
                        // PS2HOST_GPU_NO_DRAW: a measure only - everything but the
                        // primitives' drawing recorded (a software device, as
                        // SwiftShader, then times the CPU's side, not its own).
                        static const bool no_draw = std::getenv("PS2HOST_GPU_NO_DRAW") != nullptr;
                        if (!no_draw) vkCmdDraw(cmd, count, 1, first, 0);
                        g_.MarkDraw(cmd, tag);
                    }
                }
            }
        });
    }
    pending_.clear();
    ranges_.clear();
}

bool VulkanGsRenderer::ReadDepth(std::uint32_t zbp, std::uint32_t width, std::uint32_t height,
                                 std::vector<std::uint32_t>* z) {
    Flush();
    auto found = depths_.find(Key{zbp, width, height});
    if (found == depths_.end()) return false;
    Image& d = found->second;
    const VkDeviceSize bytes = VkDeviceSize(d.pw()) * d.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, d, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {d.pw(), d.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, d.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a depth read-back");
    // Z / 2^32 back to the GS's Z, a GS pixel each (the middle of each square).
    const float* in = static_cast<const float*>(mapped);
    z->resize(std::size_t(width) * height);
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x) {
            const double v = double(in[std::size_t(y * d.scale + d.scale / 2) * d.pw() + x * d.scale + d.scale / 2]) * 4294967296.0 + 0.5;
            (*z)[std::size_t(y) * width + x] = v <= 0.0 ? 0u : v >= 4294967295.0 ? 0xFFFFFFFFu : std::uint32_t(v);
        }
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadImage(Image& t, std::vector<std::uint32_t>* pixels) {
    Flush();
    const VkDeviceSize bytes = VkDeviceSize(t.width) * t.height * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.width, t.height, 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a texture read-back");
    pixels->assign(std::size_t(t.width) * t.height, 0);
    std::memcpy(pixels->data(), mapped, bytes);
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadTexture(std::uint64_t key, std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                                   std::uint32_t* height) {
    auto found = textures_.find(key);
    if (found == textures_.end() || !found->second.image.image) return false;
    *width = found->second.image.width;
    *height = found->second.image.height;
    return ReadImage(found->second.image, pixels);
}

bool VulkanGsRenderer::ReadTargetFull(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                      std::vector<std::uint32_t>* pixels, std::uint32_t* width,
                                      std::uint32_t* height) {
    Flush();
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    Image& t = found->second;
    const VkDeviceSize bytes = VkDeviceSize(t.pw()) * t.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.pw(), t.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a target read-back");
    const std::uint32_t* from = static_cast<const std::uint32_t*>(mapped);
    pixels->assign(from, from + std::size_t(t.pw()) * t.ph());
    *width = t.pw();
    *height = t.ph();
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

bool VulkanGsRenderer::ReadTarget(std::uint32_t fbp, std::uint32_t fbw, std::uint32_t psm,
                                  std::vector<std::uint32_t>* pixels, std::uint32_t* width, std::uint32_t* height) {
    Flush();
    auto found = targets_.find(Key{fbp, fbw, psm});
    if (found == targets_.end()) return false;
    Image& t = found->second;
    const VkDeviceSize bytes = VkDeviceSize(t.pw()) * t.ph() * 4;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (!Buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer, &memory, &mapped)) return false;
    g_.Run([&](VkCommandBuffer cmd) {
        Transition(cmd, t, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {t.pw(), t.ph(), 1};
        vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
        g_.Mark(cmd, kProfReadback);
    }, "a target read-back");
    pixels->assign(std::size_t(t.width) * t.height, 0);
    {
        const std::uint32_t* from = static_cast<const std::uint32_t*>(mapped);
        for (std::uint32_t y = 0; y < t.height; ++y)
            for (std::uint32_t x = 0; x < t.width; ++x)
                (*pixels)[std::size_t(y) * t.width + x] = from[std::size_t(y * t.scale + t.scale / 2) * t.pw() + x * t.scale + t.scale / 2];
    }
    *width = t.width;
    *height = t.height;
    vkUnmapMemory(g_.device, memory);
    vkDestroyBuffer(g_.device, buffer, nullptr);
    vkFreeMemory(g_.device, memory, nullptr);
    return true;
}

VulkanGsRenderer::~VulkanGsRenderer() {
    if (!pipeline_workers_.empty()) {
        {
            std::lock_guard<std::mutex> hold(pipeline_lock_);
            pipeline_stop_ = true;
        }
        pipeline_wake_.notify_all();
        for (std::thread& worker : pipeline_workers_) worker.join();
    }
    for (const auto& [key, pipeline] : pipelines_made_)
        if (pipeline) pipelines_.emplace(key, pipeline);
    while (cache_saving_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    g_.Submit();
    {
        std::lock_guard<std::mutex> hold(g_queue_lock);
        vkDeviceWaitIdle(g_.device);
    }
    for (auto& [key, t] : targets_) DestroyImage(t);
    for (auto& [key, d] : depths_) DestroyImage(d);
    for (auto& [key, texture] : textures_) DestroyImage(texture.image);
    for (auto& [size, list] : texture_pool_)
        for (Texture& texture : list) DestroyImage(texture.image);
    DestroyImage(blank_.image);
    for (auto& [size, copy] : feedbacks_) DestroyImage(copy);
    for (VkDeviceMemory block : image_blocks_) vkFreeMemory(g_.device, block, nullptr);     // (ImageMemory's)
    image_blocks_.clear();
    image_pieces_.clear();
    for (const Staged& staged : staging_free_) {
        vkUnmapMemory(g_.device, staged.memory);
        vkDestroyBuffer(g_.device, staged.buffer, nullptr);
        vkFreeMemory(g_.device, staged.memory, nullptr);
    }
    for (VertexRing* rings : {vertex_rings_, staging_rings_})
        for (int slot = 0; slot < VulkanGpu::kSlots; ++slot)
        for (VertexChunk& chunk : rings[slot].chunks) {
            vkDestroyBuffer(g_.device, chunk.buffer, nullptr);
            vkFreeMemory(g_.device, chunk.memory, nullptr);
        }
    for (auto& [key, pipeline] : pipelines_) vkDestroyPipeline(g_.device, pipeline, nullptr);
    if (date_pipeline_) vkDestroyPipeline(g_.device, date_pipeline_, nullptr);
    for (auto& [key, framebuffer] : framebuffers_) vkDestroyFramebuffer(g_.device, framebuffer, nullptr);
    for (auto& [key, pass] : render_passes_) vkDestroyRenderPass(g_.device, pass, nullptr);
    for (VkSampler s : samplers_)
        if (s) vkDestroySampler(g_.device, s, nullptr);
    if (pool_) vkDestroyDescriptorPool(g_.device, pool_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(g_.device, set_layout_, nullptr);
    if (clut_set_layout_) vkDestroyDescriptorSetLayout(g_.device, clut_set_layout_, nullptr);
    if (layout_) vkDestroyPipelineLayout(g_.device, layout_, nullptr);
    if (vert_) vkDestroyShaderModule(g_.device, vert_, nullptr);
    if (frag_) vkDestroyShaderModule(g_.device, frag_, nullptr);
    if (frag_blend_) vkDestroyShaderModule(g_.device, frag_blend_, nullptr);
    if (swizzle_pipeline_) vkDestroyPipeline(g_.device, swizzle_pipeline_, nullptr);
    if (shuffle_pipeline_) vkDestroyPipeline(g_.device, shuffle_pipeline_, nullptr);
    if (shuffle_layout_) vkDestroyPipelineLayout(g_.device, shuffle_layout_, nullptr);
    if (shuffle_) vkDestroyShaderModule(g_.device, shuffle_, nullptr);
    for (auto& format : swizzle_image_specialised_)
        for (auto& direction : format)
            for (VkPipeline pipeline : direction)
                if (pipeline) vkDestroyPipeline(g_.device, pipeline, nullptr);
    if (swizzle_layout_) vkDestroyPipelineLayout(g_.device, swizzle_layout_, nullptr);
    if (swizzle_pool_) vkDestroyDescriptorPool(g_.device, swizzle_pool_, nullptr);
    if (texture_storage_pool_) vkDestroyDescriptorPool(g_.device, texture_storage_pool_, nullptr);
    if (cache_) vkDestroyPipelineCache(g_.device, cache_, nullptr);
    if (swizzle_set_layout_) vkDestroyDescriptorSetLayout(g_.device, swizzle_set_layout_, nullptr);
    if (swizzle_) vkDestroyShaderModule(g_.device, swizzle_, nullptr);
    if (decode_pipeline_) vkDestroyPipeline(g_.device, decode_pipeline_, nullptr);
    if (decode_) vkDestroyShaderModule(g_.device, decode_, nullptr);
    for (auto [buffer, memory] : {std::pair{gsmem_, gsmem_memory_}, std::pair{linear_, linear_memory_},
                                  std::pair{clut_, clut_memory_}, std::pair{tables_, tables_memory_},
                                  std::pair{no_range_, no_range_memory_}}) {
        if (buffer) vkDestroyBuffer(g_.device, buffer, nullptr);
        if (memory) vkFreeMemory(g_.device, memory, nullptr);
    }
}

}  // namespace

std::uint64_t GpuBusyNanoseconds() { return g_gpu_ns.load(); }

std::string GpuProfileReport() {
    static const char* const kNames[kProfKinds] = {"other", "draws", "swizzle", "image<->linear", "memory copies",
                                                   "feedback copies", "decodes", "uploads", "clears", "DATE",
                                                   "read-backs", "effects"};
    std::uint64_t total = 0, ns[kProfKinds];
    for (int i = 0; i < kProfKinds; ++i) total += ns[i] = g_profile_ns[i].exchange(0);
    if (!total) return "";
    std::string out = "GPU " + std::to_string(total / 1000000) + " ms:";
    for (int i = 0; i < kProfKinds; ++i) {
        if (!ns[i]) continue;
        char text[64];
        std::snprintf(text, sizeof text, " %s %.1f%%", kNames[i], 100.0 * double(ns[i]) / double(total));
        out += text;
    }
    // The dearest kinds of draw: target (FBP, PSM), texture (none, a target
    // by its FBP, memory), the shader's blending.
    std::vector<std::pair<std::uint64_t, std::uint32_t>> draws;
    {
        std::lock_guard<std::mutex> hold(g_draw_ns_lock);
        for (const auto& [tag, t] : g_draw_ns) draws.emplace_back(t, tag);
        g_draw_ns.clear();
    }
    std::sort(draws.rbegin(), draws.rend());
    for (std::size_t i = 0; i < draws.size() && i < 12; ++i) {
        const std::uint32_t tag = draws[i].second;
        const std::uint32_t texture = tag & 0x1FFFu;
        char what[48];
        if (!texture) std::snprintf(what, sizeof what, "untextured");
        else if (texture & (1u << 12)) std::snprintf(what, sizeof what, "from target %X", unsigned(texture & 0x1FFu));
        else std::snprintf(what, sizeof what, "textured");
        char text[128];
        std::snprintf(text, sizeof text, "\n[gpu]     draws into %X/%02X, %s%s: %.1f%%", unsigned(tag >> 20),
                      unsigned((tag >> 14) & 0x3Fu), what, (tag >> 13) & 1u ? ", shader blend" : "",
                      100.0 * double(draws[i].first) / double(total));
        out += text;
    }
    return out;
}

std::string RendererCounters() {
    static std::uint64_t last[7] = {};
    const std::uint64_t now[7] = {g_barriers, g_transitions, g_dispatches, g_draws, g_passes, g_feedback, g_feedback_blend};
    char text[240];
    std::snprintf(text, sizeof text, "%llu barriers, %llu transitions, %llu dispatches, %llu draws, %llu passes, "
                  "%llu feedback copies (%llu for blending alone)",
                  (unsigned long long)(now[0] - last[0]), (unsigned long long)(now[1] - last[1]),
                  (unsigned long long)(now[2] - last[2]), (unsigned long long)(now[3] - last[3]),
                  (unsigned long long)(now[4] - last[4]), (unsigned long long)(now[5] - last[5]),
                  (unsigned long long)(now[6] - last[6]));
    for (int i = 0; i < 7; ++i) last[i] = now[i];
    static std::uint64_t last_ends[kEndWhys] = {};
    char ends[160];
    std::snprintf(ends, sizeof ends, "; passes ended by: alpha test %llu, own copy %llu, a source %llu, target %llu, "
                  "depth %llu, other work %llu",
                  (unsigned long long)(g_pass_ends[0] - last_ends[0]), (unsigned long long)(g_pass_ends[1] - last_ends[1]),
                  (unsigned long long)(g_pass_ends[2] - last_ends[2]), (unsigned long long)(g_pass_ends[3] - last_ends[3]),
                  (unsigned long long)(g_pass_ends[4] - last_ends[4]), (unsigned long long)(g_pass_ends[5] - last_ends[5]));
    for (int i = 0; i < kEndWhys; ++i) last_ends[i] = g_pass_ends[i];
    char deferred[256] = "";
    if (g_area_whole)
        std::snprintf(deferred, sizeof deferred, "; deferred passes: %.0f%% of their area drawn, depth not stored %llu, "
                      "not loaded %llu, work beside them %llu, textures renamed %llu (in all)",
                      100.0 * double(g_area_used) / double(g_area_whole),
                      (unsigned long long)g_depth_unstored, (unsigned long long)g_depth_unloaded,
                      (unsigned long long)g_beside, (unsigned long long)g_renamed);
    return std::string(text) + ends + deferred;
}

std::string SubmitStats() {
    char text[160];
    std::snprintf(text, sizeof text, "%llu submissions, the GPU busy %.1f s, %.1f s waiting for them",
                  static_cast<unsigned long long>(g_submits), double(g_gpu_ns.load()) / 1e9, g_wait_seconds);
    std::string out = text;
    for (const WaitReason& w : g_waits) {
        if (!w.why || !w.count) continue;
        std::snprintf(text, sizeof text, "; %s %llu, %.1f s", w.why, static_cast<unsigned long long>(w.count),
                      w.seconds);
        out += text;
    }
    return out;
}

void SetPipelineCacheDir(const std::string& dir) { g_pipeline_cache_dir = dir; }
void SetPipelineHints(const std::uint32_t* keys, std::size_t count) { g_pipeline_hints.assign(keys, keys + count); }

bool PipelineWarmProgress(std::size_t* made, std::size_t* total) {
    const std::size_t all = g_warm_total.load(), done = std::min(all, g_warm_made.load());
    if (made) *made = done;
    if (total) *total = all;
    return done < all;
}

std::unique_ptr<GsRenderer> CreateGsRenderer(Gpu& gpu, std::string* error) {
    if (gpu.api() == Gpu::Api::kD3D11) return CreateD3D11GsRenderer(gpu, error);
    auto renderer = std::make_unique<VulkanGsRenderer>(static_cast<VulkanGpu&>(gpu));
    if (!renderer->Init(error)) return nullptr;
    return renderer;
}

}  // namespace ps2gpu
