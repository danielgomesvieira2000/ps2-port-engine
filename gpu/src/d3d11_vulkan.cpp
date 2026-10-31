/*
 * The GS renderer on Direct3D 11: the Vulkan renderer itself (gs_renderer.cpp,
 * effects.cpp and VulkanGpu, unchanged) drawing through a small layer that
 * implements the part of Vulkan they use with Direct3D 11 - one renderer, so
 * the two backends draw the same pictures by the same rules, and what the
 * renderer learns the other gets.
 *
 * The layer, installed in volk's function table (InstallVk11):
 *   - images are textures (made at vkCreateImage; device memory is only a
 *     size), views their SRVs, UAVs, RTVs and DSVs (made when first used);
 *     an RGBA8 storage image is also a R32_UINT UAV (the swizzle's words:
 *     feature level 11_0 loads no RGBA8 from a UAV);
 *   - host-visible memory is the process's own: a buffer in it is copied to
 *     its GPU twin when the GPU reads it (a draw's vertices, a copy's
 *     source); what the GPU copies into it is read back through a staging
 *     resource when its fence is waited for;
 *   - pipelines are their shaders - the fxc-compiled HLSL of the same GLSL,
 *     found by the SPIR-V the renderer hands over (gpu/CMakeLists.txt's
 *     ps2gpu_hlsl) - and the blend, rasterizer, depth-stencil and input
 *     layout states their create infos describe; specialisation constants
 *     are left to the shaders' defaults (the whole shader: the same
 *     pictures);
 *   - descriptor set s's binding b is register s * 7 + b of the kind the
 *     shader declares (the HLSL's reflection says t, u or s); push
 *     constants are constant buffer b0;
 *   - a command buffer is a list of steps, run on the immediate context when
 *     it is submitted; barriers are Direct3D 11's own business; a fence is
 *     done when its read-backs are.
 *
 * What it does not do fails as Vulkan would (a pipeline whose shader has no
 * HLSL is not made: the renderer then goes without, as on a device without
 * the feature) and is said once in the log.
 */

#if defined(_WIN32) && defined(PS2GPU_D3D11)

#include "gpu_internal.h"
#include "d3d11_gpu.h"

#include <d3d11_4.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// The SPIR-V the renderer hands over, and the HLSL compiled from the same GLSL.
#include "gs_vert.h"
#include "gs_frag.h"
#include "gs_frag_single.h"
#include "gs_frag_blend.h"
#include "date_vert.h"
#include "date_frag.h"
#include "depth_load_frag.h"
#include "swizzle_comp.h"
#include "swizzle_image_comp.h"
#include "swizzle_depth_comp.h"
#include "decode_comp.h"
#include "decode_image_comp.h"
#include "patch_comp.h"
#include "d3d_gs_vert_dxbc.h"
#include "d3d_gs_frag_dxbc.h"
#include "d3d_gs_frag_single_dxbc.h"
#include "d3d_gs_frag_blend_dxbc.h"
#include "d3d_gs_frag_uint_dxbc.h"
#include "d3d_gs_frag_hlsl.h"
#include "d3d_gs_frag_single_hlsl.h"
#include "d3d_gs_frag_blend_hlsl.h"
#include "d3d_gs_frag_uint_hlsl.h"
#include "d3d_date_vert_dxbc.h"
#include "d3d_date_frag_dxbc.h"
#include "d3d_depth_load_frag_dxbc.h"
#include "d3d_swizzle_dxbc.h"
#include "d3d_swizzle_image_dxbc.h"
#include "d3d_swizzle_depth_dxbc.h"
#include "d3d_decode_dxbc.h"
#include "d3d_decode_image_dxbc.h"
#include "d3d_patch_dxbc.h"
#include "shuffle_comp.h"
#include "d3d_shuffle_dxbc.h"
#include "effects_0.h"
#include "d3d_effects_0_dxbc.h"
#include "effects_1.h"
#include "d3d_effects_1_dxbc.h"
#include "effects_2.h"
#include "d3d_effects_2_dxbc.h"
#include "effects_3.h"
#include "d3d_effects_3_dxbc.h"
#include "effects_4.h"
#include "d3d_effects_4_dxbc.h"
#include "effects_5.h"
#include "d3d_effects_5_dxbc.h"
#include "effects_6.h"
#include "d3d_effects_6_dxbc.h"
#include "effects_7.h"
#include "d3d_effects_7_dxbc.h"
#include "effects_8.h"
#include "d3d_effects_8_dxbc.h"
#include "effects_9.h"
#include "d3d_effects_9_dxbc.h"
#include "effects_10.h"
#include "d3d_effects_10_dxbc.h"
#include "effects_11.h"
#include "d3d_effects_11_dxbc.h"
#include "effects_12.h"
#include "d3d_effects_12_dxbc.h"
#include "effects_13.h"
#include "d3d_effects_13_dxbc.h"
#include "effects_14.h"
#include "d3d_effects_14_dxbc.h"
#include "effects_15.h"
#include "d3d_effects_15_dxbc.h"

namespace ps2gpu {
namespace vk11 {

namespace {

D3D11Gpu* g_gpu = nullptr;

template <typename T>
void Release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

void Unsupported(const char* what) {
    static std::mutex lock;
    static std::set<std::string> said;
    std::lock_guard<std::mutex> hold(lock);
    if (said.insert(what).second) std::fprintf(stderr, "[d3d11] not done by the Vulkan layer: %s\n", what);
}

std::uint64_t Fnv(const void* data, std::size_t bytes) {
    std::uint64_t h = 0xCBF29CE484222325ull;
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

// ------------------------------------------------------------------ objects

struct Memory {
    VkDeviceSize size = 0;
    std::uint32_t type = 0;                 // 0 device-local, 1 host-visible
    std::vector<std::uint8_t> host;
};

struct Buffer {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    Memory* memory = nullptr;
    VkDeviceSize offset = 0;
    ID3D11Buffer* gpu = nullptr;            // the GPU's (a host buffer's twin, when the GPU reads it)
    bool dynamic = false, mapped_once = false;      // a DYNAMIC twin (MakeGpuBuffer), and mapped before
    std::map<std::pair<VkDeviceSize, VkDeviceSize>, ID3D11ShaderResourceView*> srvs;
    std::map<std::pair<VkDeviceSize, VkDeviceSize>, ID3D11UnorderedAccessView*> uavs;
    bool host() const { return memory && memory->type == 1; }
    std::uint8_t* bytes() { return memory->host.data() + offset; }
};

struct Image {
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0, levels = 1;
    VkImageUsageFlags usage = 0;
    ID3D11Texture2D* texture = nullptr;
    bool depth = false;
};

struct View {
    Image* image = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    std::uint32_t base = 0, levels = 1;
    ID3D11ShaderResourceView* srv = nullptr;
    ID3D11UnorderedAccessView* uav_float = nullptr;
    ID3D11UnorderedAccessView* uav_uint = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    ID3D11RenderTargetView* rtv_uint = nullptr;     // (an RGBA8 image's, for logic ops)
    ID3D11DepthStencilView* dsv = nullptr;
    bool made_srv = false;
};

struct Sampler {
    ID3D11SamplerState* state = nullptr;
};

struct Descriptor {
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    View* view = nullptr;
    Sampler* sampler = nullptr;
    Buffer* buffer = nullptr;
    VkDeviceSize offset = 0, range = 0;
};

struct DescriptorSet {
    std::map<std::uint32_t, Descriptor> bindings;
};
/* Counts descriptor writes: a stage's resources bound from sets that have
   not been written since are as they were. */
std::atomic<std::uint64_t> g_descriptor_version{1};

struct DescriptorPool {
    std::set<DescriptorSet*> sets;
};

struct ShaderModule {
    const void* dxbc = nullptr;             // null: no HLSL for it
    std::size_t dxbc_bytes = 0;
    // gs.frag's colour as bytes (UINT_OUTPUT), for a pipeline with a logic op.
    const void* uint_dxbc = nullptr;
    std::size_t uint_dxbc_bytes = 0;
    // The HLSL (gs.frag's), compiled again for a pipeline's specialisation
    // (Specialised); and its UINT_OUTPUT's.
    const char* hlsl = nullptr;
    const char* uint_hlsl = nullptr;
};

/* A pipeline's specialisation of gs.frag (its kFeatures, constant 0)
   compiled from its HLSL: the whole shader - every feature, its loops
   unrolled by fxc - is 490 KB of DXBC, a variant a twentieth to a third of
   it, and on an integrated GPU the whole one halved the rate. Compiled on
   the thread making the pipeline (a player's: the pipeline workers, the
   whole shader drawing meanwhile), kept in memory and in files beside the
   pipeline cache (else in %TEMP%), by the HLSL's hash and the value. */
struct Variant {
    std::vector<std::uint8_t> dxbc;
};
std::mutex g_variants_lock;
std::map<std::pair<const char*, std::uint32_t>, std::shared_ptr<Variant>> g_variants;

std::filesystem::path VariantDir() {
    std::filesystem::path dir = PipelineCacheDir();
    if (dir.empty()) {
        const char* temp = std::getenv("TEMP");
        dir = std::filesystem::path(temp ? temp : ".") / "ps2gpu_d3d11";
    } else {
        dir /= "d3d11_shaders";
    }
    std::error_code ignored;
    std::filesystem::create_directories(dir, ignored);
    return dir;
}

std::shared_ptr<Variant> Specialised(const char* hlsl, std::uint32_t value) {
    {
        std::lock_guard<std::mutex> hold(g_variants_lock);
        auto found = g_variants.find({hlsl, value});
        if (found != g_variants.end()) return found->second;
    }
    auto v = std::make_shared<Variant>();
    const std::size_t length = std::strlen(hlsl);
    char name[64];
    std::snprintf(name, sizeof name, "%016llx_%08x.dxbc", static_cast<unsigned long long>(Fnv(hlsl, length)), value);
    const std::filesystem::path file = VariantDir() / name;
    if (std::ifstream in{file, std::ios::binary}) v->dxbc.assign(std::istreambuf_iterator<char>(in), {});
    if (v->dxbc.empty()) {
        char define[32];
        std::snprintf(define, sizeof define, "%uu", value);
        const D3D_SHADER_MACRO macros[] = {{"SPIRV_CROSS_CONSTANT_ID_0", define}, {nullptr, nullptr}};
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        if (SUCCEEDED(D3DCompile(hlsl, length, "gs.frag", macros, nullptr, "main", "ps_5_0",
                                 D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
            const auto* bytes = static_cast<const std::uint8_t*>(code->GetBufferPointer());
            v->dxbc.assign(bytes, bytes + code->GetBufferSize());
            std::ofstream out(file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(v->dxbc.data()), std::streamsize(v->dxbc.size()));
        } else {
            Unsupported("a specialisation of gs.frag that fxc did not compile (the whole shader draws)");
        }
        Release(code);
        Release(errors);
    }
    std::lock_guard<std::mutex> hold(g_variants_lock);
    return g_variants.emplace(std::make_pair(hlsl, value), v).first->second;
}

/* A shader's resources by register, from its reflection. */
struct Slot {
    D3D_SHADER_INPUT_TYPE type;
    std::uint32_t reg;
    bool uint_uav = false;                  // a typed UAV of uints (the words view)
};

struct Pipeline {
    bool compute = false;
    ID3D11ComputeShader* cs = nullptr;
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11InputLayout* layout = nullptr;
    ID3D11BlendState* blend = nullptr;
    std::vector<Slot> vs_slots, ps_slots, cs_slots;
    std::uint32_t strides[4] = {};
    // Depth and stencil as made; the dynamic ones replaced by the command
    // buffer's (dynamic: VK_DYNAMIC_STATE_* listed).
    bool depth_test = false, depth_write = false, stencil_test = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;
    VkStencilOpState front{};
    bool dyn_depth = false, dyn_stencil = false, dyn_stencil_values = false;
    bool logic = false;                     // a logic op: drawn into the target's UINT view
    bool dyn_stride = false;                // VERTEX_INPUT_BINDING_STRIDE: the strides bound stay
};

struct Fence;

/* A read-back: what the GPU copied into a staging resource, to go into host
   memory once waited for. */
struct Readback {
    ID3D11Resource* staging = nullptr;
    std::uint8_t* to = nullptr;
    std::uint32_t row_bytes = 0, rows = 0, dst_pitch = 0, texel = 0, keep = 0;   // keep: bytes of each texel kept
    bool buffer = false;
};

struct Exec;
using Step = std::function<void(Exec&)>;

struct CommandBuffer {
    std::vector<Step> steps;
    /* The host buffers its draws read, the part of each (lo, hi): copied to
       their GPU twins once as the buffer is submitted, not a draw at a time
       (a race's thousands of batches). Kept from what is recorded: the
       vertex buffers and strides bound, as the steps will have them. */
    std::map<Buffer*, std::pair<VkDeviceSize, VkDeviceSize>> uploads;
    struct Bound { Buffer* buffer = nullptr; VkDeviceSize offset = 0; std::uint32_t stride = 0; } bound[4];
    void Reads(Buffer* b, VkDeviceSize lo, VkDeviceSize hi) {
        if (!b || !b->host() || hi <= lo) return;
        auto [at, made] = uploads.try_emplace(b, lo, hi);
        if (!made) {
            at->second.first = std::min(at->second.first, lo);
            at->second.second = std::max(at->second.second, hi);
        }
    }
    void Clear() {
        steps.clear();
        uploads.clear();
        for (Bound& b : bound) b = Bound{};
    }
};

struct Fence {
    bool signaled = false;
    std::vector<Readback> readbacks;
    ID3D11Query* done = nullptr;            // the submission's end on the GPU (an event query)...
    std::uint64_t value = 0;                // ...or g_fence's value then (Direct3D 11.4: waited asleep)
};

/* Direct3D 11.4's fence (Windows 10): a submission's end signalled on it, a
   wait for it asleep on an event - a query has only polling, which kept the
   renderer's thread busy while it waited. */
ID3D11Fence* g_fence = nullptr;
ID3D11DeviceContext4* g_context4 = nullptr;
std::uint64_t g_fence_value = 0;
bool g_fence_tried = false;
bool KernelFence() {
    if (g_fence_tried) return g_fence != nullptr;
    g_fence_tried = true;
    ID3D11Device5* device5 = nullptr;
    if (FAILED(g_gpu->device->QueryInterface(__uuidof(ID3D11Device5), reinterpret_cast<void**>(&device5)))) return false;
    const bool made = SUCCEEDED(device5->CreateFence(0, D3D11_FENCE_FLAG_NONE, __uuidof(ID3D11Fence),
                                                     reinterpret_cast<void**>(&g_fence))) &&
                      SUCCEEDED(g_gpu->context->QueryInterface(__uuidof(ID3D11DeviceContext4),
                                                               reinterpret_cast<void**>(&g_context4)));
    device5->Release();
    if (!made) {
        Release(g_fence);
        Release(g_context4);
    }
    return g_fence != nullptr;
}

// ----------------------------------------------------------------- formats

DXGI_FORMAT ViewFormat(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_R32_UINT: return DXGI_FORMAT_R32_UINT;
        case VK_FORMAT_R8_UNORM: return DXGI_FORMAT_R8_UNORM;
        case VK_FORMAT_R8_UINT: return DXGI_FORMAT_R8_UINT;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case VK_FORMAT_R32G32_SFLOAT: return DXGI_FORMAT_R32G32_FLOAT;
        case VK_FORMAT_R32_SFLOAT: return DXGI_FORMAT_R32_FLOAT;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return DXGI_FORMAT_BC1_UNORM;
        case VK_FORMAT_BC2_UNORM_BLOCK: return DXGI_FORMAT_BC2_UNORM;
        case VK_FORMAT_BC3_UNORM_BLOCK: return DXGI_FORMAT_BC3_UNORM;
        case VK_FORMAT_BC7_UNORM_BLOCK: return DXGI_FORMAT_BC7_UNORM;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_FORMAT TextureFormat(VkFormat f) {
    if (f == VK_FORMAT_R8G8B8A8_UNORM) return DXGI_FORMAT_R8G8B8A8_TYPELESS;    // (its R32_UINT UAV)
    if (f == VK_FORMAT_D32_SFLOAT_S8_UINT) return DXGI_FORMAT_R32G8X24_TYPELESS;
    return ViewFormat(f);
}

bool Block(VkFormat f) {
    return f == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || f == VK_FORMAT_BC2_UNORM_BLOCK || f == VK_FORMAT_BC3_UNORM_BLOCK ||
           f == VK_FORMAT_BC7_UNORM_BLOCK;
}

/* Bytes a texel (a 4x4 block's for BC). */
std::uint32_t TexelBytes(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: return 1;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return 8;
        case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC7_UNORM_BLOCK: return 16;
        case VK_FORMAT_D32_SFLOAT_S8_UINT: return 8;
        default: return 4;
    }
}

DXGI_FORMAT AttributeFormat(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R32G32_SFLOAT: return DXGI_FORMAT_R32G32_FLOAT;
        case VK_FORMAT_R32G32B32_SFLOAT: return DXGI_FORMAT_R32G32B32_FLOAT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_R32_UINT: return DXGI_FORMAT_R32_UINT;
        case VK_FORMAT_R8_UNORM: return DXGI_FORMAT_R8_UNORM;
        case VK_FORMAT_R32_SFLOAT: return DXGI_FORMAT_R32_FLOAT;
        default: return DXGI_FORMAT_UNKNOWN;
    }
}

D3D11_COMPARISON_FUNC Compare(VkCompareOp op) {
    switch (op) {
        case VK_COMPARE_OP_NEVER: return D3D11_COMPARISON_NEVER;
        case VK_COMPARE_OP_LESS: return D3D11_COMPARISON_LESS;
        case VK_COMPARE_OP_EQUAL: return D3D11_COMPARISON_EQUAL;
        case VK_COMPARE_OP_LESS_OR_EQUAL: return D3D11_COMPARISON_LESS_EQUAL;
        case VK_COMPARE_OP_GREATER: return D3D11_COMPARISON_GREATER;
        case VK_COMPARE_OP_NOT_EQUAL: return D3D11_COMPARISON_NOT_EQUAL;
        case VK_COMPARE_OP_GREATER_OR_EQUAL: return D3D11_COMPARISON_GREATER_EQUAL;
        default: return D3D11_COMPARISON_ALWAYS;
    }
}

D3D11_STENCIL_OP StencilOp(VkStencilOp op) {
    switch (op) {
        case VK_STENCIL_OP_ZERO: return D3D11_STENCIL_OP_ZERO;
        case VK_STENCIL_OP_REPLACE: return D3D11_STENCIL_OP_REPLACE;
        case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return D3D11_STENCIL_OP_INCR_SAT;
        case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return D3D11_STENCIL_OP_DECR_SAT;
        case VK_STENCIL_OP_INVERT: return D3D11_STENCIL_OP_INVERT;
        case VK_STENCIL_OP_INCREMENT_AND_WRAP: return D3D11_STENCIL_OP_INCR;
        case VK_STENCIL_OP_DECREMENT_AND_WRAP: return D3D11_STENCIL_OP_DECR;
        default: return D3D11_STENCIL_OP_KEEP;
    }
}

D3D11_BLEND BlendFactor(VkBlendFactor f) {
    switch (f) {
        case VK_BLEND_FACTOR_ZERO: return D3D11_BLEND_ZERO;
        case VK_BLEND_FACTOR_ONE: return D3D11_BLEND_ONE;
        case VK_BLEND_FACTOR_SRC_COLOR: return D3D11_BLEND_SRC_COLOR;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return D3D11_BLEND_INV_SRC_COLOR;
        case VK_BLEND_FACTOR_DST_COLOR: return D3D11_BLEND_DEST_COLOR;
        case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return D3D11_BLEND_INV_DEST_COLOR;
        case VK_BLEND_FACTOR_SRC_ALPHA: return D3D11_BLEND_SRC_ALPHA;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return D3D11_BLEND_INV_SRC_ALPHA;
        case VK_BLEND_FACTOR_DST_ALPHA: return D3D11_BLEND_DEST_ALPHA;
        case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return D3D11_BLEND_INV_DEST_ALPHA;
        case VK_BLEND_FACTOR_CONSTANT_COLOR: return D3D11_BLEND_BLEND_FACTOR;
        case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return D3D11_BLEND_INV_BLEND_FACTOR;
        case VK_BLEND_FACTOR_SRC1_COLOR: return D3D11_BLEND_SRC1_COLOR;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return D3D11_BLEND_INV_SRC1_COLOR;
        case VK_BLEND_FACTOR_SRC1_ALPHA: return D3D11_BLEND_SRC1_ALPHA;
        case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return D3D11_BLEND_INV_SRC1_ALPHA;
        default: return D3D11_BLEND_ONE;
    }
}

D3D11_BLEND_OP BlendOp(VkBlendOp op) {
    switch (op) {
        case VK_BLEND_OP_SUBTRACT: return D3D11_BLEND_OP_SUBTRACT;
        case VK_BLEND_OP_REVERSE_SUBTRACT: return D3D11_BLEND_OP_REV_SUBTRACT;
        case VK_BLEND_OP_MIN: return D3D11_BLEND_OP_MIN;
        case VK_BLEND_OP_MAX: return D3D11_BLEND_OP_MAX;
        default: return D3D11_BLEND_OP_ADD;
    }
}

D3D11_TEXTURE_ADDRESS_MODE Address(VkSamplerAddressMode m) {
    switch (m) {
        case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return D3D11_TEXTURE_ADDRESS_MIRROR;
        case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: return D3D11_TEXTURE_ADDRESS_CLAMP;
        case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER: return D3D11_TEXTURE_ADDRESS_BORDER;
        default: return D3D11_TEXTURE_ADDRESS_WRAP;
    }
}

// --------------------------------------------------------------- shaders

struct KnownShader {
    const std::uint32_t* spirv;
    std::size_t spirv_bytes;
    const void* dxbc;
    std::size_t dxbc_bytes;
};

#define PS2GPU_SHADER(spv, hlsl) {spv, sizeof spv, hlsl, sizeof hlsl}
const KnownShader kShaders[] = {
    PS2GPU_SHADER(spirv_gs_vert, dxbc_d3d_gs_vert),
    PS2GPU_SHADER(spirv_gs_frag, dxbc_d3d_gs_frag),
    PS2GPU_SHADER(spirv_gs_frag_single, dxbc_d3d_gs_frag_single),
    PS2GPU_SHADER(spirv_gs_frag_blend, dxbc_d3d_gs_frag_blend),
    PS2GPU_SHADER(spirv_date_vert, dxbc_d3d_date_vert),
    PS2GPU_SHADER(spirv_date_frag, dxbc_d3d_date_frag),
    PS2GPU_SHADER(spirv_depth_load_frag, dxbc_d3d_depth_load_frag),
    PS2GPU_SHADER(spirv_swizzle_comp, dxbc_d3d_swizzle),
    PS2GPU_SHADER(spirv_swizzle_image_comp, dxbc_d3d_swizzle_image),
    PS2GPU_SHADER(spirv_swizzle_depth_comp, dxbc_d3d_swizzle_depth),
    PS2GPU_SHADER(spirv_decode_comp, dxbc_d3d_decode),
    PS2GPU_SHADER(spirv_decode_image_comp, dxbc_d3d_decode_image),
    PS2GPU_SHADER(spirv_patch_comp, dxbc_d3d_patch),
    PS2GPU_SHADER(spirv_shuffle_comp, dxbc_d3d_shuffle),
    PS2GPU_SHADER(spirv_effects_0, dxbc_d3d_effects_0),
    PS2GPU_SHADER(spirv_effects_1, dxbc_d3d_effects_1),
    PS2GPU_SHADER(spirv_effects_2, dxbc_d3d_effects_2),
    PS2GPU_SHADER(spirv_effects_3, dxbc_d3d_effects_3),
    PS2GPU_SHADER(spirv_effects_4, dxbc_d3d_effects_4),
    PS2GPU_SHADER(spirv_effects_5, dxbc_d3d_effects_5),
    PS2GPU_SHADER(spirv_effects_6, dxbc_d3d_effects_6),
    PS2GPU_SHADER(spirv_effects_7, dxbc_d3d_effects_7),
    PS2GPU_SHADER(spirv_effects_8, dxbc_d3d_effects_8),
    PS2GPU_SHADER(spirv_effects_9, dxbc_d3d_effects_9),
    PS2GPU_SHADER(spirv_effects_10, dxbc_d3d_effects_10),
    PS2GPU_SHADER(spirv_effects_11, dxbc_d3d_effects_11),
    PS2GPU_SHADER(spirv_effects_12, dxbc_d3d_effects_12),
    PS2GPU_SHADER(spirv_effects_13, dxbc_d3d_effects_13),
    PS2GPU_SHADER(spirv_effects_14, dxbc_d3d_effects_14),
    PS2GPU_SHADER(spirv_effects_15, dxbc_d3d_effects_15),
};
#undef PS2GPU_SHADER

std::vector<Slot> Reflect(const void* dxbc, std::size_t bytes) {
    std::vector<Slot> slots;
    ID3D11ShaderReflection* reflection = nullptr;
    if (FAILED(D3DReflect(dxbc, bytes, __uuidof(ID3D11ShaderReflection), reinterpret_cast<void**>(&reflection))))
        return slots;
    D3D11_SHADER_DESC desc{};
    reflection->GetDesc(&desc);
    for (UINT i = 0; i < desc.BoundResources; ++i) {
        D3D11_SHADER_INPUT_BIND_DESC b{};
        reflection->GetResourceBindingDesc(i, &b);
        if (b.Type == D3D_SIT_CBUFFER) continue;
        Slot s{b.Type, b.BindPoint, b.Type == D3D_SIT_UAV_RWTYPED && b.ReturnType == D3D_RETURN_TYPE_UINT};
        slots.push_back(s);
    }
    reflection->Release();
    return slots;
}

// --------------------------------------------------------- views, made lazily

ID3D11ShaderResourceView* SrvOf(View* v) {
    if (!v || !v->image || !v->image->texture) return nullptr;
    if (v->made_srv) return v->srv;
    v->made_srv = true;
    Image& img = *v->image;
    if (!(img.usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    d.Texture2D.MostDetailedMip = v->base;
    d.Texture2D.MipLevels = v->levels;
    if (img.depth) {
        d.Format = (v->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && !(v->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
                       ? DXGI_FORMAT_X32_TYPELESS_G8X24_UINT
                       : DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    } else {
        // (An RGBA8 image's view as 32-bit words is not one Direct3D 11 makes:
        // the HLSL never reads it - gs.frag's D3D11 gather.)
        if (v->format != img.format && img.format == VK_FORMAT_R8G8B8A8_UNORM) return nullptr;
        d.Format = ViewFormat(v->format);
    }
    if (FAILED(g_gpu->device->CreateShaderResourceView(img.texture, &d, &v->srv))) {
        Unsupported("a shader resource view of an image's format");
        v->srv = nullptr;
    }
    return v->srv;
}

ID3D11UnorderedAccessView* UavOf(View* v, bool uint_words) {
    if (!v || !v->image || !v->image->texture) return nullptr;
    ID3D11UnorderedAccessView*& out = uint_words ? v->uav_uint : v->uav_float;
    if (out) return out;
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    d.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    d.Texture2D.MipSlice = v->base;
    d.Format = uint_words && v->image->format == VK_FORMAT_R8G8B8A8_UNORM ? DXGI_FORMAT_R32_UINT
                                                                         : ViewFormat(v->format);
    if (FAILED(g_gpu->device->CreateUnorderedAccessView(v->image->texture, &d, &out))) {
        Unsupported("an unordered-access view of an image's format");
        out = nullptr;
    }
    return out;
}

ID3D11RenderTargetView* RtvOf(View* v, bool uint_bytes = false) {
    if (!v || !v->image || !v->image->texture) return nullptr;
    uint_bytes = uint_bytes && v->image->format == VK_FORMAT_R8G8B8A8_UNORM;
    ID3D11RenderTargetView*& out = uint_bytes ? v->rtv_uint : v->rtv;
    if (out) return out;
    D3D11_RENDER_TARGET_VIEW_DESC d{};
    d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    d.Texture2D.MipSlice = v->base;
    d.Format = uint_bytes ? DXGI_FORMAT_R8G8B8A8_UINT : ViewFormat(v->format);
    if (FAILED(g_gpu->device->CreateRenderTargetView(v->image->texture, &d, &out))) out = nullptr;
    return out;
}

ID3D11DepthStencilView* DsvOf(View* v) {
    if (!v || !v->image || !v->image->texture) return nullptr;
    if (v->dsv) return v->dsv;
    D3D11_DEPTH_STENCIL_VIEW_DESC d{};
    d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    d.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    if (FAILED(g_gpu->device->CreateDepthStencilView(v->image->texture, &d, &v->dsv))) v->dsv = nullptr;
    return v->dsv;
}

/* An image's own view of it all (copies, clears), kept with it. */
std::map<Image*, std::unique_ptr<View>> g_whole_views;
std::mutex g_whole_lock;                // (the display's present asks for one from its own thread)
View* WholeView(Image* image) {
    std::lock_guard<std::mutex> hold(g_whole_lock);
    auto& v = g_whole_views[image];
    if (!v) {
        v = std::make_unique<View>();
        v->image = image;
        v->format = image->format;
        v->aspect = image->depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        v->levels = image->levels;
    }
    return v.get();
}

void DestroyViewObjects(View* v) {
    Release(v->srv);
    Release(v->uav_float);
    Release(v->uav_uint);
    Release(v->rtv);
    Release(v->rtv_uint);
    Release(v->dsv);
}

/* A buffer's GPU twin (made at bind), and its raw views. */
bool MakeGpuBuffer(Buffer* b) {
    if (b->gpu) return true;
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = UINT((b->size + 15) & ~VkDeviceSize(15));
    d.Usage = D3D11_USAGE_DEFAULT;
    if (b->usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) d.BindFlags |= D3D11_BIND_VERTEX_BUFFER;
    // A host buffer of vertices alone (the renderer's vertex rings): its twin
    // DYNAMIC, written in place with NO_OVERWRITE - the ring hands a part out
    // again only once the GPU is past it (the fences wait for the GPU).
    b->dynamic = b->host() && b->usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (b->dynamic) {
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    }
    if (b->usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) {
        d.BindFlags |= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        d.MiscFlags |= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    }
    if (FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &b->gpu))) {
        b->gpu = nullptr;
        return false;
    }
    return true;
}

ID3D11ShaderResourceView* BufferSrv(Buffer* b, VkDeviceSize offset, VkDeviceSize range) {
    if (!b || !MakeGpuBuffer(b)) return nullptr;
    if (range == VK_WHOLE_SIZE) range = b->size - offset;
    auto& view = b->srvs[{offset, range}];
    if (view) return view;
    D3D11_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    d.BufferEx.FirstElement = UINT(offset / 4);
    d.BufferEx.NumElements = UINT(range / 4);
    d.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    if (FAILED(g_gpu->device->CreateShaderResourceView(b->gpu, &d, &view))) view = nullptr;
    return view;
}

ID3D11UnorderedAccessView* BufferUav(Buffer* b, VkDeviceSize offset, VkDeviceSize range) {
    if (!b || !MakeGpuBuffer(b)) return nullptr;
    if (range == VK_WHOLE_SIZE) range = b->size - offset;
    auto& view = b->uavs[{offset, range}];
    if (view) return view;
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.FirstElement = UINT(offset / 4);
    d.Buffer.NumElements = UINT(range / 4);
    d.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    if (FAILED(g_gpu->device->CreateUnorderedAccessView(b->gpu, &d, &view))) view = nullptr;
    return view;
}

/* A host buffer's bytes [from, from + bytes) to its GPU twin. */
void Upload(Buffer* b, VkDeviceSize from, VkDeviceSize bytes) {
    if (!b->host() || !bytes || !MakeGpuBuffer(b)) return;
    from = std::min(from, b->size);
    bytes = std::min(bytes, b->size - from);
    if (!bytes) return;
    if (b->dynamic) {
        D3D11_MAPPED_SUBRESOURCE m;
        const D3D11_MAP how = b->mapped_once ? D3D11_MAP_WRITE_NO_OVERWRITE : D3D11_MAP_WRITE_DISCARD;
        if (SUCCEEDED(g_gpu->context->Map(b->gpu, 0, how, 0, &m))) {
            // (The first map's DISCARD: the whole of it, as the buffer is new.)
            if (!b->mapped_once) std::memcpy(m.pData, b->bytes(), std::size_t(b->size));
            else std::memcpy(static_cast<std::uint8_t*>(m.pData) + from, b->bytes() + from, std::size_t(bytes));
            g_gpu->context->Unmap(b->gpu, 0);
            b->mapped_once = true;
        }
        return;
    }
    const D3D11_BOX box{UINT(from), 0, 0, UINT(from + bytes), 1, 1};
    g_gpu->context->UpdateSubresource(b->gpu, 0, &box, b->bytes() + from, 0, 0);
}

/* Bytes into a GPU buffer on the GPU's timeline (vkCmdUpdateBuffer, a host
   buffer copied into a device one): written into an upload ring with
   NO_OVERWRITE, then copied on the GPU. UpdateSubresource into a buffer the
   GPU still reads made a driver wait (an integrated GPU's: a CLUT loaded a
   primitive, each a wait). DISCARD as the ring wraps. */
constexpr UINT kUploadRing = 8u << 20;
ID3D11Buffer* g_upload_ring = nullptr;
UINT g_upload_at = kUploadRing;
bool g_upload_ring_failed = false;

void UpdateOnGpu(ID3D11Buffer* dst, UINT offset, const void* data, UINT bytes) {
    ID3D11DeviceContext* c = g_gpu->context;
    if (!g_upload_ring && !g_upload_ring_failed) {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = kUploadRing;
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_VERTEX_BUFFER;         // (a dynamic buffer needs one)
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        g_upload_ring_failed = FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &g_upload_ring));
    }
    if (!g_upload_ring || bytes > kUploadRing / 4) {
        const D3D11_BOX box{offset, 0, 0, offset + bytes, 1, 1};
        c->UpdateSubresource(dst, 0, &box, data, 0, 0);
        return;
    }
    const bool wrap = g_upload_at + bytes > kUploadRing;
    if (wrap) g_upload_at = 0;
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(c->Map(g_upload_ring, 0, wrap ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_WRITE_NO_OVERWRITE, 0, &m))) return;
    std::memcpy(static_cast<std::uint8_t*>(m.pData) + g_upload_at, data, bytes);
    c->Unmap(g_upload_ring, 0);
    const D3D11_BOX box{g_upload_at, 0, 0, g_upload_at + bytes, 1, 1};
    c->CopySubresourceRegion(dst, 0, offset, 0, 0, g_upload_ring, 0, &box);
    g_upload_at = (g_upload_at + bytes + 15) & ~15u;
}

// ------------------------------------------------------------- execution

/* What a command buffer has set, as its steps run. */
struct Exec {
    Pipeline* graphics = nullptr;
    Pipeline* compute = nullptr;
    DescriptorSet* sets[2][4] = {};         // [graphics, compute][set]
    std::uint8_t push[256] = {};
    bool push_dirty = true;
    UINT push_offset = ~0u;                 // the push constants' block in the ring (PushRing)
    UINT push_bound_graphics = ~0u, push_bound_compute = ~0u;      // ...as bound to the stages
    struct VertexBinding { Buffer* buffer = nullptr; VkDeviceSize offset = 0; std::uint32_t stride = 0; } vbs[4];
    bool depth_test = false, depth_write = false, stencil_test = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;
    VkStencilOp fail = VK_STENCIL_OP_KEEP, pass = VK_STENCIL_OP_KEEP, depth_fail = VK_STENCIL_OP_KEEP;
    VkCompareOp stencil_compare = VK_COMPARE_OP_ALWAYS;
    std::uint32_t compare_mask = 0xFF, write_mask = 0xFF, reference = 0;
    D3D11_VIEWPORT viewport{};
    D3D11_RECT scissor{};
    std::vector<Readback>* readbacks = nullptr;
    Pipeline* applied = nullptr;            // the graphics pipeline last put on the context
    View* colour = nullptr;                 // the open pass's attachments
    ID3D11DepthStencilView* dsv = nullptr;
    bool colour_uint = false;               // ...its colour bound as its UINT view (a logic op's)
    // What is on the context already: set again only when it changes.
    Buffer* vb_bound[2] = {};
    UINT vb_offset[2] = {}, vb_stride[2] = {};
    DescriptorSet* stage_sets[4] = {};      // the graphics stages' resources as bound from these
    Pipeline* stage_pipeline = nullptr;
    std::uint64_t stage_version = 0;
    ID3D11DepthStencilState* depth_bound = nullptr;
    UINT reference_bound = ~0u;
    D3D11_VIEWPORT viewport_bound{-1, -1, -1, -1, -1, -1};
    D3D11_RECT scissor_bound{-1, -1, -1, -1};
};

ID3D11Buffer* g_push_buffer = nullptr;      // b0: the push constants
std::map<std::uint64_t, ID3D11DepthStencilState*> g_depth_states;

/* The push constants' ring (Direct3D 11.1's constant buffer offsets): each
   draw's or dispatch's block written at the next 256 bytes with
   NO_OVERWRITE, bound by its offset - a DISCARD only as the ring wraps. A
   map with DISCARD a draw (the plain buffer) made the driver wait for the
   GPU at most of a race's thousands of batches. */
constexpr UINT kPushRing = 256 * 1024;
ID3D11Buffer* g_push_ring = nullptr;
ID3D11DeviceContext1* g_context1 = nullptr;
UINT g_push_at = kPushRing;                 // the next block's offset (kPushRing: wrap first)
bool g_push_ring_tried = false;

bool PushRing() {
    if (g_push_ring_tried) return g_push_ring != nullptr;
    g_push_ring_tried = true;
    D3D11_FEATURE_DATA_D3D11_OPTIONS options{};
    if (FAILED(g_gpu->device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options, sizeof options)) ||
        !options.ConstantBufferOffsetting || !options.MapNoOverwriteOnDynamicConstantBuffer)
        return false;
    if (FAILED(g_gpu->context->QueryInterface(__uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&g_context1))))
        return false;
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = kPushRing;
    d.Usage = D3D11_USAGE_DYNAMIC;
    d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &g_push_ring))) {
        Release(g_context1);
        g_push_ring = nullptr;
    }
    return g_push_ring != nullptr;
}

void PushConstants(Exec& x, bool compute) {
    if (PushRing()) {
        ID3D11DeviceContext1* c = g_context1;
        if (x.push_dirty || x.push_offset == ~0u) {
            const bool wrap = g_push_at + sizeof x.push > kPushRing;
            if (wrap) g_push_at = 0;
            D3D11_MAPPED_SUBRESOURCE m;
            if (FAILED(c->Map(g_push_ring, 0, wrap ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_WRITE_NO_OVERWRITE, 0, &m)))
                return;
            std::memcpy(static_cast<std::uint8_t*>(m.pData) + g_push_at, x.push, sizeof x.push);
            c->Unmap(g_push_ring, 0);
            x.push_offset = g_push_at;
            g_push_at += sizeof x.push;
            x.push_dirty = false;
            x.push_bound_graphics = x.push_bound_compute = ~0u;
        }
        UINT& bound = compute ? x.push_bound_compute : x.push_bound_graphics;
        if (bound == x.push_offset) return;
        const UINT first = x.push_offset / 16, constants = sizeof x.push / 16;
        if (compute) {
            c->CSSetConstantBuffers1(0, 1, &g_push_ring, &first, &constants);
        } else {
            c->VSSetConstantBuffers1(0, 1, &g_push_ring, &first, &constants);
            c->PSSetConstantBuffers1(0, 1, &g_push_ring, &first, &constants);
        }
        bound = x.push_offset;
        return;
    }
    if (!g_push_buffer) {
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = sizeof x.push;
        d.Usage = D3D11_USAGE_DYNAMIC;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &g_push_buffer))) return;
    }
    ID3D11DeviceContext* c = g_gpu->context;
    if (x.push_dirty) {
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(c->Map(g_push_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
            std::memcpy(m.pData, x.push, sizeof x.push);
            c->Unmap(g_push_buffer, 0);
        }
        x.push_dirty = false;
    }
    if (compute) {
        c->CSSetConstantBuffers(0, 1, &g_push_buffer);
    } else {
        c->VSSetConstantBuffers(0, 1, &g_push_buffer);
        c->PSSetConstantBuffers(0, 1, &g_push_buffer);
    }
}

const Descriptor* Find(DescriptorSet* const* sets, std::uint32_t reg) {
    DescriptorSet* set = sets[std::min<std::uint32_t>(reg / 7, 3)];
    if (!set) return nullptr;
    auto found = set->bindings.find(reg % 7);
    return found == set->bindings.end() ? nullptr : &found->second;
}

/* A stage's resources from the bound sets, by its reflection. */
template <typename SetSrvs, typename SetSamplers>
void BindStage(DescriptorSet* const* sets, const std::vector<Slot>& slots, SetSrvs set_srv, SetSamplers set_sampler,
               ID3D11UnorderedAccessView** uavs, UINT* uav_count) {
    for (const Slot& s : slots) {
        const Descriptor* d = Find(sets, s.reg);
        switch (s.type) {
            case D3D_SIT_TEXTURE: {
                ID3D11ShaderResourceView* srv = d && d->view ? SrvOf(d->view) : nullptr;
                set_srv(s.reg, srv);
                break;
            }
            case D3D_SIT_BYTEADDRESS:
            case D3D_SIT_STRUCTURED: {
                if (d && d->buffer && d->buffer->host()) Upload(d->buffer, d->offset, d->range == VK_WHOLE_SIZE
                                                                                       ? d->buffer->size - d->offset
                                                                                       : d->range);
                set_srv(s.reg, d && d->buffer ? BufferSrv(d->buffer, d->offset, d->range) : nullptr);
                break;
            }
            case D3D_SIT_SAMPLER: {
                ID3D11SamplerState* state = d && d->sampler ? d->sampler->state : nullptr;
                set_sampler(s.reg, state);
                break;
            }
            case D3D_SIT_UAV_RWTYPED:
                if (uavs && s.reg < 8) {
                    uavs[s.reg] = d && d->view ? UavOf(d->view, s.uint_uav) : nullptr;
                    *uav_count = std::max<UINT>(*uav_count, s.reg + 1);
                }
                break;
            case D3D_SIT_UAV_RWBYTEADDRESS:
            case D3D_SIT_UAV_RWSTRUCTURED:
                if (uavs && s.reg < 8) {
                    uavs[s.reg] = d && d->buffer ? BufferUav(d->buffer, d->offset, d->range) : nullptr;
                    *uav_count = std::max<UINT>(*uav_count, s.reg + 1);
                }
                break;
            default:
                Unsupported("a shader resource kind");
        }
    }
}

ID3D11DepthStencilState* DepthState(const Exec& x, const Pipeline& p) {
    const bool test = p.dyn_depth ? x.depth_test : p.depth_test;
    const bool write = p.dyn_depth ? x.depth_write : p.depth_write;
    const VkCompareOp compare = p.dyn_depth ? x.depth_compare : p.depth_compare;
    const bool stencil = p.dyn_stencil ? x.stencil_test : p.stencil_test;
    const VkStencilOp fail = p.dyn_stencil ? x.fail : p.front.failOp;
    const VkStencilOp pass = p.dyn_stencil ? x.pass : p.front.passOp;
    const VkStencilOp depth_fail = p.dyn_stencil ? x.depth_fail : p.front.depthFailOp;
    const VkCompareOp stencil_compare = p.dyn_stencil ? x.stencil_compare : p.front.compareOp;
    const std::uint32_t read_mask = (p.dyn_stencil_values ? x.compare_mask : p.front.compareMask) & 0xFF;
    const std::uint32_t write_mask = (p.dyn_stencil_values ? x.write_mask : p.front.writeMask) & 0xFF;
    const std::uint64_t key = std::uint64_t(test) | std::uint64_t(write) << 1 | std::uint64_t(compare) << 2 |
                              std::uint64_t(stencil) << 5 | std::uint64_t(fail) << 6 | std::uint64_t(pass) << 9 |
                              std::uint64_t(depth_fail) << 12 | std::uint64_t(stencil_compare) << 15 |
                              std::uint64_t(read_mask) << 18 | std::uint64_t(write_mask) << 26;
    auto& state = g_depth_states[key];
    if (state) return state;
    D3D11_DEPTH_STENCIL_DESC d{};
    d.DepthEnable = test || write;          // (Vulkan writes depth only where its test is on)
    d.DepthWriteMask = write && test ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
    d.DepthFunc = test ? Compare(compare) : D3D11_COMPARISON_ALWAYS;
    if (!test) d.DepthEnable = FALSE;
    d.StencilEnable = stencil;
    d.StencilReadMask = UINT8(read_mask);
    d.StencilWriteMask = UINT8(write_mask);
    d.FrontFace = {StencilOp(fail), StencilOp(depth_fail), StencilOp(pass), Compare(stencil_compare)};
    d.BackFace = d.FrontFace;
    if (FAILED(g_gpu->device->CreateDepthStencilState(&d, &state))) state = nullptr;
    return state;
}

void ApplyGraphics(Exec& x) {
    Pipeline* p = x.graphics;
    ID3D11DeviceContext* c = g_gpu->context;
    if (x.colour && p->logic != x.colour_uint) {
        ID3D11RenderTargetView* rtv = RtvOf(x.colour, p->logic);
        c->OMSetRenderTargets(rtv ? 1 : 0, rtv ? &rtv : nullptr, x.dsv);
        x.colour_uint = p->logic;
    }
    if (x.applied != p) {
        c->IASetInputLayout(p->layout);
        c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        c->VSSetShader(p->vs, nullptr, 0);
        c->PSSetShader(p->ps, nullptr, 0);
        const float factor[4] = {0, 0, 0, 0};
        c->OMSetBlendState(p->blend, factor, 0xFFFFFFFFu);
        c->RSSetState(g_gpu->scissored);
        x.applied = p;
    }
    ID3D11DepthStencilState* depth = DepthState(x, *p);
    const UINT reference = p->dyn_stencil_values ? x.reference : p->front.reference;
    if (depth != x.depth_bound || reference != x.reference_bound) {
        c->OMSetDepthStencilState(depth, reference);
        x.depth_bound = depth;
        x.reference_bound = reference;
    }
    PushConstants(x, false);
    const std::uint64_t version = g_descriptor_version.load(std::memory_order_relaxed);
    // The stages' resources, unless bound from the same sets, unwritten
    // since, for this pipeline (anything binding an output - a pass, a
    // dispatch - forgets them: Direct3D 11 unbinds what becomes an output).
    if (x.stage_pipeline != p || x.stage_version != version ||
        !std::equal(std::begin(x.stage_sets), std::end(x.stage_sets), x.sets[0])) {
        x.stage_pipeline = p;
        x.stage_version = version;
        std::copy(x.sets[0], x.sets[0] + 4, x.stage_sets);
        BindStage(x.sets[0], p->vs_slots,
                  [&](UINT reg, ID3D11ShaderResourceView* v) { c->VSSetShaderResources(reg, 1, &v); },
                  [&](UINT reg, ID3D11SamplerState* s) { c->VSSetSamplers(reg, 1, &s); }, nullptr, nullptr);
        BindStage(x.sets[0], p->ps_slots,
                  [&](UINT reg, ID3D11ShaderResourceView* v) { c->PSSetShaderResources(reg, 1, &v); },
                  [&](UINT reg, ID3D11SamplerState* s) { c->PSSetSamplers(reg, 1, &s); }, nullptr, nullptr);
    }
    if (std::memcmp(&x.viewport, &x.viewport_bound, sizeof x.viewport) != 0) {
        c->RSSetViewports(1, &x.viewport);
        x.viewport_bound = x.viewport;
    }
    if (std::memcmp(&x.scissor, &x.scissor_bound, sizeof x.scissor) != 0) {
        c->RSSetScissorRects(1, &x.scissor);
        x.scissor_bound = x.scissor;
    }
    // The vertices (a host buffer's drawn part went to its GPU twin as the
    // command buffer was submitted: CommandBuffer::uploads).
    for (std::uint32_t i = 0; i < 2; ++i) {
        Exec::VertexBinding& vb = x.vbs[i];
        if (!vb.buffer || !p->layout || !MakeGpuBuffer(vb.buffer)) continue;
        const UINT offset = UINT(vb.offset), s = vb.stride;
        if (x.vb_bound[i] == vb.buffer && x.vb_offset[i] == offset && x.vb_stride[i] == s) continue;
        c->IASetVertexBuffers(i, 1, &vb.buffer->gpu, &s, &offset);
        x.vb_bound[i] = vb.buffer;
        x.vb_offset[i] = offset;
        x.vb_stride[i] = s;
    }
}

/* A texture a dispatch both samples and writes (Vulkan's GENERAL layout:
   the effects' target, read filtered beside its storage view): Direct3D 11
   unbinds a resource's SRV while it is a UAV, so the SRV reads a copy made
   just before - what the dispatch read anyway, as no invocation waits for
   another's writes. Copies kept by size and format. */
std::map<std::tuple<UINT, UINT, UINT, DXGI_FORMAT>, std::pair<ID3D11Texture2D*, ID3D11ShaderResourceView*>> g_copies;

ID3D11ShaderResourceView* CopyForReading(ID3D11ShaderResourceView* srv) {
    ID3D11Resource* resource = nullptr;
    srv->GetResource(&resource);
    ID3D11Texture2D* source = nullptr;
    if (FAILED(resource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&source)))) {
        resource->Release();
        return nullptr;
    }
    resource->Release();
    D3D11_TEXTURE2D_DESC td{};
    source->GetDesc(&td);
    D3D11_SHADER_RESOURCE_VIEW_DESC vd{};
    srv->GetDesc(&vd);
    auto& copy = g_copies[{td.Width, td.Height, td.MipLevels, vd.Format}];
    if (!copy.first) {
        D3D11_TEXTURE2D_DESC cd = td;
        cd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        cd.MiscFlags = 0;
        cd.CPUAccessFlags = 0;
        cd.Usage = D3D11_USAGE_DEFAULT;
        if (FAILED(g_gpu->device->CreateTexture2D(&cd, nullptr, &copy.first)) ||
            FAILED(g_gpu->device->CreateShaderResourceView(copy.first, &vd, &copy.second))) {
            Release(copy.first);
            source->Release();
            return nullptr;
        }
    }
    g_gpu->context->CopyResource(copy.first, source);
    source->Release();
    return copy.second;
}

bool SameResource(ID3D11View* a, ID3D11View* b) {
    if (!a || !b) return false;
    ID3D11Resource* ra = nullptr;
    ID3D11Resource* rb = nullptr;
    a->GetResource(&ra);
    b->GetResource(&rb);
    const bool same = ra == rb;
    ra->Release();
    rb->Release();
    return same;
}

void RunCompute(Exec& x, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
    Pipeline* p = x.compute;
    if (!p || !p->cs) return;
    ID3D11DeviceContext* c = g_gpu->context;
    x.stage_pipeline = nullptr;             // (its outputs unbound from the draws' inputs)
    c->CSSetShader(p->cs, nullptr, 0);
    PushConstants(x, true);
    ID3D11UnorderedAccessView* uavs[8] = {};
    UINT uav_count = 0;
    ID3D11ShaderResourceView* srvs[16] = {};
    UINT srv_count = 0;
    BindStage(x.sets[1], p->cs_slots,
              [&](UINT reg, ID3D11ShaderResourceView* v) {
                  if (reg < 16) {
                      srvs[reg] = v;
                      srv_count = std::max<UINT>(srv_count, reg + 1);
                  }
              },
              [&](UINT reg, ID3D11SamplerState* s) { c->CSSetSamplers(reg, 1, &s); }, uavs, &uav_count);
    // An image both read and written here: read from a copy.
    for (UINT i = 0; i < srv_count; ++i)
        for (UINT u = 0; u < uav_count && srvs[i]; ++u)
            if (SameResource(srvs[i], uavs[u])) {
                srvs[i] = CopyForReading(srvs[i]);
                break;
            }
    // (Unbound first: an image read here may still be a UAV of the last.)
    ID3D11UnorderedAccessView* none_uav[8] = {};
    c->CSSetUnorderedAccessViews(0, 8, none_uav, nullptr);
    if (srv_count) c->CSSetShaderResources(0, srv_count, srvs);
    if (uav_count) c->CSSetUnorderedAccessViews(0, uav_count, uavs, nullptr);
    c->Dispatch(gx, gy, gz);
    ID3D11ShaderResourceView* none_srv[16] = {};
    if (srv_count) c->CSSetShaderResources(0, srv_count, none_srv);
    if (uav_count) c->CSSetUnorderedAccessViews(0, uav_count, none_uav, nullptr);
}

/* A device buffer's words into an RGBA8 image (vkCmdCopyBufferToImage from
   the GPU's own memory: Direct3D 11 copies no buffer to a texture). */
ID3D11ComputeShader* g_buffer_to_image = nullptr;
ID3D11Buffer* g_copy_params = nullptr;
bool g_buffer_to_image_failed = false;
const char kBufferToImage[] = R"(
ByteAddressBuffer src : register(t0);
RWTexture2D<uint> dst : register(u0);
cbuffer Copy : register(b1) { uint first, row_length, x0, y0, w, h, pad0, pad1; };
[numthreads(8, 8, 1)] void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= w || id.y >= h) return;
    dst[uint2(x0 + id.x, y0 + id.y)] = src.Load((first + id.y * row_length + id.x) * 4);
}
)";

bool CopyBufferWordsToImage(Buffer* b, Image* image, const VkBufferImageCopy& r) {
    if (image->format != VK_FORMAT_R8G8B8A8_UNORM) return false;
    if (!g_buffer_to_image && !g_buffer_to_image_failed) {
        ID3DBlob* code = nullptr;
        if (FAILED(D3DCompile(kBufferToImage, sizeof kBufferToImage - 1, "buffer_to_image", nullptr, nullptr, "main",
                              "cs_5_0", 0, 0, &code, nullptr)) ||
            FAILED(g_gpu->device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr,
                                                      &g_buffer_to_image)))
            g_buffer_to_image_failed = true;
        Release(code);
        D3D11_BUFFER_DESC d{};
        d.ByteWidth = 32;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (!g_buffer_to_image_failed && FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &g_copy_params)))
            g_buffer_to_image_failed = true;
    }
    if (g_buffer_to_image_failed) return false;
    View* view = WholeView(image);
    ID3D11UnorderedAccessView* uav = UavOf(view, true);
    ID3D11ShaderResourceView* srv = BufferSrv(b, 0, VK_WHOLE_SIZE);
    if (!uav || !srv) return false;
    const std::uint32_t params[8] = {std::uint32_t(r.bufferOffset / 4), r.bufferRowLength ? r.bufferRowLength
                                                                                           : r.imageExtent.width,
                                     std::uint32_t(r.imageOffset.x), std::uint32_t(r.imageOffset.y),
                                     r.imageExtent.width, r.imageExtent.height, 0, 0};
    ID3D11DeviceContext* c = g_gpu->context;
    c->UpdateSubresource(g_copy_params, 0, nullptr, params, 0, 0);
    ID3D11UnorderedAccessView* none_uav[8] = {};
    c->CSSetUnorderedAccessViews(0, 8, none_uav, nullptr);
    c->CSSetShader(g_buffer_to_image, nullptr, 0);
    c->CSSetConstantBuffers(1, 1, &g_copy_params);
    c->CSSetShaderResources(0, 1, &srv);
    c->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
    c->Dispatch((r.imageExtent.width + 7) / 8, (r.imageExtent.height + 7) / 8, 1);
    ID3D11ShaderResourceView* none = nullptr;
    c->CSSetShaderResources(0, 1, &none);
    c->CSSetUnorderedAccessViews(0, 1, none_uav, nullptr);
    return true;
}

/* Depth from host memory into a depth buffer (vkCmdCopyBufferToImage of its
   depth: Direct3D 11 updates no depth-stencil resource) - the floats into a
   texture, drawn into the depth with SV_Depth; the stencil as it was. */
struct DepthUpload {
    ID3D11VertexShader* vs = nullptr;
    ID3D11PixelShader* ps = nullptr;
    ID3D11DepthStencilState* write = nullptr;
    ID3D11BlendState* none = nullptr;
    bool failed = false;
} g_depth_upload;
const char kDepthUpload[] = R"(
Texture2D<float> depth : register(t0);
cbuffer Place : register(b1) { int2 origin; int2 pad; };
float4 vs_main(uint id : SV_VertexID) : SV_Position {
    const float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float ps_main(float4 at : SV_Position) : SV_Depth { return depth.Load(int3(int2(at.xy) - origin, 0)); }
)";

void UploadDepth(Image* img, const std::uint8_t* from, const VkBufferImageCopy& r) {
    DepthUpload& u = g_depth_upload;
    ID3D11Device* dev = g_gpu->device;
    if (!u.vs && !u.failed) {
        ID3DBlob* code = nullptr;
        u.failed = FAILED(D3DCompile(kDepthUpload, sizeof kDepthUpload - 1, "depth_upload", nullptr, nullptr, "vs_main",
                                     "vs_5_0", 0, 0, &code, nullptr)) ||
                   FAILED(dev->CreateVertexShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &u.vs));
        Release(code);
        u.failed = u.failed ||
                   FAILED(D3DCompile(kDepthUpload, sizeof kDepthUpload - 1, "depth_upload", nullptr, nullptr, "ps_main",
                                     "ps_5_0", 0, 0, &code, nullptr)) ||
                   FAILED(dev->CreatePixelShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &u.ps));
        Release(code);
        D3D11_DEPTH_STENCIL_DESC dd{};
        dd.DepthEnable = TRUE;
        dd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
        dd.DepthFunc = D3D11_COMPARISON_ALWAYS;
        u.failed = u.failed || FAILED(dev->CreateDepthStencilState(&dd, &u.write));
        D3D11_BLEND_DESC bd{};
        bd.RenderTarget[0].SrcBlend = bd.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        bd.RenderTarget[0].DestBlend = bd.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
        bd.RenderTarget[0].BlendOp = bd.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        u.failed = u.failed || FAILED(dev->CreateBlendState(&bd, &u.none));
    }
    if (u.failed) {
        Unsupported("a buffer copied into a depth buffer");
        return;
    }
    const UINT w = r.imageExtent.width, h = r.imageExtent.height;
    const UINT row = r.bufferRowLength ? r.bufferRowLength : w;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{from, row * 4, 0};
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
    if (FAILED(dev->CreateTexture2D(&td, &data, &texture)) ||
        FAILED(dev->CreateShaderResourceView(texture, nullptr, &srv))) {
        Release(texture);
        return;
    }
    ID3D11Buffer* place = nullptr;
    const std::int32_t origin[4] = {r.imageOffset.x, r.imageOffset.y, 0, 0};
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = sizeof origin;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    const D3D11_SUBRESOURCE_DATA place_data{origin, 0, 0};
    dev->CreateBuffer(&bd, &place_data, &place);
    ID3D11DeviceContext* c = g_gpu->context;
    ID3D11DepthStencilView* dsv = DsvOf(WholeView(img));
    c->OMSetRenderTargets(0, nullptr, dsv);
    c->OMSetDepthStencilState(u.write, 0);
    const float factor[4] = {};
    c->OMSetBlendState(u.none, factor, 0xFFFFFFFFu);
    c->RSSetState(g_gpu->scissored);
    const D3D11_VIEWPORT vp{float(r.imageOffset.x), float(r.imageOffset.y), float(w), float(h), 0.0f, 1.0f};
    const D3D11_RECT scissor{r.imageOffset.x, r.imageOffset.y, LONG(r.imageOffset.x + std::int32_t(w)),
                             LONG(r.imageOffset.y + std::int32_t(h))};
    c->RSSetViewports(1, &vp);
    c->RSSetScissorRects(1, &scissor);
    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(u.vs, nullptr, 0);
    c->PSSetShader(u.ps, nullptr, 0);
    c->PSSetShaderResources(0, 1, &srv);
    c->PSSetConstantBuffers(1, 1, &place);
    c->Draw(3, 0);
    ID3D11ShaderResourceView* none = nullptr;
    c->PSSetShaderResources(0, 1, &none);
    c->OMSetRenderTargets(0, nullptr, nullptr);
    Release(place);
    Release(srv);
    Release(texture);
}

/* A staging texture or buffer for a read-back, made a copy. */
ID3D11Texture2D* StagingTexture(DXGI_FORMAT format, UINT width, UINT height) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = width;
    d.Height = height;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D* t = nullptr;
    if (FAILED(g_gpu->device->CreateTexture2D(&d, nullptr, &t))) return nullptr;
    return t;
}

ID3D11Buffer* StagingBuffer(UINT bytes) {
    D3D11_BUFFER_DESC d{};
    d.ByteWidth = (bytes + 15) & ~15u;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* b = nullptr;
    if (FAILED(g_gpu->device->CreateBuffer(&d, nullptr, &b))) return nullptr;
    return b;
}

void FinishReadback(Readback& r) {
    ID3D11DeviceContext* c = g_gpu->context;
    D3D11_MAPPED_SUBRESOURCE m;
    if (SUCCEEDED(c->Map(r.staging, 0, D3D11_MAP_READ, 0, &m))) {
        const auto* from = static_cast<const std::uint8_t*>(m.pData);
        if (r.buffer) {
            std::memcpy(r.to, from, r.row_bytes);
        } else {
            for (std::uint32_t y = 0; y < r.rows; ++y) {
                const std::uint8_t* row = from + std::size_t(y) * m.RowPitch;
                std::uint8_t* to = r.to + std::size_t(y) * r.dst_pitch;
                if (r.keep == r.texel) {
                    std::memcpy(to, row, r.row_bytes);
                } else {
                    const std::uint32_t texels = r.row_bytes / r.texel;
                    for (std::uint32_t i = 0; i < texels; ++i) std::memcpy(to + i * r.keep, row + i * r.texel, r.keep);
                }
            }
        }
        c->Unmap(r.staging, 0);
    }
    Release(r.staging);
}

std::mutex g_objects_lock;              // the descriptor pools' sets (made from more than one thread)

/* PS2HOST_D3D11_DEBUG=1 (the device made with the SDK's debug layer): its
   messages to the log at each submission, each once. */
void SayDebugMessages() {
    static ID3D11InfoQueue* queue = [] {
        ID3D11InfoQueue* q = nullptr;
        if (FAILED(g_gpu->device->QueryInterface(__uuidof(ID3D11InfoQueue), reinterpret_cast<void**>(&q)))) return q;
        return q;
    }();
    if (!queue) return;
    static std::set<std::string> said;
    const UINT64 count = queue->GetNumStoredMessages();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T bytes = 0;
        queue->GetMessage(i, nullptr, &bytes);
        std::vector<char> buffer(bytes);
        auto* m = reinterpret_cast<D3D11_MESSAGE*>(buffer.data());
        if (FAILED(queue->GetMessage(i, m, &bytes))) continue;
        std::string text(m->pDescription, m->DescriptionByteLength ? m->DescriptionByteLength - 1 : 0);
        if (said.size() < 200 && said.insert(text).second) std::fprintf(stderr, "[d3d11 debug] %s\n", text.c_str());
    }
    queue->ClearStoredMessages();
}

// --------------------------------------------------------- Vulkan's functions

VKAPI_ATTR VkResult VKAPI_CALL CreateBuffer(VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*,
                                           VkBuffer* out) {
    auto* b = new Buffer;
    b->size = ci->size;
    b->usage = ci->usage;
    *out = reinterpret_cast<VkBuffer>(b);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    auto* b = reinterpret_cast<Buffer*>(buffer);
    if (!b) return;
    for (auto& [k, v] : b->srvs) Release(v);
    for (auto& [k, v] : b->uavs) Release(v);
    Release(b->gpu);
    delete b;
}

VKAPI_ATTR void VKAPI_CALL GetBufferMemoryRequirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* req) {
    const auto* b = reinterpret_cast<Buffer*>(buffer);
    req->size = (b->size + 255) & ~VkDeviceSize(255);
    req->alignment = 256;
    req->memoryTypeBits = 3;
}

VKAPI_ATTR VkResult VKAPI_CALL BindBufferMemory(VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    auto* b = reinterpret_cast<Buffer*>(buffer);
    b->memory = reinterpret_cast<Memory*>(memory);
    b->offset = offset;
    // A device buffer is its GPU resource; a host one gets a twin only when
    // the GPU reads it (MakeGpuBuffer, on first use).
    if (!b->host() && !MakeGpuBuffer(b)) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateMemory(VkDevice, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks*,
                                             VkDeviceMemory* out) {
    auto* m = new Memory;
    m->size = ai->allocationSize;
    m->type = ai->memoryTypeIndex;
    if (m->type == 1) m->host.assign(std::size_t(m->size), 0);
    *out = reinterpret_cast<VkDeviceMemory>(m);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    delete reinterpret_cast<Memory*>(memory);
}

VKAPI_ATTR VkResult VKAPI_CALL MapMemory(VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize,
                                        VkMemoryMapFlags, void** data) {
    auto* m = reinterpret_cast<Memory*>(memory);
    if (m->type != 1) return VK_ERROR_MEMORY_MAP_FAILED;
    *data = m->host.data() + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL UnmapMemory(VkDevice, VkDeviceMemory) {}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*,
                                          VkImage* out) {
    auto* img = new Image;
    img->format = ci->format;
    img->width = ci->extent.width;
    img->height = ci->extent.height;
    img->levels = ci->mipLevels;
    img->usage = ci->usage;
    img->depth = ci->format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    D3D11_TEXTURE2D_DESC d{};
    d.Width = std::max(1u, img->width);
    d.Height = std::max(1u, img->height);
    d.MipLevels = img->levels;
    d.ArraySize = 1;
    d.Format = TextureFormat(ci->format);
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    if (ci->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) d.BindFlags |= D3D11_BIND_SHADER_RESOURCE;
    if (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT) d.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    if (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) d.BindFlags |= D3D11_BIND_RENDER_TARGET;
    if (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) d.BindFlags |= D3D11_BIND_DEPTH_STENCIL;
    // (A clear of a colour image goes through a render target view.)
    if (!img->depth && !Block(ci->format) && (ci->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
        !(d.BindFlags & D3D11_BIND_UNORDERED_ACCESS))
        d.BindFlags |= D3D11_BIND_RENDER_TARGET;
    if (d.Format == DXGI_FORMAT_UNKNOWN ||
        FAILED(g_gpu->device->CreateTexture2D(&d, nullptr, &img->texture))) {
        delete img;
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    *out = reinterpret_cast<VkImage>(img);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice, VkImage image, const VkAllocationCallbacks*) {
    auto* img = reinterpret_cast<Image*>(image);
    if (!img) return;
    {
        std::lock_guard<std::mutex> hold(g_whole_lock);
        auto whole = g_whole_views.find(img);
        if (whole != g_whole_views.end()) {
            DestroyViewObjects(whole->second.get());
            g_whole_views.erase(whole);
        }
    }
    Release(img->texture);
    delete img;
}

VKAPI_ATTR void VKAPI_CALL GetImageMemoryRequirements(VkDevice, VkImage image, VkMemoryRequirements* req) {
    const auto* img = reinterpret_cast<Image*>(image);
    req->size = VkDeviceSize(img->width) * img->height * TexelBytes(img->format) * (img->levels > 1 ? 2 : 1);
    req->size = (req->size + 0xFFFF) & ~VkDeviceSize(0xFFFF);
    req->alignment = 256;
    req->memoryTypeBits = 1;
}

VKAPI_ATTR VkResult VKAPI_CALL BindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks*,
                                              VkImageView* out) {
    auto* v = new View;
    v->image = reinterpret_cast<Image*>(ci->image);
    v->format = ci->format;
    v->aspect = ci->subresourceRange.aspectMask;
    v->base = ci->subresourceRange.baseMipLevel;
    v->levels = ci->subresourceRange.levelCount == VK_REMAINING_MIP_LEVELS ? v->image->levels - v->base
                                                                         : ci->subresourceRange.levelCount;
    *out = reinterpret_cast<VkImageView>(v);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    auto* v = reinterpret_cast<View*>(view);
    if (!v) return;
    DestroyViewObjects(v);
    delete v;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateSampler(VkDevice, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks*,
                                            VkSampler* out) {
    D3D11_SAMPLER_DESC d{};
    const bool mag = ci->magFilter == VK_FILTER_LINEAR, min = ci->minFilter == VK_FILTER_LINEAR;
    const bool mip = ci->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (ci->anisotropyEnable) {
        d.Filter = D3D11_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = UINT(std::clamp(ci->maxAnisotropy, 1.0f, 16.0f));
    } else {
        d.Filter = D3D11_ENCODE_BASIC_FILTER(min ? D3D11_FILTER_TYPE_LINEAR : D3D11_FILTER_TYPE_POINT,
                                             mag ? D3D11_FILTER_TYPE_LINEAR : D3D11_FILTER_TYPE_POINT,
                                             mip ? D3D11_FILTER_TYPE_LINEAR : D3D11_FILTER_TYPE_POINT,
                                             D3D11_FILTER_REDUCTION_TYPE_STANDARD);
        d.MaxAnisotropy = 1;
    }
    d.AddressU = Address(ci->addressModeU);
    d.AddressV = Address(ci->addressModeV);
    d.AddressW = Address(ci->addressModeW);
    d.MipLODBias = ci->mipLodBias;
    d.ComparisonFunc = D3D11_COMPARISON_NEVER;
    d.MinLOD = ci->minLod;
    d.MaxLOD = ci->maxLod;
    auto* s = new Sampler;
    if (FAILED(g_gpu->device->CreateSamplerState(&d, &s->state))) {
        delete s;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *out = reinterpret_cast<VkSampler>(s);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroySampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*) {
    auto* s = reinterpret_cast<Sampler*>(sampler);
    if (!s) return;
    Release(s->state);
    delete s;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateShaderModule(VkDevice, const VkShaderModuleCreateInfo* ci,
                                                 const VkAllocationCallbacks*, VkShaderModule* out) {
    auto* m = new ShaderModule;
    const std::uint64_t hash = Fnv(ci->pCode, ci->codeSize);
    for (const KnownShader& k : kShaders)
        if (k.spirv_bytes == ci->codeSize && Fnv(k.spirv, k.spirv_bytes) == hash) {
            m->dxbc = k.dxbc;
            m->dxbc_bytes = k.dxbc_bytes;
            if (k.dxbc == dxbc_d3d_gs_frag || k.dxbc == dxbc_d3d_gs_frag_single) {
                m->uint_dxbc = dxbc_d3d_gs_frag_uint;
                m->uint_dxbc_bytes = sizeof dxbc_d3d_gs_frag_uint;
                m->uint_hlsl = hlsl_d3d_gs_frag_uint;
                m->hlsl = k.dxbc == dxbc_d3d_gs_frag ? hlsl_d3d_gs_frag : hlsl_d3d_gs_frag_single;
            }
            if (k.dxbc == dxbc_d3d_gs_frag_blend) m->hlsl = hlsl_d3d_gs_frag_blend;
            break;
        }
    *out = reinterpret_cast<VkShaderModule>(m);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyShaderModule(VkDevice, VkShaderModule module, const VkAllocationCallbacks*) {
    delete reinterpret_cast<ShaderModule*>(module);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                        const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    static int tag;
    *out = reinterpret_cast<VkDescriptorSetLayout>(&tag);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineLayout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                   const VkAllocationCallbacks*, VkPipelineLayout* out) {
    static int tag;
    *out = reinterpret_cast<VkPipelineLayout>(&tag);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL CreateDescriptorPool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                   const VkAllocationCallbacks*, VkDescriptorPool* out) {
    *out = reinterpret_cast<VkDescriptorPool>(new DescriptorPool);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyDescriptorPool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
    auto* p = reinterpret_cast<DescriptorPool*>(pool);
    if (!p) return;
    std::lock_guard<std::mutex> hold(g_objects_lock);
    for (DescriptorSet* s : p->sets) delete s;
    delete p;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice, VkDescriptorPool pool, VkDescriptorPoolResetFlags) {
    auto* p = reinterpret_cast<DescriptorPool*>(pool);
    std::lock_guard<std::mutex> hold(g_objects_lock);
    for (DescriptorSet* s : p->sets) delete s;
    p->sets.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* ai,
                                                     VkDescriptorSet* out) {
    auto* p = reinterpret_cast<DescriptorPool*>(ai->descriptorPool);
    std::lock_guard<std::mutex> hold(g_objects_lock);
    for (std::uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
        auto* s = new DescriptorSet;
        p->sets.insert(s);
        out[i] = reinterpret_cast<VkDescriptorSet>(s);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FreeDescriptorSets(VkDevice, VkDescriptorPool pool, std::uint32_t count,
                                                 const VkDescriptorSet* sets) {
    auto* p = reinterpret_cast<DescriptorPool*>(pool);
    std::lock_guard<std::mutex> hold(g_objects_lock);
    for (std::uint32_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<DescriptorSet*>(sets[i]);
        if (!s) continue;
        p->sets.erase(s);
        delete s;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice, std::uint32_t count, const VkWriteDescriptorSet* writes,
                                               std::uint32_t, const VkCopyDescriptorSet*) {
    g_descriptor_version.fetch_add(1, std::memory_order_relaxed);
    for (std::uint32_t i = 0; i < count; ++i) {
        const VkWriteDescriptorSet& w = writes[i];
        auto* set = reinterpret_cast<DescriptorSet*>(w.dstSet);
        for (std::uint32_t k = 0; k < w.descriptorCount; ++k) {
            Descriptor& d = set->bindings[w.dstBinding + k];
            d.type = w.descriptorType;
            if (w.pImageInfo) {
                d.view = reinterpret_cast<View*>(w.pImageInfo[k].imageView);
                d.sampler = reinterpret_cast<Sampler*>(w.pImageInfo[k].sampler);
            }
            if (w.pBufferInfo) {
                d.buffer = reinterpret_cast<Buffer*>(w.pBufferInfo[k].buffer);
                d.offset = w.pBufferInfo[k].offset;
                d.range = w.pBufferInfo[k].range;
            }
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL CreatePipelineCache(VkDevice, const VkPipelineCacheCreateInfo*,
                                                  const VkAllocationCallbacks*, VkPipelineCache* out) {
    static int tag;
    *out = reinterpret_cast<VkPipelineCache>(&tag);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyPipelineCache(VkDevice, VkPipelineCache, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL GetPipelineCacheData(VkDevice, VkPipelineCache, std::size_t* size, void*) {
    *size = 0;
    return VK_SUCCESS;
}

bool HasDynamic(const VkPipelineDynamicStateCreateInfo* d, VkDynamicState s) {
    if (!d) return false;
    for (std::uint32_t i = 0; i < d->dynamicStateCount; ++i)
        if (d->pDynamicStates[i] == s) return true;
    return false;
}

VkResult MakeGraphics(const VkGraphicsPipelineCreateInfo& ci, Pipeline* p) {
    const ShaderModule* vs = nullptr;
    const ShaderModule* ps = nullptr;
    const VkSpecializationInfo* ps_spec = nullptr;
    for (std::uint32_t i = 0; i < ci.stageCount; ++i) {
        const auto* m = reinterpret_cast<const ShaderModule*>(ci.pStages[i].module);
        if (ci.pStages[i].stage == VK_SHADER_STAGE_VERTEX_BIT) vs = m;
        if (ci.pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
            ps = m;
            ps_spec = ci.pStages[i].pSpecializationInfo;
        }
    }
    if (!vs || !vs->dxbc || (ps && !ps->dxbc)) {
        Unsupported("a graphics pipeline whose shader has no HLSL");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    ID3D11Device* dev = g_gpu->device;
    if (FAILED(dev->CreateVertexShader(vs->dxbc, vs->dxbc_bytes, nullptr, &p->vs))) return VK_ERROR_INITIALIZATION_FAILED;
    p->vs_slots = Reflect(vs->dxbc, vs->dxbc_bytes);
    const VkPipelineColorBlendStateCreateInfo* cb = ci.pColorBlendState;
    p->logic = cb && cb->logicOpEnable;
    if (p->logic && (!g_gpu->logic_op || !ps || !ps->uint_dxbc)) {
        Unsupported("a logic op without Direct3D 11.1's");
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (ps) {
        const void* code = p->logic ? ps->uint_dxbc : ps->dxbc;
        std::size_t bytes = p->logic ? ps->uint_dxbc_bytes : ps->dxbc_bytes;
        // gs.frag's features (constant 0) other than all of them: its variant.
        const char* hlsl = p->logic ? ps->uint_hlsl : ps->hlsl;
        std::uint32_t features = 0xFFFFFFFFu;
        if (ps_spec)
            for (std::uint32_t i = 0; i < ps_spec->mapEntryCount; ++i)
                if (ps_spec->pMapEntries[i].constantID == 0 && ps_spec->pMapEntries[i].size == 4)
                    std::memcpy(&features, static_cast<const std::uint8_t*>(ps_spec->pData) + ps_spec->pMapEntries[i].offset, 4);
        std::shared_ptr<Variant> variant;
        static const bool whole = std::getenv("PS2HOST_D3D11_WHOLE_SHADER") != nullptr;    // (to compare)
        if (hlsl && features != 0xFFFFFFFFu && !whole) {
            variant = Specialised(hlsl, features);
            if (!variant->dxbc.empty()) {
                code = variant->dxbc.data();
                bytes = variant->dxbc.size();
            }
        }
        if (FAILED(dev->CreatePixelShader(code, bytes, nullptr, &p->ps))) return VK_ERROR_INITIALIZATION_FAILED;
        p->ps_slots = Reflect(code, bytes);
    }
    // The vertex input: location n is TEXCOORDn (SPIRV-Cross's naming).
    if (const VkPipelineVertexInputStateCreateInfo* in = ci.pVertexInputState;
        in && in->vertexAttributeDescriptionCount) {
        std::vector<D3D11_INPUT_ELEMENT_DESC> elements;
        for (std::uint32_t i = 0; i < in->vertexAttributeDescriptionCount; ++i) {
            const VkVertexInputAttributeDescription& a = in->pVertexAttributeDescriptions[i];
            bool instance = false;
            for (std::uint32_t b = 0; b < in->vertexBindingDescriptionCount; ++b)
                if (in->pVertexBindingDescriptions[b].binding == a.binding)
                    instance = in->pVertexBindingDescriptions[b].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE;
            D3D11_INPUT_ELEMENT_DESC e{};
            e.SemanticName = "TEXCOORD";
            e.SemanticIndex = a.location;
            e.Format = AttributeFormat(a.format);
            e.InputSlot = a.binding;
            e.AlignedByteOffset = a.offset;
            e.InputSlotClass = instance ? D3D11_INPUT_PER_INSTANCE_DATA : D3D11_INPUT_PER_VERTEX_DATA;
            e.InstanceDataStepRate = instance ? 1 : 0;
            elements.push_back(e);
        }
        for (std::uint32_t b = 0; b < in->vertexBindingDescriptionCount && b < 4; ++b)
            p->strides[in->pVertexBindingDescriptions[b].binding & 3] = in->pVertexBindingDescriptions[b].stride;
        if (FAILED(dev->CreateInputLayout(elements.data(), UINT(elements.size()), vs->dxbc, vs->dxbc_bytes, &p->layout)))
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    D3D11_BLEND_DESC1 bd{};
    D3D11_RENDER_TARGET_BLEND_DESC1& rt = bd.RenderTarget[0];
    rt.RenderTargetWriteMask = 0;
    if (cb && cb->attachmentCount) {
        const VkPipelineColorBlendAttachmentState& a = cb->pAttachments[0];
        rt.BlendEnable = a.blendEnable;
        rt.SrcBlend = BlendFactor(a.srcColorBlendFactor);
        rt.DestBlend = BlendFactor(a.dstColorBlendFactor);
        rt.BlendOp = BlendOp(a.colorBlendOp);
        rt.SrcBlendAlpha = BlendFactor(a.srcAlphaBlendFactor);
        rt.DestBlendAlpha = BlendFactor(a.dstAlphaBlendFactor);
        rt.BlendOpAlpha = BlendOp(a.alphaBlendOp);
        rt.RenderTargetWriteMask = UINT8(a.colorWriteMask & 0xF);
    }
    if (!rt.BlendEnable || p->logic) {
        rt.BlendEnable = FALSE;
        rt.SrcBlend = rt.SrcBlendAlpha = D3D11_BLEND_ONE;
        rt.DestBlend = rt.DestBlendAlpha = D3D11_BLEND_ZERO;
        rt.BlendOp = rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    }
    rt.LogicOp = D3D11_LOGIC_OP_NOOP;
    if (p->logic) {
        rt.LogicOpEnable = TRUE;
        rt.LogicOp = cb->logicOp == VK_LOGIC_OP_AND ? D3D11_LOGIC_OP_AND : cb->logicOp == VK_LOGIC_OP_OR
                                                                             ? D3D11_LOGIC_OP_OR
                                                                             : D3D11_LOGIC_OP_COPY;
    }
    ID3D11Device1* dev1 = nullptr;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&dev1))))
        return VK_ERROR_INITIALIZATION_FAILED;
    ID3D11BlendState1* blend = nullptr;
    const HRESULT made = dev1->CreateBlendState1(&bd, &blend);
    dev1->Release();
    if (FAILED(made)) return VK_ERROR_INITIALIZATION_FAILED;
    p->blend = blend;
    if (const VkPipelineDepthStencilStateCreateInfo* ds = ci.pDepthStencilState) {
        p->depth_test = ds->depthTestEnable;
        p->depth_write = ds->depthWriteEnable;
        p->depth_compare = ds->depthCompareOp;
        p->stencil_test = ds->stencilTestEnable;
        p->front = ds->front;
    }
    p->dyn_depth = HasDynamic(ci.pDynamicState, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
    p->dyn_stencil = HasDynamic(ci.pDynamicState, VK_DYNAMIC_STATE_STENCIL_OP);
    p->dyn_stencil_values = HasDynamic(ci.pDynamicState, VK_DYNAMIC_STATE_STENCIL_REFERENCE);
    p->dyn_stride = HasDynamic(ci.pDynamicState, VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
    return VK_SUCCESS;
}

void DestroyPipelineObjects(Pipeline* p) {
    Release(p->cs);
    Release(p->vs);
    Release(p->ps);
    Release(p->layout);
    Release(p->blend);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateGraphicsPipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                      const VkGraphicsPipelineCreateInfo* infos,
                                                      const VkAllocationCallbacks*, VkPipeline* out) {
    VkResult result = VK_SUCCESS;
    for (std::uint32_t i = 0; i < count; ++i) {
        auto* p = new Pipeline;
        const VkResult r = MakeGraphics(infos[i], p);
        if (r != VK_SUCCESS) {
            DestroyPipelineObjects(p);
            delete p;
            out[i] = VK_NULL_HANDLE;
            result = r;
            continue;
        }
        out[i] = reinterpret_cast<VkPipeline>(p);
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateComputePipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                     const VkComputePipelineCreateInfo* infos,
                                                     const VkAllocationCallbacks*, VkPipeline* out) {
    VkResult result = VK_SUCCESS;
    for (std::uint32_t i = 0; i < count; ++i) {
        out[i] = VK_NULL_HANDLE;
        const auto* m = reinterpret_cast<const ShaderModule*>(infos[i].stage.module);
        if (!m || !m->dxbc) {
            Unsupported("a compute pipeline whose shader has no HLSL");
            result = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }
        auto* p = new Pipeline;
        p->compute = true;
        if (FAILED(g_gpu->device->CreateComputeShader(m->dxbc, m->dxbc_bytes, nullptr, &p->cs))) {
            delete p;
            result = VK_ERROR_INITIALIZATION_FAILED;
            continue;
        }
        p->cs_slots = Reflect(m->dxbc, m->dxbc_bytes);
        out[i] = reinterpret_cast<VkPipeline>(p);
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyPipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*) {
    auto* p = reinterpret_cast<Pipeline*>(pipeline);
    if (!p) return;
    DestroyPipelineObjects(p);
    delete p;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateCommandPool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                                VkCommandPool* out) {
    static int tag;
    *out = reinterpret_cast<VkCommandPool>(&tag);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice, const VkCommandBufferAllocateInfo* ai,
                                                     VkCommandBuffer* out) {
    for (std::uint32_t i = 0; i < ai->commandBufferCount; ++i) out[i] = reinterpret_cast<VkCommandBuffer>(new CommandBuffer);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer cmd, VkCommandBufferResetFlags) {
    reinterpret_cast<CommandBuffer*>(cmd)->Clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cmd, const VkCommandBufferBeginInfo*) {
    reinterpret_cast<CommandBuffer*>(cmd)->Clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer) { return VK_SUCCESS; }

CommandBuffer& Cmd(VkCommandBuffer c) { return *reinterpret_cast<CommandBuffer*>(c); }

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipeline pipeline) {
    auto* p = reinterpret_cast<Pipeline*>(pipeline);
    const bool compute = point == VK_PIPELINE_BIND_POINT_COMPUTE;
    // (A pipeline with dynamic strides keeps those bound - the renderer binds
    // its ranges' stream at stride 0 and leaves it bound across pipelines.)
    if (!compute && p && !p->dyn_stride)
        for (int i = 0; i < 4; ++i) Cmd(cmd).bound[i].stride = p->strides[i];
    Cmd(cmd).steps.push_back([p, compute](Exec& x) {
        if (compute) {
            x.compute = p;
        } else {
            x.graphics = p;
            // The vertex strides as made (a pipeline without dynamic ones).
            if (p && !p->dyn_stride)
                for (int i = 0; i < 4; ++i) x.vbs[i].stride = p->strides[i];
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout,
                                                std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets,
                                                std::uint32_t, const std::uint32_t*) {
    DescriptorSet* bound[4] = {};
    for (std::uint32_t i = 0; i < count && first + i < 4; ++i) bound[i] = reinterpret_cast<DescriptorSet*>(sets[i]);
    const int which = point == VK_PIPELINE_BIND_POINT_COMPUTE ? 1 : 0;
    Cmd(cmd).steps.push_back([which, first, count, b0 = bound[0], b1 = bound[1], b2 = bound[2], b3 = bound[3]](Exec& x) {
        DescriptorSet* b[4] = {b0, b1, b2, b3};
        for (std::uint32_t i = 0; i < count && first + i < 4; ++i) x.sets[which][first + i] = b[i];
    });
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer cmd, VkPipelineLayout, VkShaderStageFlags,
                                           std::uint32_t offset, std::uint32_t size, const void* values) {
    std::array<std::uint8_t, 256> bytes;
    size = std::min<std::uint32_t>(size, 256 - std::min<std::uint32_t>(offset, 256));
    std::memcpy(bytes.data(), values, size);
    Cmd(cmd).steps.push_back([offset, size, bytes](Exec& x) {
        std::memcpy(x.push + offset, bytes.data(), size);
        x.push_dirty = true;
    });
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers2(VkCommandBuffer cmd, std::uint32_t first, std::uint32_t count,
                                                const VkBuffer* buffers, const VkDeviceSize* offsets,
                                                const VkDeviceSize*, const VkDeviceSize* strides) {
    for (std::uint32_t i = 0; i < count && first + i < 4; ++i) {
        Buffer* b = reinterpret_cast<Buffer*>(buffers[i]);
        const VkDeviceSize offset = offsets[i];
        const bool has_stride = strides != nullptr;
        const std::uint32_t stride = strides ? std::uint32_t(strides[i]) : 0;
        const std::uint32_t slot = first + i;
        CommandBuffer::Bound& bound = Cmd(cmd).bound[slot];
        bound.buffer = b;
        bound.offset = offset;
        if (has_stride) bound.stride = stride;
        Cmd(cmd).steps.push_back([=](Exec& x) {
            x.vbs[slot].buffer = b;
            x.vbs[slot].offset = offset;
            if (has_stride) x.vbs[slot].stride = stride;
        });
    }
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer cmd, std::uint32_t, std::uint32_t, const VkViewport* v) {
    const D3D11_VIEWPORT vp{v->x, v->y, v->width, v->height, v->minDepth, v->maxDepth};
    Cmd(cmd).steps.push_back([vp](Exec& x) { x.viewport = vp; });
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer cmd, std::uint32_t, std::uint32_t, const VkRect2D* r) {
    const D3D11_RECT rect{LONG(r->offset.x), LONG(r->offset.y), LONG(r->offset.x + std::int32_t(r->extent.width)),
                          LONG(r->offset.y + std::int32_t(r->extent.height))};
    Cmd(cmd).steps.push_back([rect](Exec& x) { x.scissor = rect; });
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthTestEnable(VkCommandBuffer cmd, VkBool32 on) {
    Cmd(cmd).steps.push_back([on](Exec& x) { x.depth_test = on; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetDepthWriteEnable(VkCommandBuffer cmd, VkBool32 on) {
    Cmd(cmd).steps.push_back([on](Exec& x) { x.depth_write = on; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetDepthCompareOp(VkCommandBuffer cmd, VkCompareOp op) {
    Cmd(cmd).steps.push_back([op](Exec& x) { x.depth_compare = op; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetStencilTestEnable(VkCommandBuffer cmd, VkBool32 on) {
    Cmd(cmd).steps.push_back([on](Exec& x) { x.stencil_test = on; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetStencilOp(VkCommandBuffer cmd, VkStencilFaceFlags, VkStencilOp fail, VkStencilOp pass,
                                          VkStencilOp depth_fail, VkCompareOp compare) {
    Cmd(cmd).steps.push_back([=](Exec& x) {
        x.fail = fail;
        x.pass = pass;
        x.depth_fail = depth_fail;
        x.stencil_compare = compare;
    });
}
VKAPI_ATTR void VKAPI_CALL CmdSetStencilCompareMask(VkCommandBuffer cmd, VkStencilFaceFlags, std::uint32_t mask) {
    Cmd(cmd).steps.push_back([mask](Exec& x) { x.compare_mask = mask; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetStencilWriteMask(VkCommandBuffer cmd, VkStencilFaceFlags, std::uint32_t mask) {
    Cmd(cmd).steps.push_back([mask](Exec& x) { x.write_mask = mask; });
}
VKAPI_ATTR void VKAPI_CALL CmdSetStencilReference(VkCommandBuffer cmd, VkStencilFaceFlags, std::uint32_t ref) {
    Cmd(cmd).steps.push_back([ref](Exec& x) { x.reference = ref; });
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer cmd, std::uint32_t vertices, std::uint32_t instances,
                                  std::uint32_t first_vertex, std::uint32_t first_instance) {
    CommandBuffer& c = Cmd(cmd);
    for (const CommandBuffer::Bound& b : c.bound) {
        if (!b.buffer) continue;
        const VkDeviceSize lo = b.offset + VkDeviceSize(first_vertex) * b.stride;
        c.Reads(b.buffer, lo, lo + (b.stride ? VkDeviceSize(vertices) * b.stride : 64));
    }
    c.steps.push_back([=](Exec& x) {
        if (!x.graphics) return;
        ApplyGraphics(x);
        if (instances == 1 && first_instance == 0) g_gpu->context->Draw(vertices, first_vertex);
        else g_gpu->context->DrawInstanced(vertices, instances, first_vertex, first_instance);
    });
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cmd, std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
    Cmd(cmd).steps.push_back([=](Exec& x) { RunCompute(x, gx, gy, gz); });
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                             VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                             const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    // (Direct3D 11 orders its own work.)
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRendering(VkCommandBuffer cmd, const VkRenderingInfo* ri) {
    View* colour = ri->colorAttachmentCount ? reinterpret_cast<View*>(ri->pColorAttachments[0].imageView) : nullptr;
    const VkRenderingAttachmentInfo* z = ri->pDepthAttachment;
    const VkRenderingAttachmentInfo* s = ri->pStencilAttachment ? ri->pStencilAttachment : z;
    View* depth = z ? reinterpret_cast<View*>(z->imageView) : nullptr;
    const bool clear_colour = colour && ri->pColorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const VkClearColorValue colour_value = colour ? ri->pColorAttachments[0].clearValue.color : VkClearColorValue{};
    const bool clear_depth = z && z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const bool clear_stencil = s && s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
    const float depth_value = z ? z->clearValue.depthStencil.depth : 0.0f;
    const std::uint32_t stencil_value = s ? s->clearValue.depthStencil.stencil : 0;
    Cmd(cmd).steps.push_back([=](Exec& x) {
        ID3D11DeviceContext* c = g_gpu->context;
        // (What a draw sampled may be drawn into now: unbound first.)
        ID3D11ShaderResourceView* none[16] = {};
        c->PSSetShaderResources(0, 16, none);
        x.stage_pipeline = nullptr;
        ID3D11RenderTargetView* rtv = RtvOf(colour);
        ID3D11DepthStencilView* dsv = DsvOf(depth);
        c->OMSetRenderTargets(rtv ? 1 : 0, rtv ? &rtv : nullptr, dsv);
        x.colour = colour;
        x.dsv = dsv;
        x.colour_uint = false;
        if (rtv && clear_colour) c->ClearRenderTargetView(rtv, colour_value.float32);
        if (dsv && (clear_depth || clear_stencil))
            c->ClearDepthStencilView(dsv, (clear_depth ? D3D11_CLEAR_DEPTH : 0) | (clear_stencil ? D3D11_CLEAR_STENCIL : 0),
                                     depth_value, UINT8(stencil_value));
        x.applied = nullptr;
    });
}

VKAPI_ATTR void VKAPI_CALL CmdEndRendering(VkCommandBuffer cmd) {
    Cmd(cmd).steps.push_back([](Exec& x) {
        g_gpu->context->OMSetRenderTargets(0, nullptr, nullptr);
        x.colour = nullptr;
        x.dsv = nullptr;
        x.colour_uint = false;
    });
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer cmd, std::uint32_t count, const VkCommandBuffer* buffers) {
    for (std::uint32_t i = 0; i < count; ++i) {
        CommandBuffer* secondary = reinterpret_cast<CommandBuffer*>(buffers[i]);
        for (const auto& [b, range] : secondary->uploads) Cmd(cmd).Reads(b, range.first, range.second);
        Cmd(cmd).steps.push_back([secondary](Exec& x) {
            Exec inner;                     // (a secondary inherits no state but its pass)
            inner.readbacks = x.readbacks;
            inner.colour = x.colour;
            inner.dsv = x.dsv;
            inner.colour_uint = x.colour_uint;
            for (Step& step : secondary->steps) step(inner);
        });
    }
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, std::uint32_t count,
                                        const VkBufferCopy* regions) {
    Buffer* s = reinterpret_cast<Buffer*>(src);
    Buffer* d = reinterpret_cast<Buffer*>(dst);
    std::vector<VkBufferCopy> copies(regions, regions + count);
    Cmd(cmd).steps.push_back([s, d, copies](Exec& x) {
        ID3D11DeviceContext* c = g_gpu->context;
        for (const VkBufferCopy& r : copies) {
            if (s->host() && !d->host()) {
                UpdateOnGpu(d->gpu, UINT(r.dstOffset), s->bytes() + r.srcOffset, UINT(r.size));
            } else if (!s->host() && d->host()) {
                ID3D11Buffer* staging = StagingBuffer(UINT(r.size));
                if (!staging) continue;
                const D3D11_BOX box{UINT(r.srcOffset), 0, 0, UINT(r.srcOffset + r.size), 1, 1};
                c->CopySubresourceRegion(staging, 0, 0, 0, 0, s->gpu, 0, &box);
                Readback rb;
                rb.staging = staging;
                rb.to = d->bytes() + r.dstOffset;
                rb.row_bytes = UINT(r.size);
                rb.buffer = true;
                if (x.readbacks) x.readbacks->push_back(rb);
            } else if (!s->host() && !d->host()) {
                const D3D11_BOX box{UINT(r.srcOffset), 0, 0, UINT(r.srcOffset + r.size), 1, 1};
                c->CopySubresourceRegion(d->gpu, 0, UINT(r.dstOffset), 0, 0, s->gpu, 0, &box);
            } else {
                std::memcpy(d->bytes() + r.dstOffset, s->bytes() + r.srcOffset, std::size_t(r.size));
            }
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer cmd, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                          const void* data) {
    Buffer* d = reinterpret_cast<Buffer*>(dst);
    std::vector<std::uint8_t> bytes(static_cast<const std::uint8_t*>(data),
                                    static_cast<const std::uint8_t*>(data) + size);
    Cmd(cmd).steps.push_back([d, offset, bytes](Exec&) {
        if (d->host()) {
            std::memcpy(d->bytes() + offset, bytes.data(), bytes.size());
            return;
        }
        UpdateOnGpu(d->gpu, UINT(offset), bytes.data(), UINT(bytes.size()));
    });
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cmd, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                        std::uint32_t value) {
    Buffer* d = reinterpret_cast<Buffer*>(dst);
    Cmd(cmd).steps.push_back([d, offset, size, value](Exec&) {
        const VkDeviceSize bytes = size == VK_WHOLE_SIZE ? d->size - offset : size;
        if (d->host()) {
            for (VkDeviceSize i = 0; i + 4 <= bytes; i += 4) std::memcpy(d->bytes() + offset + i, &value, 4);
            return;
        }
        ID3D11UnorderedAccessView* uav = BufferUav(d, offset, bytes);
        if (uav) {
            const UINT v[4] = {value, value, value, value};
            g_gpu->context->ClearUnorderedAccessViewUint(uav, v);
        } else {
            std::vector<std::uint32_t> words(std::size_t(bytes / 4), value);
            const D3D11_BOX box{UINT(offset), 0, 0, UINT(offset + bytes), 1, 1};
            g_gpu->context->UpdateSubresource(d->gpu, 0, &box, words.data(), 0, 0);
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cmd, VkBuffer src, VkImage dst, VkImageLayout,
                                               std::uint32_t count, const VkBufferImageCopy* regions) {
    Buffer* s = reinterpret_cast<Buffer*>(src);
    Image* img = reinterpret_cast<Image*>(dst);
    std::vector<VkBufferImageCopy> copies(regions, regions + count);
    Cmd(cmd).steps.push_back([s, img, copies](Exec& x) {
        ID3D11DeviceContext* c = g_gpu->context;
        for (const VkBufferImageCopy& r : copies) {
            if (img->depth) {
                if (s->host()) UploadDepth(img, s->bytes() + r.bufferOffset, r);
                else Unsupported("a GPU buffer copied into a depth buffer");
                x.applied = nullptr;            // (its own shaders and states set)
                x.stage_pipeline = nullptr;
                x.depth_bound = nullptr;
                x.reference_bound = ~0u;
                x.viewport_bound = D3D11_VIEWPORT{-1, -1, -1, -1, -1, -1};
                x.scissor_bound = D3D11_RECT{-1, -1, -1, -1};
                continue;
            }
            if (!s->host()) {
                if (!CopyBufferWordsToImage(s, img, r)) Unsupported("a GPU buffer copied into a non-RGBA8 image");
                x.stage_pipeline = nullptr;
                continue;
            }
            const std::uint32_t row_texels = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
            const std::uint32_t rows = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
            std::uint32_t pitch = row_texels * TexelBytes(img->format);
            std::uint32_t slice = pitch * rows;
            if (Block(img->format)) {
                pitch = ((row_texels + 3) / 4) * TexelBytes(img->format);
                slice = pitch * ((rows + 3) / 4);
            }
            const D3D11_BOX box{UINT(r.imageOffset.x), UINT(r.imageOffset.y), 0,
                                UINT(r.imageOffset.x) + r.imageExtent.width, UINT(r.imageOffset.y) + r.imageExtent.height,
                                1};
            c->UpdateSubresource(img->texture, D3D11CalcSubresource(r.imageSubresource.mipLevel, 0, img->levels), &box,
                                 s->bytes() + r.bufferOffset, pitch, slice);
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer cmd, VkImage src, VkImageLayout, VkBuffer dst,
                                               std::uint32_t count, const VkBufferImageCopy* regions) {
    Image* img = reinterpret_cast<Image*>(src);
    Buffer* d = reinterpret_cast<Buffer*>(dst);
    std::vector<VkBufferImageCopy> copies(regions, regions + count);
    Cmd(cmd).steps.push_back([img, d, copies](Exec& x) {
        if (!d->host()) {
            Unsupported("an image copied into a GPU buffer");
            return;
        }
        ID3D11DeviceContext* c = g_gpu->context;
        for (const VkBufferImageCopy& r : copies) {
            const UINT w = r.imageExtent.width, h = r.imageExtent.height;
            // A depth buffer's whole (Direct3D 11 copies no part of one),
            // its depth kept - four bytes of each eight.
            const bool whole = img->depth;
            ID3D11Texture2D* staging = StagingTexture(TextureFormat(img->format), whole ? img->width : w,
                                                      whole ? img->height : h);
            if (!staging) continue;
            const UINT sub = D3D11CalcSubresource(r.imageSubresource.mipLevel, 0, img->levels);
            if (whole) {
                c->CopySubresourceRegion(staging, 0, 0, 0, 0, img->texture, sub, nullptr);
            } else {
                const D3D11_BOX box{UINT(r.imageOffset.x), UINT(r.imageOffset.y), 0, UINT(r.imageOffset.x) + w,
                                    UINT(r.imageOffset.y) + h, 1};
                c->CopySubresourceRegion(staging, 0, 0, 0, 0, img->texture, sub, &box);
            }
            Readback rb;
            rb.staging = staging;
            rb.texel = TexelBytes(img->format);
            rb.keep = img->depth ? 4 : rb.texel;
            rb.to = d->bytes() + r.bufferOffset;
            rb.rows = h;
            rb.row_bytes = w * rb.texel;
            rb.dst_pitch = (r.bufferRowLength ? r.bufferRowLength : w) * rb.keep;
            if (x.readbacks) x.readbacks->push_back(rb);
            else Release(rb.staging);
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer cmd, VkImage src, VkImageLayout, VkImage dst, VkImageLayout,
                                       std::uint32_t count, const VkImageCopy* regions) {
    Image* s = reinterpret_cast<Image*>(src);
    Image* d = reinterpret_cast<Image*>(dst);
    std::vector<VkImageCopy> copies(regions, regions + count);
    Cmd(cmd).steps.push_back([s, d, copies](Exec&) {
        for (const VkImageCopy& r : copies) {
            const D3D11_BOX box{UINT(r.srcOffset.x), UINT(r.srcOffset.y), 0, UINT(r.srcOffset.x) + r.extent.width,
                                UINT(r.srcOffset.y) + r.extent.height, 1};
            g_gpu->context->CopySubresourceRegion(
                d->texture, D3D11CalcSubresource(r.dstSubresource.mipLevel, 0, d->levels), UINT(r.dstOffset.x),
                UINT(r.dstOffset.y), 0, s->texture, D3D11CalcSubresource(r.srcSubresource.mipLevel, 0, s->levels),
                s->depth ? nullptr : &box);
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cmd, VkImage image, VkImageLayout,
                                             const VkClearColorValue* value, std::uint32_t,
                                             const VkImageSubresourceRange*) {
    Image* img = reinterpret_cast<Image*>(image);
    const VkClearColorValue v = *value;
    Cmd(cmd).steps.push_back([img, v](Exec&) {
        View* view = WholeView(img);
        if (ID3D11RenderTargetView* rtv = RtvOf(view)) {
            g_gpu->context->ClearRenderTargetView(rtv, v.float32);
        } else if (ID3D11UnorderedAccessView* uav = UavOf(view, false)) {
            g_gpu->context->ClearUnorderedAccessViewFloat(uav, v.float32);
        } else {
            Unsupported("a clear of an image with no view to clear it through");
        }
    });
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer cmd, VkImage image, VkImageLayout,
                                                    const VkClearDepthStencilValue* value, std::uint32_t,
                                                    const VkImageSubresourceRange*) {
    Image* img = reinterpret_cast<Image*>(image);
    const VkClearDepthStencilValue v = *value;
    Cmd(cmd).steps.push_back([img, v](Exec&) {
        if (ID3D11DepthStencilView* dsv = DsvOf(WholeView(img)))
            g_gpu->context->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, v.depth, UINT8(v.stencil));
    });
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {}

VKAPI_ATTR VkResult VKAPI_CALL CreateFence(VkDevice, const VkFenceCreateInfo* ci, const VkAllocationCallbacks*,
                                          VkFence* out) {
    auto* f = new Fence;
    f->signaled = (ci->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *out = reinterpret_cast<VkFence>(f);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks*) {
    auto* f = reinterpret_cast<Fence*>(fence);
    if (!f) return;
    std::lock_guard<std::mutex> hold(g_gpu->context_lock);
    for (Readback& r : f->readbacks) Release(r.staging);
    Release(f->done);
    delete f;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetFences(VkDevice, std::uint32_t count, const VkFence* fences) {
    for (std::uint32_t i = 0; i < count; ++i) reinterpret_cast<Fence*>(fences[i])->signaled = false;
    return VK_SUCCESS;
}

void CompleteFence(Fence* f) {
    if (!f || f->signaled) return;
    // The GPU past the submission: what it read of host memory's twins (the
    // vertex rings, written with NO_OVERWRITE) may be written again.
    if (f->value && g_fence && g_fence->GetCompletedValue() < f->value) {
        thread_local HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (SUCCEEDED(g_fence->SetEventOnCompletion(f->value, event))) WaitForSingleObject(event, 2000);
    } else if (f->done) {
        for (int spins = 0;; ++spins) {
            HRESULT hr;
            {
                std::lock_guard<std::mutex> hold(g_gpu->context_lock);
                hr = g_gpu->context->GetData(f->done, nullptr, 0, 0);
            }
            if (hr != S_FALSE) break;
            if (spins < 64) SwitchToThread();
            else Sleep(0);
        }
    }
    if (!f->readbacks.empty()) {
        std::lock_guard<std::mutex> hold(g_gpu->context_lock);
        for (Readback& r : f->readbacks) FinishReadback(r);
        f->readbacks.clear();
    }
    f->signaled = true;
}

VKAPI_ATTR VkResult VKAPI_CALL WaitForFences(VkDevice, std::uint32_t count, const VkFence* fences, VkBool32,
                                            std::uint64_t) {
    for (std::uint32_t i = 0; i < count; ++i) CompleteFence(reinterpret_cast<Fence*>(fences[i]));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    auto* f = reinterpret_cast<Fence*>(fence);
    std::vector<Readback> readbacks;
    {
        std::lock_guard<std::mutex> hold(g_gpu->context_lock);
        for (std::uint32_t i = 0; i < count; ++i)
            for (std::uint32_t k = 0; k < submits[i].commandBufferCount; ++k) {
                CommandBuffer& cb = Cmd(submits[i].pCommandBuffers[k]);
                for (const auto& [b, range] : cb.uploads) Upload(b, range.first, range.second - range.first);
                Exec x;
                x.readbacks = &readbacks;
                for (Step& step : cb.steps) step(x);
                // (What the last pass drew into goes back to being read.)
                g_gpu->context->OMSetRenderTargets(0, nullptr, nullptr);
            }
        if (f && KernelFence()) {
            f->value = ++g_fence_value;
            g_context4->Signal(g_fence, f->value);
        } else if (f) {
            if (!f->done) {
                D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT, 0};
                g_gpu->device->CreateQuery(&qd, &f->done);
            }
            if (f->done) g_gpu->context->End(f->done);
        }
        g_gpu->context->Flush();
        static const bool debug = std::getenv("PS2HOST_D3D11_DEBUG") != nullptr;
        if (debug) SayDebugMessages();
    }
    if (f) {
        f->readbacks = std::move(readbacks);
        f->signaled = false;
    } else {
        Fence done;
        done.readbacks = std::move(readbacks);
        CompleteFence(&done);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL DeviceWaitIdle(VkDevice) { return VK_SUCCESS; }

VKAPI_ATTR VkResult VKAPI_CALL GetQueryPoolResults(VkDevice, VkQueryPool, std::uint32_t, std::uint32_t, std::size_t,
                                                  void*, VkDeviceSize, VkQueryResultFlags) {
    return VK_NOT_READY;
}
VKAPI_ATTR VkResult VKAPI_CALL CreateQueryPool(VkDevice, const VkQueryPoolCreateInfo*, const VkAllocationCallbacks*,
                                              VkQueryPool*) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}
VKAPI_ATTR void VKAPI_CALL DestroyQueryPool(VkDevice, VkQueryPool, const VkAllocationCallbacks*) {}

// The physical device the renderer asks about: what Direct3D 11 at 11_0 gives.
VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties* p) {
    *p = VkPhysicalDeviceProperties{};
    p->apiVersion = VK_API_VERSION_1_3;
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    std::snprintf(p->deviceName, sizeof p->deviceName, "%s", g_gpu->Describe().c_str());
    p->limits.maxSamplerAnisotropy = 16.0f;
    p->limits.maxImageDimension2D = 16384;
    p->limits.maxPushConstantsSize = 256;
    p->limits.maxComputeWorkGroupCount[0] = p->limits.maxComputeWorkGroupCount[1] =
        p->limits.maxComputeWorkGroupCount[2] = 65535;
    p->limits.timestampPeriod = 0.0f;
    p->limits.minStorageBufferOffsetAlignment = 16;
    p->limits.nonCoherentAtomSize = 64;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceFeatures(VkPhysicalDevice, VkPhysicalDeviceFeatures* f) {
    *f = VkPhysicalDeviceFeatures{};
    f->dualSrcBlend = VK_TRUE;
    f->logicOp = g_gpu->logic_op ? VK_TRUE : VK_FALSE;     // (11.1's, through the target's UINT view)
    f->textureCompressionBC = VK_TRUE;
    f->samplerAnisotropy = VK_TRUE;
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceMemoryProperties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* m) {
    *m = VkPhysicalDeviceMemoryProperties{};
    m->memoryTypeCount = 2;
    m->memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    m->memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                             VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                         1};
    m->memoryHeapCount = 2;
    m->memoryHeaps[0] = {VkDeviceSize(4) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    m->memoryHeaps[1] = {VkDeviceSize(8) << 30, 0};
}

VKAPI_ATTR void VKAPI_CALL GetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice, std::uint32_t* count,
                                                                 VkQueueFamilyProperties* props) {
    if (!props) {
        *count = 1;
        return;
    }
    if (*count < 1) return;
    props[0] = VkQueueFamilyProperties{};
    props[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    props[0].queueCount = 1;
    props[0].timestampValidBits = 0;        // (no GPU timestamps: MakeTimestamps makes none)
    *count = 1;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice, std::uint32_t, std::uint32_t, VkQueue* queue) {
    static int tag;
    *queue = reinterpret_cast<VkQueue>(&tag);
}

}  // namespace

void Install(D3D11Gpu& gpu) {
    g_gpu = &gpu;
#define PS2GPU_VK11(name) vk##name = name
    PS2GPU_VK11(CreateBuffer);
    PS2GPU_VK11(DestroyBuffer);
    PS2GPU_VK11(GetBufferMemoryRequirements);
    PS2GPU_VK11(BindBufferMemory);
    PS2GPU_VK11(AllocateMemory);
    PS2GPU_VK11(FreeMemory);
    PS2GPU_VK11(MapMemory);
    PS2GPU_VK11(UnmapMemory);
    PS2GPU_VK11(CreateImage);
    PS2GPU_VK11(DestroyImage);
    PS2GPU_VK11(GetImageMemoryRequirements);
    PS2GPU_VK11(BindImageMemory);
    PS2GPU_VK11(CreateImageView);
    PS2GPU_VK11(DestroyImageView);
    PS2GPU_VK11(CreateSampler);
    PS2GPU_VK11(DestroySampler);
    PS2GPU_VK11(CreateShaderModule);
    PS2GPU_VK11(DestroyShaderModule);
    PS2GPU_VK11(CreateDescriptorSetLayout);
    PS2GPU_VK11(DestroyDescriptorSetLayout);
    PS2GPU_VK11(CreatePipelineLayout);
    PS2GPU_VK11(DestroyPipelineLayout);
    PS2GPU_VK11(CreateDescriptorPool);
    PS2GPU_VK11(DestroyDescriptorPool);
    PS2GPU_VK11(ResetDescriptorPool);
    PS2GPU_VK11(AllocateDescriptorSets);
    PS2GPU_VK11(FreeDescriptorSets);
    PS2GPU_VK11(UpdateDescriptorSets);
    PS2GPU_VK11(CreatePipelineCache);
    PS2GPU_VK11(DestroyPipelineCache);
    PS2GPU_VK11(GetPipelineCacheData);
    PS2GPU_VK11(CreateGraphicsPipelines);
    PS2GPU_VK11(CreateComputePipelines);
    PS2GPU_VK11(DestroyPipeline);
    PS2GPU_VK11(CreateCommandPool);
    PS2GPU_VK11(DestroyCommandPool);
    PS2GPU_VK11(AllocateCommandBuffers);
    PS2GPU_VK11(ResetCommandBuffer);
    PS2GPU_VK11(BeginCommandBuffer);
    PS2GPU_VK11(EndCommandBuffer);
    PS2GPU_VK11(CmdBindPipeline);
    PS2GPU_VK11(CmdBindDescriptorSets);
    PS2GPU_VK11(CmdPushConstants);
    PS2GPU_VK11(CmdBindVertexBuffers2);
    PS2GPU_VK11(CmdSetViewport);
    PS2GPU_VK11(CmdSetScissor);
    PS2GPU_VK11(CmdSetDepthTestEnable);
    PS2GPU_VK11(CmdSetDepthWriteEnable);
    PS2GPU_VK11(CmdSetDepthCompareOp);
    PS2GPU_VK11(CmdSetStencilTestEnable);
    PS2GPU_VK11(CmdSetStencilOp);
    PS2GPU_VK11(CmdSetStencilCompareMask);
    PS2GPU_VK11(CmdSetStencilWriteMask);
    PS2GPU_VK11(CmdSetStencilReference);
    PS2GPU_VK11(CmdDraw);
    PS2GPU_VK11(CmdDispatch);
    PS2GPU_VK11(CmdPipelineBarrier);
    PS2GPU_VK11(CmdBeginRendering);
    PS2GPU_VK11(CmdEndRendering);
    PS2GPU_VK11(CmdExecuteCommands);
    PS2GPU_VK11(CmdCopyBuffer);
    PS2GPU_VK11(CmdUpdateBuffer);
    PS2GPU_VK11(CmdFillBuffer);
    PS2GPU_VK11(CmdCopyBufferToImage);
    PS2GPU_VK11(CmdCopyImageToBuffer);
    PS2GPU_VK11(CmdCopyImage);
    PS2GPU_VK11(CmdClearColorImage);
    PS2GPU_VK11(CmdClearDepthStencilImage);
    PS2GPU_VK11(CmdWriteTimestamp);
    PS2GPU_VK11(CmdResetQueryPool);
    PS2GPU_VK11(CreateFence);
    PS2GPU_VK11(DestroyFence);
    PS2GPU_VK11(ResetFences);
    PS2GPU_VK11(WaitForFences);
    PS2GPU_VK11(QueueSubmit);
    PS2GPU_VK11(DeviceWaitIdle);
    PS2GPU_VK11(GetQueryPoolResults);
    PS2GPU_VK11(CreateQueryPool);
    PS2GPU_VK11(DestroyQueryPool);
    PS2GPU_VK11(GetPhysicalDeviceProperties);
    PS2GPU_VK11(GetPhysicalDeviceFeatures);
    PS2GPU_VK11(GetPhysicalDeviceMemoryProperties);
    PS2GPU_VK11(GetPhysicalDeviceQueueFamilyProperties);
    PS2GPU_VK11(GetDeviceQueue);
#undef PS2GPU_VK11
    // The renderer's dynamic rendering and extended dynamic state are these
    // (the 1.3 names: Vulkan 1.3's path).
    g_vk_path = kVk13;
    g_store_op_none = true;
    g_load_op_none = false;
}

ID3D11ShaderResourceView* ImageSrv(std::uint64_t image, std::uint32_t* width, std::uint32_t* height) {
    auto* img = reinterpret_cast<Image*>(image);
    if (!img || !img->texture) return nullptr;
    if (width) *width = img->width;
    if (height) *height = img->height;
    return SrvOf(WholeView(img));
}

}  // namespace vk11
}  // namespace ps2gpu

#endif  // _WIN32
