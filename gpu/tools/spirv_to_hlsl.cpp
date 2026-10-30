/*
 * spirv_to_hlsl: a shader's SPIR-V (glslang's, the Vulkan build's own) as
 * HLSL for Direct3D 11 (shader model 5.0), at build time - fxc then compiles
 * it. SPIRV-Cross does the translation; this names the registers:
 *
 *   - a resource at (set, binding) is register set * 4 + binding of its kind
 *     (t for textures and read-only buffers, s for their samplers, u for
 *     storage images and buffers): Direct3D 11 has no sets;
 *   - the push constants are constant buffer b0.
 *
 *   spirv_to_hlsl in.spv out.hlsl
 */

#include <spirv_hlsl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: spirv_to_hlsl in.spv out.hlsl\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty() || bytes.size() % 4) {
        std::fprintf(stderr, "spirv_to_hlsl: %s is not SPIR-V\n", argv[1]);
        return 1;
    }
    std::vector<std::uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    try {
        spirv_cross::CompilerHLSL compiler(std::move(words));
        spirv_cross::CompilerHLSL::Options options;
        options.shader_model = 50;
        compiler.set_hlsl_options(options);
        spirv_cross::CompilerGLSL::Options common = compiler.get_common_options();
        common.vertex.flip_vert_y = false;
        compiler.set_common_options(common);
        const spv::ExecutionModel model = compiler.get_execution_model();
        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
        auto bind = [&](const spirv_cross::Resource& r) {
            const std::uint32_t set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
            const std::uint32_t binding = compiler.get_decoration(r.id, spv::DecorationBinding);
            spirv_cross::HLSLResourceBinding b;
            b.stage = model;
            b.desc_set = set;
            b.binding = binding;
            const std::uint32_t reg = set * 4 + binding;
            b.cbv.register_binding = b.srv.register_binding = b.uav.register_binding = b.sampler.register_binding = reg;
            compiler.add_hlsl_resource_binding(b);
        };
        for (const auto* list : {&resources.sampled_images, &resources.separate_images, &resources.separate_samplers,
                                 &resources.storage_images, &resources.storage_buffers, &resources.uniform_buffers})
            for (const spirv_cross::Resource& r : *list) bind(r);
        spirv_cross::HLSLResourceBinding push;
        push.stage = model;
        push.desc_set = spirv_cross::ResourceBindingPushConstantDescriptorSet;
        push.binding = spirv_cross::ResourceBindingPushConstantBinding;
        push.cbv.register_binding = 0;
        compiler.add_hlsl_resource_binding(push);
        const std::string hlsl = compiler.compile();
        std::ofstream out(argv[2], std::ios::binary);
        out << hlsl;
        return out ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "spirv_to_hlsl: %s: %s\n", argv[1], e.what());
        return 1;
    }
}
