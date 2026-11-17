/*
 * spirv_to_hlsl: a shader's SPIR-V (glslang's, the Vulkan build's own) as
 * HLSL for Direct3D 11 (shader model 5.0), at build time - fxc then compiles
 * it. SPIRV-Cross does the translation; this names the registers:
 *
 *   - a resource at (set, binding) is register set * 7 + binding of its kind
 *     (t for textures and read-only buffers, s for their samplers, u for
 *     storage images and buffers): Direct3D 11 has no sets;
 *   - the push constants are constant buffer b0;
 *   - set 3's binding 1 (the native meshes' header: a STORAGE_BUFFER_DYNAMIC
 *     in Vulkan) read at the draw's dynamic offset: every Load of it has
 *     _ps2gpu_dynamic added, the first uint of constant buffer b12 - the D3D11
 *     layer's table of offsets bound there at the offset that holds the
 *     draw's (CmdBindDescriptorSets) - Direct3D 11 binds no view at an offset
 *     that changes each draw, and a constant buffer bound at one is no map.
 *
 *   spirv_to_hlsl in.spv out.hlsl [out.h name]
 *
 * With out.h, the HLSL is also written there as a C array (hlsl_<name>):
 * a shader the D3D11 layer compiles again for a pipeline's specialisation.
 */

#include <spirv_hlsl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::fprintf(stderr, "usage: spirv_to_hlsl in.spv out.hlsl [out.h name]\n");
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
        const spv::ExecutionModel model = compiler.get_execution_model();
        // Vulkan's clip space has Y down, Direct3D's up: a vertex shader's
        // position flipped, so the same pixels are covered.
        spirv_cross::CompilerGLSL::Options common = compiler.get_common_options();
        common.vertex.flip_vert_y = model == spv::ExecutionModelVertex;
        compiler.set_common_options(common);
        const spirv_cross::ShaderResources resources = compiler.get_shader_resources();
        auto bind = [&](const spirv_cross::Resource& r) {
            const std::uint32_t set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
            const std::uint32_t binding = compiler.get_decoration(r.id, spv::DecorationBinding);
            spirv_cross::HLSLResourceBinding b;
            b.stage = model;
            b.desc_set = set;
            b.binding = binding;
            const std::uint32_t reg = set * 7 + binding;
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
        std::string hlsl = compiler.compile();
        // The dynamic buffer (set 3, binding 1): its Loads at the draw's offset.
        std::string dynamic;
        for (const spirv_cross::Resource& r : resources.storage_buffers)
            if (compiler.get_decoration(r.id, spv::DecorationDescriptorSet) == 3 &&
                compiler.get_decoration(r.id, spv::DecorationBinding) == 1)
                dynamic = compiler.get_name(r.id);
        if (!dynamic.empty()) {
            const std::regex load("\\b" + dynamic + "\\.Load([234]?)\\(");
            const std::string with = hlsl;
            hlsl = std::regex_replace(with, load, dynamic + ".Load$1(_ps2gpu_dynamic + ");
            // (its offset: b12's first uint, before the shader's own code)
            const std::size_t at = hlsl.find("ByteAddressBuffer");
            hlsl.insert(at == std::string::npos ? 0 : at,
                        "cbuffer PS2GPU_Dynamic : register(b12)\n{\n    uint _ps2gpu_dynamic : packoffset(c0.x);\n};\n\n");
        }
        std::ofstream out(argv[2], std::ios::binary);
        out << hlsl;
        if (!out) return 1;
        if (argc == 5) {
            std::ofstream header(argv[3], std::ios::binary);
            header << "static const char hlsl_" << argv[4] << "[] = {";
            for (std::size_t i = 0; i < hlsl.size(); ++i)
                header << (i % 24 ? "" : "\n    ") << int(static_cast<unsigned char>(hlsl[i])) << ",";
            header << "\n    0};\n";
            if (!header) return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "spirv_to_hlsl: %s: %s\n", argv[1], e.what());
        return 1;
    }
}
