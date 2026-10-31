# The port's display on the GPU: SDL3 for the window, input and events, Vulkan
# for the picture. Vulkan reaches every platform the port aims at - Windows,
# Linux and Android natively, macOS through MoltenVK - and gives the control a GS
# renderer needs; SDL3 hides each platform's window and input. Both come from
# their releases at configure time (FetchContent), so nothing is installed on
# the machine and nothing of theirs is committed.
#
# Stage one presents the software GS's picture (a copy and a scaled blit, no
# shaders). The GPU GS renderer builds on this device and swapchain.

cmake_minimum_required(VERSION 3.20)
project(ps2gpu CXX C)

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.363
    GIT_SHALLOW TRUE)
FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE)
FetchContent_Declare(sdl3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.18
    GIT_SHALLOW TRUE)

set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(vulkan_headers)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
FetchContent_MakeAvailable(volk sdl3)

# Shaders: GLSL compiled to SPIR-V at build time by glslang, from its release
# like the others, and embedded as arrays (glslang --vn). A build for another
# machine (Android's) runs the PC's: PS2_HOST_GLSLANG, the glslang a PC build
# made (engine/tools/port.py's android passes it).
set(PS2_HOST_GLSLANG "" CACHE FILEPATH "glslang to run at build time (a cross build's: the PC's)")
if(PS2_HOST_GLSLANG)
    set(GLSLANG ${PS2_HOST_GLSLANG})
else()
    set(GLSLANG glslang-standalone)
FetchContent_Declare(glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE)
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glslang)
endif()

set(PS2GPU_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${PS2GPU_SHADER_DIR})
set(PS2GPU_SHADER_HEADERS)
foreach(shader gs.vert gs.frag date.vert date.frag swizzle.comp decode.comp depth_load.frag patch.comp shuffle.comp fxaa.comp downsample.comp
               smaa_edges.comp smaa_weights.comp smaa_blend.comp)
    string(REPLACE "." "_" name ${shader})
    set(out ${PS2GPU_SHADER_DIR}/${name}.h)
    add_custom_command(OUTPUT ${out}
        COMMAND ${GLSLANG} -V --target-env vulkan1.1 --vn spirv_${name}
                -I${CMAKE_CURRENT_LIST_DIR}/third_party/smaa
                -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/${shader}
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/${shader} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
                ${CMAKE_CURRENT_LIST_DIR}/shaders/smaa_common.glsl ${CMAKE_CURRENT_LIST_DIR}/third_party/smaa/SMAA.glsl
                ${GLSLANG}
        COMMENT "glslang ${shader}")
    list(APPEND PS2GPU_SHADER_HEADERS ${out})
endforeach()
# The fragment shader without its second output, for devices without
# dual-source blending.
set(out ${PS2GPU_SHADER_DIR}/gs_frag_single.h)
add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DSINGLE_OUTPUT --vn spirv_gs_frag_single
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag ${GLSLANG}
    COMMENT "glslang gs.frag (single output)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# The swizzle straight to and from a colour target (a storage image).
set(out ${PS2GPU_SHADER_DIR}/swizzle_image_comp.h)
add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DIMAGE --vn spirv_swizzle_image_comp
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
            ${GLSLANG}
    COMMENT "glslang swizzle.comp (a target's image)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# The texture decode straight into a texture's image (a storage image).
set(out ${PS2GPU_SHADER_DIR}/decode_image_comp.h)
add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DTO_IMAGE --vn spirv_decode_image_comp
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/decode.comp
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/decode.comp ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
            ${GLSLANG}
    COMMENT "glslang decode.comp (into a texture's image)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# ...and from a depth buffer, sampled.
set(out ${PS2GPU_SHADER_DIR}/swizzle_depth_comp.h)
add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DDEPTH_IMAGE --vn spirv_swizzle_depth_comp
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/swizzle.comp ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
            ${GLSLANG}
    COMMENT "glslang swizzle.comp (a depth buffer)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# ...and blending in the shader, from a copy of the target (a factor over 1).
set(out ${PS2GPU_SHADER_DIR}/gs_frag_blend.h)
add_custom_command(OUTPUT ${out}
    COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DSHADER_BLEND --vn spirv_gs_frag_blend
            -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/gs.frag ${GLSLANG}
    COMMENT "glslang gs.frag (blending in the shader)")
list(APPEND PS2GPU_SHADER_HEADERS ${out})
# The PC's effects (effects.cpp): effects.comp once for each pass.
foreach(pass RANGE 0 15)
    set(out ${PS2GPU_SHADER_DIR}/effects_${pass}.h)
    add_custom_command(OUTPUT ${out}
        COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DPASS=${pass} --vn spirv_effects_${pass}
                -o ${out} ${CMAKE_CURRENT_LIST_DIR}/shaders/effects.comp
        DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/effects.comp ${GLSLANG}
        COMMENT "glslang effects.comp (pass ${pass})")
    list(APPEND PS2GPU_SHADER_HEADERS ${out})
endforeach()
# Direct3D 11's shaders (Windows): the same GLSL, its SPIR-V turned into HLSL
# by SPIRV-Cross (tools/spirv_to_hlsl.cpp names the registers) and compiled
# by fxc (the Windows SDK's) at shader model 5.0, embedded as arrays (dxbc_*).
# A cross build (Android's) has no Direct3D.
set(PS2GPU_D3D11 OFF)
if(WIN32 AND NOT PS2_HOST_GLSLANG)
    find_program(PS2GPU_FXC fxc HINTS "$ENV{WindowsSdkVerBinPath}x64" "$ENV{WindowsSdkBinPath}x64")
    if(NOT PS2GPU_FXC)
        file(GLOB PS2GPU_FXC_FOUND "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64/fxc.exe")
        if(PS2GPU_FXC_FOUND)
            list(SORT PS2GPU_FXC_FOUND)
            list(GET PS2GPU_FXC_FOUND -1 PS2GPU_FXC)
        endif()
    endif()
    if(PS2GPU_FXC)
        set(PS2GPU_D3D11 ON)
    else()
        message(WARNING "no fxc.exe (Windows SDK): Direct3D 11's GS renderer left out")
    endif()
endif()
if(PS2GPU_D3D11)
    FetchContent_Declare(spirv_cross
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
        GIT_TAG vulkan-sdk-1.4.363.0
        GIT_SHALLOW TRUE)
    set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_MSL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spirv_cross)
    add_executable(ps2gpu_spirv_to_hlsl ${CMAKE_CURRENT_LIST_DIR}/tools/spirv_to_hlsl.cpp)
    target_link_libraries(ps2gpu_spirv_to_hlsl PRIVATE spirv-cross-hlsl spirv-cross-glsl spirv-cross-core)
    # ps2gpu_hlsl(<name> <source> <fxc profile> [glslang -D...]): dxbc_<name> in <name>_dxbc.h.
    function(ps2gpu_hlsl name source profile)
        set(spv ${PS2GPU_SHADER_DIR}/${name}.spv)
        set(hlsl ${PS2GPU_SHADER_DIR}/${name}.hlsl)
        set(out ${PS2GPU_SHADER_DIR}/${name}_dxbc.h)
        # (the HLSL as an array too, for every shader: ${name}_hlsl.h)
        add_custom_command(OUTPUT ${out} ${PS2GPU_SHADER_DIR}/${name}_hlsl.h
            COMMAND ${GLSLANG} -V --target-env vulkan1.1 -DD3D11 ${ARGN}
                    -I${CMAKE_CURRENT_LIST_DIR}/third_party/smaa -o ${spv} ${CMAKE_CURRENT_LIST_DIR}/shaders/${source}
            COMMAND ps2gpu_spirv_to_hlsl ${spv} ${hlsl} ${PS2GPU_SHADER_DIR}/${name}_hlsl.h ${name}
            COMMAND ${PS2GPU_FXC} /nologo /O3 /T ${profile} /E main /Fh ${out} /Vn dxbc_${name} ${hlsl}
            DEPENDS ${CMAKE_CURRENT_LIST_DIR}/shaders/${source} ${CMAKE_CURRENT_LIST_DIR}/shaders/gs_address.glsl
                    ${CMAKE_CURRENT_LIST_DIR}/shaders/smaa_common.glsl ${CMAKE_CURRENT_LIST_DIR}/third_party/smaa/SMAA.glsl
                    ${CMAKE_CURRENT_LIST_DIR}/tools/spirv_to_hlsl.cpp ${GLSLANG} ps2gpu_spirv_to_hlsl
            COMMENT "HLSL ${name} (${source})")
        set(PS2GPU_SHADER_HEADERS ${PS2GPU_SHADER_HEADERS} ${out} PARENT_SCOPE)
    endfunction()
    ps2gpu_hlsl(d3d_gs_vert gs.vert vs_5_0)
    ps2gpu_hlsl(d3d_gs_frag gs.frag ps_5_0)
    ps2gpu_hlsl(d3d_gs_frag_single gs.frag ps_5_0 -DSINGLE_OUTPUT)
    ps2gpu_hlsl(d3d_gs_frag_blend gs.frag ps_5_0 -DSHADER_BLEND)
    ps2gpu_hlsl(d3d_gs_frag_uint gs.frag ps_5_0 -DSINGLE_OUTPUT -DUINT_OUTPUT)
    ps2gpu_hlsl(d3d_date_vert date.vert vs_5_0)
    ps2gpu_hlsl(d3d_date_frag date.frag ps_5_0)
    ps2gpu_hlsl(d3d_depth_load_frag depth_load.frag ps_5_0)
    ps2gpu_hlsl(d3d_swizzle swizzle.comp cs_5_0)
    ps2gpu_hlsl(d3d_swizzle_image swizzle.comp cs_5_0 -DIMAGE -DWORD_IMAGE)
    ps2gpu_hlsl(d3d_swizzle_depth swizzle.comp cs_5_0 -DDEPTH_IMAGE)
    ps2gpu_hlsl(d3d_decode decode.comp cs_5_0)
    ps2gpu_hlsl(d3d_decode_image decode.comp cs_5_0 -DTO_IMAGE)
    ps2gpu_hlsl(d3d_patch patch.comp cs_5_0)
    ps2gpu_hlsl(d3d_shuffle shuffle.comp cs_5_0)
    ps2gpu_hlsl(d3d_fxaa fxaa.comp cs_5_0)
    ps2gpu_hlsl(d3d_downsample downsample.comp cs_5_0)
    ps2gpu_hlsl(d3d_smaa_edges smaa_edges.comp cs_5_0)
    ps2gpu_hlsl(d3d_smaa_weights smaa_weights.comp cs_5_0)
    ps2gpu_hlsl(d3d_smaa_blend smaa_blend.comp cs_5_0)
    foreach(pass RANGE 0 15)
        ps2gpu_hlsl(d3d_effects_${pass} effects.comp cs_5_0 -DPASS=${pass})
    endforeach()
endif()
add_custom_target(ps2gpu_shaders DEPENDS ${PS2GPU_SHADER_HEADERS})

add_library(ps2gpu STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/window_display.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/vulkan_display.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/d3d11_gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/d3d11_display.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/d3d11_renderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/d3d11_vulkan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/vk_device.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs_renderer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/effects.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/threaded_renderer.cpp)
add_dependencies(ps2gpu ps2gpu_shaders)
target_include_directories(ps2gpu PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include PRIVATE ${PS2GPU_SHADER_DIR}
                           ${CMAKE_CURRENT_LIST_DIR}/third_party/smaa)
target_compile_features(ps2gpu PUBLIC cxx_std_17)
target_link_libraries(ps2gpu PUBLIC SDL3::SDL3-static PRIVATE volk::volk Vulkan::Headers)
if(PS2GPU_D3D11)
    target_compile_definitions(ps2gpu PRIVATE PS2GPU_D3D11=1)
endif()
if(MSVC)
    target_compile_options(ps2gpu PRIVATE /W3 /Z7)        # symbols for --sample, as ps2host has
endif()

if(NOT ANDROID)
    add_executable(ps2gpu_smoke ${CMAKE_CURRENT_LIST_DIR}/tests/display_smoke.cpp)
    target_link_libraries(ps2gpu_smoke PRIVATE ps2gpu)
    add_executable(ps2gpu_tests ${CMAKE_CURRENT_LIST_DIR}/tests/renderer_tests.cpp)
    target_link_libraries(ps2gpu_tests PRIVATE ps2gpu)
endif()
