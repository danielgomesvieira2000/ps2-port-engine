# The native host: the shared runtime (PS2 instruction semantics, also used by
# verification) and the PC platform layer that serves the translated code's
# hooks. A game port adds its translated functions with ps2_add_port().

cmake_minimum_required(VERSION 3.20)

set(PS2_ENGINE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

add_library(ps2rt STATIC
    ${PS2_ENGINE_DIR}/runtime/src/eefpu.cpp
    ${PS2_ENGINE_DIR}/runtime/src/eeint.cpp
    ${PS2_ENGINE_DIR}/runtime/src/vu0.cpp
    ${PS2_ENGINE_DIR}/runtime/src/lifted_support.cpp)
target_include_directories(ps2rt PUBLIC ${PS2_ENGINE_DIR}/runtime/include)
target_compile_features(ps2rt PUBLIC cxx_std_17)

add_library(ps2host STATIC
    ${CMAKE_CURRENT_LIST_DIR}/src/host.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/devices.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/ipu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/vif1_worker.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/threads.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/fiber.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/platform.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/high_fps.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/display_list.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/iop.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/sdk.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/disc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/cdvd.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/fileio.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/vif.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/dma.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/loadfile.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/dbc.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/mc2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/mcserv.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/iop_vm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/iop_libs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/iop_inet.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/iopheap.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/spu2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/audio.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/music.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/presence.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/vu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/timers.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/pc_menu.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/window.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/settings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/hud.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/libnet.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/logitech.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/bindings.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/timeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs_thread.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs_draw.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gs_memory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu_bridge.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/profile.cpp)
# The IOP interpreter names library functions from engine/data/iop_exports.csv,
# built in (iop_exports.inc: a line of it a string), so that the executable
# reads nothing from where it was built.
file(STRINGS ${PS2_ENGINE_DIR}/data/iop_exports.csv iop_export_lines)
set(iop_exports_inc "// From engine/data/iop_exports.csv (engine/host/CMakeLists.txt): generated.\n")
foreach(line IN LISTS iop_export_lines)
    string(REPLACE "\\" "\\\\" line "${line}")
    string(REPLACE "\"" "\\\"" line "${line}")
    string(APPEND iop_exports_inc "\"${line}\\n\"\n")
endforeach()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated/iop_exports.inc.new "${iop_exports_inc}")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/generated/iop_exports.inc.new
               ${CMAKE_CURRENT_BINARY_DIR}/generated/iop_exports.inc COPYONLY)     # (touched only when it changes)
target_include_directories(ps2host PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${PS2_ENGINE_DIR}/data/iop_exports.csv)
target_include_directories(ps2host PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(ps2host PUBLIC ps2rt)

# Texture packs (texture_pack.cpp): single headers, pinned by hash - xxHash
# (BSD-2-Clause; PCSX2's names are XXH3 hashes), bcdec (MIT or Unlicense; DDS
# BC1-BC7) and stb_image (MIT or public domain; PNG). And for the host's music
# (music.cpp), minimp3 (CC0: public domain; MP3); for the overlay's text where
# there is no GDI (window.cpp), stb_easy_font (MIT or public domain).
set(third_party_dir ${CMAKE_CURRENT_BINARY_DIR}/third_party)
foreach(entry
        "xxhash.h|https://raw.githubusercontent.com/Cyan4973/xxHash/v0.8.3/xxhash.h|17973c0dc49d9854ca26caa191f0e12f7a424b68858d9a78de3860d959d85e4b"
        "bcdec.h|https://raw.githubusercontent.com/iOrange/bcdec/80859ed3b7afb1c527a2a99d70c61457bea72d0c/bcdec.h|134520764d96f70a27db814173616c89ad85db95fbd3328417c7a8c77e7ca189"
        "stb_image.h|https://raw.githubusercontent.com/nothings/stb/2c980bb59875b0d32144a71867fbdebb2f77cd20/stb_image.h|594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3"
        "stb_easy_font.h|https://raw.githubusercontent.com/nothings/stb/2c980bb59875b0d32144a71867fbdebb2f77cd20/stb_easy_font.h|7b48b52a316477766c0a7a096b33a29bcbd847b6fab4fabd32bb3fdb7cea1a60"
        "minimp3.h|https://raw.githubusercontent.com/lieff/minimp3/afb604c06bc8beb145fecd42c0ceb5bda8795144/minimp3.h|5fb296a790734772b65a5514438cc06d14f24e539c13475ff2a7f737681c82c0")
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 name)
    list(GET parts 1 url)
    list(GET parts 2 sha)
    if(NOT EXISTS ${third_party_dir}/${name})
        file(DOWNLOAD ${url} ${third_party_dir}/${name} EXPECTED_HASH SHA256=${sha} TLS_VERIFY ON)
    endif()
endforeach()
target_include_directories(ps2host PRIVATE ${third_party_dir})
target_sources(ps2host PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/texture_pack.cpp)

# The window through SDL3 and Vulkan (engine/gpu), with GDI as the fallback.
option(PS2HOST_VULKAN "Present the picture through SDL3 and Vulkan" ON)
if(PS2HOST_VULKAN)
    add_subdirectory(${PS2_ENGINE_DIR}/gpu ${CMAKE_BINARY_DIR}/ps2gpu)
    target_link_libraries(ps2host PUBLIC ps2gpu)
    target_compile_definitions(ps2host PRIVATE PS2HOST_VULKAN=1)
endif()

# AVX2: three-operand SIMD for the VU's lanes and the rest (6% of a race).
# Off: a build for processors without it (Intel's Pentium and Celeron of
# before 2021), SSE4.1 at least - a port ships both, the AVX2 one starting
# the other where it must (src/cpu_check.cpp).
option(PS2HOST_AVX2 "Build for processors with AVX2 (off: SSE4.1)" ON)
# A game executable's entry (ps2_add_port's /ENTRY), before the runtime: the
# processor checked by code built without AVX - a library of its own, apart
# from ps2rt's flags - and the SSE4 build started instead where it must be.
if(MSVC)
    add_library(ps2_cpu_check STATIC ${CMAKE_CURRENT_LIST_DIR}/src/cpu_check.cpp)
    if(PS2HOST_AVX2)
        target_compile_definitions(ps2_cpu_check PRIVATE PS2HOST_BUILT_FOR_AVX2=1)
    endif()
endif()

if(MSVC)
    # Translated code throws lifted::Abort; its chunks are large.
    target_compile_options(ps2rt PUBLIC /EHsc /bigobj /W3 /wd4100 /wd4189)
    if(PS2HOST_AVX2)
        target_compile_options(ps2rt PUBLIC /arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # MSVC takes SSE4.1's intrinsics without a flag; clang-cl needs the
        # floor named (the one cpu_check.cpp holds the SSE4 build to).
        target_compile_options(ps2rt PUBLIC /clang:-msse4.1)
    endif()
    target_compile_definitions(ps2rt PUBLIC _CRT_SECURE_NO_WARNINGS)
    # PS2HOST_PGO (clang-cl): "generate" - a build that counts what runs
    # (LLVM_PROFILE_FILE names where it writes them) - or the .profdata
    # merged from such runs (llvm-profdata), to build with what they ran
    # (engine/tools/port.py's pgo).
    set(PS2HOST_PGO "" CACHE STRING "Profile-guided build: 'generate', or the .profdata to build with")
    if(PS2HOST_PGO AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(PS2HOST_PGO STREQUAL "generate")
            target_compile_options(ps2rt PUBLIC /clang:-fprofile-instr-generate)
            target_compile_definitions(ps2rt PUBLIC PS2HOST_PGO_GENERATE=1)    # main.cpp writes the profile
            get_filename_component(llvm_bin ${CMAKE_CXX_COMPILER} DIRECTORY)
            file(GLOB profile_runtime ${llvm_bin}/../lib/clang/*/lib/windows/clang_rt.profile-x86_64.lib)
            if(NOT profile_runtime)
                message(FATAL_ERROR "PS2HOST_PGO=generate: LLVM's clang_rt.profile-x86_64.lib not found")
            endif()
            list(GET profile_runtime 0 profile_runtime)
            target_link_libraries(ps2rt PUBLIC ${profile_runtime})
        else()
            # The profile's time in the flags: a new profile builds everything
            # again (Ninja does not follow the file - objects from the last
            # one beside the new one's would not link with ThinLTO).
            file(TIMESTAMP ${PS2HOST_PGO} pgo_time "%Y%m%d%H%M%S")
            target_compile_definitions(ps2rt PUBLIC PS2HOST_PGO_TIME=${pgo_time})
            target_compile_options(ps2rt PUBLIC "/clang:-fprofile-instr-use=${PS2HOST_PGO}"
                                   /clang:-Wno-profile-instr-unprofiled /clang:-Wno-profile-instr-out-of-date
                                   /clang:-Wno-backend-plugin)
        endif()
    endif()
    # PS2HOST_LTO (clang-cl, lld-link): ThinLTO - the host's functions the
    # game's code calls (memory, the dispatch, the VU's) inlined across files.
    # The link compiles the files again (the VU's take 2 GB and more each):
    # as many at once as PS2HOST_LTO_JOBS (port.py: as for the compiler, by
    # the memory there is - all the processors' worth ran out of it).
    option(PS2HOST_LTO "ThinLTO (clang-cl and lld-link)" OFF)
    set(PS2HOST_LTO_JOBS "all" CACHE STRING "ThinLTO's code generators at once in the link")
    if(PS2HOST_LTO AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(ps2rt PUBLIC /clang:-flto=thin)
        target_link_options(ps2rt PUBLIC /opt:lldltojobs=${PS2HOST_LTO_JOBS})
    endif()
    target_compile_options(ps2host PRIVATE /Z7)     # symbols for --sample, as the ports'
    target_compile_options(ps2rt PRIVATE /Z7)
    target_link_libraries(ps2host PRIVATE dbghelp winmm)  # ...which resolves them; 1 ms sleeps
endif()
if(WIN32)
    target_link_libraries(ps2host PRIVATE ws2_32)         # the IOP's inet (iop_inet.cpp)
endif()
if(NOT MSVC)
    # clang or GCC (Android's NDK): translated code throws lifted::Abort; no
    # multiply-add fused into one rounding (AArch64's clang fuses by default)
    # - the game's floats as the PC's builds have them.
    target_compile_options(ps2rt PUBLIC -fexceptions -ffp-contract=off -Wno-unused-parameter)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
        if(PS2HOST_AVX2)
            target_compile_options(ps2rt PUBLIC -mavx2 -mfma -mbmi -mbmi2)
        else()
            target_compile_options(ps2rt PUBLIC -msse4.1)
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        # Arm: the SSE intrinsics the runtime and the VU's code use, as NEON -
        # sse2neon (MIT, pinned by hash), behind the x86 headers' names.
        set(sse2neon_dir ${CMAKE_CURRENT_BINARY_DIR}/sse2neon)
        if(NOT EXISTS ${sse2neon_dir}/sse2neon.h)
            file(DOWNLOAD https://raw.githubusercontent.com/DLTcollab/sse2neon/v1.8.0/sse2neon.h
                 ${sse2neon_dir}/sse2neon.h
                 EXPECTED_HASH SHA256=07723c9f9457dd4316f1fde3dd4eb6f31dd67d9955f6c21f4e609ac1698be48a TLS_VERIFY ON)
        endif()
        foreach(header immintrin.h emmintrin.h xmmintrin.h pmmintrin.h tmmintrin.h smmintrin.h nmmintrin.h)
            file(WRITE ${sse2neon_dir}/${header} "#pragma once\n#include \"sse2neon.h\"\n")
        endforeach()
        target_include_directories(ps2rt SYSTEM BEFORE PUBLIC ${sse2neon_dir})
    endif()
    if(ANDROID)
        target_link_libraries(ps2host PUBLIC android log)
        # The game a shared library (SDL's activity loads it): its symbols
        # hidden but for what Java looks up (SDL_main, the JNI natives - which
        # say so themselves), so the game's calls to its own functions and its
        # globals' reads are direct rather than through the PLT and the GOT;
        # no stack canaries in the translated code's every frame.
        target_compile_options(ps2rt PUBLIC -fvisibility=hidden -fvisibility-inlines-hidden -fno-plt
                               -fno-stack-protector)
        # ThinLTO here too (the NDK's clang and lld): the host's functions
        # the game's code calls inlined across files.
        option(PS2HOST_ANDROID_LTO "ThinLTO in the Android build" ON)
        set(PS2HOST_LTO_JOBS "all" CACHE STRING "ThinLTO's code generators at once in the link")
        set(PS2HOST_PGO "" CACHE STRING "Profile-guided build: the .profdata to build with")
        if(PS2HOST_ANDROID_LTO)
            target_compile_options(ps2rt PUBLIC -flto=thin)
            target_link_options(ps2rt PUBLIC -flto=thin -Wl,--thinlto-jobs=${PS2HOST_LTO_JOBS})
        endif()
        # PS2HOST_PGO: "generate" - a build that counts what runs (written
        # beside the settings: main.cpp) - or the .profdata merged from such
        # runs (engine/tools/android.py's pgo, on the emulator: the same
        # sources, the same counts as a phone's).
        if(PS2HOST_PGO STREQUAL "generate")
            target_compile_options(ps2rt PUBLIC -fprofile-instr-generate)
            target_link_options(ps2rt PUBLIC -fprofile-instr-generate)
            target_compile_definitions(ps2rt PUBLIC PS2HOST_PGO_GENERATE=1)
        elseif(PS2HOST_PGO)
            file(TIMESTAMP ${PS2HOST_PGO} pgo_time "%Y%m%d%H%M%S")
            target_compile_definitions(ps2rt PUBLIC PS2HOST_PGO_TIME=${pgo_time})
            target_compile_options(ps2rt PUBLIC "-fprofile-instr-use=${PS2HOST_PGO}" -Wno-profile-instr-unprofiled
                                   -Wno-profile-instr-out-of-date -Wno-backend-plugin)
        endif()
    endif()
endif()

# The host's own tests, which need no game.
# vu_corpus.cpp: random microprograms, compiled by engine/tools/vurecomp.py
# --corpus as the build goes, which VuCompiledIsTheInterpreter runs both ways.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(PS2HOST_VURECOMP ${CMAKE_CURRENT_LIST_DIR}/../tools/vurecomp.py)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vu_corpus.cpp
    COMMAND ${Python3_EXECUTABLE} ${PS2HOST_VURECOMP} --corpus --count 120 --seed 1
            --out ${CMAKE_CURRENT_BINARY_DIR}/vu_corpus.cpp
    DEPENDS ${PS2HOST_VURECOMP}
    COMMENT "vurecomp: the VU test corpus")
if(ANDROID)
    # (The tests run on the PC's build.)
    add_library(ps2host_tests OBJECT EXCLUDE_FROM_ALL ${CMAKE_CURRENT_LIST_DIR}/tests/host_tests.cpp)
else()
add_executable(ps2host_tests ${CMAKE_CURRENT_LIST_DIR}/tests/host_tests.cpp
                             ${CMAKE_CURRENT_BINARY_DIR}/vu_corpus.cpp)
endif()
target_link_libraries(ps2host_tests PRIVATE ps2host)
if(PS2HOST_VULKAN)
    target_compile_definitions(ps2host_tests PRIVATE PS2HOST_VULKAN=1)
endif()
# The IOP interpreter's internals, for its instruction tests.
target_include_directories(ps2host_tests PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)

# ps2_add_port(<target>
#     GENERATED <dir with dispatch.cpp from engine/tools/port.py>
#     TRANSLATED <dir with the game's translated sources (src/lifted)>
#     SOURCES <the game's own sources: overrides, enhancements>
#     GAME <the game's directory, with game.json>)
# With GAME, the port started without arguments (from Explorer, say) plays the
# game from the player's own disc image: asked for the first time (and checked
# to be the disc game.json's executable comes from), remembered after. And the
# executable carries the game's icon (the window's and the taskbar's too): the
# port's own, port/icon.ico - or build/port/icon.ico where a player made one
# for their own build (from their disc: a game's port/tools/make_icon.py). The
# game's project needs the RC language for it.
function(ps2_add_port target)
    cmake_parse_arguments(PORT "" "GENERATED;TRANSLATED;GAME" "SOURCES" ${ARGN})
    file(GLOB translated ${PORT_TRANSLATED}/*.cpp)
    # dispatch.cpp, and discovered.cpp: code found copied at run time, translated.
    file(GLOB generated ${PORT_GENERATED}/*.cpp)
    if(ANDROID)
        # Android: the game as a library SDL's activity loads (lib<target>.so,
        # its SDL_main main.cpp's main), in an APK (engine/tools/android.py).
        add_library(${target} SHARED
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src/main.cpp
            ${generated}
            ${translated}
            ${PORT_SOURCES})
    else()
    add_executable(${target}
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src/main.cpp
        ${generated}
        ${translated}
        ${PORT_SOURCES})
    endif()
    if(MSVC)
        target_link_libraries(${target} PRIVATE ps2_cpu_check)
    endif()
    target_link_libraries(${target} PRIVATE ps2host)
    if(PORT_GAME)
        get_filename_component(game_dir ${PORT_GAME} ABSOLUTE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${game_dir}/game.json)
        file(READ ${game_dir}/game.json game_json)
        string(JSON game_name GET ${game_json} name)
        string(JSON game_executable GET ${game_json} executable)
        string(JSON game_title ERROR_VARIABLE no_title GET ${game_json} title)
        if(no_title)
            set(game_title ${game_name})
        endif()
        string(JSON game_version ERROR_VARIABLE no_version GET ${game_json} version)
        if(no_version)
            set(game_version "0.0.0")
        endif()
        target_compile_definitions(${target} PRIVATE PS2HOST_GAME_NAME="${game_name}"
                                                     PS2HOST_GAME_TITLE="${game_title}"
                                                     PS2HOST_GAME_VERSION="${game_version}"
                                                     PS2HOST_GAME_EXECUTABLE="${game_executable}"
                                                     PS2HOST_GAME_TARGET="${target}")
        if(WIN32)
            target_link_libraries(${target} PRIVATE comdlg32)
        endif()
        set(icon ${game_dir}/build/port/icon.ico)
        if(NOT EXISTS ${icon})
            set(icon ${game_dir}/port/icon.ico)
        endif()
        if(WIN32 AND EXISTS ${icon} AND CMAKE_RC_COMPILER)
            # The icon, and the version Explorer shows (game.json's "version",
            # its numbers for the fixed fields: 0.1.0-alpha is 0,1,0,0).
            string(REGEX MATCHALL "[0-9]+" version_numbers "${game_version}")
            list(APPEND version_numbers 0 0 0 0)
            list(SUBLIST version_numbers 0 4 version_numbers)
            string(REPLACE ";" "," version_fixed "${version_numbers}")
            file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${target}_icon.rc
                 CONTENT "1 ICON \"${icon}\"
1 VERSIONINFO
FILEVERSION ${version_fixed}
PRODUCTVERSION ${version_fixed}
BEGIN
  BLOCK \"StringFileInfo\"
  BEGIN
    BLOCK \"040904B0\"
    BEGIN
      VALUE \"FileDescription\", \"${game_title} - PC port\"
      VALUE \"ProductName\", \"${game_title} - PC port\"
      VALUE \"FileVersion\", \"${game_version}\"
      VALUE \"ProductVersion\", \"${game_version}\"
      VALUE \"OriginalFilename\", \"${target}.exe\"
    END
  END
  BLOCK \"VarFileInfo\"
  BEGIN
    VALUE \"Translation\", 0x409, 1200
  END
END
")
            target_sources(${target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${target}_icon.rc)
            set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${target}_icon.rc PROPERTIES
                                        OBJECT_DEPENDS ${icon})
        endif()
        if(MSVC)
            # A windowed program: started from Explorer, no console beside the
            # game (main.cpp attaches to a terminal's when there is one).
            target_link_options(${target} PRIVATE /SUBSYSTEM:WINDOWS /ENTRY:Ps2CheckedStart)
        endif()
    endif()
    if(MSVC)
        # Symbols in the optimised build too, for --sample (profile.cpp): /Z7
        # keeps them in the objects (no PDB contention in parallel builds),
        # /DEBUG gathers them; /OPT keeps what it does without symbols.
        target_compile_options(${target} PRIVATE /Z7)
        # The symbols' file named by its name alone in the executable, not the
        # path it was built at (a release carries no one's folders).
        target_link_options(${target} PRIVATE /DEBUG /OPT:REF /OPT:ICF "/PDBALTPATH:%_PDB%")
    endif()
endfunction()
