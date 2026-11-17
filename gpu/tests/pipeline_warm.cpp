/* A first start's pipelines, timed: the renderer made on a folder's
   gpu_pipelines.txt (and gpu_pipeline_cache.bin, when there is one) as a
   player's start makes it, and how long its workers take to make them all.

       ps2gpu_warm <folder>

   A cold driver: the folder without gpu_pipeline_cache.bin, and the driver's
   own shader cache emptied or turned off. No window. */

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "ps2gpu/gs_renderer.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: ps2gpu_warm <folder>\n");
        return 2;
    }
    std::string error;
    auto gpu = ps2gpu::CreateGpu(&error);
    if (!gpu) {
        std::printf("no GPU: %s\n", error.c_str());
        return 1;
    }
    ps2gpu::SetPipelineCacheDir(argv[1]);
    const auto began = std::chrono::steady_clock::now();
    auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
    if (!renderer) {
        std::printf("renderer: %s\n", error.c_str());
        return 1;
    }
    const auto made_renderer = std::chrono::steady_clock::now();
    std::size_t made = 0, total = 0, said = ~std::size_t(0);
    while (ps2gpu::PipelineWarmProgress(&made, &total)) {
        if (made != said) {
            said = made;
            std::printf("  %zu of %zu after %.1f s\n", made, total,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto now = std::chrono::steady_clock::now();
    std::printf("renderer made in %.1f s; %zu pipelines in %.1f s in all (%s)\n",
                std::chrono::duration<double>(made_renderer - began).count(), total,
                std::chrono::duration<double>(now - began).count(), gpu->Describe().c_str());
    return 0;
}
