/*
 * The GS renderer on Direct3D 11 (BACKENDS_ROADMAP.md, X3).
 *
 * Until it draws, a D3D11 display shows the software GS's pictures: the host
 * falls back to it when no GPU renderer can be made.
 */

#include "gpu_internal.h"

namespace ps2gpu {

std::unique_ptr<GsRenderer> CreateD3D11GsRenderer(Gpu& gpu, std::string* error) {
    (void)gpu;
    if (error) *error = "no Direct3D 11 GS renderer yet (the software GS draws)";
    return nullptr;
}

}  // namespace ps2gpu
