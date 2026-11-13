/*
 * The GPU GS renderer's tests: draw into a target, read it back, check pixels
 * as the software GS (engine/host/src/gs_draw.cpp) would have them.
 */

#include <cstdio>
#include <cstdlib>

#include "ps2gpu/gs_renderer.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }
}

std::uint32_t Rgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a) {
    return std::uint32_t(r) | std::uint32_t(g) << 8 | std::uint32_t(b) << 16 | std::uint32_t(a) << 24;
}

/* Each channel within `slack` of the expected value: blending on the GPU
   rounds where the GS truncates. */
bool Near(std::uint32_t got, std::uint32_t want, int slack = 1) {
    for (int shift = 0; shift < 32; shift += 8) {
        const int g = int((got >> shift) & 0xFF), w = int((want >> shift) & 0xFF);
        if (g - w > slack || w - g > slack) return false;
    }
    return true;
}

ps2gpu::GsDrawState Target(std::uint32_t fbp) {
    ps2gpu::GsDrawState state;
    state.fbp = fbp;
    state.fbw = 1;
    state.width = 64;
    state.height = 64;
    state.scissor_x1 = 63;
    state.scissor_y1 = 63;
    return state;
}

/* A rectangle as two triangles, texture coordinates 0..1 across it. */
void Rect(ps2gpu::GsRenderer& r, const ps2gpu::GsDrawState& state, float x0, float y0, float x1, float y1,
          std::uint32_t rgba) {
    const std::uint8_t cr = rgba & 0xFF, cg = (rgba >> 8) & 0xFF, cb = (rgba >> 16) & 0xFF, ca = rgba >> 24;
    const ps2gpu::GsVertex v[6] = {
        {x0, y0, 0, cr, cg, cb, ca, 0, 0, 1}, {x1, y0, 0, cr, cg, cb, ca, 1, 0, 1},
        {x0, y1, 0, cr, cg, cb, ca, 0, 1, 1}, {x1, y0, 0, cr, cg, cb, ca, 1, 0, 1},
        {x1, y1, 0, cr, cg, cb, ca, 1, 1, 1}, {x0, y1, 0, cr, cg, cb, ca, 0, 1, 1}};
    r.DrawTriangles(state, v, 6);
}

std::uint32_t Pixel8(ps2gpu::GsRenderer& r, std::uint32_t fbp, int x, int y) {
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    if (!r.ReadTarget(fbp, 8, 0, &pixels, &w, &h) || pixels.size() != std::size_t(w) * h) return 0xDEADBEEF;
    return pixels[std::size_t(y) * w + x];
}

std::uint32_t Pixel(ps2gpu::GsRenderer& r, std::uint32_t fbp, int x, int y) {
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    if (!r.ReadTarget(fbp, 1, 0, &pixels, &w, &h) || pixels.size() != std::size_t(w) * h) return 0xDEADBEEF;
    return pixels[std::size_t(y) * w + x];
}

void Untextured(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x10);
    state.gouraud = false;
    // A flat right triangle covering the top-left half of 0..32.
    const ps2gpu::GsVertex flat[3] = {{0, 0, 0, 0, 0, 0, 0x80, 0, 0, 1},
                                      {32, 0, 0, 0, 0, 0, 0x80, 0, 0, 1},
                                      {0, 32, 0, 0x40, 0x80, 0xFF, 0x80, 0, 0, 1}};
    renderer.DrawTriangles(state, flat, 3);
    // Scissored away entirely.
    ps2gpu::GsDrawState clipped = state;
    clipped.scissor_x0 = 60;
    clipped.scissor_y0 = 60;
    const ps2gpu::GsVertex hidden[3] = {{0, 40, 0, 0xFF, 0, 0, 0x80, 0, 0, 1},
                                        {20, 40, 0, 0xFF, 0, 0, 0x80, 0, 0, 1},
                                        {0, 60, 0, 0xFF, 0, 0, 0x80, 0, 0, 1}};
    renderer.DrawTriangles(clipped, hidden, 3);

    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    Check(renderer.ReadTarget(0x10, 1, 0, &pixels, &w, &h) && w == 64 && h == 64, "the target reads back");
    if (pixels.size() == 64 * 64) {
        Check(pixels[5 * 64 + 5] == Rgba(0x40, 0x80, 0xFF, 0x80), "flat shading takes the last vertex's colour");
        Check(pixels[40 * 64 + 40] == 0, "outside the triangle stays as local memory was (zero)");
        Check(pixels[45 * 64 + 5] == 0, "a scissored triangle draws nothing");
        Check(pixels[0] == Rgba(0x40, 0x80, 0xFF, 0x80), "the GS samples pixel (0,0) at its corner");
    }
}

void Textured(ps2gpu::GsRenderer& renderer) {
    // 2x2: red, green / blue, white at half alpha.
    const std::uint32_t texels[4] = {Rgba(0xFF, 0, 0, 0x80), Rgba(0, 0xFF, 0, 0x80), Rgba(0, 0, 0xFF, 0x80),
                                     Rgba(0xFF, 0xFF, 0xFF, 0x40)};
    renderer.UploadTexture(7, texels, 2, 2);
    ps2gpu::GsDrawState state = Target(0x20);
    state.textured = true;
    state.texture = 7;
    state.wms = state.wmt = 1;                     // CLAMP

    Rect(renderer, state, 0, 0, 32, 32, Rgba(0x80, 0x80, 0x80, 0x80));
    Check(Pixel(renderer, 0x20, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "MODULATE by 0x80 is the texel");
    Check(Pixel(renderer, 0x20, 20, 4) == Rgba(0, 0xFF, 0, 0x80), "the texel to the right");
    Check(Pixel(renderer, 0x20, 20, 20) == Rgba(0xFF, 0xFF, 0xFF, 0x40), "the texel below and right");

    Rect(renderer, state, 32, 0, 64, 32, Rgba(0x40, 0x40, 0x40, 0x40));
    Check(Pixel(renderer, 0x20, 36, 4) == Rgba(0x7F, 0, 0, 0x40), "MODULATE by 0x40 halves, truncating");

    ps2gpu::GsDrawState decal = state;
    decal.tfx = ps2gpu::TextureFunction::Decal;
    Rect(renderer, decal, 0, 32, 32, 64, Rgba(0x10, 0x10, 0x10, 0x10));
    Check(Pixel(renderer, 0x20, 4, 36) == Rgba(0xFF, 0, 0, 0x80), "DECAL ignores the vertex colour");

    ps2gpu::GsDrawState no_tcc = decal;
    no_tcc.tcc = false;
    Rect(renderer, no_tcc, 32, 32, 64, 64, Rgba(0x10, 0x10, 0x10, 0x22));
    Check(Pixel(renderer, 0x20, 36, 36) == Rgba(0xFF, 0, 0, 0x22), "without TCC the alpha is the vertex's");
}

void Blended(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x30);
    Rect(renderer, state, 0, 0, 64, 64, Rgba(0xFF, 0xFF, 0xFF, 0x80));

    // (Cs - Cd) * As + Cd: the usual transparency, As 0x40 is a quarter... of
    // 0x80 = 1.0, so half.
    ps2gpu::GsDrawState over = state;
    over.blend = true;
    over.a = ps2gpu::BlendColour::Source;
    over.b = ps2gpu::BlendColour::Dest;
    over.c = ps2gpu::BlendAlpha::Source;
    over.d = ps2gpu::BlendColour::Dest;
    Rect(renderer, over, 0, 0, 32, 32, Rgba(0, 0, 0xFF, 0x40));
    Check(Near(Pixel(renderer, 0x30, 4, 4), Rgba(0x80, 0x80, 0xFF, 0x40)),
          "(Cs - Cd) * As + Cd blends half way at As 0x40, and writes As");

    // (Cs - 0) * FIX + Cd: additive.
    Rect(renderer, state, 32, 0, 64, 32, Rgba(10, 20, 30, 0x80));
    ps2gpu::GsDrawState add = state;
    add.blend = true;
    add.a = ps2gpu::BlendColour::Source;
    add.b = ps2gpu::BlendColour::Zero;
    add.c = ps2gpu::BlendAlpha::Fix;
    add.d = ps2gpu::BlendColour::Dest;
    add.fix = 0x80;
    Rect(renderer, add, 32, 0, 64, 32, Rgba(5, 6, 7, 0x80));
    Check(Near(Pixel(renderer, 0x30, 36, 4), Rgba(15, 26, 37, 0x80)), "(Cs - 0) * FIX + Cd adds");

    // (Cd - Cs) * FIX + 0 with FIX 0x80: subtracts, clamped at zero.
    Rect(renderer, state, 0, 32, 32, 64, Rgba(100, 100, 100, 0x80));
    ps2gpu::GsDrawState sub = add;
    sub.a = ps2gpu::BlendColour::Dest;
    sub.b = ps2gpu::BlendColour::Source;
    sub.d = ps2gpu::BlendColour::Zero;
    Rect(renderer, sub, 0, 32, 32, 64, Rgba(30, 120, 0, 0x80));
    Check(Near(Pixel(renderer, 0x30, 4, 36), Rgba(70, 0, 100, 0x80)), "(Cd - Cs) * FIX subtracts, clamped");
}

void AlphaTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x40);
    state.alpha_test = true;
    state.atst = 5;                                 // GEQUAL
    state.aref = 0x80;
    Rect(renderer, state, 0, 0, 32, 32, Rgba(0xFF, 0, 0, 0x40));
    Rect(renderer, state, 32, 0, 64, 32, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x40, 4, 4) == 0, "alpha 0x40 fails GEQUAL 0x80: nothing is written");
    Check(Pixel(renderer, 0x40, 36, 4) == Rgba(0, 0xFF, 0, 0x80), "alpha 0x80 passes GEQUAL 0x80");

    // NEVER with AFAIL FB_ONLY: the frame buffer is written all the same
    // (Burnout 3 copies its frame this way, leaving depth alone).
    ps2gpu::GsDrawState never = state;
    never.atst = 0;
    never.afail = 1;
    Rect(renderer, never, 0, 32, 32, 64, Rgba(0, 0, 0xFF, 0x80));
    Check(Pixel(renderer, 0x40, 4, 36) == Rgba(0, 0, 0xFF, 0x80), "a failed test with AFAIL FB_ONLY writes colour");
}

/* A rectangle at one depth. */
void DepthRect(ps2gpu::GsRenderer& r, const ps2gpu::GsDrawState& state, float x0, float y0, float x1, float y1,
               std::uint32_t z, std::uint32_t rgba) {
    const std::uint8_t cr = rgba & 0xFF, cg = (rgba >> 8) & 0xFF, cb = (rgba >> 16) & 0xFF, ca = rgba >> 24;
    const ps2gpu::GsVertex v[6] = {
        {x0, y0, z, cr, cg, cb, ca, 0, 0, 1}, {x1, y0, z, cr, cg, cb, ca, 1, 0, 1},
        {x0, y1, z, cr, cg, cb, ca, 0, 1, 1}, {x1, y0, z, cr, cg, cb, ca, 1, 0, 1},
        {x1, y1, z, cr, cg, cb, ca, 1, 1, 1}, {x0, y1, z, cr, cg, cb, ca, 0, 1, 1}};
    r.DrawTriangles(state, v, 6);
}

void DepthTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x50);
    state.depth = true;
    state.zbp = 0x60;
    state.ztst = 2;                                 // GEQUAL: the larger Z wins
    // Near then far: the far one is hidden.
    DepthRect(renderer, state, 0, 0, 32, 32, 0x200000, Rgba(0xFF, 0, 0, 0x80));
    DepthRect(renderer, state, 0, 0, 32, 32, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x50, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "GEQUAL keeps the nearer (larger) Z");
    // Far then near: the near one covers it.
    DepthRect(renderer, state, 32, 0, 64, 32, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    DepthRect(renderer, state, 32, 0, 64, 32, 0x200000, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x50, 36, 4) == Rgba(0xFF, 0, 0, 0x80), "a nearer Z draws over a farther one");
    // Equal Z: GEQUAL passes, GREATER does not.
    DepthRect(renderer, state, 0, 32, 32, 64, 0x300000, Rgba(0, 0, 0xFF, 0x80));
    DepthRect(renderer, state, 0, 32, 32, 64, 0x300000, Rgba(0xFF, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x50, 4, 36) == Rgba(0xFF, 0xFF, 0, 0x80), "GEQUAL passes an equal Z");
    ps2gpu::GsDrawState greater = state;
    greater.ztst = 3;
    DepthRect(renderer, greater, 0, 32, 32, 64, 0x300000, Rgba(0, 0xFF, 0xFF, 0x80));
    Check(Pixel(renderer, 0x50, 4, 36) == Rgba(0xFF, 0xFF, 0, 0x80), "GREATER fails an equal Z");
    // ZMSK: tested, not written.
    ps2gpu::GsDrawState masked = state;
    masked.depth_write = false;
    DepthRect(renderer, masked, 32, 32, 64, 64, 0x500000, Rgba(0xFF, 0, 0xFF, 0x80));
    DepthRect(renderer, state, 32, 32, 64, 64, 0x400000, Rgba(0, 0, 0xFF, 0x80));
    Check(Pixel(renderer, 0x50, 36, 36) == Rgba(0, 0, 0xFF, 0x80), "with ZMSK the Z is not written");
    // What local memory held: uploaded depth hides a farther draw.
    std::vector<std::uint32_t> z(64 * 64, 0x800000);
    renderer.UploadDepth(0x61, 64, 64, z.data());
    ps2gpu::GsDrawState uploaded = state;
    uploaded.fbp = 0x51;
    uploaded.zbp = 0x61;
    DepthRect(renderer, uploaded, 0, 0, 64, 64, 0x700000, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x51, 10, 10) == 0, "an uploaded depth buffer hides what is behind it");
    // Read back: what was uploaded, and what a draw wrote, as the GS's Z.
    DepthRect(renderer, uploaded, 0, 0, 16, 16, 0x900000, Rgba(0xFF, 0, 0, 0x80));
    std::vector<std::uint32_t> back;
    Check(renderer.ReadDepth(0x61, 64, 64, &back) && back.size() == 64 * 64, "a depth buffer reads back");
    Check(back[40 * 64 + 40] == 0x800000, "...uploaded Z as it was");
    Check(back[4 * 64 + 4] == 0x900000, "...drawn Z as the GS's Z");
    Check(!renderer.ReadDepth(0x62, 64, 64, &back), "...and a buffer never made does not");
}

void AlphaFailWrites(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7A);
    state.depth = true;
    state.zbp = 0x7B;
    state.ztst = 2;                                 // GEQUAL
    ps2gpu::GsDrawState failing = state;
    failing.alpha_test = true;
    failing.atst = 5;                               // GEQUAL 0x80: alpha 0x40 fails
    failing.aref = 0x80;
    // FB_ONLY: the colour is written, the depth is not - a farther draw still lands.
    failing.afail = 1;
    DepthRect(renderer, failing, 0, 0, 16, 16, 0x500000, Rgba(0xFF, 0, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 4, 4) == Rgba(0xFF, 0, 0, 0x40), "FB_ONLY: a failing pixel's colour is written");
    DepthRect(renderer, state, 0, 0, 16, 16, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7A, 4, 4) == Rgba(0, 0xFF, 0, 0x80), "FB_ONLY: ...and its depth is not");
    // ZB_ONLY: the depth is written, the colour is not.
    failing.afail = 2;
    DepthRect(renderer, failing, 16, 0, 32, 16, 0x500000, Rgba(0xFF, 0, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 20, 4) == 0, "ZB_ONLY: a failing pixel's colour is not written");
    DepthRect(renderer, state, 16, 0, 32, 16, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7A, 20, 4) == 0, "ZB_ONLY: ...but its depth is, and hides a farther draw");
    // RGB_ONLY: colour without alpha (and no depth).
    DepthRect(renderer, state, 32, 0, 48, 16, 0x100000, Rgba(0, 0, 0xFF, 0x22));
    failing.afail = 3;
    DepthRect(renderer, failing, 32, 0, 48, 16, 0x500000, Rgba(0xFF, 0xFF, 0, 0x40));
    Check(Pixel(renderer, 0x7A, 36, 4) == Rgba(0xFF, 0xFF, 0, 0x22), "RGB_ONLY: colour written, alpha kept");
    DepthRect(renderer, state, 32, 0, 48, 16, 0x200000, Rgba(0, 0xFF, 0xFF, 0x80));
    Check(Pixel(renderer, 0x7A, 36, 4) == Rgba(0, 0xFF, 0xFF, 0x80), "RGB_ONLY: ...and no depth");
}

void DestinationAlphaTested(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7C);
    Rect(renderer, state, 0, 0, 32, 64, Rgba(0x10, 0x10, 0x10, 0x80));    // alpha bit 1
    Rect(renderer, state, 32, 0, 64, 64, Rgba(0x20, 0x20, 0x20, 0x7F));   // alpha bit 0
    ps2gpu::GsDrawState date = state;
    date.date = true;
    date.datm = 1;
    Rect(renderer, date, 0, 0, 64, 32, Rgba(0xFF, 0, 0, 0x80));
    Check(Pixel(renderer, 0x7C, 4, 4) == Rgba(0xFF, 0, 0, 0x80), "DATM 1 draws where the alpha bit is 1");
    Check(Pixel(renderer, 0x7C, 36, 4) == Rgba(0x20, 0x20, 0x20, 0x7F), "...and not where it is 0");
    date.datm = 0;
    Rect(renderer, date, 0, 32, 64, 64, Rgba(0, 0xFF, 0, 0x80));
    Check(Pixel(renderer, 0x7C, 4, 36) == Rgba(0x10, 0x10, 0x10, 0x80), "DATM 0 does not draw where the bit is 1");
    Check(Pixel(renderer, 0x7C, 36, 36) == Rgba(0, 0xFF, 0, 0x80), "...and draws where it is 0");
}

void Addressed(ps2gpu::GsRenderer& renderer) {
    // 4x1: texels 0..3 with red 10, 20, 30, 40.
    const std::uint32_t texels[4] = {Rgba(10, 0, 0, 0x80), Rgba(20, 0, 0, 0x80), Rgba(30, 0, 0, 0x80),
                                     Rgba(40, 0, 0, 0x80)};
    renderer.UploadTexture(9, texels, 4, 1);
    ps2gpu::GsDrawState state = Target(0x7F);
    state.textured = true;
    state.texture = 9;
    state.tfx = ps2gpu::TextureFunction::Decal;
    // REGION_CLAMP to texels 1..2: the left quarter of the rectangle clamps up to 1.
    state.wms = 2;
    state.minu = 1;
    state.maxu = 2;
    Rect(renderer, state, 0, 0, 64, 8, 0);
    Check(Pixel(renderer, 0x7F, 4, 4) == Rgba(20, 0, 0, 0x80), "REGION_CLAMP raises texel 0 to MINU");
    Check(Pixel(renderer, 0x7F, 60, 4) == Rgba(30, 0, 0, 0x80), "...and lowers texel 3 to MAXU");
    // REGION_REPEAT: (u & MINU) | MAXU - MINU 1, MAXU 2 maps 0, 1, 2, 3 to 2, 3, 2, 3.
    state.wms = 3;
    Rect(renderer, state, 0, 8, 64, 16, 0);
    Check(Pixel(renderer, 0x7F, 4, 12) == Rgba(30, 0, 0, 0x80), "REGION_REPEAT: texel 0 reads texel 2");
    Check(Pixel(renderer, 0x7F, 20, 12) == Rgba(40, 0, 0, 0x80), "REGION_REPEAT: texel 1 reads texel 3");
    // Bilinear, clamped: between texels 1 and 2 at u = 2.0 the mix is half way, 25.
    state.wms = 1;
    state.bilinear = true;
    Rect(renderer, state, 0, 16, 64, 24, 0);
    Check(Pixel(renderer, 0x7F, 32, 20) == Rgba(25, 0, 0, 0x80), "bilinear mixes the two texels around u - 0.5");
}

void RenderToTexture(ps2gpu::GsRenderer& renderer) {
    // Draw into target 0x90, then sample it into 0x91.
    ps2gpu::GsDrawState a = Target(0x90);
    Rect(renderer, a, 0, 0, 32, 64, Rgba(0x11, 0x22, 0x33, 0x80));
    Rect(renderer, a, 32, 0, 64, 64, Rgba(0x44, 0x55, 0x66, 0x10));
    ps2gpu::GsDrawState b = Target(0x91);
    b.textured = true;
    b.tfx = ps2gpu::TextureFunction::Decal;
    b.texture_target = true;
    b.tex_fbp = 0x90;
    b.tex_fbw = 1;
    b.tex_target_psm = 0;
    b.tw = b.th = 64;
    Rect(renderer, b, 0, 0, 64, 64, 0);
    Check(Pixel(renderer, 0x91, 4, 4) == Rgba(0x11, 0x22, 0x33, 0x80), "a target sampled as a texture");
    Check(Pixel(renderer, 0x91, 40, 4) == Rgba(0x44, 0x55, 0x66, 0x10), "...all of it");
    // Read as 16-bit texels: 5 bits a channel, alpha TA1 or TA0 by the alpha bit.
    ps2gpu::GsDrawState b16 = b;
    b16.tex_format = 2;
    b16.ta0 = 0x20;
    b16.ta1 = 0x70;
    Rect(renderer, b16, 0, 0, 64, 32, 0);
    Check(Pixel(renderer, 0x91, 4, 4) == Rgba(0x10, 0x20, 0x30, 0x70), "16-bit texels: 5 bits, TA1 for the bit");
    Check(Pixel(renderer, 0x91, 40, 4) == Rgba(0x40, 0x50, 0x60, 0x20), "...TA0 without it");
    // Sampling the target being drawn: a copy is taken first.
    ps2gpu::GsDrawState self = b;
    self.fbp = 0x90;
    self.texture_target = true;
    self.tex_fbp = 0x90;
    Rect(renderer, self, 0, 0, 32, 64, 0);                        // left half reads itself: unchanged
    Check(Pixel(renderer, 0x90, 4, 4) == Rgba(0x11, 0x22, 0x33, 0x80), "a target drawing itself reads a copy");
}

// FBMSK masking part of a channel keeps exactly those bits: Burnout 3's race
// writes two bits of each half of its frame's pixels through a 16-bit view.
void PartlyMaskedChannels(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x94);
    Rect(renderer, state, 0, 0, 16, 16, 0x12345678u);
    state.fbmsk = 0x0F0F0F0Fu;
    Rect(renderer, state, 0, 0, 8, 16, 0xFFFFFFFFu);
    Check(Pixel(renderer, 0x94, 2, 2) == 0xF2F4F6F8u, "a partly masked channel keeps its masked bits");
    Check(Pixel(renderer, 0x94, 12, 2) == 0x12345678u, "...and nothing outside the draw changes");
    // A 16-bit target: FBMSK in RGBA32 bit positions all the same, as the GS
    // masks the colour before making it 16-bit (KeptBits): 0x3FFF keeps R and
    // G's low six bits; G's top two, B and A are written.
    ps2gpu::GsDrawState sixteen = Target(0x95);
    sixteen.psm = 0x02;
    Rect(renderer, sixteen, 0, 0, 16, 16, 0x00102030u);        // A 0, B 0x10, G 0x20, R 0x30
    sixteen.fbmsk = 0x3FFF;
    Rect(renderer, sixteen, 0, 0, 16, 16, 0x80F8F8F8u);
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    renderer.ReadTarget(0x95, 1, 0x02, &pixels, &w, &h);
    Check(!pixels.empty() && pixels[0] == 0x80F8E030u, "a 16-bit target's FBMSK in RGBA32 bit positions");
}

void TargetFromMemory(ps2gpu::GsRenderer& renderer) {
    std::vector<std::uint32_t> pixels(64 * 64);
    for (std::uint32_t i = 0; i < pixels.size(); ++i) pixels[i] = Rgba(std::uint8_t(i), std::uint8_t(i >> 6), 7, 0x80);
    renderer.UploadTarget(0x92, 1, 0, pixels.data(), 64, 64);
    Check(Pixel(renderer, 0x92, 5, 3) == pixels[3 * 64 + 5], "a target loaded from memory reads back the same");
    ps2gpu::GsDrawState state = Target(0x92);
    Rect(renderer, state, 0, 0, 8, 8, Rgba(1, 2, 3, 4));
    Check(Pixel(renderer, 0x92, 2, 2) == Rgba(1, 2, 3, 4) && Pixel(renderer, 0x92, 20, 20) == pixels[20 * 64 + 20],
          "drawing over it keeps the rest");
}

// Memory patched on the GPU between two draws on one target (a CLUT's words,
// a primitive at a time): a compute dispatch with push constants of its own.
// The second draw's target size, pushed for the first, must still be there
// (Direct3D 11's layer once shared one set between compute and drawing:
// Burnout 3's eliminator).
void ComputeBetweenDraws(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x96);
    Rect(renderer, state, 0, 0, 8, 8, Rgba(0x80, 0, 0, 0x80));
    const std::uint32_t patch[3] = {0x3000 * 64, 0x12345678, 0xFFFFFFFF};    // (word, value, mask)
    renderer.MemoryPatch(patch, 1);
    Rect(renderer, state, 40, 40, 48, 48, Rgba(0, 0x80, 0, 0x80));
    Check(Pixel(renderer, 0x96, 4, 4) == Rgba(0x80, 0, 0, 0x80) && Pixel(renderer, 0x96, 44, 44) == Rgba(0, 0x80, 0, 0x80),
          "a draw after memory patched on the GPU lands where it should");
}

void PerPixelBlending(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x7E);
    Rect(renderer, state, 0, 0, 64, 64, Rgba(0xFF, 0xFF, 0xFF, 0x80));
    ps2gpu::GsDrawState pabe = state;
    pabe.blend = true;                              // (Cs - Cd) * FIX + Cd, FIX half
    pabe.c = ps2gpu::BlendAlpha::Fix;
    pabe.fix = 0x40;
    pabe.pabe = true;
    Rect(renderer, pabe, 0, 0, 32, 32, Rgba(0, 0, 0, 0x80));
    Rect(renderer, pabe, 32, 0, 64, 32, Rgba(0, 0, 0, 0x40));
    Check(Near(Pixel(renderer, 0x7E, 4, 4), Rgba(0x80, 0x80, 0x80, 0x80)), "PABE: As 0x80 blends");
    Check(Pixel(renderer, 0x7E, 36, 4) == Rgba(0, 0, 0, 0x40), "PABE: As below 0x80 does not");
}

void Masked(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x78);
    Rect(renderer, state, 0, 0, 32, 32, Rgba(0x10, 0x20, 0x30, 0x40));
    ps2gpu::GsDrawState masked = state;
    masked.fbmsk = 0xFF000000u;                     // alpha kept, as Burnout 3 draws
    Rect(renderer, masked, 0, 0, 32, 32, Rgba(0xA0, 0xB0, 0xC0, 0x80));
    Check(Pixel(renderer, 0x78, 4, 4) == Rgba(0xA0, 0xB0, 0xC0, 0x40), "FBMSK 0xFF000000 keeps the frame's alpha");
    masked.fbmsk = 0x0000FF00u;
    Rect(renderer, masked, 0, 0, 32, 32, Rgba(1, 2, 3, 4));
    Check(Pixel(renderer, 0x78, 4, 4) == Rgba(1, 0xB0, 3, 4), "FBMSK 0x0000FF00 keeps green");
}

void Fogged(ps2gpu::GsRenderer& renderer) {
    ps2gpu::GsDrawState state = Target(0x70);
    state.fog = true;
    state.fogcol = 0x0000FF;                        // red
    ps2gpu::GsVertex v[6];
    const float corners[6][2] = {{0, 0}, {32, 0}, {0, 32}, {32, 0}, {32, 32}, {0, 32}};
    for (int i = 0; i < 6; ++i) v[i] = {corners[i][0], corners[i][1], 0, 0xFF, 0xFF, 0xFF, 0x80, 0, 0, 1, 0x80};
    renderer.DrawTriangles(state, v, 6);
    Check(Pixel(renderer, 0x70, 4, 4) == Rgba(0xFE, 0x7F, 0x7F, 0x80),
          "fog F 0x80: (128 * C + 127 * FOGCOL) >> 8, as the software GS");
}

}  // namespace

/* A native mesh (gs_mesh.vert): a square of the room program's vertices, its
   header placing them as they are - drawn as DrawTriangles would draw the same
   square; and culled where all of a triangle is out past the guard band. */
void Meshes(ps2gpu::GsRenderer& renderer) {
    if (!renderer.DrawsMeshes()) {
        std::printf("meshes: not drawn by this renderer (skipped)\n");
        return;
    }
    // Four vertices, four quadwords each: place, normal, colour, texture coordinates.
    const float quads[4][16] = {
        {8, 8, 1, 0, 0, 0, 1, 0, 100, 50, 25, 128, 0, 0, 1, 0},
        {24, 8, 1, 0, 0, 0, 1, 0, 100, 50, 25, 128, 1, 0, 1, 0},
        {8, 24, 1, 0, 0, 0, 1, 0, 100, 50, 25, 128, 0, 1, 1, 0},
        {24, 24, 1, 0, 0, 0, 1, 0, 100, 50, 25, 128, 1, 1, 1, 0}};
    const std::uint32_t indices[6] = {0, 1, 2, 1, 3, 2};
    renderer.UploadMesh(7, quads, 16, indices, 6);
    float header[ps2gpu::GsRenderer::kMeshHeaderFloats] = {};
    auto row = [&](int r, float x, float y, float z, float w) {
        header[r * 4] = x, header[r * 4 + 1] = y, header[r * 4 + 2] = z, header[r * 4 + 3] = w;
    };
    row(0, 1, 1, 1, 0);             // the colour's own share
    row(1, 1, 1, 1, 1);             // ...times 1
    row(3, 1, 0, 0, 0), row(4, 0, 1, 0, 0), row(5, 0, 0, 1, 0), row(6, 0, 0, 0, 1);    // the view: as it is
    row(7, 0.01f, 0, 0, 0), row(8, 0, 0.01f, 0, 0), row(9, 0, 0, 0.01f, 0), row(10, 0, 0, 0, 1);    // a wide guard band
    row(15, 1, 0, 0, 0), row(16, 0, 1, 0, 0);
    row(24, 1, 0, 0, 0);            // vf11: the fog (none)
    row(28, 0, 0, 0, 0), row(29, 0, 0, 0, 0), row(30, 10, 0, 0, 0);     // vf15-vf17: the light, its strength 0
    ps2gpu::GsDrawState state = Target(0x1C0);
    renderer.DrawMesh(state, 7, header);
    Check(Pixel(renderer, 0x1C0, 16, 16) == Rgba(100, 50, 25, 128), "a mesh draws its vertices' colour");
    Check(Pixel(renderer, 0x1C0, 4, 4) == 0, "...and nothing outside it");
    Rect(renderer, state, 40, 8, 56, 24, Rgba(100, 50, 25, 128));
    Check(Pixel(renderer, 0x1C0, 8, 8) == Pixel(renderer, 0x1C0, 40, 8) &&
              Pixel(renderer, 0x1C0, 23, 23) == Pixel(renderer, 0x1C0, 55, 23) &&
              Pixel(renderer, 0x1C0, 24, 16) == Pixel(renderer, 0x1C0, 56, 16),
          "...covering the pixels the same square of triangles covers");
    // The guard band at the square's own size: every vertex out past x's plane (|x| > |w|) - culled.
    row(7, 1, 0, 0, 0);
    row(0, 0, 1, 0, 0);
    ps2gpu::GsDrawState other = Target(0x1C1);
    renderer.DrawMesh(other, 7, header);
    Check(Pixel(renderer, 0x1C1, 16, 16) == 0, "a mesh's triangle all out past one plane of the guard band is culled");
    renderer.ReleaseMesh(7);
    renderer.DrawMesh(other, 7, header);
}


// MSAA (SetGpuMsaa): a frame buffer drawn into through its multisampled twin - what it held kept (the twin
// filled from it), the draw resolved into it; and off again, the same.
void Multisampled(ps2gpu::GsRenderer& renderer) {
    ps2gpu::SetGpuMsaa(4);
    renderer.Kick();
    std::vector<std::uint32_t> pixels(512 * 448);
    for (std::uint32_t i = 0; i < pixels.size(); ++i) pixels[i] = Rgba(std::uint8_t(i), std::uint8_t(i >> 9), 9, 0x80);
    renderer.UploadTarget(0x1A0, 8, 0, pixels.data(), 512, 448);
    ps2gpu::GsDrawState state = Target(0x1A0);
    state.fbw = 8;
    state.width = 512;
    state.height = 448;
    state.scissor_x1 = 511;
    state.scissor_y1 = 447;
    Rect(renderer, state, 10, 10, 100, 100, Rgba(200, 100, 50, 0x80));
    const std::uint32_t inside = Pixel8(renderer, 0x1A0, 50, 50), outside = Pixel8(renderer, 0x1A0, 300, 300);
    std::printf("  msaa: inside %08X (want %08X), outside %08X (want %08X)\n", inside, Rgba(200, 100, 50, 0x80), outside,
                pixels[300 * 512 + 300]);
    Check(inside == Rgba(200, 100, 50, 0x80), "MSAA: a draw resolved into its target");
    Check(outside == pixels[300 * 512 + 300], "MSAA: what the target held kept");
    ps2gpu::SetGpuMsaa(1);
    renderer.Kick();
    // ...and as a game goes: memory into the target, a draw, the target back into memory - the same bytes with MSAA
    // as without (axis-aligned rectangles with their edges between the pixels' centres: no edge to smooth. Local
    // memory's 512 pages hold both targets and the texture).
    auto through_memory = [&](std::uint32_t samples, std::uint32_t fbp) {
        ps2gpu::SetGpuMsaa(samples);
        renderer.Kick();
        const std::uint32_t blocks = 512 * 448 * 4 / 256;
        std::vector<std::uint8_t> bytes(std::size_t(blocks) * 256);
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::uint8_t(i * 7 + (i >> 9));
        renderer.MemoryWrite(fbp * 32, fbp * 32 + blocks, bytes.data());
        renderer.TargetFromMemory(fbp, 8, 0, 512, 448);
        ps2gpu::GsDrawState st = Target(fbp);
        st.fbw = 8;
        st.width = 512;
        st.height = 448;
        st.scissor_x1 = 511;
        st.scissor_y1 = 447;
        st.fbmsk = 1;                   // (Outbreak's every draw: red's lowest bit kept)
        Rect(renderer, st, 9.5f, 9.5f, 99.5f, 99.5f, Rgba(200, 100, 50, 0x80));
        Rect(renderer, st, 199.5f, 199.5f, 299.5f, 259.5f, Rgba(10, 250, 50, 0x80));
        // a textured sprite, blended (a logo over its screen)
        const std::uint32_t texels[4] = {Rgba(0xFF, 0, 0, 0x80), Rgba(0, 0xFF, 0, 0x40), Rgba(0, 0, 0xFF, 0x80),
                                         Rgba(0xFF, 0xFF, 0xFF, 0x20)};
        renderer.UploadTexture(77, texels, 2, 2);
        ps2gpu::GsDrawState tex = st;
        tex.textured = true;
        tex.texture = 77;
        tex.wms = tex.wmt = 1;
        tex.sprite = true;
        tex.blend = true;
        Rect(renderer, tex, 319.5f, 39.5f, 479.5f, 199.5f, Rgba(0x80, 0x80, 0x80, 0x80));
        // a texture decoded from GPU memory (as a game's are), drawn twice - between, decoded again
        for (int k = 0; k < 2; ++k) {
            std::vector<std::uint8_t> texmem(64 * 64 * 4);
            for (std::size_t i = 0; i < texmem.size(); ++i) texmem[i] = std::uint8_t(i * 3 + k * 50);
            const std::uint32_t tbp = 0x3000;
            renderer.MemoryWrite(tbp, tbp + 64, texmem.data());
            ps2gpu::GsTextureDecode d;
            d.tbp = tbp;
            d.tbw = 1;
            d.psm = 0;
            d.tw = 64;
            d.th = 64;
            renderer.TextureFromMemory(900 + k, d);
            ps2gpu::GsDrawState dec = st;
            dec.textured = true;
            dec.texture = 900 + k;
            dec.wms = dec.wmt = 1;
            Rect(renderer, dec, 39.5f + 200 * float(k), 299.5f, 139.5f + 200 * float(k), 399.5f, Rgba(0x80, 0x80, 0x80, 0x80));
        }
        renderer.TargetToMemory(fbp, 8, 0, 512, 448);
        renderer.MemoryRead(fbp * 32, fbp * 32 + blocks, bytes.data());
        ps2gpu::SetGpuMsaa(1);
        renderer.Kick();
        return bytes;
    };
    const std::vector<std::uint8_t> plain = through_memory(1, 0x40), multi = through_memory(4, 0xC0);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < plain.size(); ++i) differ += plain[i] != multi[i];
    std::printf("  msaa through memory: %zu of %zu bytes differ\n", differ, plain.size());
    if (differ) {                       // (where: pages of 64x32 pixels, eight a row)
        std::vector<std::uint32_t> pages(plain.size() / 8192 + 1);
        for (std::size_t i = 0; i < plain.size(); ++i) pages[i / 8192] += plain[i] != multi[i];
        for (std::size_t g = 0; g < pages.size(); ++g)
            if (pages[g]) std::printf("    page at (%zu, %zu): %u bytes\n", g % 8 * 64, g / 8 * 32, pages[g]);
    }
    Check(differ == 0, "MSAA: through memory, the same as without");
}

// A target taller than a depth buffer (512 rows, whatever draws with it) with the destination alpha test and
// depth: the test's setup pass and the batch's own within the depth buffer's rows - over the target's the setup
// wrote past the depth buffer's memory (a device lost, the images beside it noise; the validation layer's
// VUID-VkRenderingInfo-pNext-06080). At 3x, as played; and its feedback copy and shader blend the same way.
void TallerThanDepth(ps2gpu::GsRenderer& renderer) {
    renderer.SetScale(3);
    ps2gpu::GsDrawState state = Target(0x200);
    state.fbw = 8;
    state.width = 512;
    state.height = 1024;
    state.scissor_x1 = 511;
    state.scissor_y1 = 1023;
    Rect(renderer, state, 0, 0, 256, 1024, Rgba(0x10, 0x10, 0x10, 0x80));     // alpha bit 1
    Rect(renderer, state, 256, 0, 512, 1024, Rgba(0x20, 0x20, 0x20, 0x7F));   // alpha bit 0
    ps2gpu::GsDrawState date = state;
    date.date = true;
    date.datm = 1;
    date.depth = true;
    date.zbp = 0x300;
    date.ztst = 1;
    DepthRect(renderer, date, 0, 0, 512, 1024, 0x100000, Rgba(0xFF, 0, 0, 0x80));
    std::vector<std::uint32_t> pixels;
    std::uint32_t w = 0, h = 0;
    Check(renderer.ReadTarget(0x200, 8, 0, &pixels, &w, &h) && w == 512 && h == 1024, "a tall target reads back");
    if (pixels.size() == 512 * 1024) {
        Check(pixels[100 * 512 + 40] == Rgba(0xFF, 0, 0, 0x80), "a tall target: DATM 1 draws where the alpha bit is 1");
        Check(pixels[100 * 512 + 300] == Rgba(0x20, 0x20, 0x20, 0x7F), "...and not where it is 0");
        Check(pixels[900 * 512 + 40] == Rgba(0x10, 0x10, 0x10, 0x80), "...and nothing below the depth buffer's rows");
    }
    // The same without the test, blended in the shader (its copy of the target) and MSAA's twins.
    ps2gpu::SetGpuMsaa(4);
    renderer.Kick();
    ps2gpu::GsDrawState deep = state;
    deep.depth = true;
    deep.zbp = 0x300;
    deep.ztst = 1;
    DepthRect(renderer, deep, 0, 0, 512, 1024, 0x100000, Rgba(0, 0xFF, 0, 0x80));
    deep.date = true;
    deep.datm = 1;
    DepthRect(renderer, deep, 0, 0, 512, 1024, 0x100000, Rgba(0, 0, 0xFF, 0x80));
    const std::uint32_t got = Pixel8(renderer, 0x200, 40, 100);
    if (got != Rgba(0, 0, 0xFF, 0x80)) std::printf("  tall, MSAA: %08X\n", got);
    Check(got == Rgba(0, 0, 0xFF, 0x80), "a tall target with MSAA: drawn within the depth buffer");
    ps2gpu::SetGpuMsaa(1);
    renderer.Kick();
    renderer.SetScale(1);
}

int main() {
    std::string error;
    // PS2HOST_RENDERER=d3d11: the same tests on Direct3D 11.
    const char* api = std::getenv("PS2HOST_RENDERER");
    auto gpu = api && std::string(api) == "d3d11" ? ps2gpu::CreateD3D11Gpu(&error) : ps2gpu::CreateGpu(&error);
    if (!gpu) {
        std::printf("no GPU: %s (skipped)\n", error.c_str());
        return 0;
    }
    auto renderer = ps2gpu::CreateGsRenderer(*gpu, &error);
    if (!renderer) {
        std::printf("FAIL: renderer: %s\n", error.c_str());
        return 1;
    }
    Untextured(*renderer);
    Textured(*renderer);
    Blended(*renderer);
    AlphaTested(*renderer);
    DepthTested(*renderer);
    Fogged(*renderer);
    Masked(*renderer);
    AlphaFailWrites(*renderer);
    DestinationAlphaTested(*renderer);
    PerPixelBlending(*renderer);
    Addressed(*renderer);
    RenderToTexture(*renderer);
    TargetFromMemory(*renderer);
    PartlyMaskedChannels(*renderer);
    ComputeBetweenDraws(*renderer);
    Meshes(*renderer);
    Multisampled(*renderer);
    TallerThanDepth(*renderer);
    if (g_failures) std::printf("gpu renderer tests: %d failed\n", g_failures);
    else std::printf("gpu renderer tests: all passed (%s)\n", gpu->Describe().c_str());
    return g_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
