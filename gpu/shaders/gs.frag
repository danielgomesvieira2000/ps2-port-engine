#version 450
// The GS's per-pixel work, in its own terms (0-255, with 0x80 as 1.0 where it
// multiplies): the texture function, then the alpha test. Blending is the
// fixed-function blender's; the GS's factor - source alpha or FIX over 128 -
// goes out as the second output (dual-source blending), since the alpha
// written is the source alpha as it is. Compiled a second time with
// SINGLE_OUTPUT for devices without dual-source blending, and a third with
// SHADER_BLEND: the GS's blend done here, from a copy of the target - for a
// factor the blender cannot have (FIX over 0x80: more than 1.0, which a
// fixed-point target clamps to 1).
#ifdef SHADER_BLEND
#define SINGLE_OUTPUT
layout(set = 2, binding = 0) uniform sampler2D destination;
#else
// The scene's depth as the effects saw it (GsEffects' scene depth: the
// distance in r, the depth's mapping a / 1024 in g and b in b), for a
// game's soft particles (blend_ops bit 21).
layout(set = 2, binding = 0) uniform sampler2D scene_depth;
#endif

layout(location = 0) in vec4 shade;
layout(location = 1) noperspective in vec3 coordinate;
layout(location = 2) noperspective in float fog;
layout(location = 3) flat in vec4 range;   // a sprite's S, T at its first and last pixels (gs.vert)

layout(set = 0, binding = 0) uniform sampler2D texture_image;
// The same image's texels as 32-bit words (R in bits 0-7, A in 24-31): the
// four texels a bilinear sample mixes in one gather - one read of the
// texture rather than four (Sample).
layout(set = 0, binding = 1) uniform usampler2D texture_words;
// The GPU's CLUT (decode.comp's), 256 entries as the GS expands them: a target
// read as PSMT8H (tex_format 4) indexes it by its alpha byte.
layout(std430, set = 1, binding = 0) readonly buffer Clut { uint clut[256]; };

layout(push_constant) uniform Params {
    layout(offset = 16) uint textured;
    uint tfx;           // 0 MODULATE, 1 DECAL, 2 HIGHLIGHT, 3 HIGHLIGHT2
    uint tcc;           // the texel's alpha counts
    uint atst;          // 0 NEVER, 1 ALWAYS, 2 LESS, 3 LEQUAL, 4 EQUAL, 5 GEQUAL, 6 GREATER, 7 NOTEQUAL
    uint aref;
    uint fix_factor;    // blend with FIX rather than the source alpha
    uint fix;
    uint afail;         // which pixels this pass draws: 0 passing, 1 failing, 2 all
    uint fge;           // fog
    uint fogcol;        // FOGCOL, R G B in bytes 0-2
    uint pabe;          // which pixels this pass draws: 0 all, 1 As >= 0x80, 2 As < 0x80
    uint bilinear;      // bit 0: the four texels around, else the nearest; bits 1 and on: Mip
    uint wms, wmt;      // CLAMP: 0 REPEAT, 1 CLAMP, 2 REGION_CLAMP, 3 REGION_REPEAT
    uint minu, maxu, minv, maxv;
    uint tw, th;        // TEX0's size (0: the image's own)
    uint tex_format;    // a target read as 0 32-bit, 1 24-bit, 2 16-bit texels, 4 PSMT8H through the CLUT
    uint texa;          // TEXA: TA0 in bits 0-7, TA1 8-15, AEM 16
    uint image_size;    // the bound image's size in its pixels, width in bits 0-15, height 16-31 - not
                        // asked of the image (textureSize: a sampler's message per pixel on some GPUs)
    uint logic;         // FBMSK masking part of a channel: 1 the AND pass, 2 the OR pass, 0 neither
    uint keep;          // ...the bits FBMSK keeps, RGBA bytes as the target holds them
    uint scale;         // a target drawn larger (the internal resolution): its pixels to a GS texel
    uint blend_ops;     // bits 0-7, SHADER_BLEND: ALPHA's A, B, C, D, two bits each; 8-15: Snap's scale;
                        // 16: a 16-bit target (five bits a channel stored); 17-18: its blend made
                        // by the blender (gs_renderer.cpp's Blender16): 1 S written, 2 added, 3 taken away;
                        // 19 the paint's Fresnel term; 20 modern filtering, 22-24 log2 of its most taps;
                        // 21 a soft particle; 25-26 the player's filter (1 nearest, 2 bilinear,
                        // 3 trilinear or anisotropic: a replacement's too)
} p;

// What a draw may use, from its parameters (gs_renderer.cpp's Features): a
// pipeline made for a kind of draw leaves out the rest - the same results,
// the work and registers of what it never does gone (an integrated GPU's
// time at 2x was this shader's: the whole of it, everything a draw might
// do, for every pixel). Everything by default: the shader as it is.
layout(constant_id = 0) const uint kFeatures = 0xFFFFFFFFu;
const uint kTextured = 1u, kReplaced = 2u, kMip = 4u, kLinear = 8u, kExpand = 16u, kFine = 32u, kSnap = 64u,
           kFog = 128u, kAlphaTest = 256u, kPabe = 512u, kLogic = 1024u, kSixteen = 2048u;
bool Has(uint feature) { return (kFeatures & feature) != 0u; }
const float kEdge = 1.0 / 16384.0;       // Sample's nudge past a texel's edge

float Ta0() { return float(p.texa & 255u); }
float Ta1() { return float((p.texa >> 8) & 255u); }
bool Aem() { return ((p.texa >> 16) & 1u) != 0u; }
ivec2 ImageSize() { return ivec2(int(p.image_size & 0xFFFFu), int(p.image_size >> 16)); }

// A target drawn larger: its pixels to a GS texel (1 where none can be).
uint Scale() { return Has(kFine) ? p.scale : 1u; }

// A texel coordinate wrapped as CLAMP says, as the software GS's Wrap. A
// target drawn larger (Fine) is addressed in its own pixels: its size and a
// region's bounds scaled; REGION_REPEAT's masks do not scale (Texel takes
// those in GS texels).
int Wrap(int i, int size, uint mode, uint lo, uint hi, int k) {
    // (A power of two - TEX0's sizes are, and a target drawn at 2x or 4x -
    // wraps by its mask, the same for negative i: two integer divisions a
    // call otherwise, four calls a bilinear pixel.)
    if (mode == 0u) return k == 1 || (size & (size - 1)) == 0 ? (i & (size - 1)) : ((i % size) + size) % size;
    if (mode == 1u) return clamp(i, 0, size - 1);
    if (mode == 2u) return clamp(i, int(lo) * k, int(hi) * k + k - 1);
    return (i & int(lo)) | int(hi);
}

// Whether the texture is a target drawn larger, addressed in its own pixels.
bool Fine() { return Scale() > 1u && p.wms != 3u && p.wmt != 3u; }

ivec2 TextureSize() {
    return p.tw != 0u ? ivec2(int(p.tw), int(p.th)) : ImageSize();
}

// A texel as the GS expands it: TEXA's alpha for 24- and 16-bit texels, a
// target read as PSMT8H through the CLUT.
vec4 Expand(vec4 t) {
    if (!Has(kExpand)) return t;
    if (p.tex_format == 4u) {                       // PSMT8H: the alpha byte indexes the CLUT
        const uint c = clut[uint(t.a) & 255u];
        return vec4(float(c & 255u), float((c >> 8) & 255u), float((c >> 16) & 255u), float(c >> 24));
    }
    bool black = t.r + t.g + t.b == 0.0;
    if (p.tex_format == 1u) {                       // 24-bit: TEXA's alpha
        t.a = (Aem() && black) ? 0.0 : Ta0();
    } else if (p.tex_format == 2u) {                // 16-bit: 5 bits a channel, alpha by its bit
        bool bit = t.a >= 128.0;
        t.rgb = floor(t.rgb / 8.0) * 8.0;
        black = t.r + t.g + t.b == 0.0;
        t.a = bit ? Ta1() : ((Aem() && black) ? 0.0 : Ta0());
    }
    return t;
}

vec4 Fetch(ivec2 at) { return Expand(floor(texelFetch(texture_image, at, 0) * 255.0 + 0.5)); }
vec4 Unpack(uint w) { return Expand(vec4(float(w & 255u), float((w >> 8) & 255u), float((w >> 16) & 255u), float(w >> 24))); }

vec4 Texel(int u, int v) {
    const int k = Fine() ? int(Scale()) : 1;
    ivec2 size = TextureSize() * k;
    ivec2 at = ivec2(Wrap(u, size.x, p.wms, p.minu, p.maxu, k), Wrap(v, size.y, p.wmt, p.minv, p.maxv, k));
    if (k == 1) at &= 2047;
    // A target drawn larger with REGION_REPEAT: the GS texel, and the middle
    // of its square.
    if (k == 1 && Scale() > 1u) at = at * int(Scale()) + int(Scale() / 2u);
    // A target may be smaller than TEX0 says: what lies beyond is not drawn here.
    at = min(at, ImageSize() - 1);
    return Fetch(at);
}

// A texel of mipmap level n (n > 0: a texture from memory, never a target):
// TW-n by TH-n, below the levels before it at the image's left edge
// (gs_renderer.cpp's TextureFromMemory), a region's bounds shifted as its
// coordinates are.
vec4 Texel(int u, int v, int n) {
    if (n == 0) return Texel(u, v);
    const ivec2 size = max(TextureSize() >> n, ivec2(1));
    ivec2 at = ivec2(Wrap(u, size.x, p.wms, p.minu >> n, p.maxu >> n, 1),
                     Wrap(v, size.y, p.wmt, p.minv >> n, p.maxv >> n, 1));
    at = clamp(at, ivec2(0), size - 1);
    for (int i = 0; i < n; ++i) at.y += max(TextureSize().y >> i, 1);
    return Fetch(at);
}

// Texel's addresses for the four texels around (u0, v0) in level n, worked
// out together (x0, x1, y0, y1): a bilinear sample otherwise repeats the
// size, wrap and mip work four times. The same texels as Texel's.
ivec4 Neighbours(ivec2 i0, int n) {
    const int k = n == 0 && Fine() ? int(Scale()) : 1;
    const ivec2 size = n == 0 ? TextureSize() * k : max(TextureSize() >> n, ivec2(1));
    ivec2 a = ivec2(Wrap(i0.x, size.x, p.wms, p.minu >> n, p.maxu >> n, k),
                    Wrap(i0.y, size.y, p.wmt, p.minv >> n, p.maxv >> n, k));
    ivec2 b = ivec2(Wrap(i0.x + 1, size.x, p.wms, p.minu >> n, p.maxu >> n, k),
                    Wrap(i0.y + 1, size.y, p.wmt, p.minv >> n, p.maxv >> n, k));
    if (n == 0) {
        if (k == 1) { a &= 2047; b &= 2047; }
        if (k == 1 && Scale() > 1u) { a = a * int(Scale()) + int(Scale() / 2u); b = b * int(Scale()) + int(Scale() / 2u); }
        a = min(a, ImageSize() - 1);
        b = min(b, ImageSize() - 1);
    } else {
        a = clamp(a, ivec2(0), size - 1);
        b = clamp(b, ivec2(0), size - 1);
        int offset = 0;
        for (int i = 0; i < n; ++i) offset += max(TextureSize().y >> i, 1);
        a.y += offset;
        b.y += offset;
    }
    return ivec4(a.x, b.x, a.y, b.y);
}

// The texel at (u, v) in level 0's texels, in level n: the nearest, or the
// four around (u - 0.5, v - 0.5) mixed per channel as the GS mixes them: in
// sixteenths (its coordinates' 4 fractional bits), each step truncated - a
// game that filters its last frame back in relies on the loss (the software
// GS's Texel).
vec4 Sample(vec2 uv, int n, bool linear) {
    if (Fine()) uv *= float(Scale());               // in the target's own pixels
    uv /= float(1 << n);
    // A coordinate the GS has on a texel's edge (its fixed point: sprites'
    // texels one to a pixel) comes out of the GPU's interpolation a hair to
    // either side, as the GPU rounds: nudged on (kEdge: about a float's
    // last bit at a coordinate of 1024 - larger nudges took coordinates
    // genuinely just short of an edge over it too), every GPU takes the same
    // texel.
    if (!linear) return Texel(int(floor(uv.x + kEdge)), int(floor(uv.y + kEdge)), n);
    const ivec2 xy = ivec2(floor((uv - 0.5) * 16.0 + kEdge * 16.0));
    const ivec2 i0 = xy >> 4;
    const vec4 f = vec4(xy & 15, 0, 0);
    const ivec4 at = Neighbours(i0, n);
    vec4 c00, c10, c01, c11;
#ifdef D3D11
    // (Direct3D 11 views no RGBA8 image as 32-bit words: four fetches, the
    // same texels.)
    if (false) {
#else
    if (at.y == at.x + 1 && at.w == at.z + 1) {
#endif
        // Side by side in the image (almost always - not where a wrap or a
        // clamp parts them): gathered at the corner they share.
        const uvec4 w = textureGather(texture_words, vec2(at.yw) / vec2(ImageSize()), 0);
        c01 = Unpack(w.x);
        c11 = Unpack(w.y);
        c10 = Unpack(w.z);
        c00 = Unpack(w.w);
    } else {
        c00 = Fetch(at.xz);
        c10 = Fetch(at.yz);
        c01 = Fetch(at.xw);
        c11 = Fetch(at.yw);
    }
    const vec4 top = c00 + floor((c10 - c00) * f.x / 16.0), bottom = c01 + floor((c11 - c01) * f.x / 16.0);
    return top + floor((bottom - top) * f.y / 16.0);
}

// A texture pack's replacement (GsRenderer::UploadReplacement): finer than
// the GS's TW x TH, with mipmap levels of its own (levels - 1 in tex_format's
// bits 9-11), its texels RGBA as they are (alpha 0x80 is 1.0, as the GS's). Placed
// over the GS's texels: CLAMP's modes and region in them, k replacement
// texels to one; filtered as the game filters, with the level the pixel's
// footprint in the replacement's texels asks for (the GS's LOD knows only its
// own texels).
bool Replaced() { return Has(kReplaced) && (p.tex_format & 0x100u) != 0u; }

ivec2 ReplacedSize(int n) { return textureSize(texture_image, n); }

int ReplacedWrap(int i, int size, uint mode, uint lo, uint hi, float k) {
    if (mode == 0u) return ((i % size) + size) % size;
    if (mode == 1u) return clamp(i, 0, size - 1);
    if (mode == 2u) return clamp(i, int(floor(float(lo) * k)), int(ceil(float(hi + 1u) * k)) - 1);
    // REGION_REPEAT: the GS's texel masked, the replacement's texels within it kept.
    const int g = int(floor(float(i) / k));
    return int(float((g & int(lo)) | int(hi)) * k) + (i - int(float(g) * k));
}

vec4 ReplacedTexel(int u, int v, int n, vec2 k) {
    const ivec2 size = ReplacedSize(n);
    ivec2 at = ivec2(ReplacedWrap(u, size.x, p.wms, p.minu, p.maxu, k.x),
                     ReplacedWrap(v, size.y, p.wmt, p.minv, p.maxv, k.y));
    at = clamp(at, ivec2(0), size - 1);
    return floor(texelFetch(texture_image, at, n) * 255.0 + 0.5);
}

vec4 ReplacedLevel(vec2 r, int n, bool linear) {
    const vec2 k = vec2(ReplacedSize(n)) / vec2(TextureSize());      // its texels to a GS texel, at this level
    r /= float(1 << n);
    if (!linear) return ReplacedTexel(int(floor(r.x)), int(floor(r.y)), n, k);
    const vec2 xy = r - 0.5;
    const ivec2 i0 = ivec2(floor(xy));
    const vec2 f = xy - vec2(i0);
    const vec4 c00 = ReplacedTexel(i0.x, i0.y, n, k), c10 = ReplacedTexel(i0.x + 1, i0.y, n, k);
    const vec4 c01 = ReplacedTexel(i0.x, i0.y + 1, n, k), c11 = ReplacedTexel(i0.x + 1, i0.y + 1, n, k);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// `r`: where in level 0's texels; `dx`, `dy`: its change a pixel. As the
// game filters - or as the player chose (blend_ops 25-26): nearest, the
// nearest level's nearest texel; bilinear, the nearest level; trilinear, two
// levels mixed; anisotropic, up to 1 << blend_ops 22-24 taps along the
// footprint's long axis, as SampleModern takes them.
vec4 ReplacedSample(vec2 r, vec2 dx, vec2 dy) {
    const uint m = p.bilinear;
    const int last = int((p.tex_format >> 9) & 7u);
    const uint forced = (p.blend_ops >> 25) & 3u;
    const float lx = length(dx), ly = length(dy);
    // REPEAT or CLAMP: the GPU's sampler filters it (gs_renderer.cpp's
    // ReplacementSampler: linear in and between levels, the player's
    // anisotropy) - one sample, where the levels by hand were 4 to 8 texels
    // (and anisotropy's taps each); the region modes by hand still.
    const bool by_sampler = p.wms <= 1u && p.wmt <= 1u;
    const vec2 size0 = vec2(ReplacedSize(0));
    if (forced == 0u) {
        const bool linear = (m & 2u) == 0u ? (m & 1u) != 0u : ((m >> 8) & 1u) != 0u || ((m >> 5) & 7u) >= 4u;
        if (linear && by_sampler) return floor(textureGrad(texture_image, r / size0, dx / size0, dy / size0) * 255.0 + 0.5);
        const float lod = clamp(log2(max(max(lx, ly), 1e-6)), 0.0, float(last));
        const int n = int(floor(lod));
        const vec4 c0 = ReplacedLevel(r, n, linear);
        if (!linear || n >= last || lod == float(n)) return floor(c0 + 0.5);
        return floor(mix(c0, ReplacedLevel(r, n + 1, linear), lod - float(n)) + 0.5);
    }
    const float major = max(lx, ly), minor = max(min(lx, ly), 1e-6);
    const float most = forced == 3u ? float(1u << ((p.blend_ops >> 22) & 7u)) : 1.0;
    const float taps = clamp(ceil(major / minor), 1.0, most);
    const float bias = most > 1.0 ? 1.0 : 0.0;
    const float lod = clamp(log2(max(major / taps, 1e-6)) - bias, 0.0, float(last));
    if (forced < 3u) {
        const int n = int(min(floor(lod + 0.5), float(last)));
        if (forced == 2u && by_sampler) return floor(textureLod(texture_image, r / size0, float(n)) * 255.0 + 0.5);
        return floor(ReplacedLevel(r, n, forced == 2u) + 0.5);
    }
    if (by_sampler) return floor(textureGrad(texture_image, r / size0, dx / size0, dy / size0) * 255.0 + 0.5);
    const int n = int(floor(lod));
    const float f = lod - float(n);
    const vec2 axis = lx > ly ? dx : dy;
    vec4 sum = vec4(0.0);
    for (int i = 0; i < int(taps); ++i) {
        const vec2 at = r + axis * ((float(i) + 0.5) / taps - 0.5);
        vec4 c = ReplacedLevel(at, n, true);
        if (n < last && f > 0.0) c = mix(c, ReplacedLevel(at, n + 1, true), f);
        sum += c;
    }
    return floor(sum / taps + 0.5);
}

// TEX1's mipmapping (GsDrawState::mip, in p.bilinear): the GS's LOD from the
// pixel's Q - (log2(1/|Q|) << L) + K, or K alone (LCM) - magnifies (MMAG,
// level 0) up to 0, else minifies by MMIN: 2 and 4 the nearest level, 3 and 5
// the two around mixed; 4 and 5 bilinear within a level. The GS's own
// selection, not the GPU's: the levels it picks at any internal resolution
// (the software GS's ChooseLevel).
/* A mipmapped texture as a modern GPU filters it (blend_ops bit 20: a
   game's choice, the bridge's SetGpuModernTextures): the level from the
   footprint on screen - at an internal resolution the finer levels it can
   show, where the GS's depth-based level is the PS2's picture's - blended
   between levels (no bands where the level changes), and up to eight taps
   along the footprint's long axis (anisotropic: the road ahead sharp, not
   smeared). `dx`, `dy` the coordinate's change a pixel, in level 0's
   texels; `most` the taps at most (the player's anisotropy; 1: trilinear,
   the level the footprint's as it is - finer would shimmer, nothing to
   calm it). */
// Level n bilinear at uv (level 0's texels) for SampleModern, by the GPU's
// sampler (binding 0 linear for it: gs_renderer.cpp's Flush) - one fetch,
// where Sample's GS-exact sixteenths are a gather and their arithmetic: the
// difference was MODERN's cost (a 3x explosion 7.35 ms a picture against
// 6.8 as the game filters; 6.9 so). The level's texels only, never its
// neighbour's in the image: CLAMP held to its outer texels' middles, REPEAT
// across its edge two fetches there (each held so) mixed - small levels are
// mostly edge. By hand still for a region's modes, texels that are not the
// image's as they are (TEXA) and a target drawn larger: the same for the
// whole draw. Its weights the GS's sixteenths (the position held to them,
// and a 1024th on: the sampler's 8-bit weights then take exactly those) and
// its values truncated: a texel of alpha 0 beside an opaque one is 0 a
// sixteenth away, as Sample has it - finer weights gave such pixels an alpha
// of 1, past an alpha test that hides a whole polygon.
vec4 Bilinear(vec2 uv, int n) {
    if (p.wms > 1u || p.wmt > 1u || p.tex_format != 0u || Scale() != 1u || ImageSize().x < TextureSize().x)
        return Sample(uv, n, true);
    const vec2 size = vec2(max(TextureSize() >> n, ivec2(1)));
    vec2 q = floor((uv / float(1 << n) - 0.5) * 16.0 + kEdge * 16.0) / 16.0 + (0.5 + 1.0 / 1024.0);
    // Across REPEAT's edge: the last texel's middle (a) and the first's (b),
    // w of the way from a to b; within it, a alone.
    vec2 a, b, w = vec2(0.0);
    for (int i = 0; i < 2; ++i) {
        float x = q[i];
        if ((i == 0 ? p.wms : p.wmt) == 0u) {
            x -= floor(x / size[i]) * size[i];
            a[i] = x;
            b[i] = 0.5;
            if (x < 0.5) { a[i] = size[i] - 0.5; w[i] = floor((x + 0.5) * 16.0) / 16.0; }
            else if (x > size[i] - 0.5) { a[i] = size[i] - 0.5; w[i] = floor((x - (size[i] - 0.5)) * 16.0) / 16.0; }
        } else {
            a[i] = b[i] = clamp(x, 0.5, size[i] - 0.5);
        }
    }
    int offset = 0;
    for (int i = 0; i < n; ++i) offset += max(TextureSize().y >> i, 1);
    const vec2 at = vec2(0.0, float(offset)), image = vec2(ImageSize());
    vec4 c = textureLod(texture_image, (a + at) / image, 0.0);
    if (w.x > 0.0) c = mix(c, textureLod(texture_image, (vec2(b.x, a.y) + at) / image, 0.0), w.x);
    if (w.y > 0.0) {
        vec4 d = textureLod(texture_image, (vec2(a.x, b.y) + at) / image, 0.0);
        if (w.x > 0.0) d = mix(d, textureLod(texture_image, (b + at) / image, 0.0), w.x);
        c = mix(c, d, w.y);
    }
    return floor(c * 255.0 + 1.0 / 64.0);
}

vec4 SampleModern(vec2 uv, vec2 dx, vec2 dy, int mxl, float most) {
    const float lx = length(dx), ly = length(dy);
    const float major = max(lx, ly), minor = max(min(lx, ly), 1e-4);
    const float taps = clamp(ceil(major / minor), 1.0, most);
    const float bias = most > 1.0 ? 1.0 : 0.0;     // (a level finer: sharp, the taps keep it calm)
    const float lod = clamp(log2(max(major / taps, 1e-4)) - bias, 0.0, float(mxl));
    const int n = int(floor(lod));
    const float f = lod - float(n);
    const vec2 axis = lx > ly ? dx : dy;
    vec4 sum = vec4(0.0);
    for (int i = 0; i < int(taps); ++i) {
        const vec2 at = uv + axis * ((float(i) + 0.5) / taps - 0.5);
        vec4 c = Bilinear(at, n);
        if (n < mxl && f > 0.0) c = mix(c, Bilinear(at, n + 1), f);
        sum += c;
    }
    return floor(sum / taps + 0.5);
}

vec4 Sample(vec2 uv) {
    const uint m = p.bilinear;
    if (!Has(kMip) || (m & 2u) == 0u) return Sample(uv, 0, Has(kLinear) && (m & 1u) != 0u);
    const int mxl = int((m >> 2) & 7u);
    const uint mmin = (m >> 5) & 7u;
    float lod = float(int(m << 4) >> 20) / 16.0;    // K, 4 fractional bits
    const float q = coordinate.z != 0.0 ? abs(coordinate.z) : 1.0;     // as the software GS's
    if (((m >> 9) & 1u) == 0u) lod += -log2(q) * float(1u << ((m >> 10) & 3u));
    if (lod <= 0.0) return Sample(uv, 0, ((m >> 8) & 1u) != 0u);
    const bool linear = mmin == 4u || mmin == 5u;
    if (mmin == 2u || mmin == 4u) return Sample(uv, int(min(floor(lod + 0.5), float(mxl))), linear);
    lod = min(lod, float(mxl));
    const int n = int(floor(lod));
    const float f = lod - float(n);
    const vec4 c0 = Sample(uv, n, linear);
    if (n >= mxl || f == 0.0) return c0;
    const vec4 c1 = Sample(uv, n + 1, linear);
    return floor(c0 + (c1 - c0) * f + 0.5);
}

#ifdef UINT_OUTPUT
// (Direct3D 11.1's logic ops - FBMSK's partly masked channels - are for UINT
// targets alone: the colour written as its bytes, to a UINT view of the
// target.)
layout(location = 0) out uvec4 colour_bits;
vec4 colour;
#elif defined(SINGLE_OUTPUT)
layout(location = 0) out vec4 colour;
#else
layout(location = 0, index = 0) out vec4 colour;
layout(location = 0, index = 1) out vec4 factor;
#endif

void main() {
    vec4 v = round(shade * 255.0);
    vec4 c = v;
    if (Has(kTextured) && p.textured != 0u) {
        vec2 uv = coordinate.xy / coordinate.z * vec2(TextureSize());
        // A sprite on a target drawn larger: sampled as finely as the target
        // is drawn, within the coordinates the GS reads at its first and last
        // pixels (gs_renderer.cpp's DrawTriangles: further, where it meets
        // another), and - REPEAT, the sprite within one copy of the texture -
        // not wrapping round to its far side (a tile's edge: seams).
        const uint snap = (p.blend_ops >> 8) & 255u;
        if (Has(kSnap) && snap > 1u) {
            const vec2 size = vec2(TextureSize());
            const vec2 lo = min(range.xy, range.zw), hi = max(range.xy, range.zw);
            uv = clamp(uv, lo * size, hi * size);
            const float inset = p.bilinear != 0u ? 0.5 : 0.001;
            if (p.wms == 0u && lo.x >= 0.0 && hi.x <= 1.0) uv.x = clamp(uv.x, inset, size.x - inset);
            if (p.wmt == 0u && lo.y >= 0.0 && hi.y <= 1.0) uv.y = clamp(uv.y, inset, size.y - inset);
        }
        // The footprint in a replacement's texels, from the coordinates before
        // the sprite's clamp above (whose edges would read as a large one).
        vec4 t;
        if (Replaced()) {
            const vec2 k = vec2(ReplacedSize(0)) / vec2(TextureSize());
            const vec2 r = coordinate.xy / coordinate.z * vec2(TextureSize()) * k;
            const vec2 dx = dFdx(r), dy = dFdy(r);
            t = ReplacedSample(uv * k, dx, dy);
        } else if (Has(kMip) && ((p.blend_ops >> 20) & 1u) != 0u && (p.bilinear & 2u) != 0u) {
            const vec2 r = coordinate.xy / coordinate.z * vec2(TextureSize());
            t = SampleModern(uv, dFdx(r), dFdy(r), int((p.bilinear >> 2) & 7u),
                             float(1u << ((p.blend_ops >> 22) & 7u)));
        } else if (((p.blend_ops >> 19) & 1u) != 0u) {
            // A car's paint reflecting its sphere map (the Fresnel tag, below):
            // a map drawn each frame, no mipmaps - where it is minified, four
            // taps over the pixel's footprint, as a mip would: no shimmer on
            // a distant or a curved body.
            const vec2 dx = dFdx(uv), dy = dFdy(uv);
            if (max(length(dx), length(dy)) > 1.5) {
                const vec2 a = 0.25 * (dx + dy), b = 0.25 * (dx - dy);
                t = floor((Sample(uv + a) + Sample(uv + b) + Sample(uv - b) + Sample(uv - a)) * 0.25 + 0.5);
            } else {
                t = Sample(uv);
            }
        } else {
            t = Sample(uv);
        }
        vec3 product = floor(t.rgb * v.rgb / 128.0);
        if (p.tfx == 0u) {
            c.rgb = product;
            c.a = p.tcc != 0u ? floor(t.a * v.a / 128.0) : v.a;
        } else if (p.tfx == 1u) {
            c.rgb = t.rgb;
            c.a = p.tcc != 0u ? t.a : v.a;
        } else if (p.tfx == 2u) {
            c.rgb = product + v.a;
            c.a = p.tcc != 0u ? t.a + v.a : v.a;
        } else {
            c.rgb = product + v.a;
            c.a = p.tcc != 0u ? t.a : v.a;
        }
        c = min(c, vec4(255.0));
    }

    if (Has(kFog) && p.fge != 0u) {
        // Towards FOGCOL as F falls, colour only.
        float f = round(fog * 255.0);
        vec3 fc = vec3(float(p.fogcol & 0xFFu), float((p.fogcol >> 8) & 0xFFu), float((p.fogcol >> 16) & 0xFFu));
        c.rgb = floor((f * c.rgb + (255.0 - f) * fc) / 256.0);
    }

    // (Without kAlphaTest: the test ALWAYS, and no pass of failing pixels -
    // nothing discarded, as the GPU likes it: depth tested before shading.)
    if (Has(kAlphaTest)) {
        uint a = uint(c.a);
        bool pass = true;
        switch (p.atst) {
            case 0u: pass = false; break;
            case 2u: pass = a < p.aref; break;
            case 3u: pass = a <= p.aref; break;
            case 4u: pass = a == p.aref; break;
            case 5u: pass = a >= p.aref; break;
            case 6u: pass = a > p.aref; break;
            case 7u: pass = a != p.aref; break;
            default: break;
        }
        // The renderer draws AFAIL as passes: 0 keeps the passing pixels, 1 the
        // failing ones (with only what AFAIL writes), 2 all of them.
        if (p.afail == 0u && !pass) discard;
        if (p.afail == 1u && pass) discard;
    }
    if (Has(kPabe)) {
        if (p.pabe == 1u && c.a < 128.0) discard;
        if (p.pabe == 2u && c.a >= 128.0) discard;
    }

    if (((p.blend_ops >> 19) & 1u) != 0u) {
        // A car's paint taking its reflection from a sphere map (a game's
        // tag): the map's centre the view straight back, its rim grazing -
        // N.V = sqrt(1 - r^2) - and a Fresnel term (Schlick's, softened) on
        // the reflection: dimmer face on, brighter towards the body's edges,
        // as lacquer. (On the colour, the blend by the paint's alpha as it
        // was: scaling the share itself needs the target in the shader, a
        // copy of it for each of the thousands of triangles.) Not to the rim
        // itself: a warped map is least like the world there.
        const vec2 st = coordinate.xy / (coordinate.z != 0.0 ? coordinate.z : 1.0) * 2.0 - 1.0;
        const float r2 = min(dot(st, st), 0.85);
        const float facing = 1.0 - sqrt(1.0 - r2);
        const float fresnel = 0.04 + 0.96 * facing * facing * facing;
        // The brightest of what it reflects - the sun and the sky's brightest,
        // near the picture's white (128: the 3D is drawn at half intensity) -
        // over white, as a lacquer's highlight: the glow takes them up.
        const float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));
        const float glint = 1.0 + 1.1 * smoothstep(96.0, 128.0, lum);
        c.rgb = min(c.rgb * (0.7 + 3.0 * fresnel) * glint, vec3(255.0));
    }

#ifndef SHADER_BLEND
    if (((p.blend_ops >> 21) & 1u) != 0u) {
        // A soft particle (a game's tag): faded as it nears what is behind
        // it, so smoke and fire meet the road and the cars without a seam.
        const vec4 scene = texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0);
        const float z = gl_FragCoord.z * 4294967296.0;
        if (scene.a > 0.5 && z > scene.b + 0.5) {
            const float mine = scene.g * 1024.0 / (z - scene.b);
            const float soft = 0.35 + 0.025 * mine;
            c.a = floor(c.a * clamp((scene.r - mine) / soft, 0.0, 1.0) + 0.5);
        }
    }
#endif

#ifdef SHADER_BLEND
    // Cv = ((A - B) * C >> 7) + D, clamped (COLCLAMP), as the software GS.
    vec4 dst = round(texelFetch(destination, ivec2(gl_FragCoord.xy), 0) * 255.0);
    // (A 16-bit target's five bits: a sum the blender clamped at 255 is 248.)
    if (Has(kSixteen) && ((p.blend_ops >> 16) & 1u) != 0u) dst.rgb = floor(dst.rgb / 8.0) * 8.0;
    uint ops = p.blend_ops & 255u;
    vec3 op_a = (ops & 3u) == 0u ? c.rgb : (ops & 3u) == 1u ? dst.rgb : vec3(0.0);
    vec3 op_b = ((ops >> 2) & 3u) == 0u ? c.rgb : ((ops >> 2) & 3u) == 1u ? dst.rgb : vec3(0.0);
    float op_c = ((ops >> 4) & 3u) == 0u ? c.a : ((ops >> 4) & 3u) == 1u ? dst.a : float(p.fix);
    vec3 op_d = ((ops >> 6) & 3u) == 0u ? c.rgb : ((ops >> 6) & 3u) == 1u ? dst.rgb : vec3(0.0);
    c.rgb = clamp(floor((op_a - op_b) * op_c / 128.0) + op_d, 0.0, 255.0);
#else
    // A 16-bit target's blend the blender makes (the renderer's Blender16):
    // here S, the blend with the target as 0 wherever it is an operand, cut
    // to five bits a channel - up when it is taken away - for the blender to
    // add to the target, or take away, by a factor of 1 (or written, 1).
    // Not PABE's pass of the pixels left unblended.
    const uint way = Has(kSixteen) && p.pabe != 2u ? (p.blend_ops >> 17) & 3u : 0u;
    if (way != 0u) {
        uint ops = p.blend_ops & 255u;
        vec3 op_a = (ops & 3u) == 0u ? c.rgb : vec3(0.0);
        vec3 op_b = ((ops >> 2) & 3u) == 0u ? c.rgb : vec3(0.0);
        float op_c = ((ops >> 4) & 3u) == 0u ? c.a : float(p.fix);
        vec3 op_d = ((ops >> 6) & 3u) == 0u ? c.rgb : vec3(0.0);
        vec3 s = floor((op_a - op_b) * op_c / 128.0) + op_d;
        c.rgb = way == 1u ? clamp(s, 0.0, 255.0)
              : min(way == 3u ? ceil(-s / 8.0) * 8.0 : floor(s / 8.0) * 8.0, vec3(255.0));
    }
#endif
    // A 16-bit target keeps the top five bits of each colour byte.
    if (Has(kSixteen) && ((p.blend_ops >> 16) & 1u) != 0u) c.rgb = floor(c.rgb / 8.0) * 8.0;
    colour = c / 255.0;
#ifdef SHADER_BLEND
    // What FBMSK keeps, from the target (a channel masked in part, on a GPU
    // without logic ops: gs_renderer.cpp's MaskedInShader). Nothing where no
    // channel is: a channel kept whole isn't written, one written whole keeps
    // no bit.
    if (p.logic == 0u && p.keep != 0u)
        colour = unpackUnorm4x8((packUnorm4x8(dst / 255.0) & p.keep) | (packUnorm4x8(colour) & ~p.keep));
#endif
    // A partly masked channel: the target ANDed with what FBMSK keeps, then
    // ORed with the rest of the source's bits (the pipelines' logic ops).
    if (Has(kLogic)) {
        if (p.logic == 1u) colour = unpackUnorm4x8(p.keep);
        if (p.logic == 2u) colour = unpackUnorm4x8(packUnorm4x8(c / 255.0) & ~p.keep);
    }
#ifdef UINT_OUTPUT
    colour_bits = uvec4(round(colour * 255.0));
#endif
#ifndef SINGLE_OUTPUT
    float k = p.fix_factor != 0u ? float(p.fix) : c.a;
    factor = vec4(way != 0u ? 1.0 : k / 128.0);
#endif
}
