/* Texture packs (texture_pack.h): PCSX2's names, the pack's index, its files
   read in the background. */

#include "ps2host/texture_pack.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#define XXH_INLINE_ALL
#include "xxhash.h"
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "stb_image.h"

namespace ps2host {
namespace texture_pack {

using gs::u32;
using gs::u64;
using gs::u8;

namespace {

u32 Bits(u64 value, int low, int count) { return u32((value >> low) & ((1ull << count) - 1)); }

// ------------------------------------------------------------------ names

enum : u32 {
    kCT32 = 0x00, kCT24 = 0x01, kCT16 = 0x02, kCT16S = 0x0A, kT8 = 0x13, kT4 = 0x14, kT8H = 0x1B,
    kT4HL = 0x24, kT4HH = 0x2C, kZ32 = 0x30, kZ24 = 0x31, kZ16 = 0x32, kZ16S = 0x3A,
};

/* What PCSX2's psm_t says of a format: its block's size, its CLUT's entries
   (0: none), whether every bit of a texel is the texture's (fmsk all ones),
   and its colour's size in bits (32, 24, 16). */
struct PsmInfo {
    u32 bw = 8, bh = 8, pal = 0;
    bool full = false;
    u32 colour = 32;
    bool known = false;
};

PsmInfo FormatOf(u32 psm) {
    switch (psm) {
        case kCT32: case kZ32: return {8, 8, 0, true, 32, true};
        case kCT24: case kZ24: return {8, 8, 0, false, 24, true};
        case kCT16: case kCT16S: case kZ16: case kZ16S: return {16, 8, 0, false, 16, true};
        case kT8: return {16, 16, 256, true, 32, true};
        case kT4: return {32, 16, 16, true, 32, true};
        case kT8H: return {8, 8, 256, false, 32, true};
        case kT4HL: case kT4HH: return {8, 8, 16, false, 32, true};
        default: return {};
    }
}

/* PCSX2's SourceRegion::Create: the part of the texture CLAMP's region modes
   read, on each axis that has one ([min, max), 0 and 0 for none). */
struct Region {
    u32 min_x = 0, max_x = 0, min_y = 0, max_y = 0;
    bool HasX() const { return min_x != 0 || max_x != 0; }
    bool HasY() const { return min_y != 0 || max_y != 0; }
};

Region RegionOf(u64 tex0, u64 clamp) {
    Region r;
    const u32 tw = 1u << Bits(tex0, 26, 4), th = 1u << Bits(tex0, 30, 4);
    const u32 wms = Bits(clamp, 0, 2), wmt = Bits(clamp, 2, 2);
    const u32 minu = Bits(clamp, 4, 10), maxu = Bits(clamp, 14, 10);
    const u32 minv = Bits(clamp, 24, 10), maxv = Bits(clamp, 34, 10);
    if (wms == 2 && maxu >= minu) {
        if (maxu - minu + 1 < tw || maxu >= tw) r.min_x = minu, r.max_x = maxu + 1;
    } else if (wms == 3 && minu != 0) {
        const u32 rw = ((minu | maxu) - maxu) + 1;
        if (rw < tw || (maxu != 0 && rw <= tw)) r.min_x = maxu, r.max_x = (minu | maxu) + 1;
    }
    if (wmt == 2 && maxv >= minv) {
        if (maxv - minv + 1 < th || maxv >= th) r.min_y = minv, r.max_y = maxv + 1;
    } else if (wmt == 3 && minv != 0) {
        const u32 rh = ((minv | maxv) - maxv) + 1;
        if (rh < th || (maxv != 0 && rh <= th)) r.min_y = maxv, r.max_y = (minv | maxv) + 1;
    }
    return r;
}

/* A texel as PCSX2's rtx expands it for hashing: RGBA (R in the low byte). */
u32 Expand(u32 psm, u32 raw, u64 texa) {
    const u32 ta0 = Bits(texa, 0, 8), aem = Bits(texa, 15, 1), ta1 = Bits(texa, 32, 8);
    const PsmInfo f = FormatOf(psm);
    if (f.colour == 24) return (raw & 0xFFFFFF) | ((aem && (raw & 0xFFFFFF) == 0) ? 0u : ta0 << 24);
    if (f.colour == 16) {
        const u32 rgb = (raw & 31) << 3 | ((raw >> 5) & 31) << 11 | ((raw >> 10) & 31) << 19;
        const u32 a = (raw & 0x8000) ? ta1 : (aem && raw == 0) ? 0u : ta0;
        return rgb | a << 24;
    }
    return raw;
}

/* PCSX2's HashTextureLevel, level 0. */
u64 TextureHash(const gs::Memory& memory, u64 tex0, u64 texa, const Region& region) {
    const u32 psm = Bits(tex0, 20, 6), tbp = Bits(tex0, 0, 14), tbw = Bits(tex0, 14, 6);
    const PsmInfo f = FormatOf(psm);
    const int tw = region.HasX() ? int(region.max_x - region.min_x) : 1 << Bits(tex0, 26, 4);
    const int th = region.HasY() ? int(region.max_y - region.min_y) : 1 << Bits(tex0, 30, 4);
    // GetRect: the region's bounds on an axis that has one, else 0 to the size.
    const int left = region.HasX() ? int(region.min_x) : 0, top = region.HasY() ? int(region.min_y) : 0;
    const int right = region.HasX() ? int(region.max_x) : tw, bottom = region.HasY() ? int(region.max_y) : th;
    const int bl = left / int(f.bw) * int(f.bw), bt = top / int(f.bh) * int(f.bh);
    const int br = (right + int(f.bw) - 1) / int(f.bw) * int(f.bw);
    const int bb = (bottom + int(f.bh) - 1) / int(f.bh) * int(f.bh);
    XXH3_state_t state;
    XXH3_64bits_reset(&state);
    const bool expanded = tw < int(f.bw) || th < int(f.bh) || !f.full || region.max_x > 0 || region.min_y > 0;
    if (!expanded) {
        // The blocks as memory holds them, a row of blocks at a time.
        for (int y = bt; y < bb; y += int(f.bh))
            for (int x = bl; x < br; x += int(f.bw)) {
                u32 nibble = 0;
                const u32 block = (gs::Memory::Address(tbp, tbw, u32(x), u32(y), psm, &nibble) / 256) & 0x3FFF;
                XXH3_64bits_update(&state, memory.data() + std::size_t(block) * 256, 256);
            }
        return XXH3_64bits_digest(&state);
    }
    // Expanded: indices a byte each, or RGBA - laid out as PCSX2's temporary
    // buffer is (its pitch from the block rectangle's right edge, aligned to
    // 32 bytes), the rows hashed from the rectangle's left edge - in bytes,
    // even for four-byte texels (as PCSX2 does).
    const bool pal = f.pal != 0;
    const u32 texel_bytes = pal ? 1 : 4;
    const std::size_t pitch = (std::size_t(br) * texel_bytes + 31) & ~std::size_t(31);
    std::vector<u8> buffer(pitch * std::size_t(bb - bt) + 64, 0);
    for (int y = bt; y < bb; ++y) {
        u8* row = buffer.data() + pitch * std::size_t(y - bt);
        for (int x = bl; x < br; ++x) {
            const u32 raw = memory.Read(tbp, tbw, u32(x), u32(y), psm);
            if (pal) {
                row[x - bl] = u8(raw);
            } else {
                const u32 c = Expand(psm, raw, texa);
                std::memcpy(row + std::size_t(x - bl) * 4, &c, 4);
            }
        }
    }
    const u8* at = buffer.data() + pitch * std::size_t(top - bt) + std::size_t(left - bl);
    const std::size_t row_size = std::size_t(tw) * texel_bytes;
    for (int y = 0; y < th; ++y, at += pitch) XXH3_64bits_update(&state, at, row_size);
    return XXH3_64bits_digest(&state);
}

// ------------------------------------------------------------------ files

struct NameHash {
    std::size_t operator()(const Name& n) const {
        return std::size_t(n.texture * 0x9E3779B97F4A7C15ull ^ n.clut ^ (u64(n.bits) << 32 | n.region_w << 16 |
                                                                           n.region_h));
    }
};

bool ParseHex(const std::string& s, u64* out) {
    if (s.empty() || s.size() > 16) return false;
    u64 v = 0;
    for (char ch : s) {
        int d;
        if (ch >= '0' && ch <= '9') d = ch - '0';
        else if (ch >= 'a' && ch <= 'f') d = ch - 'a' + 10;
        else if (ch >= 'A' && ch <= 'F') d = ch - 'A' + 10;
        else return false;
        v = v << 4 | u64(d);
    }
    *out = v;
    return true;
}

/* A file's name (without its extension) as PCSX2's parser reads it. */
bool ParseName(const std::string& stem, Name* name) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= stem.size(); ++i)
        if (i == stem.size() || stem[i] == '-') {
            parts.push_back(stem.substr(start, i - start));
            start = i + 1;
        }
    if (parts.size() < 2 || parts.size() > 4) return false;
    for (const std::string& part : parts)
        if (part.empty()) return false;
    Name n;
    u64 value = 0;
    if (!ParseHex(parts[0], &n.texture) || !ParseHex(parts.back(), &value) || parts.back().size() != 8) return false;
    n.bits = u32(value) & ~0x4000u;                     // TCC: PCSX2 clears it
    std::size_t next = 1;
    if (next + 1 < parts.size() && parts[next][0] != 'r') {
        if (!ParseHex(parts[next], &n.clut)) return false;
        ++next;
    }
    if (next + 1 < parts.size()) {
        const std::string& r = parts[next];
        const std::size_t x = r.find('x');
        if (r[0] != 'r' || x == std::string::npos) return false;     // (the old r<64-bit> form: not read)
        u64 w = 0, h = 0;
        try {
            w = std::stoull(r.substr(1, x - 1));
            h = std::stoull(r.substr(x + 1));
        } catch (...) {
            return false;
        }
        n.region_w = u32(w);
        n.region_h = u32(h);
        ++next;
    }
    if (next + 1 != parts.size()) return false;
    *name = n;
    return true;
}

/* Halves an RGBA level (a box filter), down to 1 x 1. */
void Halve(const std::vector<u32>& from, u32 w, u32 h, std::vector<u32>* to, u32* nw, u32* nh) {
    *nw = std::max(1u, w / 2);
    *nh = std::max(1u, h / 2);
    to->assign(std::size_t(*nw) * *nh, 0);
    for (u32 y = 0; y < *nh; ++y)
        for (u32 x = 0; x < *nw; ++x) {
            u32 sum[4] = {};
            for (u32 dy = 0; dy < 2; ++dy)
                for (u32 dx = 0; dx < 2; ++dx) {
                    const u32 c = from[std::size_t(std::min(h - 1, y * 2 + dy)) * w + std::min(w - 1, x * 2 + dx)];
                    for (int k = 0; k < 4; ++k) sum[k] += (c >> (8 * k)) & 255;
                }
            u32 c = 0;
            for (int k = 0; k < 4; ++k) c |= ((sum[k] + 2) / 4) << (8 * k);
            (*to)[std::size_t(y) * *nw + x] = c;
        }
}

std::atomic<bool> g_block_compression{false};

/* RGBA levels (largest first) as one image; more made by halving, down to
   1 x 1 or 8 in all, when there are fewer. */
std::shared_ptr<Image> Rgba(std::vector<std::vector<u32>> levels, u32 width, u32 height) {
    constexpr u32 kMaxLevels = 8;
    u32 w = width, h = height;
    for (std::size_t n = 1; n < levels.size(); ++n) w = std::max(1u, w / 2), h = std::max(1u, h / 2);
    while (levels.size() < kMaxLevels && (w > 1 || h > 1)) {
        std::vector<u32> next;
        u32 nw, nh;
        Halve(levels.back(), w, h, &next, &nw, &nh);
        levels.push_back(std::move(next));
        w = nw;
        h = nh;
    }
    levels.resize(std::min<std::size_t>(levels.size(), kMaxLevels));
    auto image = std::make_shared<Image>();
    image->format = kRgba8;
    image->width = width;
    image->height = height;
    image->levels = u32(levels.size());
    for (const std::vector<u32>& level : levels) {
        const std::size_t at = image->data.size();
        image->data.resize(at + level.size() * 4);
        std::memcpy(image->data.data() + at, level.data(), level.size() * 4);
    }
    return image;
}

u32 Le32(const u8* p) { return u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24; }

// ------------------------------------------------------------------ BC encoding

/* The mipmap levels a block-compressed file lacks, made here: its smallest
   level decoded, halved, and each half encoded again as the file's format
   (BC1, BC2, BC3; BC7 keeps the levels it has). A simple encoder - the box
   around the block's colours, inset, each texel the nearest of the four -
   for levels seen small and far off. Without them a pack's texture far away
   shimmers, and trilinear or anisotropic filtering has no levels to use. */
u32 To565(int r, int g, int b) {
    return u32((std::clamp(r, 0, 255) * 31 + 127) / 255) << 11 | u32((std::clamp(g, 0, 255) * 63 + 127) / 255) << 5 |
           u32((std::clamp(b, 0, 255) * 31 + 127) / 255);
}
void From565(u32 c, int* rgb) {
    rgb[0] = int((c >> 11) & 31) * 255 / 31;
    rgb[1] = int((c >> 5) & 63) * 255 / 63;
    rgb[2] = int(c & 31) * 255 / 31;
}

/* A 4x4 tile's colours (RGBA, R in the low byte) as a BC1 colour block.
   `punch`: BC1 with texels of alpha under 128 - the three-colour mode, their
   index 3 (transparent). */
void EncodeColour(const u32* tile, u8* out, bool punch) {
    int lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
    bool any = false;
    for (int i = 0; i < 16; ++i) {
        if (punch && (tile[i] >> 24) < 128) continue;
        any = true;
        for (int k = 0; k < 3; ++k) {
            const int v = int((tile[i] >> (8 * k)) & 255);
            lo[k] = std::min(lo[k], v);
            hi[k] = std::max(hi[k], v);
        }
    }
    if (!any) lo[0] = lo[1] = lo[2] = hi[0] = hi[1] = hi[2] = 0;
    for (int k = 0; k < 3; ++k) {               // inset by a sixteenth of the range
        const int inset = (hi[k] - lo[k]) / 16;
        lo[k] += inset;
        hi[k] -= inset;
    }
    u32 c0 = To565(hi[0], hi[1], hi[2]), c1 = To565(lo[0], lo[1], lo[2]);
    // Four colours: c0 > c1; three and transparent: c0 <= c1.
    if (punch ? c0 > c1 : c0 < c1) std::swap(c0, c1);
    int p[4][3];
    From565(c0, p[0]);
    From565(c1, p[1]);
    const bool four = c0 > c1;
    for (int k = 0; k < 3; ++k) {
        if (four) {
            p[2][k] = (2 * p[0][k] + p[1][k]) / 3;
            p[3][k] = (p[0][k] + 2 * p[1][k]) / 3;
        } else {
            p[2][k] = (p[0][k] + p[1][k]) / 2;
            p[3][k] = 0;
        }
    }
    u32 indices = 0;
    if (four) {
        // Each texel's place along the line from c1 to c0, to the nearest
        // of its four colours (1, 3, 2, 0 in that order).
        const int axis[3] = {p[0][0] - p[1][0], p[0][1] - p[1][1], p[0][2] - p[1][2]};
        const int length = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        static const u32 kOrder[4] = {1, 3, 2, 0};
        for (int i = 0; i < 16; ++i) {
            int along = 0;
            for (int k = 0; k < 3; ++k) along += (int((tile[i] >> (8 * k)) & 255) - p[1][k]) * axis[k];
            const int q = length ? std::clamp((along * 6 + length) / (2 * length), 0, 3) : 0;
            indices |= kOrder[q] << (2 * i);
        }
    }
    for (int i = 0; i < 16 && !four; ++i) {
        u32 best = 0;
        if (punch && (tile[i] >> 24) < 128) {
            best = 3;
        } else {
            int best_d = 1 << 30;
            for (u32 j = 0; j < (four ? 4u : 3u); ++j) {
                int d = 0;
                for (int k = 0; k < 3; ++k) {
                    const int e = int((tile[i] >> (8 * k)) & 255) - p[j][k];
                    d += e * e;
                }
                if (d < best_d) best_d = d, best = j;
            }
        }
        indices |= best << (2 * i);
    }
    out[0] = u8(c0);
    out[1] = u8(c0 >> 8);
    out[2] = u8(c1);
    out[3] = u8(c1 >> 8);
    for (int k = 0; k < 4; ++k) out[4 + k] = u8(indices >> (8 * k));
}

/* A tile's alphas as BC3's alpha block: its largest and smallest, the eight
   values between, each texel the nearest. */
void EncodeAlpha(const u32* tile, u8* out) {
    int a0 = 0, a1 = 255;
    for (int i = 0; i < 16; ++i) {
        const int a = int(tile[i] >> 24);
        a0 = std::max(a0, a);
        a1 = std::min(a1, a);
    }
    // Each texel's place between a1 (0) and a0 (7), to the nearest of the
    // eight: 7 is index 0, 0 index 1, those between 8 - q.
    u64 indices = 0;
    const int range = a0 - a1;
    for (int i = 0; i < 16 && range > 0; ++i) {
        const int q = ((int(tile[i] >> 24) - a1) * 14 + range) / (2 * range);
        indices |= u64(q == 7 ? 0 : q == 0 ? 1 : 8 - q) << (3 * i);
    }
    out[0] = u8(a0);
    out[1] = u8(a1);
    for (int k = 0; k < 6; ++k) out[2 + k] = u8(indices >> (8 * k));
}

/* An RGBA level as BC1, BC2 or BC3 blocks, appended to `out`. */
void EncodeLevel(const std::vector<u32>& level, u32 w, u32 h, int bc, std::vector<u8>* out) {
    const u32 bx = (w + 3) / 4, by = (h + 3) / 4;
    const std::size_t block = bc == kBc1 ? 8 : 16;
    const std::size_t at = out->size();
    out->resize(at + std::size_t(bx) * by * block);
    u32 tile[16];
    bool punch = false;
    if (bc == kBc1)
        for (u32 c : level) punch |= (c >> 24) < 128;
    for (u32 y = 0; y < by; ++y)
        for (u32 x = 0; x < bx; ++x) {
            for (u32 j = 0; j < 4; ++j)
                for (u32 i = 0; i < 4; ++i)
                    tile[j * 4 + i] = level[std::size_t(std::min(h - 1, y * 4 + j)) * w + std::min(w - 1, x * 4 + i)];
            u8* b = out->data() + at + (std::size_t(y) * bx + x) * block;
            if (bc == kBc1) {
                EncodeColour(tile, b, punch);
            } else {
                if (bc == kBc2) {
                    for (int i = 0; i < 16; i += 2)
                        b[i / 2] = u8((tile[i] >> 28) | ((tile[i + 1] >> 28) << 4));
                } else {
                    EncodeAlpha(tile, b);
                }
                EncodeColour(tile, b + 8, false);
            }
        }
}

/* A level of a block-compressed file, decoded to RGBA. */
std::vector<u32> DecodeLevel(const u8* data, u32 w, u32 h, int bc) {
    std::vector<u32> level(std::size_t(w) * h);
    const u32 bx = (w + 3) / 4, by = (h + 3) / 4;
    const std::size_t block = bc == kBc1 ? 8 : 16;
    u32 tile[16];
    for (u32 y = 0; y < by; ++y)
        for (u32 x = 0; x < bx; ++x) {
            const u8* src = data + (std::size_t(y) * bx + x) * block;
            if (bc == kBc1) bcdec_bc1(src, tile, 16);
            else if (bc == kBc2) bcdec_bc2(src, tile, 16);
            else if (bc == kBc3) bcdec_bc3(src, tile, 16);
            else bcdec_bc7(src, tile, 16);
            for (u32 j = 0; j < 4 && y * 4 + j < h; ++j)
                for (u32 i = 0; i < 4 && x * 4 + i < w; ++i) level[std::size_t(y * 4 + j) * w + x * 4 + i] = tile[j * 4 + i];
        }
    return level;
}

std::shared_ptr<Image> LoadDds(const std::vector<u8>& file) {
    if (file.size() < 128 || std::memcmp(file.data(), "DDS ", 4) != 0) return nullptr;
    const u8* h = file.data() + 4;
    const u32 height = Le32(h + 8), width = Le32(h + 12), flags = Le32(h + 4);
    const u32 mips = (flags & 0x20000) ? std::max(1u, Le32(h + 24)) : 1;
    const u8* pf = h + 72;
    const u32 pf_flags = Le32(pf + 4), fourcc = Le32(pf + 8), bitcount = Le32(pf + 12);
    std::size_t offset = 128;
    // Block compression (a Format), or -1 RGBA8, -2 BGRA8.
    int bc = 0;
    if (pf_flags & 0x4) {
        if (fourcc == 0x31545844) bc = kBc1;                    // DXT1
        else if (fourcc == 0x33545844) bc = kBc2;               // DXT3
        else if (fourcc == 0x35545844) bc = kBc3;               // DXT5
        else if (fourcc == 0x30315844) {                        // DX10
            if (file.size() < 148) return nullptr;
            const u32 dxgi = Le32(file.data() + 128);
            offset = 148;
            if (dxgi == 71 || dxgi == 72) bc = kBc1;
            else if (dxgi == 74 || dxgi == 75) bc = kBc2;
            else if (dxgi == 77 || dxgi == 78) bc = kBc3;
            else if (dxgi == 98 || dxgi == 99) bc = kBc7;
            else if (dxgi == 28 || dxgi == 29) bc = -1;
            else if (dxgi == 87 || dxgi == 91) bc = -2;
            else return nullptr;
        } else {
            return nullptr;
        }
    } else if ((pf_flags & 0x40) && bitcount == 32) {
        bc = Le32(pf + 16) == 0x00FF0000 ? -2 : -1;             // RGB masks: BGRA or RGBA
    } else {
        return nullptr;
    }
    if (width == 0 || height == 0 || width > 8192 || height > 8192) return nullptr;
    // The levels the file has, as many as are whole.
    std::vector<std::pair<std::size_t, std::size_t>> spans;     // offset, bytes
    u32 w = width, hh = height;
    for (u32 n = 0; n < mips && n < 14; ++n) {
        const std::size_t bytes = bc > 0 ? std::size_t((w + 3) / 4) * ((hh + 3) / 4) * (bc == kBc1 ? 8 : 16)
                                         : std::size_t(w) * hh * 4;
        if (offset + bytes > file.size()) break;
        spans.emplace_back(offset, bytes);
        offset += bytes;
        if (w == 1 && hh == 1) break;
        w = std::max(1u, w / 2);
        hh = std::max(1u, hh / 2);
    }
    if (spans.empty()) return nullptr;
    if (bc > 0 && g_block_compression) {
        auto image = std::make_shared<Image>();
        image->format = u32(bc);
        image->width = width;
        image->height = height;
        image->levels = u32(spans.size());
        image->data.assign(file.begin() + std::ptrdiff_t(spans.front().first),
                           file.begin() + std::ptrdiff_t(spans.back().first + spans.back().second));
        // The levels it lacks (8 in all, as the renderer takes), made from
        // its smallest (EncodeLevel).
        constexpr u32 kMaxLevels = 8;
        if (bc != kBc7 && spans.size() < kMaxLevels) {
            u32 lw = width, lh = height;
            for (std::size_t n = 1; n < spans.size(); ++n) lw = std::max(1u, lw / 2), lh = std::max(1u, lh / 2);
            if (lw > 1 || lh > 1) {
                std::vector<u32> level = DecodeLevel(file.data() + spans.back().first, lw, lh, bc);
                while (image->levels < kMaxLevels && (lw > 1 || lh > 1)) {
                    std::vector<u32> next;
                    u32 nw, nh;
                    Halve(level, lw, lh, &next, &nw, &nh);
                    EncodeLevel(next, nw, nh, bc, &image->data);
                    ++image->levels;
                    level = std::move(next);
                    lw = nw;
                    lh = nh;
                }
            }
        }
        return image;
    }
    std::vector<std::vector<u32>> levels;
    w = width;
    hh = height;
    for (const auto& [at, bytes] : spans) {
        std::vector<u32> level(std::size_t(w) * hh);
        if (bc > 0) {
            const u32 bx = (w + 3) / 4, by = (hh + 3) / 4;
            const std::size_t block = bc == kBc1 ? 8 : 16;
            u32 tile[16];
            for (u32 y = 0; y < by; ++y)
                for (u32 x = 0; x < bx; ++x) {
                    const u8* src = file.data() + at + (std::size_t(y) * bx + x) * block;
                    if (bc == kBc1) bcdec_bc1(src, tile, 16);
                    else if (bc == kBc2) bcdec_bc2(src, tile, 16);
                    else if (bc == kBc3) bcdec_bc3(src, tile, 16);
                    else bcdec_bc7(src, tile, 16);
                    for (u32 j = 0; j < 4 && y * 4 + j < hh; ++j)
                        for (u32 i = 0; i < 4 && x * 4 + i < w; ++i)
                            level[std::size_t(y * 4 + j) * w + x * 4 + i] = tile[j * 4 + i];
                }
        } else {
            for (std::size_t i = 0; i < level.size(); ++i) {
                u32 c = Le32(file.data() + at + i * 4);
                if (bc == -2) c = (c & 0xFF00FF00u) | (c >> 16 & 0xFF) | (c & 0xFF) << 16;
                level[i] = c;
            }
        }
        levels.push_back(std::move(level));
        w = std::max(1u, w / 2);
        hh = std::max(1u, hh / 2);
    }
    // (Decoded here, a compressed file's levels below its own are made too.)
    return Rgba(std::move(levels), width, height);
}

std::shared_ptr<Image> LoadPng(const std::vector<u8>& file) {
    int w = 0, h = 0, channels = 0;
    unsigned char* data = stbi_load_from_memory(file.data(), int(file.size()), &w, &h, &channels, 4);
    if (!data) return nullptr;
    std::vector<u32> level(std::size_t(w) * h);
    std::memcpy(level.data(), data, level.size() * 4);
    stbi_image_free(data);
    // A PNG without alpha: 0x80, the GS's 1.0 (as PCSX2 gives it).
    if (channels == 3 || channels == 1)
        for (u32& c : level) c = (c & 0xFFFFFF) | 0x80000000u;
    return Rgba({std::move(level)}, u32(w), u32(h));
}

std::shared_ptr<Image> Load(const std::filesystem::path& path) {
#ifdef _WIN32
    std::FILE* f = _wfopen(path.wstring().c_str(), L"rb");
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) return nullptr;
    std::vector<u8> bytes;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size > 0) {
        bytes.resize(std::size_t(size));
        if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
    }
    std::fclose(f);
    if (bytes.empty()) return nullptr;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](char ch) { return char(std::tolower(ch)); });
    return ext == ".dds" ? LoadDds(bytes) : LoadPng(bytes);
}

// ------------------------------------------------------------------ the pack

/* The images read stay while they come to under this many bytes (the least
   recently drawn go first: read again if drawn again). */
constexpr std::size_t kKeptBytes = std::size_t(1536) << 20;

struct Pack {
    std::mutex lock;
    std::condition_variable wake;
    struct File {
        std::filesystem::path path;
        u64 id = 0;                     // the renderer's name for its image (UploadReplacement)
    };
    std::unordered_map<Name, File, NameHash> files;
    struct Kept {
        std::shared_ptr<const Image> image;
        std::list<Name>::iterator lru;
    };
    std::unordered_map<Name, Kept, NameHash> kept;
    std::list<Name> lru;
    std::size_t kept_bytes = 0;
    std::unordered_set<Name, NameHash> failed, queued;
    std::deque<Name> queue;
    // The readers: a file's mipmap levels made as it is read (EncodeLevel,
    // ~1-3 ms a file) - a few at a time, so that a view's new textures come
    // in as quickly as they did read whole.
    std::vector<std::thread> readers;
    bool stop = false;

    ~Pack() {
        {
            std::lock_guard<std::mutex> hold(lock);
            stop = true;
        }
        wake.notify_all();
        for (std::thread& reader : readers)
            if (reader.joinable()) reader.join();
    }

    void Read() {
        for (;;) {
            Name name;
            File file;
            {
                std::unique_lock<std::mutex> hold(lock);
                wake.wait(hold, [&] { return stop || !queue.empty(); });
                if (stop) return;
                name = queue.front();
                queue.pop_front();
                file = files[name];
            }
            std::shared_ptr<Image> loaded = Load(file.path);
            if (loaded) loaded->id = file.id;
            std::shared_ptr<const Image> image = loaded;
            std::lock_guard<std::mutex> hold(lock);
            queued.erase(name);
            if (!image) {
                failed.insert(name);
                std::fprintf(stderr, "[texture pack] could not read %s\n", file.path.string().c_str());
                continue;
            }
            lru.push_front(name);
            kept[name] = {image, lru.begin()};
            kept_bytes += image->data.size();
            while (kept_bytes > kKeptBytes && lru.size() > 1) {
                auto old = kept.find(lru.back());
                kept_bytes -= old->second.image->data.size();
                kept.erase(old);
                lru.pop_back();
            }
        }
    }
};

Pack& ThePack() {
    static Pack pack;
    return pack;
}

std::atomic<bool> g_enabled{true};
std::atomic<u64> g_generation{1};
std::atomic<std::size_t> g_count{0};

}  // namespace

std::string Name::Text() const {
    char text[96];
    int n = std::snprintf(text, sizeof text, "%llx", static_cast<unsigned long long>(texture));
    if (clut) n += std::snprintf(text + n, sizeof text - n, "-%llx", static_cast<unsigned long long>(clut));
    if (region_w || region_h) n += std::snprintf(text + n, sizeof text - n, "-r%ux%u", region_w, region_h);
    std::snprintf(text + n, sizeof text - n, "-%08x", bits);
    return text;
}

std::size_t Open(const std::string& directory) {
    namespace fs = std::filesystem;
    Pack& pack = ThePack();
    std::error_code error;
    std::size_t count = 0;
    {
        std::lock_guard<std::mutex> hold(pack.lock);
        pack.files.clear();
        const fs::path root = fs::u8path(directory);
        if (fs::is_directory(root, error)) {
            for (fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, error), end;
                 it != end; it.increment(error)) {
                if (error) break;
                if (!it->is_regular_file(error)) continue;
                std::string ext = it->path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](char ch) { return char(std::tolower(ch)); });
                if (ext != ".dds" && ext != ".png") continue;
                Name name;
                if (!ParseName(it->path().stem().string(), &name)) continue;
                pack.files.emplace(name, Pack::File{it->path(), pack.files.size() + 1});    // the first found, as PCSX2
            }
        }
        count = pack.files.size();
        if (count && pack.readers.empty()) {
            const unsigned cores = std::thread::hardware_concurrency();
            const unsigned n = cores >= 8 ? 3 : cores >= 4 ? 2 : 1;
            for (unsigned i = 0; i < n; ++i)
                pack.readers.emplace_back([&pack] {
#ifdef _WIN32
                    // Below the game's threads: a view's new files a moment
                    // later, never the game's work.
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
                    pack.Read();
                });
        }
    }
    g_count = count;
    ++g_generation;
    return count;
}

std::size_t Count() { return g_count; }
void SetBlockCompression(bool on) { g_block_compression = on; }
bool Enabled() { return g_enabled; }
void SetEnabled(bool on) {
    if (g_enabled.exchange(on) != on) ++g_generation;
}
u64 Generation() { return g_generation; }

Name NameOf(const gs::Memory& memory, u64 tex0, u64 clamp, u64 texa, const u32* clut) {
    Name name;
    const u32 psm = Bits(tex0, 20, 6);
    const PsmInfo f = FormatOf(psm);
    if (!f.known) return name;
    const Region region = RegionOf(tex0, clamp);
    name.texture = TextureHash(memory, tex0, texa, region);
    name.bits = psm | Bits(tex0, 26, 4) << 6 | Bits(tex0, 30, 4) << 10;
    if (!f.pal && f.colour != 32)
        name.bits |= Bits(texa, 0, 8) << 15 | Bits(texa, 15, 1) << 23 | Bits(texa, 32, 8) << 24;
    name.region_w = region.HasX() ? region.max_x - region.min_x : 0;
    name.region_h = region.HasY() ? region.max_y - region.min_y : 0;
    if (f.pal) {
        // The CLUT as PCSX2's Read32 gives it: a 4-bit texture's 16 entries
        // from CSA's, an 8-bit one's 256.
        const u32 csa = Bits(tex0, 56, 5);
        const u32* entries = f.pal == 16 ? clut + (csa * 16) % 256 : clut;
        name.clut = XXH3_64bits(entries, std::size_t(f.pal) * 4);
    }
    return name;
}

namespace {

// ------------------------------------------------------------------ overrides

struct Overrides {
    std::mutex lock;
    std::unordered_map<Name, std::shared_ptr<const Image>, NameHash> images;
    std::unordered_map<const void*, std::shared_ptr<const Image>> decoded;     // by the PNG's address
    u64 next_id = 1ull << 40;           // past any pack file's id
};
Overrides& TheOverrides() {
    static Overrides overrides;
    return overrides;
}
// Wanted's test: a bit per PSM | TW << 6 | TH << 10 an override has.
std::atomic<u64> g_override_bits[(1u << 14) / 64];
std::atomic<bool> g_any_override{false};

void IndexOverrides(Overrides& o) {
    for (auto& word : g_override_bits) word = 0;
    for (const auto& [name, image] : o.images) {
        const u32 bits = name.bits & 0x3FFFu;
        g_override_bits[bits / 64] |= 1ull << (bits % 64);
    }
    g_any_override = !o.images.empty();
    ++g_generation;
}

}  // namespace

bool SetOverride(const std::string& text, const std::uint8_t* png, std::size_t bytes) {
    Name name;
    if (!ParseName(text, &name)) return false;
    Overrides& o = TheOverrides();
    std::lock_guard<std::mutex> hold(o.lock);
    if (!png) {
        if (o.images.erase(name)) IndexOverrides(o);
        return true;
    }
    std::shared_ptr<const Image>& image = o.decoded[png];
    if (!image) {
        std::shared_ptr<Image> loaded = LoadPng(std::vector<u8>(png, png + bytes));
        if (!loaded) {
            o.decoded.erase(png);
            return false;
        }
        loaded->id = o.next_id++;
        image = loaded;
    }
    auto found = o.images.find(name);
    if (found != o.images.end() && found->second == image) return true;
    o.images[name] = image;
    IndexOverrides(o);
    return true;
}

std::shared_ptr<const Image> MakeImage(std::vector<u32> rgba, u32 width, u32 height) {
    if (!width || !height || rgba.size() != std::size_t(width) * height) return nullptr;
    std::shared_ptr<Image> image = Rgba({std::move(rgba)}, width, height);
    Overrides& o = TheOverrides();
    std::lock_guard<std::mutex> hold(o.lock);
    image->id = o.next_id++;
    return image;
}

bool SetOverride(const std::string& text, std::shared_ptr<const Image> image) {
    Name name;
    if (!ParseName(text, &name)) return false;
    Overrides& o = TheOverrides();
    std::lock_guard<std::mutex> hold(o.lock);
    auto found = o.images.find(name);
    if (!image) {
        if (found != o.images.end()) {
            o.images.erase(found);
            IndexOverrides(o);
        }
        return true;
    }
    if (found != o.images.end() && found->second == image) return true;
    o.images[name] = std::move(image);
    IndexOverrides(o);
    return true;
}

void ClearOverrides() {
    Overrides& o = TheOverrides();
    std::lock_guard<std::mutex> hold(o.lock);
    if (o.images.empty()) return;
    o.images.clear();
    IndexOverrides(o);
}

bool Active() { return g_any_override || (g_count != 0 && g_enabled); }

bool Wanted(u32 bits) {
    if (g_count != 0 && g_enabled) return true;
    if (!g_any_override) return false;
    bits &= 0x3FFFu;
    return (g_override_bits[bits / 64] >> (bits % 64)) & 1;
}

std::shared_ptr<const Image> Find(const Name& name, bool* pending) {
    *pending = false;
    if (g_any_override) {
        Overrides& o = TheOverrides();
        std::lock_guard<std::mutex> hold(o.lock);
        auto found = o.images.find(name);
        if (found != o.images.end()) return found->second;
    }
    if (g_count == 0 || !g_enabled) return nullptr;
    Pack& pack = ThePack();
    std::lock_guard<std::mutex> hold(pack.lock);
    auto kept = pack.kept.find(name);
    if (kept != pack.kept.end()) {
        pack.lru.splice(pack.lru.begin(), pack.lru, kept->second.lru);
        return kept->second.image;
    }
    if (pack.failed.count(name) || !pack.files.count(name)) return nullptr;
    *pending = true;
    if (pack.queued.insert(name).second) {
        pack.queue.push_back(name);
        pack.wake.notify_one();
    }
    return nullptr;
}

}  // namespace texture_pack
}  // namespace ps2host
