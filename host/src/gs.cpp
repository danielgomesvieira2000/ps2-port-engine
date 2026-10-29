/*
 * The GS (gs.h): GIF packets, registers, image transfers, the display.
 * Drawing is in gs_draw.cpp, local memory's layout in gs_memory.cpp.
 *
 * A GIF tag (GS User's Manual, chapter 7) is a quadword:
 *   bits 0-14 NLOOP, 15 EOP, 46 PRE, 47-57 PRIM, 58-59 FLG, 60-63 NREG,
 *   64-127 REGS (NREG 4-bit register descriptors)
 * FLG 0 PACKED: a quadword per register, in the descriptor's packed format;
 * 1 REGLIST: a 64-bit register value per descriptor, two per quadword;
 * 2 IMAGE (and 3): NLOOP quadwords of image data for HWREG.
 */

#include "ps2host/gs.h"
#include "ps2host/profile.h"
#include "ps2host/dma.h"
#include "gs_decode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2host {
namespace gs {

namespace {

float AsFloat(u32 bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

u32 Bits(u64 value, int low, int count) { return u32((value >> low) & ((1ull << count) - 1)); }

}  // namespace

std::string Gs::Summary() const {
    char text[160];
    std::snprintf(text, sizeof text, "GIF tags: PATH1 %llu, PATH2 %llu, PATH3 %llu; image %llu quadwords;",
                  static_cast<unsigned long long>(gif.tags[1]), static_cast<unsigned long long>(gif.tags[2]),
                  static_cast<unsigned long long>(gif.tags[3]), static_cast<unsigned long long>(gif.image_qwords));
    std::string out = text;
    out += " register writes:";
    for (u32 r = 0; r < 256; ++r)
        if (gif.register_writes[r]) {
            std::snprintf(text, sizeof text, " %02X:%llu", unsigned(r),
                          static_cast<unsigned long long>(gif.register_writes[r]));
            out += text;
        }
    return out;
}

namespace detail {
Gs* g_the_gs = nullptr;
Gs& MakeGs() {
    static Gs gs;
    g_the_gs = &gs;
    return gs;
}
}  // namespace detail

// ---------------------------------------------------------------- GIF

void Gs::Word(int path, u32 word) { GifDecode<Gs>::Word(gif, *this, path, word); }

void Gs::BeginRecord(std::FILE* out) {
    record_ = nullptr;
    if (!out) return;
    std::fwrite("GSRP1", 1, 5, out);
    std::fwrite(memory.data(), 1, 4u << 20, out);
    auto reg = [&](u32 address, u64 value) {
        std::fputc('R', out);
        std::fwrite(&address, 4, 1, out);
        std::fwrite(&value, 8, 1, out);
    };
    for (u32 c = 0; c < 2; ++c) {
        const Context& x = ctx_[c];
        // TEX0 with CLD 1: the CLUT loaded again from where it was.
        reg(0x06 + c, (x.tex0 & ~(7ull << 61)) | 1ull << 61);
        reg(0x08 + c, x.clamp);
        reg(0x14 + c, x.tex1);
        reg(0x18 + c, x.xyoffset);
        reg(0x34 + c, x.miptbp1);
        reg(0x36 + c, x.miptbp2);
        reg(0x40 + c, x.scissor);
        reg(0x42 + c, x.alpha);
        reg(0x47 + c, x.test);
        reg(0x4A + c, x.fba);
        reg(0x4C + c, x.frame);
        reg(0x4E + c, x.zbuf);
    }
    reg(0x1A, prmodecont_);
    reg(0x1B, prmode_);
    reg(0x1C, texclut_);
    reg(0x3B, texa_);
    reg(0x3D, fogcol_);
    reg(0x45, dthe_);
    reg(0x46, colclamp_);
    reg(0x49, pabe_);
    reg(0x50, bitbltbuf_);
    reg(0x51, trxpos_);
    reg(0x52, trxreg_);
    reg(0x0A, u64(gif.fog) << 56);
    reg(0x00, prim_);                                   // PRIM last: it starts no primitive
    reg(0x01, gif.rgbaq);
    reg(0x02, gif.st);
    reg(0x03, gif.uv);
    auto privileged = [&](u32 offset, u64 value) {
        std::fputc('P', out);
        const u32 address = 0x12000000u + offset;
        std::fwrite(&address, 4, 1, out);
        std::fwrite(&value, 8, 1, out);
    };
    privileged(0x0000, pmode_);
    privileged(0x0070, dispfb_[0]);
    privileged(0x0080, display_[0]);
    privileged(0x0090, dispfb_[1]);
    privileged(0x00A0, display_[1]);
    privileged(0x00E0, bgcolor_);
    record_ = out;
}

void Gs::RecordBlank() {
    if (record_) std::fputc('V', record_);
}

void Gs::EndRecord() {
    if (!record_) return;
    std::fputc('V', record_);
    std::fclose(record_);
    record_ = nullptr;
}

void Gs::Words(int path, const u32* words, u32 count, u32 from) {
    GifDecode<Gs>::Words(gif, *this, path, words, count, from);
}

void Gs::Qword(int path, u64 low, u64 high) { GifDecode<Gs>::Qword(gif, *this, path, low, high); }

// ---------------------------------------------------------------- registers

void Gs::WriteRegister(u32 address, u64 value) { GifDecode<Gs>::Register(gif, *this, address, value); }

/* A vertex the decoding made (GifDecode::Register): queued, and the
   primitive drawn once there are enough (Kick). */
void Gs::PushVertex(const Vertex& v, bool draw) {
    if (queued_ == 3) {                         // keep the newest two
        queue_[0] = queue_[1];
        queue_[1] = queue_[2];
        queued_ = 2;
    }
    queue_[queued_++] = v;
    Kick(draw);
}

/* The registers the GS keeps (every one but those only the decoding reads). */
void Gs::Apply(u32 address, u64 value) {
    ++register_writes_;
    switch (address) {
        case 0x00: prim_ = value; queued_ = 0; strip_ = 0; break;
        case 0x06: case 0x07:
            ctx_[address - 0x06].tex0 = value;
            LoadClut(value);
            break;
        case 0x08: case 0x09: ctx_[address - 0x08].clamp = value; break;
        case 0x14: case 0x15: ctx_[address - 0x14].tex1 = value; break;
        case 0x34: case 0x35: ctx_[address - 0x34].miptbp1 = value; break;
        case 0x36: case 0x37: ctx_[address - 0x36].miptbp2 = value; break;
        case 0x16: case 0x17: {
            // TEX2: TEX0's format and CLUT fields only.
            u64& tex0 = ctx_[address - 0x16].tex0;
            const u64 mask = (0x3Full << 20) | (0x7FFFFFFull << 37);
            tex0 = (tex0 & ~mask) | (value & mask);
            LoadClut(tex0);
            break;
        }
        case 0x18: case 0x19: ctx_[address - 0x18].xyoffset = value; break;
        case 0x1A: prmodecont_ = value; break;
        case 0x1B: prmode_ = value; break;
        case 0x1C: texclut_ = value; break;
        case 0x3B: texa_ = value; break;
        case 0x3D: fogcol_ = value; break;
        case 0x40: case 0x41: ctx_[address - 0x40].scissor = value; break;
        case 0x42: case 0x43: ctx_[address - 0x42].alpha = value; break;
        case 0x45: dthe_ = value; break;
        case 0x46: colclamp_ = value; break;
        case 0x47: case 0x48: ctx_[address - 0x47].test = value; break;
        case 0x49: pabe_ = value; break;
        case 0x4A: case 0x4B: ctx_[address - 0x4A].fba = value; break;
        case 0x4C: case 0x4D: ctx_[address - 0x4C].frame = value; break;
        case 0x4E: case 0x4F: ctx_[address - 0x4E].zbuf = value; break;
        case 0x50: bitbltbuf_ = value; break;
        case 0x51: trxpos_ = value; break;
        case 0x52: trxreg_ = value; break;
        case 0x53:
            trxdir_ = value & 3;
            trx_x_ = 0;
            trx_y_ = 0;
            trx_active_ = trxdir_ == 0;
            if (trace)
                std::fprintf(trace_out, "[gs] transfer dir %u: src %X/%u psm %02X (%u,%u) -> dst %X/%u psm %02X (%u,%u), %ux%u\n",
                             unsigned(trxdir_), Bits(bitbltbuf_, 0, 14), Bits(bitbltbuf_, 16, 6), Bits(bitbltbuf_, 24, 6),
                             Bits(trxpos_, 0, 11), Bits(trxpos_, 16, 11), Bits(bitbltbuf_, 32, 14), Bits(bitbltbuf_, 48, 6),
                             Bits(bitbltbuf_, 56, 6), Bits(trxpos_, 32, 11), Bits(trxpos_, 48, 11), Bits(trxreg_, 0, 12),
                             Bits(trxreg_, 32, 12));
            if (on_memory) {
                const u32 w = Bits(trxreg_, 0, 12), h = Bits(trxreg_, 32, 12);
                if (trxdir_ == 0 || trxdir_ == 2)       // into local memory
                    Touch(Bits(bitbltbuf_, 32, 14), Bits(bitbltbuf_, 48, 6), Bits(bitbltbuf_, 56, 6),
                          Bits(trxpos_, 32, 11), Bits(trxpos_, 48, 11), w, h, true);
                if (trxdir_ == 1 || trxdir_ == 2)       // out of it
                    Touch(Bits(bitbltbuf_, 0, 14), Bits(bitbltbuf_, 16, 6), Bits(bitbltbuf_, 24, 6),
                          Bits(trxpos_, 0, 11), Bits(trxpos_, 16, 11), w, h, false);
            }
            if (trxdir_ == 2) {
                LocalToLocal();
                if (on_transfer_done) on_transfer_done();
            }
            break;
        case 0x54: Transfer(value, 0); break;          // HWREG: 64 bits
        case 0x70: case 0x71: case 0x72: case 0x73: case 0x74: case 0x75: case 0x76:
        case 0x77: case 0x78: case 0x79: case 0x7A: case 0x7B: case 0x7C: case 0x7D:
            effect_params_[address - 0x70] = value;
            break;
        case 0x7E:
            if (trace) std::fprintf(trace_out, "[gs] PC effect %llX\n", static_cast<unsigned long long>(value));
            if (std::getenv("PS2HOST_EFFECT_LOG")) std::fprintf(stderr, "[gs] effect %llX set %d\n", static_cast<unsigned long long>(value), on_effect ? 1 : 0);
            if (on_effect) on_effect(value, effect_params_);
            break;
        default: break;
    }
}

// ---------------------------------------------------------------- transfers

void Gs::Transfer(u64 low, u64 high) {
    u8 bytes[16];
    std::memcpy(bytes, &low, 8);
    std::memcpy(bytes + 8, &high, 8);
    TransferBlock(bytes, 1);
}

void Gs::TransferBlock(const u8* data, u32 qwords) {
    if (!trx_active_) return;
    Finish();                                   // the draws before it land first
    uploads_ += qwords;
    const u32 dbp = Bits(bitbltbuf_, 32, 14), dbw = Bits(bitbltbuf_, 48, 6), psm = Bits(bitbltbuf_, 56, 6);
    const u32 dx = Bits(trxpos_, 32, 11), dy = Bits(trxpos_, 48, 11);
    const u32 w = Bits(trxreg_, 0, 12), h = Bits(trxreg_, 32, 12);
    if (!w || !h) { trx_active_ = false; return; }
    // Up to 16 quadwords' pixels at a time, then written a row's run at a
    // time (WriteRun: one call, not one a pixel).
    constexpr u32 kChunk = 16;
    u32 pixels[32 * kChunk];
    for (u32 q = 0; q < qwords && trx_active_; q += kChunk) {
        const u8* bytes = data + 16 * std::size_t(q);
        const u32 size = 16 * std::min(kChunk, qwords - q);
        u32 n = 0;
        switch (psm) {
            case PSMCT32: case PSMZ32:
                std::memcpy(pixels, bytes, size);       // little-endian words, as the GS takes them
                n = size / 4;
                break;
            case PSMCT24: case PSMZ24:
                // 24-bit pixels run across quadwords: gather bytes.
                for (u32 i = 0; i < size; ++i) {
                    trx_partial_ |= u32(bytes[i]) << (8 * trx_partial_bytes_);
                    if (++trx_partial_bytes_ == 3) { pixels[n++] = trx_partial_; trx_partial_ = 0; trx_partial_bytes_ = 0; }
                }
                break;
            case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S:
                for (u32 i = 0; i < size; i += 2) {
                    std::uint16_t half;
                    std::memcpy(&half, bytes + i, 2);
                    pixels[n++] = half;
                }
                break;
            case PSMT8: case PSMT8H:
                for (u32 i = 0; i < size; ++i) pixels[n++] = bytes[i];
                break;
            case PSMT4: case PSMT4HL: case PSMT4HH:
                for (u32 i = 0; i < size; ++i) { pixels[n++] = bytes[i] & 15; pixels[n++] = bytes[i] >> 4; }
                break;
            default:
                trx_active_ = false;
                break;
        }
        for (u32 i = 0; i < n && trx_active_;) {
            const u32 x = (dx + trx_x_) % 2048, y = (dy + trx_y_) % 2048;
            const u32 run = std::min({n - i, w - trx_x_, 2048 - x});
            memory.WriteRun(dbp, dbw, x, y, psm, pixels + i, run);
            i += run;
            trx_x_ += run;
            if (trx_x_ == w) {
                trx_x_ = 0;
                if (++trx_y_ == h) {
                    trx_active_ = false;
                    if (on_transfer_done) on_transfer_done();
                }
            }
        }
    }
}

void Gs::LocalToLocal() {
    Finish();
    Timed timed(Section::GsUpload);
    ++uploads_;
    const u32 sbp = Bits(bitbltbuf_, 0, 14), sbw = Bits(bitbltbuf_, 16, 6), spsm = Bits(bitbltbuf_, 24, 6);
    const u32 dbp = Bits(bitbltbuf_, 32, 14), dbw = Bits(bitbltbuf_, 48, 6), dpsm = Bits(bitbltbuf_, 56, 6);
    const u32 sx = Bits(trxpos_, 0, 11), sy = Bits(trxpos_, 16, 11);
    const u32 dx = Bits(trxpos_, 32, 11), dy = Bits(trxpos_, 48, 11);
    const u32 w = Bits(trxreg_, 0, 12), h = Bits(trxreg_, 32, 12);
    std::vector<u32> copy(std::size_t(w) * h);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x) copy[y * w + x] = memory.Read(sbp, sbw, (sx + x) % 2048, (sy + y) % 2048, spsm);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x) memory.Write(dbp, dbw, (dx + x) % 2048, (dy + y) % 2048, dpsm, copy[y * w + x]);
}

// ---------------------------------------------------------------- CLUT

namespace {

u32 From16(u32 c, u64 texa) {
    const u32 r = (c & 31) << 3, g = ((c >> 5) & 31) << 3, b = ((c >> 10) & 31) << 3;
    const u32 a = (c & 0x8000) ? Bits(texa, 32, 8) : ((Bits(texa, 15, 1) && !(c & 0x7FFF)) ? 0 : Bits(texa, 0, 8));
    return r | g << 8 | b << 16 | a << 24;
}

}  // namespace

void Gs::LoadClut(u64 tex0) {
    const u32 psm = Bits(tex0, 20, 6), cld = Bits(tex0, 61, 3);
    if (psm != PSMT8 && psm != PSMT4 && psm != PSMT8H && psm != PSMT4HL && psm != PSMT4HH) return;
    const u32 cbp = Bits(tex0, 37, 14), cpsm = Bits(tex0, 51, 4), csa = Bits(tex0, 56, 5);
    bool load = cld == 1 || cld == 2 || cld == 3;
    if (cld == 4) load = cbp != cbp0_;
    if (cld == 5) load = cbp != cbp1_;
    if (cld == 2 || cld == 4) cbp0_ = cbp;
    if (cld == 3 || cld == 5) cbp1_ = cbp;
    if (!load) return;
    Finish();                                   // the CLUT may have been drawn
    ++clut_loads_;
    if (on_clut_load && on_clut_load(tex0, texa_)) return;
    if (on_memory) on_memory(cbp, cbp + 8, false);
    auto clut = std::make_shared<std::array<u32, 256>>(*clut_);
    const bool eight = psm == PSMT8 || psm == PSMT8H;
    const u32 count = eight ? 256 : 16;
    // CSM2 (TEX0 bit 55): the CLUT is one row of a CT16 buffer, entry i at
    // (TEXCLUT COU * 16 + i, COV) in a buffer CBW wide - no swizzle, no CSA.
    const bool csm2 = Bits(tex0, 55, 1);
    const u32 cbw = csm2 ? std::max<u32>(1, Bits(texclut_, 0, 6)) : 1;
    const u32 format = cpsm == 0 ? PSMCT32 : cpsm;
    if (!csm2 && (format == PSMCT32 || format == PSMCT24 || format == PSMCT16 || format == PSMCT16S)) {
        // CSM1, the usual: every entry inside the CLUT's first page, so its
        // byte is the base block's (cbp * 256) plus where the page puts it -
        // the same each load: the entries' offsets once a format, the bytes
        // read straight (Burnout 3 loads ~650 CLUTs a frame in a pile-up,
        // 256 Memory::Read each were a twentieth of the GS thread).
        struct Offsets {
            u32 at[4][2][256];      // [format: CT32, CT24, CT16, CT16S][eight][entry]
            Offsets() {
                const u32 formats[4] = {PSMCT32, PSMCT24, PSMCT16, PSMCT16S};
                for (u32 f = 0; f < 4; ++f)
                    for (u32 e = 0; e < 2; ++e)
                        for (u32 i = 0; i < (e ? 256u : 16u); ++i) {
                            const u32 p = (i & 0xE7) | ((i & 0x08) << 1) | ((i & 0x10) >> 1);
                            u32 nibble = 0;
                            at[f][e][i] = e ? Memory::Address(0, 1, p % 16, p / 16, formats[f], &nibble)
                                            : Memory::Address(0, 1, i % 8, i / 8, formats[f], &nibble);
                        }
            }
        };
        static const Offsets offsets;
        const u32 f = format == PSMCT32 ? 0 : format == PSMCT24 ? 1 : format == PSMCT16 ? 2 : 3;
        const u32* at = offsets.at[f][eight ? 1 : 0];
        const u8* bytes = memory.data();
        const u32 base = cbp * 256;
        constexpr u32 kMemoryMask = (4u << 20) - 1;
        for (u32 i = 0; i < count; ++i) {
            const u8* p = bytes + ((base + at[i]) & kMemoryMask);
            u32 color;
            if (f == 0) {
                color = u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16 | u32(p[3]) << 24;
            } else if (f == 1) {
                const u32 rgb = u32(p[0]) | u32(p[1]) << 8 | u32(p[2]) << 16;
                color = rgb | ((Bits(texa_, 15, 1) && !rgb) ? 0 : Bits(texa_, 0, 8)) << 24;
            } else {
                color = From16(u32(p[0]) | u32(p[1]) << 8, texa_);
            }
            (*clut)[eight ? i : (csa * 16 + i) % 256] = color;
        }
        clut_ = clut;
        return;
    }
    for (u32 i = 0; i < count; ++i) {
        // CSM1: an 8-bit CLUT is a 16x16 image with index bits 3 and 4 swapped;
        // a 4-bit one an 8x2 image.
        u32 x, y;
        if (csm2) {
            x = Bits(texclut_, 6, 6) * 16 + i;
            y = Bits(texclut_, 12, 10);
        } else if (eight) {
            const u32 p = (i & 0xE7) | ((i & 0x08) << 1) | ((i & 0x10) >> 1);
            x = p % 16;
            y = p / 16;
        } else {
            x = i % 8;
            y = i / 8;
        }
        const u32 raw = memory.Read(cbp, cbw, x, y, cpsm == 0 ? PSMCT32 : cpsm);
        u32 color;
        if (cpsm == PSMCT32) {
            color = raw;
        } else if (cpsm == PSMCT24) {
            // A 24-bit CLUT (Mister Mosquito's 8-bit textures): its colour,
            // and TEXA's alpha as for a 24-bit texel - as the GPU's load does.
            const u32 rgb = raw & 0xFFFFFF;
            color = rgb | ((Bits(texa_, 15, 1) && !rgb) ? 0 : Bits(texa_, 0, 8)) << 24;
        } else {
            color = From16(raw, texa_);
        }
        (*clut)[eight || csm2 ? i : (csa * 16 + i) % 256] = color;
    }
    clut_ = clut;
}

// ---------------------------------------------------------------- display

void Gs::SetPrivileged(u32 address, u64 value) { GifDecode<Gs>::Privileged(gif, *this, address, value); }

void Gs::ApplyPrivileged(u32 address, u64 value) {
    switch (address & 0xFFFF) {
        case 0x0000: pmode_ = value; break;
        case 0x0020: smode2_ = value; break;
        case 0x0070: dispfb_[0] = value; break;
        case 0x0080: display_[0] = value; break;
        case 0x0090: dispfb_[1] = value; break;
        case 0x00A0: display_[1] = value; break;
        case 0x00E0: bgcolor_ = value; break;
        default: break;
    }
}

void Gs::Touch(u32 bp, u32 bw, u32 psm, u32 x, u32 y, u32 w, u32 h, bool write) const {
    if (!on_memory) return;
    last_transfer = TouchSeen{bp, bw, psm, x, y, w, h, write};
    (void)x;
    // Whole rows of pages: buffers are laid out a page (8 KB, 32 blocks) at a
    // time, rows of FBW pages, 32 to 128 pixels tall by format.
    const u32 page_rows = (psm == PSMCT32 || psm == PSMCT24 || psm == PSMZ32 || psm == PSMZ24) ? 32
                        : (psm == PSMT4) ? 128 : 64;
    const u32 pages_wide = std::max(1u, (psm == PSMT8 || psm == PSMT4) ? bw / 2 : bw);
    const u32 rows = (y + h + page_rows - 1) / page_rows;
    const u32 begin = bp, end = bp + rows * pages_wide * 32;
    // Written, which blocks the rectangle covers, whole or in part, a block's
    // pixels at a time (8x8 32-bit, 16x8 16-bit, 16x16 8-bit, 32x16 4-bit):
    // only those in part need what was there (with --gpu, brought back from
    // the GPU, which waits for it). Formats that write part of a pixel
    // (8H, 4HL, 4HH), and a rectangle past the buffer's width, as before.
    u32 cw = 0, ch = 0;
    switch (psm) {
        case PSMCT32: case PSMCT24: case PSMZ32: case PSMZ24: cw = 8; ch = 8; break;
        case PSMCT16: case PSMCT16S: case PSMZ16: case PSMZ16S: cw = 16; ch = 8; break;
        case PSMT8: cw = 16; ch = 16; break;
        case PSMT4: cw = 32; ch = 16; break;
        default: break;
    }
    if (!write || !on_write || !cw || !bw || x + w > bw * 64 || end > 16384 || end <= begin) {
        on_memory(begin, end, write);
        return;
    }
    std::vector<u8> cover(end - begin, 0);
    for (u32 cy = y / ch * ch; cy < y + h; cy += ch)
        for (u32 cx = x / cw * cw; cx < x + w; cx += cw) {
            u32 nibble = 0;
            const u32 block = Memory::Address(bp, bw, cx, cy, psm, &nibble) / 256;
            if (block < begin || block >= end) {        // (not expected: as before)
                on_memory(begin, end, write);
                return;
            }
            const bool whole = cx >= x && cx + cw <= x + w && cy >= y && cy + ch <= y + h;
            u8& c = cover[block - begin];
            c = std::max<u8>(c, whole ? 2 : 1);
        }
    on_write(begin, end, cover.data());
}

bool Gs::Picture(std::vector<u32>* pixels, u32* width, u32* height) const {
    Finish();
    // The read circuit shown: 2 if enabled, else 1.
    int circuit = Bits(pmode_, 1, 1) ? 1 : Bits(pmode_, 0, 1) ? 0 : -1;
    if (circuit < 0) {
        if (!dispfb_[1] && !dispfb_[0]) return false;
        circuit = dispfb_[1] ? 1 : 0;
    }
    const u64 fb = dispfb_[circuit], disp = display_[circuit];
    const u32 fbp = Bits(fb, 0, 9) * 32, fbw = Bits(fb, 9, 6), psm = Bits(fb, 15, 5);
    const u32 dbx = Bits(fb, 32, 11), dby = Bits(fb, 43, 11);
    const u32 magh = Bits(disp, 23, 4) + 1, magv = Bits(disp, 27, 2) + 1;
    u32 w = (Bits(disp, 32, 12) + 1) / magh, h = (Bits(disp, 44, 11) + 1) / magv;
    if (!w || w > 1024) w = 640;
    if (!h || h > 1024) h = 448;
    if (!fbw) return false;
    const u32 lines = ShownLines(), repeat = h / std::max(lines, 1u);
    Touch(fbp, fbw, psm, dbx, dby, w, lines, false);
    pixels->assign(std::size_t(w) * h, 0);
    for (u32 y = 0; y < h; ++y)
        for (u32 x = 0; x < w; ++x) {
            u32 c = memory.Read(fbp, fbw, dbx + x, dby + y / std::max(repeat, 1u), psm);
            if (psm == PSMCT16 || psm == PSMCT16S) c = From16(c, 0);
            // RGBA in memory, BGRA in the picture.
            (*pixels)[std::size_t(y) * w + x] = (c & 0xFF00FF00u) | ((c & 0xFF) << 16) | ((c >> 16) & 0xFF);
        }
    *width = w;
    *height = h;
    return true;
}

u32 Gs::ShownLines() const {
    const int circuit = Bits(pmode_, 1, 1) || (!Bits(pmode_, 0, 1) && dispfb_[1]) ? 1 : 0;
    const u64 disp = display_[circuit];
    u32 h = (Bits(disp, 44, 11) + 1) / (Bits(disp, 27, 2) + 1);
    if (!h || h > 1024) h = 448;
    const bool field = Bits(smode2_, 0, 1) && Bits(smode2_, 1, 1);    // interlaced, FFMD
    return field ? h / 2 : h;
}

bool Gs::FieldMode() const { return Bits(smode2_, 0, 1) && Bits(smode2_, 1, 1); }

bool Gs::WriteBmp(const std::string& path) const {
    std::vector<u32> pixels;
    u32 w = 0, h = 0;
    if (!Picture(&pixels, &w, &h)) return false;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const u32 data = w * h * 4, file = 54 + data;
    u8 header[54] = {'B', 'M'};
    auto put32 = [&](int at, u32 v) { std::memcpy(header + at, &v, 4); };
    put32(2, file);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, u32(-s32(h)));                   // negative: top-down rows
    header[26] = 1;
    header[28] = 32;
    put32(34, data);
    std::fwrite(header, 1, 54, f);
    std::fwrite(pixels.data(), 4, pixels.size(), f);
    std::fclose(f);
    return true;
}

}  // namespace gs
}  // namespace ps2host
