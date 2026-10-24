/*
 * The GIF's decoding (gs.h's GifState): tags, PACKED, REGLIST and IMAGE
 * data, and the vertices the XYZ registers make - what turns a GIF packet
 * into register writes, vertices and image data. It hands them to a Sink:
 *
 *   Apply(address, value)      a register write the GS model keeps (every
 *                              register but RGBAQ, ST, UV, FOG and the XYZs,
 *                              which only the decoding reads)
 *   PushVertex(vertex, draw)   a vertex, and whether it draws (XYZ2) or
 *                              only joins the queue (XYZ3)
 *   Image(data, qwords)        IMAGE data, whole quadwords
 *   ApplyPrivileged(a, value)  a privileged register write
 *   Record(...)                the stream being recorded (--gs-replay)
 *   Tracing(), TraceLine(text) --gs-trace's lines, in the stream's order
 *
 * Gs is a sink itself (its own calls, as before); with the GS's work split
 * over two threads (gs_thread.cpp, PS2HOST_GS_SPLIT) the decoding runs on a
 * thread of its own and its sink is a ring the GS thread takes them from, in
 * order - the same calls, the same pictures.
 */

#ifndef PS2HOST_GS_DECODE_H
#define PS2HOST_GS_DECODE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ps2host/dma.h"
#include "ps2host/gs.h"

namespace ps2host {
namespace gs {

template <class Sink>
struct GifDecode {
    static u32 Bits(u64 value, int low, int count) { return u32((value >> low) & ((1ull << count) - 1)); }
    static float AsFloat(u32 bits) {
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }

    static void Words(GifState& s, Sink& k, int path, const u32* words, u32 count, u32 from) {
        if (count) k.Record(path, words, count);
        GifState::Path& p = s.paths[path];
        for (u32 i = 0; i < count;) {
            if (p.pending_count == 0 && p.loops && p.format >= 2 && count - i >= 4) {     // IMAGE
                const u32 qwords = p.loops < (count - i) / 4 ? p.loops : (count - i) / 4;
                s.image_qwords += qwords;
                k.Image(reinterpret_cast<const u8*>(words + i), qwords);
                p.loops -= qwords;
                i += 4 * qwords;
                continue;
            }
            // PACKED, whole quadwords, no trace wanting each one's source:
            // a run of them with the tag's descriptors taken once.
            if (p.pending_count == 0 && p.loops && p.format == 0 && count - i >= 4 && !k.Tracing()) {
                i = PackedRun(s, k, p, words, count, i);
                continue;
            }
            // DMA already supplies complete quadwords. Avoid assembling those
            // through four Word calls; retain that path for fragmented packets.
            // Word's fourth write supplies the diagnostic source address.
            if (p.pending_count == 0 && count - i >= 4) {
                if (from != Gs::kNoSource) s.dma_source = from + 4 * (i + 3);
                Qword(s, k, path, u64(words[i]) | u64(words[i + 1]) << 32, u64(words[i + 2]) | u64(words[i + 3]) << 32);
                i += 4;
                continue;
            }
            if (from != Gs::kNoSource) s.dma_source = from + 4 * i;
            Word(s, k, path, words[i++]);
        }
    }

    /* PACKED quadwords from words[i] on, as Qword would take them, until the
       tag's loops end or the words do (a whole quadword left at least): each
       register's descriptor from the tag once, the place in the loop in
       locals (a quadword's descriptor shifted out of the tag, and the tag's
       fields read again, each time were most of the decoding). Returns
       where it stopped. */
    static u32 PackedRun(GifState& s, Sink& k, GifState::Path& p, const u32* words, u32 count, u32 i) {
        u32 descriptors[16];
        const u32 nregs = p.nregs;
        for (u32 r = 0; r < nregs; ++r) descriptors[r] = u32(p.tag_high >> (4 * r)) & 15;
        u32 reg = p.reg, loops = p.loops;
        while (loops && count - i >= 4) {
            const u64 low = u64(words[i]) | u64(words[i + 1]) << 32;
            const u64 high = u64(words[i + 2]) | u64(words[i + 3]) << 32;
            i += 4;
            Packed(s, k, descriptors[reg], low, high);
            if (++reg == nregs) {
                reg = 0;
                --loops;
            }
        }
        p.reg = reg;
        p.loops = loops;
        return i;
    }

    static void Word(GifState& s, Sink& k, int path, u32 word) {
        GifState::Path& p = s.paths[path];
        p.pending[p.pending_count++] = word;
        if (p.pending_count < 4) return;
        p.pending_count = 0;
        Qword(s, k, path, u64(p.pending[0]) | u64(p.pending[1]) << 32, u64(p.pending[2]) | u64(p.pending[3]) << 32);
    }

    static void Qword(GifState& s, Sink& k, int path, u64 low, u64 high) {
        GifState::Path& p = s.paths[path];
        if (!p.loops) {
            // A tag.
            p.tag_low = low;
            p.tag_high = high;
            p.loops = Bits(low, 0, 15);
            p.format = Bits(low, 58, 2);
            p.nregs = Bits(low, 60, 4);
            if (!p.nregs) p.nregs = 16;
            p.reg = 0;
            s.q = 1.0f;
            ++s.tags[path];
            if (k.Tracing()) {
                char line[160];
                std::snprintf(line, sizeof line, "[gs] tag path %d nloop %u eop %u pre %u prim %X flg %u nreg %u regs %016llX\n",
                              path, p.loops, Bits(low, 15, 1), Bits(low, 46, 1), Bits(low, 47, 11), p.format,
                              p.nregs, (unsigned long long)high);
                k.TraceLine(line);
            }
            if (Bits(low, 46, 1) && p.format == 0) Register(s, k, 0x00, Bits(low, 47, 11));
            return;
        }
        switch (p.format) {
            case 0:                                         // PACKED
                Packed(s, k, u32(p.tag_high >> (4 * p.reg)) & 15, low, high);
                if (++p.reg == p.nregs) { p.reg = 0; --p.loops; }
                break;
            case 1:                                         // REGLIST
                for (u64 value : {low, high}) {
                    if (!p.loops) break;
                    const u32 descriptor = u32(p.tag_high >> (4 * p.reg)) & 15;
                    if (descriptor != 0xE && descriptor != 0xF) Register(s, k, descriptor, value);
                    if (++p.reg == p.nregs) { p.reg = 0; --p.loops; }
                }
                break;
            default: {                                      // IMAGE
                ++s.image_qwords;
                u8 bytes[16];
                std::memcpy(bytes, &low, 8);
                std::memcpy(bytes + 8, &high, 8);
                k.Image(bytes, 1);
                --p.loops;
                break;
            }
        }
    }

    static void Packed(GifState& s, Sink& k, u32 descriptor, u64 low, u64 high) {
        switch (descriptor) {
            case 0x0: Register(s, k, 0x00, low & 0x7FF); break;
            case 0x1: {                                     // RGBA, Q from ST
                s.rgba_source = s.dma_source;
                u32 qbits;
                std::memcpy(&qbits, &s.q, 4);
                s.rgbaq = Bits(low, 0, 8) | Bits(low, 32, 8) << 8 | u64(Bits(high, 0, 8)) << 16 |
                          u64(Bits(high, 32, 8)) << 24 | u64(qbits) << 32;
                break;
            }
            case 0x2: s.st = low; s.q = AsFloat(u32(high)); break;   // Q waits for the next RGBA
            case 0x3: s.uv = Bits(low, 0, 14) | u64(Bits(low, 32, 14)) << 16; break;
            case 0x4: case 0xC: {                           // XYZF2 / XYZF3
                const u64 value = Bits(low, 0, 16) | u64(Bits(low, 32, 16)) << 16 |
                                  u64(Bits(high, 4, 24)) << 32 | u64(Bits(high, 36, 8)) << 56;
                const bool adc = descriptor == 0xC || Bits(high, 47, 1);
                Register(s, k, adc ? 0x0C : 0x04, value);
                break;
            }
            case 0x5: case 0xD: {                           // XYZ2 / XYZ3
                const u64 value = Bits(low, 0, 16) | u64(Bits(low, 32, 16)) << 16 | u64(u32(high)) << 32;
                const bool adc = descriptor == 0xD || Bits(high, 47, 1);
                Register(s, k, adc ? 0x0D : 0x05, value);
                break;
            }
            case 0xA: Register(s, k, 0x0A, u64(Bits(high, 36, 8)) << 56); break;  // FOG
            case 0xE: Register(s, k, u32(high & 0xFF), low); break;               // A+D
            case 0xF: break;                                                    // NOP
            default: Register(s, k, descriptor, low); break; // TEX0, CLAMP
        }
    }

    /* A register write: the decoding's own registers kept here, a vertex
       made, everything else (PRIM, XYOFFSET and PRMODE/PRMODECONT here too)
       handed on. */
    static void Register(GifState& s, Sink& k, u32 address, u64 value) {
        ++s.register_writes[address & 0xFF];
        switch (address) {
            case 0x00: s.prim = value; k.Apply(address, value); break;
            case 0x01: s.rgbaq = value; s.rgba_source = g_dma_source; break;
            case 0x02: s.st = value; break;
            case 0x03: s.uv = value; break;
            case 0x0A: s.fog = u8(value >> 56); break;
            case 0x04: case 0x05: case 0x0C: case 0x0D: {
                // A vertex: position from the register, the rest latched.
                if (k.Tracing()) s.xyz_source = s.dma_source ? s.dma_source : g_dma_source;
                const u64 prim = (s.prmodecont & 1) ? s.prim : ((s.prim & 7) | (s.prmode & ~7ull));
                const u64 xyoffset = s.xyoffset[(prim >> 9) & 1];
                Vertex v;
                const bool f = address == 0x04 || address == 0x0C;
                v.x = (float(Bits(value, 0, 16)) - float(Bits(xyoffset, 0, 16))) / 16.0f;
                // Field mode: a game draws one field half a pixel lower than the
                // other (sceGsSetHalfOffset), for a television to scan between
                // the lines. Shown a field at a time on a progressive screen that
                // shook the picture up and down; drawn on the whole pixel, both
                // fields stand still (PS2HOST_HALF_OFFSET=1 keeps the console's).
                static const bool half_offset = std::getenv("PS2HOST_HALF_OFFSET") != nullptr;
                const bool field_mode = Bits(s.smode2, 0, 1) && Bits(s.smode2, 1, 1);
                const u32 ofy = field_mode && !half_offset ? Bits(xyoffset, 32, 16) & ~15u : Bits(xyoffset, 32, 16);
                v.y = (float(Bits(value, 16, 16)) - float(ofy)) / 16.0f;
                v.z = f ? Bits(value, 32, 24) : u32(value >> 32);
                v.fog = f ? u8(value >> 56) : s.fog;
                v.r = u8(s.rgbaq); v.g = u8(s.rgbaq >> 8); v.b = u8(s.rgbaq >> 16); v.a = u8(s.rgbaq >> 24);
                v.q = AsFloat(u32(s.rgbaq >> 32));
                v.s = AsFloat(u32(s.st));
                v.t = AsFloat(u32(s.st >> 32));
                v.u = Bits(s.uv, 0, 14);
                v.v = Bits(s.uv, 16, 14);
                k.PushVertex(v, address == 0x04 || address == 0x05);
                break;
            }
            case 0x18: case 0x19:
                if (k.Tracing() && value != s.xyoffset[address - 0x18]) {
                    char line[64];
                    std::snprintf(line, sizeof line, "[gs] xyoffset_%u (%u, %u)\n", unsigned(address - 0x17),
                                  unsigned(value & 0xFFFF) / 16, unsigned((value >> 32) & 0xFFFF) / 16);
                    k.TraceLine(line);
                }
                s.xyoffset[address - 0x18] = value;
                k.Apply(address, value);
                break;
            case 0x1A: s.prmodecont = value; k.Apply(address, value); break;
            case 0x1B: s.prmode = value; k.Apply(address, value); break;
            default: k.Apply(address, value); break;
        }
    }

    static void Privileged(GifState& s, Sink& k, u32 address, u64 value) {
        k.RecordPrivileged(address, value);
        if ((address & 0xFFFF) == 0x0020) s.smode2 = value;
        k.ApplyPrivileged(address, value);
    }
};

}  // namespace gs
}  // namespace ps2host

#endif  /* PS2HOST_GS_DECODE_H */
