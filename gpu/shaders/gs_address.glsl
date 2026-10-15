// GS local memory's layout: a pixel's byte address (GS User's Manual,
// chapter 8; the tables are engine/host/src/gs_memory.cpp's). The including
// shader declares push constants `p` with bp (base, 256-byte blocks), bw
// (width, 64-pixel units) and psm.
const uint PSMCT32 = 0x00u, PSMCT24 = 0x01u, PSMCT16 = 0x02u, PSMCT16S = 0x0Au;
const uint PSMZ32 = 0x30u, PSMZ24 = 0x31u, PSMZ16 = 0x32u, PSMZ16S = 0x3Au;

const uint kBlock32[32] = uint[](0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23,
                                 8, 9, 12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31);
const uint kBlockZ32[32] = uint[](24, 25, 28, 29, 8, 9, 12, 13, 26, 27, 30, 31, 10, 11, 14, 15,
                                  16, 17, 20, 21, 0, 1, 4, 5, 18, 19, 22, 23, 2, 3, 6, 7);
const uint kBlock16[32] = uint[](0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15,
                                 16, 18, 24, 26, 17, 19, 25, 27, 20, 22, 28, 30, 21, 23, 29, 31);
const uint kBlock16S[32] = uint[](0, 2, 16, 18, 1, 3, 17, 19, 8, 10, 24, 26, 9, 11, 25, 27,
                                  4, 6, 20, 22, 5, 7, 21, 23, 12, 14, 28, 30, 13, 15, 29, 31);
const uint kBlockZ16[32] = uint[](24, 26, 16, 18, 25, 27, 17, 19, 28, 30, 20, 22, 29, 31, 21, 23,
                                  8, 10, 0, 2, 9, 11, 1, 3, 12, 14, 4, 6, 13, 15, 5, 7);
const uint kBlockZ16S[32] = uint[](24, 26, 8, 10, 25, 27, 9, 11, 16, 18, 0, 2, 17, 19, 1, 3,
                                   28, 30, 12, 14, 29, 31, 13, 15, 20, 22, 4, 6, 21, 23, 5, 7);
const uint kColumn32[64] = uint[](0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                  16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31,
                                  32, 33, 36, 37, 40, 41, 44, 45, 34, 35, 38, 39, 42, 43, 46, 47,
                                  48, 49, 52, 53, 56, 57, 60, 61, 50, 51, 54, 55, 58, 59, 62, 63);
const uint kColumn16[128] = uint[](
    0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27,
    4, 6, 12, 14, 20, 22, 28, 30, 5, 7, 13, 15, 21, 23, 29, 31,
    32, 34, 40, 42, 48, 50, 56, 58, 33, 35, 41, 43, 49, 51, 57, 59,
    36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63,
    64, 66, 72, 74, 80, 82, 88, 90, 65, 67, 73, 75, 81, 83, 89, 91,
    68, 70, 76, 78, 84, 86, 92, 94, 69, 71, 77, 79, 85, 87, 93, 95,
    96, 98, 104, 106, 112, 114, 120, 122, 97, 99, 105, 107, 113, 115, 121, 123,
    100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127);

const uint kMask = (4u << 20) - 1u;

uint Page(uint px, uint py) { return p.bp + 32u * (py * max(p.bw, 1u) + px); }

// Generic callers retain the push-constant format. Common image transfers
// specialize it at pipeline creation, allowing the driver to remove branches.
layout(constant_id = 0) const uint specialized_format = 0xFFFFFFFFu;
uint Format() { return specialized_format == 0xFFFFFFFFu ? p.psm : specialized_format; }
bool Is16() { return Format() == PSMCT16 || Format() == PSMCT16S || Format() == PSMZ16 || Format() == PSMZ16S; }

// The byte address of pixel (x, y).
uint Address(uint x, uint y) {
    if (Is16()) {
        // These tables are bit permutations. Express them directly rather
        // than dynamically indexing shader-local constant arrays per pixel.
        const uint bx = (x >> 4u) & 3u, by = (y >> 3u) & 7u;
        const bool s = Format() == PSMCT16S || Format() == PSMZ16S;
        uint b = ((bx & 1u) << 1u) | (by & 1u);
        b |= s ? ((bx & 2u) << 3u) | ((by & 2u) << 2u) | (by & 4u)
               : ((bx & 2u) << 2u) | ((by & 2u) << 1u) | ((by & 4u) << 2u);
        if (Format() == PSMZ16 || Format() == PSMZ16S) b ^= 24u;
        const uint block = Page(x / 64u, y / 64u) + b;
        const uint column = ((x & 1u) << 1u) | ((x & 6u) << 2u) | ((x & 8u) >> 3u) |
                            ((y & 1u) << 2u) | ((y & 6u) << 4u);
        return (block * 256u + column * 2u) & kMask;
    }
    const uint bx = (x >> 3u) & 7u, by = (y >> 3u) & 3u;
    uint b = (bx & 1u) | ((bx & 2u) << 1u) | ((bx & 4u) << 2u) | ((by & 1u) << 1u) |
             ((by & 2u) << 2u);
    if (Format() == PSMZ32 || Format() == PSMZ24) b ^= 24u;
    const uint block = Page(x / 64u, y / 32u) + b;
    const uint column = (x & 1u) | ((x & 6u) << 1u) | ((y & 1u) << 1u) | ((y & 6u) << 3u);
    return (block * 256u + column * 4u) & kMask;
}
