// The GS's Z (unsigned, 32 bits; larger is nearer) as the depth images hold
// it (a float), and back - for every shader that writes or reads them (the
// renderer's DepthOfZ / ZOfDepth are the same, on the CPU).
//
// A float has 24 bits: Z / 2^32 told Z apart to 1 below 2^24 (Z16, Z24, and
// Outbreak's 3D in Z32, under 2^20), but only to 256 near the top - where 2D
// overlays put theirs. Sony's network configuration draws its text at Z
// FFFFFFE0, then its boxes at FFFFFFD0 tested GEQUAL against it: both were
// 1.0, the boxes passed, and covered every letter. So: Z / 2^33 below
// kDepthTop - the same comparisons as Z / 2^32 (a power of two: the same
// rounding, halved) - and the 2^23 Z from kDepthTop up each its own float in
// the free upper half, 2^-24 apart. Each piece is a straight line, so a
// primitive inside one interpolates its Z as the GS does; only one across
// FF800000 (within 0.2% of the top) is bent.
const uint kDepthTop = 0xFF800000u;
const float kDepthTopAt = 0.4990234375;         // kDepthTop / 2^33 (0.5 - 2^-10)

float DepthOfZ(uint z) {
    return z < kDepthTop ? float(z) * (1.0 / 8589934592.0)
                         : kDepthTopAt + float(z - kDepthTop) * (1.0 / 16777216.0);
}

// Back to Z: truncated below kDepthTop (as Z / 2^32 was), the nearest above.
uint ZOfDepth(float d) {
    if (d < kDepthTopAt) return uint(max(d, 0.0) * 8589934592.0);
    return kDepthTop + uint(min((d - kDepthTopAt) * 16777216.0 + 0.5, 8388607.0));
}

// ...as a float (distances: the effects, soft particles).
float ZOfDepthF(float d) {
    return d < kDepthTopAt ? d * 8589934592.0 : float(kDepthTop) + (d - kDepthTopAt) * 16777216.0;
}
