// RenderWare's programs' arithmetic beyond gs_mesh_common.glsl's (gs_mesh_rw.vert; host's ps2host/rw_native.h
// the CPU's twin): SQRT toward zero, ITOF12, MAX's sign-and-magnitude order.

// sqrt(|x|) toward zero: the nearest, one step down where its square is past x
float SqrtTZ(float x) {
    const float ax = abs(x);
    if (ax == 0.0) return 0.0;
    precise float r = sqrt(ax);
    for (int i = 0; i < 2; ++i) {
        precise float e = fma(r, r, -ax);
        if (e > 0.0) r = TowardZero(r);
    }
    precise float up = AwayFromZero(r);
    precise float eu = fma(up, up, -ax);
    if (eu <= 0.0) r = up;
    return r;
}

// ITOF0 toward zero (the GPU's int to float rounds to the nearest)
float ItofTZ(int i) {
    float f = float(i);
    if (abs(f) >= 2147483648.0) return i < 0 ? (i == int(0x80000000u) ? -2147483648.0 : -2147483520.0) : 2147483520.0;
    const int back = int(f);
    if (i >= 0 ? back > i : back < i) f = TowardZero(f);
    return f;
}

float Itof12(uint i) { return ItofTZ(int(i)) / 4096.0; }

int VuOrder(float f) {
    const uint b = floatBitsToUint(f);
    const int m = int(b & 0x7FFFFFFFu);
    return (b >> 31) != 0u ? -m : m;
}
float VuMax(float a, float b) { return VuOrder(a) < VuOrder(b) ? b : a; }
float VuMin(float a, float b) { return VuOrder(b) < VuOrder(a) ? b : a; }
