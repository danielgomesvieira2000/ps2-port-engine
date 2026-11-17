// What the native programs' vertex shaders share (gs_mesh.vert, gs_mesh_chars.vert): the VU's arithmetic -
// float rounded toward zero, made from the GPU's nearest and its exact error - FTOI, and the UNORM8
// attribute's values (gs.vert's colour and fog).

// c / 255 for each c, rounded to the nearest, as a UNORM8 attribute is (gs.vert's colour and fog)
const uint kUnorm8[256] = uint[](
    0u, 998277249u, 1006665857u, 1010876609u, 1015054465u, 1017159841u, 1019265217u, 1021370593u,
    1023443073u, 1024495761u, 1025548449u, 1026601137u, 1027653825u, 1028706513u, 1029759201u, 1030811889u,
    1031831681u, 1032358025u, 1032884369u, 1033410713u, 1033937057u, 1034463401u, 1034989745u, 1035516089u,
    1036042433u, 1036568777u, 1037095121u, 1037621465u, 1038147809u, 1038674153u, 1039200497u, 1039726841u,
    1040220289u, 1040483461u, 1040746633u, 1041009805u, 1041272977u, 1041536149u, 1041799321u, 1042062493u,
    1042325665u, 1042588837u, 1042852009u, 1043115181u, 1043378353u, 1043641525u, 1043904697u, 1044167869u,
    1044431041u, 1044694213u, 1044957385u, 1045220557u, 1045483729u, 1045746901u, 1046010073u, 1046273245u,
    1046536417u, 1046799589u, 1047062761u, 1047325933u, 1047589105u, 1047852277u, 1048115449u, 1048378621u,
    1048608897u, 1048740483u, 1048872069u, 1049003655u, 1049135241u, 1049266827u, 1049398413u, 1049529999u,
    1049661585u, 1049793171u, 1049924757u, 1050056343u, 1050187929u, 1050319515u, 1050451101u, 1050582687u,
    1050714273u, 1050845859u, 1050977445u, 1051109031u, 1051240617u, 1051372203u, 1051503789u, 1051635375u,
    1051766961u, 1051898547u, 1052030133u, 1052161719u, 1052293305u, 1052424891u, 1052556477u, 1052688063u,
    1052819649u, 1052951235u, 1053082821u, 1053214407u, 1053345993u, 1053477579u, 1053609165u, 1053740751u,
    1053872337u, 1054003923u, 1054135509u, 1054267095u, 1054398681u, 1054530267u, 1054661853u, 1054793439u,
    1054925025u, 1055056611u, 1055188197u, 1055319783u, 1055451369u, 1055582955u, 1055714541u, 1055846127u,
    1055977713u, 1056109299u, 1056240885u, 1056372471u, 1056504057u, 1056635643u, 1056767229u, 1056898815u,
    1056997505u, 1057063298u, 1057129091u, 1057194884u, 1057260677u, 1057326470u, 1057392263u, 1057458056u,
    1057523849u, 1057589642u, 1057655435u, 1057721228u, 1057787021u, 1057852814u, 1057918607u, 1057984400u,
    1058050193u, 1058115986u, 1058181779u, 1058247572u, 1058313365u, 1058379158u, 1058444951u, 1058510744u,
    1058576537u, 1058642330u, 1058708123u, 1058773916u, 1058839709u, 1058905502u, 1058971295u, 1059037088u,
    1059102881u, 1059168674u, 1059234467u, 1059300260u, 1059366053u, 1059431846u, 1059497639u, 1059563432u,
    1059629225u, 1059695018u, 1059760811u, 1059826604u, 1059892397u, 1059958190u, 1060023983u, 1060089776u,
    1060155569u, 1060221362u, 1060287155u, 1060352948u, 1060418741u, 1060484534u, 1060550327u, 1060616120u,
    1060681913u, 1060747706u, 1060813499u, 1060879292u, 1060945085u, 1061010878u, 1061076671u, 1061142464u,
    1061208257u, 1061274050u, 1061339843u, 1061405636u, 1061471429u, 1061537222u, 1061603015u, 1061668808u,
    1061734601u, 1061800394u, 1061866187u, 1061931980u, 1061997773u, 1062063566u, 1062129359u, 1062195152u,
    1062260945u, 1062326738u, 1062392531u, 1062458324u, 1062524117u, 1062589910u, 1062655703u, 1062721496u,
    1062787289u, 1062853082u, 1062918875u, 1062984668u, 1063050461u, 1063116254u, 1063182047u, 1063247840u,
    1063313633u, 1063379426u, 1063445219u, 1063511012u, 1063576805u, 1063642598u, 1063708391u, 1063774184u,
    1063839977u, 1063905770u, 1063971563u, 1064037356u, 1064103149u, 1064168942u, 1064234735u, 1064300528u,
    1064366321u, 1064432114u, 1064497907u, 1064563700u, 1064629493u, 1064695286u, 1064761079u, 1064826872u,
    1064892665u, 1064958458u, 1065024251u, 1065090044u, 1065155837u, 1065221630u, 1065287423u, 1065353216u);
float Unorm8(uint c) { return uintBitsToFloat(kUnorm8[c & 0xFFu]); }

// FTOI: toward zero, saturated.
int Ftoi(float y) {
    if (y >= 2147483648.0) return 2147483647;
    if (y < -2147483648.0) return -2147483647 - 1;
    return int(y);
}

// The VU rounds toward zero (the emulated VU's arithmetic, under VuRounding:
// what the recorded pictures hold), the GPU to the nearest: each operation
// made toward zero from the nearest and its exact error (a product's by fma,
// a sum's by TwoSum) - one step toward zero where the nearest went past.
// Exact for normal values (the native programs decline the rest).
const float kMax = uintBitsToFloat(0x7F7FFFFFu);
float TowardZero(float x) { return uintBitsToFloat(floatBitsToUint(x) - 1u); }     // (x not 0: its magnitude one less)
float AwayFromZero(float x) { return uintBitsToFloat(floatBitsToUint(x) + 1u); }
float Fix(float r, float e) {
    // r the nearest, e the exact value's excess over it: past it (the other sign) - one toward zero
    return (e != 0.0 && r != 0.0 && ((e < 0.0) != (r < 0.0))) ? TowardZero(r) : r;
}
#ifdef NO_FMA
// Direct3D 11's shaders have no fused multiply-add for floats (fxc's mad may or may not fuse): a product's exact
// error by Dekker's - each factor split in two halves (Veltkamp's, 2^12 + 1), their products exact - and each
// residual below so arranged that its sign, all a decision reads, is the exact value's, as fma's single rounding
// keeps it. The same results as fma's (every scripted run as recorded with it: PS2HOST_MESH_NO_FMA=1).
void Split(float a, out float hi, out float lo) {
    precise float c = 4097.0 * a;
    precise float h = c - (c - a);
    hi = isinf(c) ? a : h;
    lo = isinf(c) ? 0.0 : a - h;
}
// a * b = r + e exactly (r the nearest).
float ProductError(float a, float b, float r) {
    float ah, al, bh, bl;
    Split(a, ah, al);
    Split(b, bh, bl);
    precise float e = ((ah * bh - r) + ah * bl + al * bh) + al * bl;
    return e;
}
// 1 - a * b, its sign the exact value's (a * b near 1: 1 - p exact, Sterbenz's).
float OneLess(float a, float b) {
    precise float p = a * b;
    precise float pe = ProductError(a, b, p);
    precise float d = (1.0 - p) - pe;
    return d;
}
#endif
float Mul(float a, float b) {
    precise float r = a * b;
#ifdef NO_FMA
    precise float e = ProductError(a, b, r);
#else
    precise float e = fma(a, b, -r);
#endif
    return Fix(r, e);
}
float Add(float a, float b) {
    precise float r = a + b;
    precise float bb = r - a;
    precise float e = (a - (r - bb)) + (b - bb);
    return Fix(r, e);
}
float Sub(float a, float b) { return Add(a, -b); }
// 1/x toward zero (the VU's DIV): the largest magnitude whose product with x is not past 1.
float Recip(float x) {
    const float ax = abs(x);
    if (ax == 0.0) return x < 0.0 || floatBitsToUint(x) == 0x80000000u ? -kMax : kMax;   // (the VU's: the largest)
    precise float q = 1.0 / ax;
    if (isinf(q) || q == 0.0) return x < 0.0 ? -q : q;
#ifdef NO_FMA
    precise float e0 = OneLess(ax, q);
    precise float q1 = q * e0 + q;
#else
    precise float e0 = fma(-ax, q, 1.0);
    precise float q1 = fma(q, e0, q);
#endif
    q = q1;
    for (int i = 0; i < 2; ++i) {
#ifdef NO_FMA
        precise float r = OneLess(q, ax);
#else
        precise float r = fma(-q, ax, 1.0);
#endif
        if (r < 0.0) q = TowardZero(q);
    }
    precise float up = AwayFromZero(q);
#ifdef NO_FMA
    precise float ru = OneLess(up, ax);
#else
    precise float ru = fma(-up, ax, 1.0);
#endif
    if (ru >= 0.0) q = up;
    return x < 0.0 ? -q : q;
}
// 1/sqrt(x) toward zero (the VU's RSQRT: the double's 1/sqrt truncated): the largest y with y*y*x not past 1.
bool Past(float y, float x) {
    // y*y*x > 1, near enough exactly: y*y = p + pe (exact), then p*x - 1 and pe*x
    precise float p = y * y;
#ifdef NO_FMA
    // (t: p*x - 1 rounded once, as fma's - s - 1 exact, s near 1; then pe*x + t's sign, the exact sum's)
    precise float pe = ProductError(y, y, p);
    precise float s = p * x;
    precise float se = ProductError(p, x, s);
    precise float t = (s - 1.0) + se;
    precise float q = pe * x;
    precise float qe = ProductError(pe, x, q);
    precise float u = t + q;
    precise float ub = u - t;
    precise float ue = (t - (u - ub)) + (q - ub);
    precise float total = u + (ue + qe);
#else
    precise float pe = fma(y, y, -p);
    precise float t = fma(p, x, -1.0);
    precise float total = fma(pe, x, t);
#endif
    return total > 0.0;
}
float Rsqrt(float x) {
    if (x == 0.0) return kMax;      // (the VU's: the largest)
    precise float y = inversesqrt(x);
    if (isinf(y) || y == 0.0) return y;
    // (a Newton step: the GPU's is approximate)
    precise float h = 0.5 * x * y;
#ifdef NO_FMA
    precise float n = 1.5 - h * y;
#else
    precise float n = fma(-h, y, 1.5);
#endif
    y = y * n;
    for (int i = 0; i < 2; ++i)
        if (Past(y, x)) y = TowardZero(y);
    if (!Past(AwayFromZero(y), x)) y = AwayFromZero(y);
    return y;
}

// MAX / MINI: the VU's order, sign and magnitude (-0 below +0, as integers) - (from gs_mesh_rw.glsl: the generated
// functions, engine/tools/vutrans, use them in every native mesh shader)
int VuOrder(float f) {
    const uint b = floatBitsToUint(f);
    const int m = int(b & 0x7FFFFFFFu);
    return (b >> 31) != 0u ? -m : m;
}
float VuMax(float a, float b) { return VuOrder(a) < VuOrder(b) ? b : a; }
float VuMin(float a, float b) { return VuOrder(b) < VuOrder(a) ? b : a; }
