#version 450
#extension GL_GOOGLE_include_directive : require
// A native microprogram's vertices made on the GPU (docs/NATIVE_RENDERER.md,
// the GPU's way): a mesh of the game's own vertices kept in a buffer, and per
// draw the header the microprogram would have been given - the GS's values
// made here, as the microcode makes them, then on as gs.vert takes them from
// the bridge. The first such program, Outbreak's room (VU1 C43D662E; its
// reference on the CPU FloatOf in games/re_outbreak2/port/vu_native_room.cpp,
// the same arithmetic - float rounded toward zero, the emulated VU's, which
// matches the microcode word for word in every scripted run).
//
// A vertex, four quadwords: its place, normal, colour and texture coordinates.
// The header: rows 0-23 of VU1's data memory, and the program's constant
// registers vf11-vf17; the context's XYOFFSET (sixteenths) beside them.
// Which triangles: the strip's restarts (the vertices marked in their data)
// are the index buffer's; a triangle whose three vertices are all out past
// one plane of the guard band (rows 7-10: CLIP, |x| > |w| and so on - the
// microcode's ADC) is culled by the six cull distances, the same rule. The
// places stay the GS's (w 1, as gs.vert's): a triangle across a plane - the
// microcode's clipper - is not for this shader (its batch the microcode's,
// for now; later homogeneous places and the GPU's clipping).

layout(set = 3, binding = 0, std430) readonly buffer Mesh {
    vec4 quad[];
} mesh;

layout(set = 3, binding = 1, std430) readonly buffer Header {
    vec4 row[24];       // VU1 data memory 0-23
    vec4 k[7];          // vf11-vf17
    vec4 offset;        // XYOFFSET's OFX, OFY (sixteenths of a pixel)
} header;

layout(push_constant) uniform Target {
    vec2 size;          // the target, in pixels
} target;

#include "gs_depth.glsl"

out gl_PerVertex {
    vec4 gl_Position;
    float gl_CullDistance[6];
};

layout(location = 0) out vec4 shade;
layout(location = 1) noperspective out vec3 coordinate;
layout(location = 2) noperspective out float fog_out;
layout(location = 3) flat out vec4 range_out;

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
float Mul(float a, float b) {
    precise float r = a * b;
    precise float e = fma(a, b, -r);
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
    precise float e0 = fma(-ax, q, 1.0);
    precise float q1 = fma(q, e0, q);
    q = q1;
    for (int i = 0; i < 2; ++i) {
        precise float r = fma(-q, ax, 1.0);
        if (r < 0.0) q = TowardZero(q);
    }
    precise float up = AwayFromZero(q);
    precise float ru = fma(-up, ax, 1.0);
    if (ru >= 0.0) q = up;
    return x < 0.0 ? -q : q;
}
// 1/sqrt(x) toward zero (the VU's RSQRT: the double's 1/sqrt truncated): the largest y with y*y*x not past 1.
bool Past(float y, float x) {
    // y*y*x > 1, near enough exactly: y*y = p + pe (exact), then p*x - 1 and pe*x
    precise float p = y * y;
    precise float pe = fma(y, y, -p);
    precise float t = fma(p, x, -1.0);
    precise float total = fma(pe, x, t);
    return total > 0.0;
}
float Rsqrt(float x) {
    if (x == 0.0) return kMax;      // (the VU's: the largest)
    precise float y = inversesqrt(x);
    if (isinf(y) || y == 0.0) return y;
    // (a Newton step: the GPU's is approximate)
    precise float h = 0.5 * x * y;
    precise float n = fma(-h, y, 1.5);
    y = y * n;
    for (int i = 0; i < 2; ++i)
        if (Past(y, x)) y = TowardZero(y);
    if (!Past(AwayFromZero(y), x)) y = AwayFromZero(y);
    return y;
}
// A row's dot with the place, as MULA, MADDA, MADDA, MADD (w: 1) make it.
float Dot(int r, vec4 p, int lane) {
    return Add(Add(Add(Mul(header.row[r][lane], p.x), Mul(header.row[r + 1][lane], p.y)),
                   Mul(header.row[r + 2][lane], p.z)), header.row[r + 3][lane]);
}

void main() {
    const uint at = uint(gl_VertexIndex) * 4u;
    const vec4 p = mesh.quad[at];
    const vec4 nrm = mesh.quad[at + 1u];
    const vec4 col = mesh.quad[at + 2u];
    const vec4 tx = mesh.quad[at + 3u];
    // the place through rows 3-6 (the view) and 7-10 (the guard band)
    vec4 cp, g;
    for (int l = 0; l < 4; ++l) {
        cp[l] = Dot(3, p, l);
        g[l] = Dot(7, p, l);
    }
    // the guard band's six planes, as CLIP judges (|x| > |w| out)
    const float gw = abs(g.w);
    gl_CullDistance[0] = gw - g.x;
    gl_CullDistance[1] = gw + g.x;
    gl_CullDistance[2] = gw - g.y;
    gl_CullDistance[3] = gw + g.y;
    gl_CullDistance[4] = gw - g.z;
    gl_CullDistance[5] = gw + g.z;
    const float q = Recip(cp.w);
    // the texture coordinates through rows 15-17 (Q: the vertex's own z)
    const float s0 = Add(Add(Mul(header.row[15].x, tx.x), Mul(header.row[16].x, tx.y)), header.row[17].x);
    const float t0 = Add(Add(Mul(header.row[15].y, tx.x), Mul(header.row[16].y, tx.y)), header.row[17].y);
    const vec3 stq = vec3(Mul(s0, q), Mul(t0, q), Mul(tx.z, q));
    // a point light: vf15-vf17's x its place, vf12's x its strength over the distance
    const float lx = Sub(header.k[4].x, p.x), ly = Sub(header.k[5].x, p.y), lz = Sub(header.k[6].x, p.z);
    const float rs = Rsqrt(Add(Add(Mul(lx, lx), Mul(ly, ly)), Mul(lz, lz)));
    const float atten = min(Mul(header.k[1].x, rs), 1.0);
    const float nx = Mul(lx, rs), ny = Mul(ly, rs), nz = Mul(lz, rs);
    const float lambert = max(Add(Add(Mul(nx, nrm.x), Mul(ny, nrm.y)), Mul(nz, nrm.z)), 0.0);
    vec4 c;
    for (int l = 0; l < 3; ++l) {
        const float lit = Add(Mul(header.k[2][l], lambert), Mul(header.k[3][l], col[l]));
        c[l] = Add(Mul(lit, atten), Mul(header.row[0][l], col[l]));
    }
    c.w = col.w;
    for (int l = 0; l < 4; ++l) c[l] = Mul(min(c[l], 128.0), header.row[1][l]);
    const uvec4 rgba = uvec4(ivec4(c)) & 0xFFu;
    // the fog from w through vf11
    const float fog = max(min(Mul(Mul(Sub(header.k[0].x, cp.w), header.k[0].y), 255.0), 255.0), 0.0);
    const uint f = (uint(Ftoi(fog * 16.0)) >> 4) & 0xFFu;
    // X, Y, Z as XYZF2 takes them (12.4 fixed X and Y, 24-bit Z; FTOI4's 16 exact), X and Y less XYOFFSET
    const uint x = uint(Ftoi(Mul(cp.x, q) * 16.0)) & 0xFFFFu;
    const uint y = uint(Ftoi(Mul(cp.y, q) * 16.0)) & 0xFFFFu;
    const uint z = (uint(Ftoi(Mul(cp.z, q) * 16.0)) >> 4) & 0xFFFFFFu;
    const vec2 position = (vec2(float(x), float(y)) - header.offset.xy) / 16.0;
    // ...on as gs.vert
    const vec2 ndc = (position + 0.5) / target.size * 2.0 - 1.0;
    gl_Position = vec4(ndc, DepthOfZ(z), 1.0);
    shade = vec4(Unorm8(rgba.r), Unorm8(rgba.g), Unorm8(rgba.b), Unorm8(rgba.a));
    coordinate = stq;
    fog_out = Unorm8(f);
    range_out = vec4(-1e9, -1e9, 1e9, 1e9);
}
