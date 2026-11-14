"""Emits a generalized per-vertex function (generalize.py's fit) as Python (an exact evaluator, checked by
translate.py against the emulator's kicks) and as GLSL (gs_mesh_common.glsl's Mul / Add / Sub / Recip / Rsqrt /
Ftoi and gs_mesh_rw.glsl's SqrtTZ / ItofTZ / Itof12 / VuMax / VuMin; helpers it lacks are emitted as stubs
and listed).

Function interface (both languages): vertex i of a batch; M(q, lane) the batch's data memory words (GLSL:
mesh.quad[q][lane], the batch's VU memory image) with `base` the batch's vertex base qword; U[j] the batch's
uniform words in slot order (the CPU's header for the batch); the result the vertex's significant output words
in slice.significant order (fit["names"]).
"""
SIMPLE = {"add": "f_add", "sub": "f_sub", "mul": "f_mul", "max": "f_max", "mini": "f_mini", "div": "f_div",
          "rsqrt": "f_rsqrt"}
ONE = 0x3F800000


def _leaf_py(r, j, uidx):
    if r[0] == "U":
        return "U[%d]" % uidx[j]
    if r[0] == "V":
        _, stride, rel, lane = r
        return "M(base + %d * i + %d, %d)" % (stride, rel, lane)
    if r[0] == "G4":
        _, B, rel = r
        return "M(base + %d + %d * (i // 4), i %% 4)" % (rel, B)
    if r[0] == "P":
        _, p, slope, table = r
        rels = [a if a is not None else 0 for a, _ in table]
        lanes = [l if l is not None else 0 for _, l in table]
        return "M(base + %d * i + %r[i %% %d], %r[i %% %d])" % (slope, tuple(rels), p, tuple(lanes), p)
    raise ValueError(r)


def _uidx(fit):
    out, k = {}, 0
    for j, r in enumerate(fit["roles"]):
        if r[0] == "U":
            out[j] = k
            k += 1
    return out


def _ref_py(a):
    return "0x%s" % a[1:] if a[0] == "c" else a


def python_function(name, fit):
    uidx = _uidx(fit)
    lines = ["def %s(i, M, base, U):" % name,
             '    """outputs: %s"""' % " ".join(fit["names"])]
    for j, r in enumerate(fit["abstract"]):
        lines.append("    L%d = %s" % (j, _leaf_py(r, j, uidx)))
    for t, op, attr, args in fit["prog"]:
        a = [_ref_py(x) for x in args]
        if op in SIMPLE:
            e = "%s(%s)" % (SIMPLE[op], ", ".join(a))
        elif op == "ftoi":
            e = "f_ftoi(%s, %d)" % (a[0], attr)
        elif op == "itof":
            e = "f_itof(%s, %d)" % (a[0], attr)
        elif op == "abs":
            e = "f_abs(%s)" % a[0]
        elif op == "g":
            e = a[0]
        else:
            e = "OPS[%r](%r, %s)" % (op, attr, ", ".join(a))
        lines.append("    %s = %s" % (t, e))
    lines.append("    return [%s]" % ", ".join(_ref_py(o) for o in fit["outs"]))
    return "\n".join(lines) + "\n"


def python_module(image, funcs):
    head = ['"""Per-vertex functions of VU1 image %08X, derived by engine/tools/vutrans (generated)."""' % image,
            "import os", "import sys",
            "sys.path.insert(0, %r)" % __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
            "from vu1sym import OPS, f_add, f_sub, f_mul, f_max, f_mini, f_div, f_rsqrt, f_ftoi, f_itof, "
            "f_abs  # noqa: E402,F401", "", ""]
    return "\n".join(head) + "\n\n".join(python_function(n, f) for n, f in funcs)


# ---------------------------------------------------------------- GLSL

KNOWN = {"Mul", "Add", "Sub", "Recip", "Rsqrt", "Ftoi", "SqrtTZ", "ItofTZ", "Itof12", "VuMax", "VuMin"}
STUBS = {
    "VuDiv": "// MISSING: DIV toward zero with a numerator other than 1.0 (Recip(d) times n is two roundings)\n"
             "float VuDiv(float n, float d) { return n == 0.0 ? 0.0 : (d == 0.0 ? kMax : n / d); }\n",
    "VuRsqrtDiv": "// MISSING: RSQRT n / sqrt(|d|) toward zero with n other than 1.0\n"
                  "float VuRsqrtDiv(float n, float d) { return n * Rsqrt(abs(d)); }\n",
    "VuClipJudge": "// MISSING (gs_mesh_rw.vert has a local Judge): CLIP's 6-bit judgement of x, y, z against |w|\n"
                   "uint VuClipJudge(float x, float y, float z, float w) {\n"
                   "    const float aw = abs(w);\n    uint j = 0u;\n"
                   "    if (x > aw) j |= 1u; if (x < -aw) j |= 2u;\n    if (y > aw) j |= 4u; if (y < -aw) j |= 8u;\n"
                   "    if (z > aw) j |= 16u; if (z < -aw) j |= 32u;\n    return j;\n}\n",
    "VuMacFlags": "// MISSING: MAC flags (zero bits 3-0, sign bits 7-4, x highest) of an FMAC's written lanes\n"
                  "uint VuMacFlag(float r, uint lane) {\n"
                  "    const uint b = floatBitsToUint(r);\n"
                  "    return (((b & 0x7FFFFFFFu) == 0u) ? (8u >> lane) : 0u) | (((b >> 31) != 0u) ? (0x80u >> lane) "
                  ": 0u);\n}\n",
    "VuItof": "// MISSING: ITOF4 / ITOF15 (ItofTZ then an exact power-of-two scale, as Itof12)\n"
              "float VuItof(uint i, float scale) { return ItofTZ(int(i)) / scale; }\n",
}


def _glsl_function(name, fit, used):
    uidx = _uidx(fit)
    out = ["// outputs: %s" % " ".join(fit["names"]),
           "void %s(uint i, uint base, out uint o[%d]) {" % (name, len(fit["outs"]))]
    typ = {}
    expr = {}

    def F(r):
        if r[0] == "c":
            return "uintBitsToFloat(0x%su)" % r[1:]
        return r if typ[r] == "f" else "uintBitsToFloat(%s)" % r

    def Ub(r):
        if r[0] == "c":
            return "0x%su" % r[1:]
        return r if typ[r] == "u" else "floatBitsToUint(%s)" % r

    for j, r in enumerate(fit["abstract"]):
        if r[0] == "U":
            e = "VuU(%du)" % uidx[j]
        elif r[0] == "V":
            _, stride, k, lane = r
            e = "VuM(base + %du * i %s %du, %d)" % (stride, "+" if k >= 0 else "-", abs(k), lane)
        elif r[0] == "G4":
            _, B, rel = r
            e = "VuM(base + %du + %du * (i / 4u), int(i %% 4u))" % (rel, B)
        else:
            _, p, slope, table = r
            rels = ", ".join("%du" % (a if a is not None else 0) for a, _ in table)
            lanes = ", ".join("%d" % (l if l is not None else 0) for _, l in table)
            e = "VuM(base + %du * i + uint[%d](%s)[i %% %du], int[%d](%s)[i %% %du])" % (
                slope % (1 << 32), p, rels, p, p, lanes, p)
        out.append("    const uint L%d = %s;" % (j, e))
        typ["L%d" % j] = "u"
    for t, op, attr, args in fit["prog"]:
        if op in ("add", "sub", "mul"):
            h = {"add": "Add", "sub": "Sub", "mul": "Mul"}[op]
            used.add(h)
            e, ty = "%s(%s, %s)" % (h, F(args[0]), F(args[1])), "f"
        elif op in ("max", "mini"):
            h = "VuMax" if op == "max" else "VuMin"
            used.add(h)
            e, ty = "%s(%s, %s)" % (h, F(args[0]), F(args[1])), "f"
        elif op == "abs":
            e, ty = "abs(%s)" % F(args[0]), "f"
        elif op == "ftoi":
            used.add("Ftoi")
            e, ty = "uint(Ftoi(%s * %.1f))" % (F(args[0]), float(1 << attr)), "u"
        elif op == "itof":
            if attr == 0:
                used.add("ItofTZ")
                e = "ItofTZ(int(%s))" % Ub(args[0])
            elif attr == 12:
                used.add("Itof12")
                e = "Itof12(%s)" % Ub(args[0])
            else:
                used.add("VuItof")
                e = "VuItof(%s, %.1f)" % (Ub(args[0]), float(1 << attr))
            ty = "f"
        elif op == "div":
            if args[0] == "c%08X" % ONE:
                used.add("Recip")
                e = "Recip(%s)" % F(args[1])
            else:
                used.add("VuDiv")
                e = "VuDiv(%s, %s)" % (F(args[0]), F(args[1]))
            ty = "f"
        elif op == "rsqrt":
            if args[0] == "c%08X" % ONE:
                used.add("Rsqrt")
                e = "Rsqrt(abs(%s))" % F(args[1])
            else:
                used.add("VuRsqrtDiv")
                e = "VuRsqrtDiv(%s, %s)" % (F(args[0]), F(args[1]))
            ty = "f"
        elif op in ("sqrt", "esqrt"):
            used.add("SqrtTZ")
            e, ty = "SqrtTZ(%s)" % F(args[0]), "f"
        elif op == "ersqrt":
            used.add("Rsqrt")
            e, ty = "Rsqrt(abs(%s))" % F(args[0]), "f"
        elif op in ("iadd", "isub", "iand", "ior"):
            sym = {"iadd": "+", "isub": "-", "iand": "&", "ior": "|"}[op]
            e, ty = "((%s %s %s) & 0xFFFFu)" % (Ub(args[0]), sym, Ub(args[1])), "u"
        elif op == "lo16":
            e, ty = "(%s & 0xFFFFu)" % Ub(args[0]), "u"
        elif op == "sext16":
            e, ty = "uint(int(%s << 16) >> 16)" % Ub(args[0]), "u"
        elif op == "clipj":
            used.add("VuClipJudge")
            e, ty = "VuClipJudge(%s)" % ", ".join(F(x) for x in args), "u"
        elif op == "fcand":
            terms = " | ".join("(%s & %du)" % (Ub(x), m) for x, m in zip(args, attr))
            e, ty = "((%s) != 0u ? 1u : 0u)" % terms, "u"
        elif op == "fcget":
            e, ty = "(%s | (%s << 6))" % (Ub(args[0]), Ub(args[1])), "u"
        elif op in ("fcor", "fceq"):
            win = " | ".join("(%s << %d)" % (Ub(x), 6 * k) for k, x in enumerate(args))
            if op == "fcor":
                e = "((((%s) | 0x%Xu) & 0xFFFFFFu) == 0xFFFFFFu ? 1u : 0u)" % (win, attr[1])
            else:
                e = "((%s) == 0x%Xu ? 1u : 0u)" % (win, attr)
            ty = "u"
        elif op == "mac":
            used.add("VuMacFlags")
            lanes = [l for l in range(4) if attr & (8 >> l)]
            e = " | ".join("VuMacFlag(%s, %du)" % (F(x), l) for x, l in zip(args, lanes))
            ty = "u"
        elif op == "g":
            e, ty = args[0], typ.get(args[0], "u")
        else:
            e, ty = "0u /* unsupported %s */" % op, "u"
            used.add("unsupported:" + op)
        typ[t] = ty
        expr[t] = e
        out.append("    %s %s = %s;" % ("precise float" if ty == "f" else "uint", t, e))
    for k, o in enumerate(fit["outs"]):
        out.append("    o[%d] = %s;" % (k, Ub(o)))
    out.append("}")
    return "\n".join(out) + "\n"


def glsl_module(image, funcs):
    used = set()
    bodies = [_glsl_function(n, f, used) for n, f in funcs]
    missing = sorted(x for x in used if x not in KNOWN)
    hdr = """#version 450
#extension GL_GOOGLE_include_directive : require
// VU1 image %08X's per-vertex functions, derived by engine/tools/vutrans (generated; milestone 1: not yet
// compiled or wired). A compute harness: invocation i evaluates vertex i of one batch with function `func`.
// Bindings: mesh = the batch's VU data memory qwords, header = its uniform words (slot order), result = the
// significant output words.
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) readonly buffer Mesh { uvec4 quad[]; } mesh;
layout(set = 0, binding = 1, std430) readonly buffer Header { uint u[]; } header;
layout(set = 0, binding = 2, std430) writeonly buffer Result { uint w[]; } result;
layout(push_constant) uniform Batch { uint func; uint base; uint count; uint stride_out; } batch;

#include "gs_mesh_common.glsl"
#include "gs_mesh_rw.glsl"

uint VuU(uint j) { return header.u[j]; }
uint VuM(uint q, int lane) { return mesh.quad[q & 0x3FFu][lane]; }
""" % image
    stubs = "".join(STUBS[m] for m in missing if m in STUBS)
    calls = []
    for k, (n, f) in enumerate(funcs):
        no = len(f["outs"])
        calls.append("    if (batch.func == %du) { uint o[%d]; %s(i, batch.base, o); "
                     "for (int k = 0; k < %d; ++k) result.w[i * batch.stride_out + uint(k)] = o[k]; }" %
                     (k, no, n, no))
    main = "void main() {\n    const uint i = gl_GlobalInvocationID.x;\n    if (i >= batch.count) return;\n%s\n}\n" \
        % "\n".join(calls)
    return hdr + "\n" + stubs + "\n" + "\n".join(bodies) + "\n" + main, missing
