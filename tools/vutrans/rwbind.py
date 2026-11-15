"""vutrans milestone 3 (translate.py --bind rw): the projected functions bound to RenderWare's native header
(rwheader.py), merged by what they read, checked, and written as gs_mesh_rw.vert's drop-in (--emit-glsl-rw).

Steps, per projected function (translate.project_all's):
  1. bind it on every batch it serves (rwheader.bind): its uniform part read from the header the port makes for
     that run (an instance's, a stream's) - each batch's header and mode recorded;
  2. merge the bound functions by their signature (rebuilt or carried transforms, 2678's and 01CE's world: the
     same arithmetic over the same header words);
  3. per mode (the header's misc.z: the shader's dispatch), the function serving most vertices evaluated on every
     other's vertices: bit-equal on all of them - the other is the same function (a lane variant of an unrolled
     loop, the strip's first vertices) and is dropped; else the mode keeps both and is left to the hand-written
     shader;
  4. each function checked as the shader runs it: its evaluator, header words from the header (the port's), mesh
     words from data memory, against every kicked word the consumer reads.
"""
import collections
import importlib.util
import os
import sys

import emit
import generalize
import rwheader
import slice as gifslice


def _load(src, name):
    out_py = os.path.join(os.environ.get("TEMP", "."), name + ".py")
    with open(out_py, "w", newline="\n") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, out_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, out_py


def _check(fn, fit, batch):
    """(exact, checked, first mismatch) of fn over one batch (si, head, d, p, verts, hdr values)."""
    si, head, d, p, verts, values = batch
    words = memoryview(d).cast("I")

    def M(q, lane):
        return words[(q & 0x3FF) * 4 + lane]
    U = [values[r[1][1:]] for r in fit["roles"] if r[0] == "U"]
    exact = checked = 0
    bad = None
    base = p["base"] if p["base"] is not None else 0
    for v in verts:
        out = fn(v.loop, M, base, U)
        have = {n: b for n, b, _node, _m in gifslice.significant(v)}
        want = [have[n] & mk for n, mk in zip(fit["names"], fit["out_masks"])]
        got = [o & mk for o, mk in zip(out, fit["out_masks"])]
        checked += 1
        if got == want:
            exact += 1
        elif bad is None:
            bad = (si, v.loop, ["%08X" % x for x in got], ["%08X" % x for x in want])
    return exact, checked, bad


def bind_all(a, pr, run_image, nruns, regval_head):
    pf, order, pnames = pr["funcs"], pr["order"], pr["names"]
    hcache = {}

    def hdrs_of(si, head, d):
        r = hcache.get(si)
        if r is None:
            hs = rwheader.headers(run_image[si], head, d) or []
            st = rwheader.State(head, d)
            r = []
            for mode, slots in hs:
                memo = {}
                r.append((mode, slots, {k: st.value(e, memo) for k, e in slots.items()}))
            hcache[si] = r
        return r

    bound = collections.OrderedDict()
    failed = []
    unproven = collections.Counter()
    stats = collections.Counter()
    for sig in order:
        g = pf[sig]
        batches, full = [], []
        why = None
        for si, head, d, p, verts in g["batches"]:
            if p["base"] is None or (p["base"] - p["top"]) % 1024 != 0:
                why = "vertex base not TOP"
                break
            words = memoryview(d).cast("I")
            U = [regval_head(head, r[1]) if r[1][0] != "m" else words[(r[1][1] & 0x3FF) * 4 + r[1][2]]
                 for r in p["roles"] if r[0] == "U"]
            hd = hdrs_of(si, head, d)
            if not hd:
                why = "no header (the port declines the run, or path C)"
                break
            batches.append((p, U, hd))
            full.append((si, head, d, p, verts))
        if why is None:
            b, info = rwheader.bind(g["fit"], batches, None)
            if b is None:
                why = "unbound uniforms %s" % sorted(set(info["unbound"]))[:6]
        if why is not None:
            failed.append((pnames[sig], g["vertices"], why))
            # (its vertices' modes, where the run's header says one: those modes not proven - left to the hand)
            for _p, _U, hd in batches:
                if len(hd) == 1:
                    unproven[hd[0][0]] += len(_p["idxs"]) if "idxs" in _p else 1
            continue
        stats["structural"] += info["structural"]
        stats["value"] += info["value"]
        e = bound.get(b["signature"])
        if e is None:
            e = bound[b["signature"]] = {"fit": b, "batches": [], "vertices": 0, "from": [], "modes": collections.Counter(),
                                         "images": collections.Counter(), "loops": collections.Counter(),
                                         "how": collections.Counter()}
        e["from"].append(pnames[sig])
        e.setdefault("by_value", set()).update(info.get("by_value", []))
        e["how"]["s"] += info["structural"]
        e["how"]["v"] += info["value"]
        for (si, head, d, p, verts), (pb, U, hd), mode, c in zip(full, batches, info["modes"], info["choice"]):
            e["batches"].append((si, head, d, p, verts, hd[c][2], mode))
            e["vertices"] += len(verts)
            e["modes"][mode] += len(verts)
            e["images"]["%08X" % run_image[si]] += len(verts)
            e["loops"].update(v.loop for v in verts)
    nlanes = unify_lanes(bound)
    order_b = sorted(bound, key=lambda k: -bound[k]["vertices"])
    prefix = (a.rw_name or "VuRw").replace("Generated", "")
    names = {sig: "%s_f%d" % (prefix, k) for k, sig in enumerate(order_b)}
    mod, out_py = _load(emit.python_module(images_id(run_image), [(names[s], bound[s]["fit"]) for s in order_b]),
                        "vutrans_rw_bound")
    ok = True
    for sig in order_b:
        e = bound[sig]
        fn = getattr(mod, names[sig])
        e["exact"] = e["checked"] = 0
        e["bad"] = None
        for bt in e["batches"]:
            x, c, bad = _check(fn, e["fit"], bt[:6])
            e["exact"] += x
            e["checked"] += c
            e["bad"] = e["bad"] or bad
        ok = ok and e["exact"] == e["checked"]
    # per mode: the function serving most of its vertices, the others evaluated with it
    by_mode = collections.defaultdict(list)
    for sig in order_b:
        for mode in bound[sig]["modes"]:
            by_mode[mode].append(sig)
    dispatch = {}
    merged = collections.defaultdict(list)
    conflicts = {}
    for mode, sigs in sorted(by_mode.items()):
        sigs.sort(key=lambda s: -bound[s]["modes"][mode])
        main = sigs[0]
        fn = getattr(mod, names[main])
        rest = []
        for other in sigs[1:]:
            ex = ch = 0
            bad = None
            for bt in bound[other]["batches"]:
                if rwheader_mode(bt) != mode:
                    continue
                x, c, b_ = _check_with(fn, bound[main]["fit"], bound[other]["fit"], bt)
                ex += x
                ch += c
                bad = bad or b_
            if ex == ch and ch:
                merged[main].append((names[other], mode, ch))
            else:
                rest.append((names[other], ex, ch, bad))
        if unproven.get(mode):
            rest.append(("(not bound)", 0, unproven[mode], None))
        if rest:
            conflicts[mode] = rest
        dispatch[mode] = main
    # The world's lit modes (1332's: Lit, AlphaZero) as the port draws them by default: its alphas made on the CPU
    # into the mesh and those bits cleared - the mode's vertices drawn by the function of the mode without them,
    # every word but the alpha (the mesh's): checked here too
    LIT = rwheader.kRwWLit | rwheader.kRwWAlphaZero
    for mode in sorted(by_mode):
        alias = mode & ~LIT
        if not (mode & rwheader.kRwWorld) or alias == mode or alias not in dispatch:
            continue
        main = dispatch[alias]
        fn = getattr(mod, names[main])
        fit = bound[main]["fit"]
        fit_na = dict(fit, out_masks=tuple(0 if n == "RGBAQ.w" else mk for n, mk in zip(fit["names"],
                                                                                      fit["out_masks"])))
        ex = ch = 0
        bad = None
        for s in by_mode[mode]:
            for bt in bound[s]["batches"]:
                if rwheader_mode(bt) != mode:
                    continue
                x, c, b_ = _check(fn, fit_na, bt[:6])
                ex += x
                ch += c
                bad = bad or b_
        print("mode %X as drawn by default (mode %X, the CPU's alpha): %s on %d of %d of its vertices "
              "(every word but RGBAQ.w)" % (mode, alias, names[main], ex, ch))
        if ex != ch:
            conflicts.setdefault(alias, []).append(("(mode %X's vertices)" % mode, ex, ch, bad))
    # ---- report
    print("\nbound to RenderWare's header (rwheader.py): %d functions from %d projected (%d header words matched "
          "structurally, %d by value); not bound: %d (%d vertices)" % (
              len(bound), len(order) - len(failed), stats["structural"], stats["value"], len(failed),
              sum(f[1] for f in failed)))
    print("  (functions merged with another's lanes or first vertices - one function, its mesh inputs fitted again "
          "over all their vertices: %d)" % nlanes)
    for n, nv, why in failed[:12]:
        print("  not bound %s (%d vertices): %s" % (n, nv, why))
    if len(failed) > 12:
        print("  ... %d more" % (len(failed) - 12))
    for sig in order_b:
        e = bound[sig]
        fit = e["fit"]
        nh = sum(1 for r in fit["abstract"] if r[0] == "H")
        print("  %s: %d vertices in %d batches (from %s); %d ops, %d leaves (%d header words; matched %d structurally, "
              "%d by value); images %s; modes %s; serves %s; evaluator %d of %d exact" % (
                  names[sig], e["vertices"], len(e["batches"]), " ".join(e["from"][:6]) +
                  (" ..." if len(e["from"]) > 6 else ""), len(fit["prog"]), len(fit["abstract"]), nh, e["how"]["s"],
                  e["how"]["v"], dict(e["images"]), {"%X" % k: v for k, v in e["modes"].items()},
                  selector_of(e["loops"]), e["exact"], e["checked"]))
        if e.get("lanes"):
            print("     (one function for %d whose mesh inputs differed: %s)" % (
                e["lanes"], " ".join("%s%s" % (r[0], r[1:]) for r in fit["abstract"] if r[0] in MESH)))
        if e["bad"]:
            print("     MISMATCH run %d vertex %d: %s vs %s" % e["bad"])
        if a.verbose:
            for x, row, lane in sorted(e.get("by_value", ())):
                print("     by value: header.q[%d].%s = %s" % (row, "xyzw"[lane], x))
            print("     " + fit["text"].replace("\n", "\n     "))
        for other, mode, ch in merged.get(sig, []):
            print("     = %s on mode %X: bit-equal on all its %d vertices (merged)" % (other, mode, ch))
    print("dispatch (mode -> function): %s" % ", ".join("%X -> %s" % (m_, names[s]) for m_, s in sorted(dispatch.items())))
    for mode, rest in conflicts.items():
        for n, ex, ch, bad in rest:
            print("  mode %X: %s differs from %s on %d of %d vertices (e.g. %s) - the mode left to the hand-written "
                  "shader" % (mode, n, names[dispatch[mode]], ch - ex, ch, bad))
    if a.emit_glsl_rw:
        good = {m_: s for m_, s in dispatch.items() if m_ not in conflicts and bound[s]["exact"] == bound[s]["checked"]}
        src = glsl_rw(a, bound, names, good, merged, nruns, run_image)
        with open(a.emit_glsl_rw, "w", newline="\n") as fh:
            fh.write(src)
        print("wrote %s (modes %s)" % (a.emit_glsl_rw, " ".join("%X" % m_ for m_ in sorted(good))))
    return ok


MESH = ("V", "G4", "P")


def _points(ab, loops):
    """The (vertex, ('m', quadword from TOP, lane)) a mesh input role reads at each vertex of loops."""
    out = []
    for i in loops:
        if ab[0] == "V":
            q, lane = ab[1] * i + ab[2], ab[3]
        elif ab[0] == "G4":
            q, lane = ab[2] + ab[1] * (i // 4), i % 4
        else:
            _, p, slope, table = ab
            rel, lane = table[i % p]
            if rel is None:
                continue
            q = slope * i + rel
        out.append((i, ("m", q % 1024, lane)))
    return out


def _signed(a):
    return a - 1024 if a >= 512 else a


def _unify(entries):
    """One function for entries whose bound programs are the same but for their mesh inputs' roles (an unrolled
    loop's lanes, a strip's first vertices): each mesh slot fitted again over the union of the vertices it read
    (generalize._fit_slot: a lane i % 4's V roles become one G4); None if a slot fits no role over the union."""
    ab0 = entries[0]["fit"]["abstract"]
    new = []
    for k, r in enumerate(ab0):
        if r[0] not in MESH:
            new.append(r)
            continue
        pts = {}
        for e in entries:
            for i, d in _points(e["fit"]["abstract"][k], sorted(set(e["loops"]))):
                if pts.setdefault(i, d) != d:
                    return None
        pl = sorted(pts.items())
        if len(pl) < 2:
            if all(e["fit"]["abstract"][k] == r for e in entries):
                new.append(r)
                continue
            return None
        role = generalize._fit_slot(pl, None)
        if role[0] == "V":
            new.append(("V", role[1], _signed(role[2]), role[3]))
        elif role[0] == "G4":
            new.append(("G4", role[1], role[2]))
        elif role[0] == "P":
            new.append(("P", role[1], role[2], tuple((None, None) if a is None else (_signed(a), lane)
                                                     for a, lane in role[3])))
        else:
            return None
    return tuple(new)


def unify_lanes(bound):
    """Merges (in place) the bound functions that differ only in their mesh inputs' roles (_unify), greedily from
    the one serving most vertices; returns the merges made."""
    def shape(fit):
        return (fit["names"], fit["text"], tuple(("M",) if r[0] in MESH else r for r in fit["abstract"]),
                fit["out_masks"])
    groups = collections.defaultdict(list)
    for sig, e in bound.items():
        groups[shape(e["fit"])].append(sig)
    merges = 0
    for sh, sigs in groups.items():
        if len(sigs) < 2:
            continue
        sigs.sort(key=lambda s: -bound[s]["vertices"])
        while sigs:
            head = sigs.pop(0)
            members = [head]
            ab = bound[head]["fit"]["abstract"]
            for s in list(sigs):
                got = _unify([bound[x] for x in members + [s]])
                if got is not None:
                    members.append(s)
                    sigs.remove(s)
                    ab = got
            if len(members) == 1:
                continue
            e = bound[head]
            fit = dict(e["fit"], abstract=ab)
            fit["signature"] = (fit["names"], fit["text"], ab, fit["out_masks"])
            for s in members[1:]:
                o = bound.pop(s)
                e["batches"] += o["batches"]
                e["vertices"] += o["vertices"]
                e["from"] += o["from"]
                e["modes"].update(o["modes"])
                e["images"].update(o["images"])
                e["loops"].update(o["loops"])
                e["how"].update(o["how"])
                e.setdefault("by_value", set()).update(o.get("by_value", ()))
                merges += 1
            e["fit"] = fit
            e["lanes"] = len(members)
    return merges


def rwheader_mode(bt):
    return bt[6] if len(bt) > 6 else None


def _check_with(fn, fit_main, fit_other, bt):
    """fn (fit_main's evaluator) on fit_other's batch: its header words by fit_main's leaves."""
    return _check(fn, dict(fit_main, out_masks=fit_main["out_masks"]), bt[:6])


def images_id(run_image):
    ims = sorted(set(run_image.values()))
    return ims[0] if ims else 0


def selector_of(loops):
    idx = set(loops)
    if len(idx) <= 6 and max(idx) < 8:
        return "i in {%s}" % ",".join(str(i) for i in sorted(idx))
    lo = min(idx)
    rs = set(i % 4 for i in idx)
    if len(rs) < 4:
        return "i %% 4 in {%s}, i >= %d" % (",".join(str(r) for r in sorted(rs)), lo)
    return "all i >= %d" % lo


# ---------------------------------------------------------------- GLSL

def _rw_leaf(j, role, abstract):
    """gs_mesh_rw.vert's inputs: the header's quadwords header.q[row] (vec4), the batch's mesh mesh.quad[k] (uvec4:
    VU1 data memory from TOP), v the vertex (the strip's)."""
    if abstract[0] == "H":
        return "floatBitsToUint(header.q[%d][%d])" % (abstract[1], abstract[2])
    if abstract[0] == "V":
        _, stride, rel, lane = abstract
        return "mesh.quad[%du * v %s %du][%d]" % (stride, "+" if rel >= 0 else "-", abs(rel), lane)
    if abstract[0] == "G4":
        _, B, rel = abstract
        return "mesh.quad[%du + %du * (v / 4u)][v %% 4u]" % (rel, B)
    if abstract[0] == "P":
        _, p, slope, table = abstract
        rels = ", ".join("%du" % (x if x is not None else 0) for x, _ in table)
        lanes = ", ".join("%du" % (l if l is not None else 0) for _, l in table)
        return "mesh.quad[%du * v + uint[%d](%s)[v %% %du]][uint[%d](%s)[v %% %du]]" % (
            slope % (1 << 32), p, rels, p, p, lanes, p)
    raise ValueError("gs_mesh_rw: an input it cannot read: %r" % (abstract,))


def glsl_rw(a, bound, names, dispatch, merged, nruns, run_image):
    sigs = []
    for m_, s in sorted(dispatch.items()):
        if s not in sigs:
            sigs.append(s)
    first = bound[sigs[0]]["fit"]
    onames, omasks = first["names"], first["out_masks"]
    used = set()
    bodies = []
    for s in sigs:
        fit = bound[s]["fit"]
        if fit["names"] != onames:
            raise ValueError("%s: outputs %s, not %s" % (names[s], fit["names"], onames))
        bodies.append(emit._glsl_function(names[s], fit, used, leaf=_rw_leaf, params="uint v"))
    missing = sorted(x for x in used if x not in emit.KNOWN)
    if missing:
        raise ValueError("helpers gs_mesh_common.glsl / gs_mesh_rw.glsl lack: %s" % " ".join(missing))
    fname = a.rw_name or "VuRwGenerated"
    prefix = fname.replace("Generated", "")
    consts = "\n".join("const int k%s_%s = %d;" % (prefix, n.replace(".", "_"), k) for k, n in enumerate(onames))
    lines = []
    for s in sigs:
        e = bound[s]
        lines.append("//   %s: %d vertices in %d batches (images %s; serves %s), its evaluator bit-exact on %d of %d; "
                     "modes %s" % (names[s], e["vertices"], len(e["batches"]),
                                   " ".join("%s x%d" % kv for kv in sorted(e["images"].items())),
                                   selector_of(e["loops"]), e["exact"], e["checked"],
                                   " ".join("%X" % m_ for m_, s2 in sorted(dispatch.items()) if s2 == s)))
        for other, mode, ch in merged.get(s, []):
            if mode in dispatch:
                lines.append("//     (and on mode %X the vertices of %s, %d: bit-equal - one function)" % (
                    mode, other, ch))
    cases = []
    for m_, s in sorted(dispatch.items()):
        cases.append("    case 0x%Xu: %s(v, o); return true;" % (m_, names[s]))
    cmd = "python -I engine/tools/vutrans/translate.py " + " ".join(
        os.path.basename(x) if ("/" in x or "\\" in x) else x for x in sys.argv[1:])
    head = """// GENERATED by engine/tools/vutrans (translate.py --bind rw --emit-glsl-rw) - do not edit by hand.
//
// RenderWare's VU1 images %s: their per-vertex functions derived from the snapshot %s (%d runs), projected
// (masks %s) and bound to the native header (rwheader.py: the port's vu_native_rw.cpp, as expressions of VU1's
// state) - each one's uniform part read from header.q, the rest made per vertex as the microcode makes it:
%s
//   %s
// Needs gs_mesh_rw.vert's Mesh (uvec4 quad[]: VU1 data memory from TOP) and Header (vec4 q[]), gs_mesh_common.glsl
// and gs_mesh_rw.glsl. %s(mode, v, o): vertex v's words (o[k%s_*], as the microcode kicks them: the GS reads
// their low bits) for a header of that mode; false - a mode no function serves (the hand-written arithmetic's).
""" % (" ".join(sorted(set(e for s in sigs for e in bound[s]["images"]))), os.path.basename(a.snap), nruns,
       "{%s}" % ", ".join("%s: 0x%X" % kv for kv in pr_masks(a).items()), "\n".join(lines), cmd, fname, prefix)
    disp = "bool %s(uint mode, uint v, out uint o[%d]) {\n    switch (mode) {\n%s\n    default: break;\n    }\n" \
           "    for (int k = 0; k < %d; ++k) o[k] = 0u;\n    return false;\n}\n" % (fname, len(onames), "\n".join(cases),
                                                                                len(onames))
    return head + "\n" + consts + "\n\n" + "\n".join(bodies) + "\n" + disp


def pr_masks(a):
    out = {}
    for x in a.mask:
        k, v = x.split("=")
        out[k] = int(v, 0)
    return out
