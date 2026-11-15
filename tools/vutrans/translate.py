"""vutrans milestone 1: derive a VU1 microprogram's per-vertex function from a --vu-snapshot automatically.

    python -I engine/tools/vutrans/translate.py SNAP IMAGE [--runs N] [--skip N] [--emit-py out.py]
                                                [--emit-glsl out.glsl] [--verbose]

For each run of IMAGE: run vu1sym (concrete + symbolic), split the kicks into vertices (slice.py), fit each
tag's vertices to one function (generalize.py). Reports, per run path (the set of counter-like branch outcomes),
the functions found (stride, base relative to TOP, uniform rows, data-dependent words), the data-dependent
branches (and which words they guard), and checks every function across all batches that share it: the emitted
Python evaluator (--emit-py; else an in-memory one) must reproduce every kicked significant word of every vertex
bit for bit. --emit-glsl writes the same functions as GLSL (gs_mesh_common.glsl / gs_mesh_rw.glsl helpers).

Milestone 2 (--project, --mask, --assume, --emit-glsl-mesh, --binding): the functions as a native mesh shader reads
them (generalize.project), each checked the same way on its words as read, and the one serving most vertices
written as a drop-in for the shader (emit.glsl_bound: its inputs the shader's own buffers). Outbreak's room and
characters (engine/gpu/shaders/generated/):
    translate.py SNAP 990887E7 --runs 2000 --mask XYZF2.w=0xFF0 --assume V,4,0,3=0x7FFF --emit-glsl-mesh F
    translate.py SNAP 6BD7E3E9 --runs 2000 --mask XYZF2.w=0xFF0 --assume V,5,0,3=0x7FFF --binding gs_mesh_chars
                 --emit-glsl-mesh F
(XYZF2.w: the fog alone, ADC being the shader's index buffer and guard band; the mark word 0 or 0x8000, others
declined by the native path.)

Milestone 3 (--bind rw, --emit-glsl-rw, --no-data-branches; IMAGE a comma-separated list): a second engine's
programs - Burnout 3's RenderWare - whose native shader (gs_mesh_rw.vert) reads a header the port computes on the
CPU (transforms rebuilt, an instance's matrix): each projected function's uniform part bound to that header
(rwheader.py), merged by what it reads (rebuilt or carried transforms, images, an unrolled loop's lanes, a strip's
first vertices: rwbind.py), checked and written with a dispatch by the header's mode (engine/gpu/shaders/generated/):
    translate.py SNAP D4B61373,34C3CF67 --runs 5000 --mask XYZF2.w=0 --bind rw
                 --emit-glsl-rw vu_rw_instanced.glsl --rw-name VuRwInstancedGenerated
    translate.py SNAP 2678EB26,1332C99F,01CE30D5,0835A68B --runs 5000 --no-data-branches --mask XYZF2.w=0 --bind rw
                 --emit-glsl-rw vu_rw_world.glsl --rw-name VuRwWorldGenerated
(XYZF2.w: ADC and fog, the shader's own - the strip's triangles its indices; --no-data-branches: the world's paths A
and B, path C's clipper the CPU's.)
"""
import argparse
import collections
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vu1sym  # noqa: E402
import slice as gifslice  # noqa: E402
import generalize  # noqa: E402
import emit  # noqa: E402


def path_key(branches):
    """A run's path: the counter-like branch outcomes it took (as a set: loop trip counts drop out)."""
    return tuple(sorted(set((pc, bool(t)) for pc, t, _n, kind in branches if kind == "counter")))


def describe_cond(dag, n):
    return dag.fmt(n, depth=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap")
    ap.add_argument("image")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--emit-py")
    ap.add_argument("--emit-glsl")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--engine-moves", action="store_true",
                    help="MOVE/MR32/MFIR timing as vu.cpp (no write when unchanged); default: as vu1emu")
    ap.add_argument("--mask", action="append", default=[], metavar="WORD=BITS",
                    help="project: the bits of an output word the consumer reads (0: dropped), e.g. XYZF2.w=0xFF0")
    ap.add_argument("--assume", action="append", default=[], metavar="ROLE=BITS",
                    help="project: an input zero in BITS (a precondition the consumer checks), ROLE an abstract "
                         "vertex role 'V,stride,rel,lane', e.g. V,4,0,3=0x7FFF")
    ap.add_argument("--project", action="store_true",
                    help="also report the functions as the consumer reads them (g dropped, --mask / --assume)")
    ap.add_argument("--emit-glsl-mesh", metavar="FILE",
                    help="write the projected function serving most vertices as a drop-in for a native mesh shader "
                         "(--binding)")
    ap.add_argument("--indirect", action="append", default=[], metavar="REL,LANE",
                    help="a vertex word (qword rel from the vertex's base, lane) holding an address its inputs are "
                         "read at (a matrix's rows), tried for slots otherwise uniform; found ones are added")
    ap.add_argument("--binding", default="gs_mesh", help="the shader the drop-in is for (emit.BINDINGS)")
    ap.add_argument("--no-data-branches", action="store_true",
                    help="only the runs that took no data-dependent branch (e.g. RenderWare's world: paths A and B, "
                         "not path C's clipper)")
    ap.add_argument("--bind", choices=["rw"],
                    help="milestone 3: bind the projected functions to a native header (rw: RenderWare's, "
                         "rwheader.py) and merge them by what they read from it")
    ap.add_argument("--emit-glsl-rw", metavar="FILE",
                    help="write the bound functions and a dispatch by the header's mode as a drop-in for "
                         "gs_mesh_rw.vert")
    ap.add_argument("--rw-name", default=None, help="the drop-in's dispatch function (default by the program)")
    a = ap.parse_args()
    if a.mask or a.assume or a.emit_glsl_mesh or a.bind:
        a.project = True
    images = [int(x, 16) for x in a.image.split(",")]
    image = images[0]

    funcs = collections.OrderedDict()       # signature -> {"fit": first fit, "batches": [...]}
    paths = collections.Counter()
    path_funcs = collections.defaultdict(collections.Counter)
    data_branches = collections.defaultdict(lambda: {"count": 0, "taken": 0, "cond": None, "leaves": set()})
    counter_branches = collections.defaultdict(lambda: [0, 0])
    guarded = collections.defaultdict(set)    # word name -> branch pcs guarding it
    flagged = set()                           # words computed from flags
    nruns = nverts = nbatch = 0
    t_sym = 0.0
    failures = collections.Counter()
    ipdom = dec = None
    stride_seen = collections.Counter()
    ncontrolled = nsingle = nsingle_batches = 0
    diag = None
    sources = set()                           # (rel, lane): vertex words holding addresses (role I)
    for x in a.indirect:
        rel, lane = x.split(",")
        sources.add((int(rel), int(lane)))
    run_image = {}                            # snapshot index -> image (several images: one report)
    nskipped = 0

    def all_runs():
        for im in images:
            for si, head, m, d in vu1sym.image_runs(a.snap, im, a.runs, a.skip):
                yield im, si, head, m, d
    current = None
    for im, si, head, m, d in all_runs():
        if im != current:
            current = im
            ipdom = vu1sym.build_ipdom(m)
            dec = vu1sym.decode_image(m)
            stride_seen.clear()
        run_image[si] = im
        t0 = time.perf_counter()
        s = vu1sym.SymVu(head, m, d, decoded=dec, ipdom=ipdom, engine_moves=a.engine_moves)
        s.run()
        t_sym += time.perf_counter() - t0
        if a.no_data_branches and any(kind == "data" for _pc, _t, _n, kind in s.branches):
            nskipped += 1
            del s
            continue
        nruns += 1
        dag = s.dag
        pk = path_key(s.branches)
        paths[pk] += 1
        for pc, taken, cn, kind in s.branches:
            if kind == "data":
                e = data_branches[pc]
                e["count"] += 1
                e["taken"] += bool(taken)
                if e["cond"] is None:
                    e["cond"] = describe_cond(dag, cn)
            elif kind == "counter":
                c = counter_branches[pc]
                c[0] += 1
                c[1] += bool(taken)
        tags = gifslice.parse_kicks(s.kicks)
        dwords = memoryview(d).cast("I")

        def memf(q, lane, dwords=dwords):
            return dwords[(q & 0x3FF) * 4 + lane]
        leafval = s.leafval_fn(head, d)
        mem = s.leafval_fn(head, d)
        top = s.top
        for t in tags:
            if not t.vertices:
                continue
            nbatch += 1
            hint = None
            if stride_seen:
                hint = stride_seen.most_common(1)[0][0]
            b = generalize.fit_batch(dag, t, gifslice, top, hint, mem=memf, sources=sources or None)
            sources |= b.sources
            for fit in b.fits:
                if fit["stride"] is not None and fit["base"] is not None and len(fit["idxs"]) > 2:
                    stride_seen[(fit["stride"], (fit["base"] - top) % 1024)] += 1
            for name, w in b.words.items():
                if w["ctl"]:
                    guarded[name] |= w["ctl"]
                if w["flags"]:
                    flagged.add(name)
            for fit in b.fits:
                if not fit["ok"]:
                    failures["slot without a role"] += len(fit["idxs"])
                    continue
                sig = fit["signature"]
                f = funcs.get(sig)
                if f is None:
                    f = funcs[sig] = {"fit": fit, "batches": [], "vertices": 0, "runs": set(), "paths": set(),
                                      "uniform_rows": collections.Counter(), "bases": collections.Counter(),
                                      "loops": collections.Counter()}
                f["batches"].append((si, head, d, fit, [t.vertices[i] for i in fit["idxs"]]))
                f["vertices"] += len(fit["idxs"])
                f["loops"].update(fit["loops"])
                f["runs"].add(si)
                f["paths"].add(pk)
                rows = generalize.uniform_rows(fit)
                f["uniform_rows"][tuple(sorted(rows.items()))] += 1
                f["bases"][(fit["base"] - top) if fit["base"] is not None else None] += 1
                path_funcs[pk][sig] += len(fit["idxs"])
            nverts += len(t.vertices)
            ncontrolled += b.controlled
            singles = sum(1 for f in b.fits if len(f["idxs"]) == 1)
            nsingle += singles
            if singles == len(t.vertices) and len(t.vertices) >= 4:
                nsingle_batches += 1
                if diag is None or (diag[1] and b.controlled == 0):
                    diag = (induction_diag(dag, t), b.controlled)
            if len(b.fits) > 1 and a.verbose:
                print("  run %d tag %d: %d shapes: %s" % (si, t.index, len(b.fits),
                                                          [len(f["idxs"]) for f in b.fits]))
        del s, dag

    # ---- emit and verify
    names = {}
    for k, sig in enumerate(funcs):
        names[sig] = "vu_%08X_f%d" % (image, k)
    pysrc = emit.python_module(image, [(names[s], f["fit"]) for s, f in funcs.items()])
    out_py = a.emit_py or os.path.join(os.environ.get("TEMP", "."), "vutrans_%08X.py" % image)
    with open(out_py, "w", newline="\n") as fh:
        fh.write(pysrc)
    spec = importlib.util.spec_from_file_location("vutrans_emitted_%08X" % image, out_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checked = exact = 0
    bad_examples = []
    for sig, f in funcs.items():
        fn = getattr(mod, names[sig])
        for si, head, d, fit, verts in f["batches"]:
            words = memoryview(d).cast("I")
            vc = None

            def M(q, lane, words=words):
                return words[(q & 0x3FF) * 4 + lane]
            U = []
            for r in fit["roles"]:
                if r[0] == "U":
                    U.append(regval_head(head, r[1]) if r[1][0] != "m" else M(r[1][1], r[1][2]))
            for v in verts:
                out = fn(v.loop, M, fit["base"] if fit["base"] is not None else 0, U)
                want = [b for _n, b, _node, _m in gifslice.significant(v)]
                checked += 1
                if out == want:
                    exact += 1
                elif len(bad_examples) < 5:
                    bad_examples.append((names[sig], si, v.loop, out, want))
            del vc
    projected = None
    if a.project:
        projected = project_all(a, funcs, image)
    if a.emit_glsl:
        src, missing = emit.glsl_module(image, [(names[s], f["fit"]) for s, f in funcs.items()])
        with open(a.emit_glsl, "w", newline="\n") as fh:
            fh.write(src)
    else:
        _src, missing = emit.glsl_module(image, [(names[s], f["fit"]) for s, f in funcs.items()])

    # ---- report
    print("image %s: %d runs (%.1f runs/s symbolic), %d batches, %d vertices%s" %
          (",".join("%08X" % x for x in images), nruns, nruns / max(t_sym, 1e-9), nbatch, nverts,
           ("; %d runs skipped (a data-dependent branch)" % nskipped) if a.no_data_branches else ""))
    print("vertices in a generalized function: %d; not generalized: %s" %
          (sum(f["vertices"] for f in funcs.values()), dict(failures) or 0))
    print("emitted evaluator (%s): %d of %d vertices bit-exact (all significant words)" % (out_py, exact, checked))
    for e in bad_examples:
        print("  MISMATCH %s run %d vertex %d: %s vs %s" % e)
    print("vertices with a word under data-dependent control flow: %d of %d" % (ncontrolled, nverts))
    if nsingle:
        print("vertices whose function no other vertex of their batch shares: %d (%d batches entirely so)" %
              (nsingle, nsingle_batches))
    if diag:
        print("why (vertex 0 against vertex 2 of %s batch): %s" % (
            "a" if not diag[1] else "a (data-dependent)", diag[0]))
    print("\npaths (counter-like branch outcomes): %d" % len(paths))
    for pk, n in paths.most_common():
        fs = path_funcs.get(pk, {})
        print("  path %s: %d runs; functions %s" % (short_path(pk), n,
              ", ".join("%s x%d" % (names[s].split("_")[-1], c) for s, c in fs.items())))
    print("\nfunctions: %d" % len(funcs))
    for sig, f in funcs.items():
        fit = f["fit"]
        nv = sum(1 for r in fit["abstract"] if r[0] in ("V", "G4", "P"))
        print("  %s: %d vertices in %d batches over %d runs, %d paths; %d ops, %d leaves (%d per-vertex, %d uniform); "
              "outputs %s" % (names[sig], f["vertices"], len(f["batches"]), len(f["runs"]), len(f["paths"]),
                              len(fit["prog"]), len(fit["roles"]), nv, len(fit["roles"]) - nv,
                              " ".join(fit["names"])))
        st = fit["stride"]
        slots = sorted(set(divmod(r[2], st) + ("xyzw"[r[3]],) for r in fit["abstract"] if r[0] == "V" and r[1] == st))
        other = sorted(set((r[1], r[2], "xyzw"[r[3]]) for r in fit["abstract"] if r[0] == "V" and r[1] != st))
        print("     stride %s, base - TOP %s; vertex slots (vertex delta, qword, lane) %s%s" % (
            st, dict(f["bases"]), " ".join("%+d.%d.%s" % x for x in slots),
            ("; other streams (stride, base+, lane) %s" % other) if other else ""))
        g4 = [r for r in fit["abstract"] if r[0] == "G4"]
        if g4:
            print("     SoA groups of 4 (qword base+rel + B*(i//4), lane i%%4): %s" % sorted(set(g4)))
        ind = sorted(set(r[2:] for r in fit["abstract"] if r[0] == "I"))
        if ind:
            print("     indirect (rows at an address the vertex holds: (address word's qword, lane), row, lane): %s"
                  % " ".join("(%d.%d)+%d.%s" % (q, l, k, "xyzw"[lo]) for q, l, k, lo in ind))
        pp = [r for r in fit["abstract"] if r[0] == "P"]
        if pp:
            print("     periodic slots (period, slope, per-residue (base+, lane)): %s" % sorted(set(pp), key=str))
        print("     serves %s" % selector(f["loops"]))
        regs = sorted(set(r[1:] for r in fit["abstract"] if r[0] == "U" and r[1] != "m"))
        if regs:
            print("     uniform registers: %s" % regs)
        for rows, c in f["uniform_rows"].most_common(4):
            print("     uniform rows (x%d batches): %s" % (c, " ".join("%d.%s" % kv for kv in rows)))
        if len(f["uniform_rows"]) > 4:
            print("     ... %d uniform layouts" % len(f["uniform_rows"]))
    print("\ncounter-like branches: %d pcs (%s)" % (len(counter_branches), " ".join(
        "%04X:%d/%d" % (pc, c[1], c[0]) for pc, c in sorted(counter_branches.items()))))
    print("data-dependent branches: %d" % len(data_branches))
    for pc, e in sorted(data_branches.items()):
        words = sorted(w for w, pcs in guarded.items() if pc in pcs)
        print("  %04X: %d executions, %d taken; cond %s\n     guards words: %s" %
              (pc, e["count"], e["taken"], e["cond"][:300], " ".join(words) or "(none kicked)"))
    if flagged:
        print("words computed from flags (CLIP/MAC, e.g. ADC): %s" % " ".join(sorted(flagged)))
    if missing:
        print("GLSL helpers missing from gs_mesh_common.glsl / gs_mesh_rw.glsl (emitted as stubs): %s" %
              " ".join(sorted(missing)))
    ok = exact == checked
    if projected is not None:
        ok = report_projected(a, projected, image, nruns) and ok
    if a.bind:
        import rwbind
        ok = rwbind.bind_all(a, projected, run_image, nruns, regval_head) and ok
    return 0 if ok else 1


def parse_bits(text):
    k, v = text.split("=")
    return k, int(v, 0)


def project_all(a, funcs, image):
    """The functions projected (generalize.project) and merged by their projected signature; each batch checked
    with its own projected fit (its uniforms): the emitted evaluator's words, masked, against the kicks'."""
    masks = dict(parse_bits(x) for x in a.mask)
    assume = {}
    for x in a.assume:
        k, v = parse_bits(x)
        parts = k.split(",")
        assume[(parts[0],) + tuple(int(p) for p in parts[1:])] = v
    pf = collections.OrderedDict()
    for sig, f in funcs.items():
        for si, head, d, fit, verts in f["batches"]:
            p = generalize.project(fit, masks, assume)
            if p is None:
                continue
            g = pf.get(p["signature"])
            if g is None:
                g = pf[p["signature"]] = {"fit": p, "batches": [], "vertices": 0, "runs": set(), "from": set(),
                                          "loops": collections.Counter()}
            g["batches"].append((si, head, d, p, verts))
            g["vertices"] += len(verts)
            g["runs"].add(si)
            g["from"].add(sig)
            g["loops"].update(fit["loops"])
    order = sorted(pf, key=lambda k: -pf[k]["vertices"])
    names = {sig: "vu_%08X_p%d" % (image, k) for k, sig in enumerate(order)}
    src = emit.python_module(image, [(names[s], pf[s]["fit"]) for s in order])
    out_py = os.path.join(os.environ.get("TEMP", "."), "vutrans_%08X_projected.py" % image)
    with open(out_py, "w", newline="\n") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("vutrans_projected_%08X" % image, out_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for sig in order:
        g = pf[sig]
        fn = getattr(mod, names[sig])
        g["checked"] = g["exact"] = 0
        g["bad"] = []
        for si, head, d, p, verts in g["batches"]:
            words = memoryview(d).cast("I")

            def M(q, lane, words=words):
                return words[(q & 0x3FF) * 4 + lane]
            U = [regval_head(head, r[1]) if r[1][0] != "m" else M(r[1][1], r[1][2]) for r in p["roles"]
                 if r[0] == "U"]
            for v in verts:
                out = fn(v.loop, M, p["base"] if p["base"] is not None else 0, U)
                have = {n: b for n, b, _node, _m in gifslice.significant(v)}
                want = [have[n] & m for n, m in zip(p["names"], p["out_masks"])]
                got = [o & m for o, m in zip(out, p["out_masks"])]
                g["checked"] += 1
                if got == want:
                    g["exact"] += 1
                elif len(g["bad"]) < 3:
                    g["bad"].append((si, v.loop, got, want))
    return {"order": order, "funcs": pf, "names": names, "masks": masks, "assume": assume, "py": out_py}


def report_projected(a, pr, image, nruns):
    pf, order, names = pr["funcs"], pr["order"], pr["names"]
    print("\nprojected (g dropped; masks %s; assumed zero %s): %d functions" % (
        {k: "0x%X" % v for k, v in pr["masks"].items()}, {k: "0x%X" % v for k, v in pr["assume"].items()},
        len(pf)))
    ok = True
    for sig in order:
        g = pf[sig]
        fit = g["fit"]
        nu = sum(1 for r in fit["abstract"] if r[0] == "U")
        print("  %s: %d vertices in %d batches over %d runs (from %d functions); %d ops, %d leaves (%d uniform); "
              "serves %s; evaluator %d of %d exact (masked)" % (
                  names[sig], g["vertices"], len(g["batches"]), len(g["runs"]), len(g["from"]), len(fit["prog"]),
                  len(fit["abstract"]), nu, selector(g["loops"]), g["exact"], g["checked"]))
        for b in g["bad"]:
            print("     MISMATCH run %d vertex %d: %s vs %s" % b)
        ok = ok and g["exact"] == g["checked"]
    if a.emit_glsl_mesh:
        sig = order[0]
        g = pf[sig]
        src = emit.glsl_bound(a.binding, names[sig], g["fit"], {
            "snapshot": os.path.basename(a.snap), "image": "%08X" % image, "runs": nruns,
            "vertices": g["vertices"], "batches": len(g["batches"]), "exact": g["exact"], "checked": g["checked"],
            "masks": pr["masks"], "assume": pr["assume"], "serves": selector(g["loops"]),
            "command": "python -I engine/tools/vutrans/translate.py " + " ".join(
                os.path.basename(x) if ("/" in x or "\\" in x) else x for x in sys.argv[1:])})
        with open(a.emit_glsl_mesh, "w", newline="\n") as fh:
            fh.write(src)
        print("wrote %s (%s, binding %s)" % (a.emit_glsl_mesh, names[sig], a.binding))
    return ok


def induction_diag(dag, tag):
    """For a batch whose vertices all differ in shape: vertex 0 against 2 (a strip's same side) - the
    differing subterms, and whether each later one contains the earlier (an accumulator: x' = f(x, ...))."""
    v = tag.vertices
    out = []
    for name in gifslice.significant(v[0]):
        a, b = v[0].words[name[0]][1], v[2].words[name[0]][1]
        for x, y in generalize.holes(dag, a, b):
            acc = generalize.contains(dag, y, x)
            item = "%s: %s -> %s%s" % (name[0], dag.fmt(x, 3)[:80], dag.fmt(y, 3)[:120],
                                       " (accumulator)" if acc else "")
            if item not in out:
                out.append(item)
        if len(out) >= 4:
            break
    return "; ".join(out[:4])


def selector(loops):
    """Which vertex indices i a function served: 'all', 'i%4 in {..}, i>=k', 'i in {..}'."""
    idx = set(loops)
    if len(idx) <= 6 and max(idx) < 8:
        return "i in {%s}" % ",".join(str(i) for i in sorted(idx))
    lo = min(idx)
    rs = set(i % 4 for i in idx)
    if len(rs) < 4:
        return "i %% 4 in {%s}, i >= %d" % (",".join(str(r) for r in sorted(rs)), lo)
    return "all i >= %d" % lo


def short_path(pk):
    return " ".join("%04X%s" % (pc, "+" if t else "-") for pc, t in pk)


def regval_head(head, desc):
    import struct
    op = desc[0]
    if op == "vf":
        return struct.unpack_from("<I", head, 28 + 16 * desc[1] + 4 * desc[2])[0]
    if op == "vi":
        return struct.unpack_from("<I", head, 28 + 512 + 4 * desc[1])[0] & 0xFFFF
    if op == "acc":
        return struct.unpack_from("<I", head, 28 + 512 + 128 + 4 * desc[1])[0]
    if op == "Q":
        return struct.unpack_from("<I", head, 28 + 512 + 4 * 22)[0]
    if op == "I":
        return struct.unpack_from("<I", head, 28 + 512 + 4 * 21)[0]
    fl = struct.unpack_from("<6I", head, 4)
    if op == "P":
        return fl[3]
    if op == "top":
        return fl[4]
    if op == "itop":
        return fl[5]
    if op == "mac":
        return fl[0]
    if op == "clipold":
        return (fl[2] >> (6 * desc[1])) & 63
    raise KeyError(op)


if __name__ == "__main__":
    sys.exit(main())
