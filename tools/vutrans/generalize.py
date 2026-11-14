"""Does every output vertex of a batch (one GIF tag's vertices) share one per-vertex function?

canon(): a vertex's significant output words (slice.significant) as one canonical straight-line program: the
DAG under them in post-order, leaves as placeholders L0, L1, ... in first-visit order, constants inline. Two
vertices have the same function shape when their canonical texts are equal.

fit_batch(): for the vertices of one shape, each leaf slot is given a role from its leaves over the vertices
(i = the vertex's loop index in its tag):
  U     the same leaf for every vertex (uniform: header / constant rows, carried registers)
  V     data memory qword base + rel + stride*i, one lane (a vertex input; rel = stride*delta + off, delta != 0:
        a neighbour, e.g. the CLIP flags of the two previous vertices; a second stride: a second stream)
  G4    qword base + rel + B*(i // 4), lane i % 4  (structure-of-arrays groups of four, e.g. normals)
  P     periodic, p = 2 or 4 (an unrolled loop's lanes): qword base + slope*i + rel[i % p], lane lane[i % p]
  ?     none of these (the function does not generalize)
stride and base are inferred (the stride the V slots share; base so that most V slots have delta 0).

signature(): the function's identity across batches and runs - the canonical text plus the roles with the
uniform slots abstracted (their addresses are the CPU-side header layout, which may move per instance).
"""
import collections

import vu1sym


def canon(dag, roots):
    """roots: node ids. Returns (text, leaves, program) - program: [(ref, op, attr, argrefs)] in post-order,
    leaves: node ids by placeholder index, the roots' refs at the text's end."""
    nodes = dag.nodes
    ref = {}
    leaves = []
    prog = []
    lines = []
    for r in roots:
        stack = [(r, False)]
        while stack:
            n, expanded = stack.pop()
            if n in ref:
                continue
            op, args, attr = nodes[n]
            if op == "c":
                ref[n] = "c%08X" % attr
                continue
            if not args:
                ref[n] = "L%d" % len(leaves)
                leaves.append(n)
                continue
            if not expanded:
                stack.append((n, True))
                for a in reversed(args):
                    if a not in ref:
                        stack.append((a, False))
                continue
            t = "t%d" % len(prog)
            ref[n] = t
            argrefs = tuple(ref[a] for a in args)
            ctl = dag.ctl[n]
            prog.append((t, op, attr, argrefs))
            lines.append("%s=%s%s(%s)%s" % (t, op, "" if attr is None else "<%s>" % (attr,), ",".join(argrefs),
                                            "" if not ctl or op != "g" else "{%s}" % ",".join(
                                                "%04X" % x for x in sorted(ctl))))
    out = tuple(ref[r] for r in roots)
    return "\n".join(lines) + "\n=>" + ",".join(out), leaves, prog, out


def leaf_desc(dag, n):
    op, _, attr = dag.nodes[n]
    return (op,) + tuple(attr)


def _fit_slot(points, slope_hint=None):
    """points: [(i, desc)]. Returns a role tuple. slope_hint: the batch's stride, for periodic slots whose
    residues are seen once each (an unrolled loop's first group)."""
    d0 = points[0][1]
    if all(d == d0 for _, d in points):
        return ("U", d0)
    if not all(d[0] == "m" for _, d in points):
        return ("?", tuple(points[:4]))
    (i0, a0), (i1, a1) = points[0], points[1]
    lanes = set(d[2] for _, d in points)
    if len(lanes) == 1 and i1 != i0:
        q0, q1 = a0[1], a1[1]
        b = (q1 - q0) // (i1 - i0) if (q1 - q0) % (i1 - i0) == 0 else None
        if b:
            a = q0 - b * i0
            if all((d[1] - a - b * i) % 1024 == 0 for i, d in points):
                return ("V", b, a % 1024, a0[2])
    # SoA groups of four: lane i % 4, qword a + B*(i // 4)
    if all(d[2] == i % 4 for i, d in points):
        gs = sorted(set((i // 4, d[1]) for i, d in points))
        if len(gs) >= 2:
            (g0, x0), (g1, x1) = gs[0], gs[1]
            if (x1 - x0) % (g1 - g0) == 0:
                B = (x1 - x0) // (g1 - g0)
                a = x0 - B * g0
                if all((d[1] - a - B * (i // 4)) % 1024 == 0 for i, d in points):
                    return ("G4", B, a % 1024)
        elif len(gs) == 1:
            return ("G4", 0, gs[0][1])
    # periodic (an unrolled loop's lanes): per residue r = i % p its own qword offset and lane, one slope
    for p in (2, 4):
        res = {}
        for i, d in points:
            res.setdefault(i % p, []).append((i, d))
        if len(res) < 2:
            continue
        slope = None
        for r, pts in res.items():
            if len(set(d[2] for _, d in pts)) != 1:
                break
            if len(pts) >= 2:
                (i0, d0), (i1, d1) = pts[0], pts[1]
                if (d1[1] - d0[1]) % (i1 - i0):
                    break
                sl = (d1[1] - d0[1]) // (i1 - i0)
                if slope is None:
                    slope = sl
                elif sl != slope:
                    break
        else:
            if slope is None:
                if slope_hint is None:
                    continue
                slope = slope_hint
            table = {}
            ok = True
            for r, pts in res.items():
                i0, d0 = pts[0]
                a = (d0[1] - slope * i0) % 1024
                if not all((d[1] - a - slope * i) % 1024 == 0 for i, d in pts):
                    ok = False
                    break
                table[r] = (a, d0[2])
            if ok:
                return ("P", p, slope, tuple(table.get(r, (None, None)) for r in range(p)))
    return ("?", tuple(points[:4]))


class Batch:
    """One tag's vertices, fitted."""

    def __init__(self):
        self.tag = None
        self.classes = []      # [(text, [vertex indices])], biggest first
        self.fits = []         # per class: dict with roles, stride, base, uniforms, ok
        self.words = {}        # output word name -> {'ctl': set, 'flags': bool}


def _flag_ops(dag, root, cache):
    """Whether the DAG under root holds a flag op (clipj / fc* / mac) - an output computed from flags."""
    if root in cache:
        return cache[root]
    nodes = dag.nodes
    stack, seen, found = [root], set(), False
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        op, args, _ = nodes[x]
        if op in ("clipj", "fcand", "fcor", "fceq", "fcget", "mac"):
            found = True
            break
        stack.extend(args)
    cache[root] = found
    return found


def fit_batch(dag, tag, slicer, top=None, hint=None):
    """tag: a slice.Tag. Returns a Batch. hint: (stride, base - TOP) seen in earlier batches of the image, for
    batches too short to infer them (every vertex its own shape)."""
    b = Batch()
    b.tag = tag
    groups = collections.OrderedDict()
    per = []
    flagcache = {}
    b.controlled = 0          # vertices with a word under a data-dependent branch
    for idx, v in enumerate(tag.vertices):
        sig = slicer.significant(v)
        if any(dag.ctl[s[2]] for s in sig):
            b.controlled += 1
        names = tuple(s[0] for s in sig)
        roots = [s[2] for s in sig]
        text, leaves, prog, outs = canon(dag, roots)
        key = (names, text)
        groups.setdefault(key, []).append(idx)
        per.append((leaves, prog, outs, names))
        for name, bits, node, mask in sig:
            w = b.words.setdefault(name, {"ctl": set(), "flags": False})
            if dag.ctl[node]:
                w["ctl"] |= dag.ctl[node]
            if not w["flags"] and _flag_ops(dag, node, flagcache):
                w["flags"] = True
    anchor = None          # (base, stride) of the batch's biggest shape: places lone vertices' leaves
    if hint is not None and top is not None and (len(groups) == len(tag.vertices) or
                                                 max(len(v) for v in groups.values()) < 3):
        anchor = ((top + hint[1]) % 1024, hint[0])
    for key, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        names, text = key
        leaves0 = per[idxs[0]][0]
        roles = []
        for j in range(len(leaves0)):
            pts = [(tag.vertices[i].loop, leaf_desc(dag, per[i][0][j])) for i in idxs]
            if len(idxs) == 1 and anchor is not None and pts[0][1][0] == "m":
                base_, stride_ = anchor
                i, desc = pts[0]
                rel = (desc[1] - base_ - stride_ * i) % 1024
                if rel >= 512:
                    rel -= 1024
                # within the batch's vertex input (from base to base + stride * vertices): a vertex input
                if 0 <= desc[1] - base_ < stride_ * (len(tag.vertices) + 4):
                    roles.append(("V", stride_, (base_ + stride_ * i + rel - stride_ * i) % 1024, desc[2]))
                    continue
            roles.append(_fit_slot(pts, anchor[1] if anchor is not None else (hint[0] if hint else None)))
        fit = {"names": names, "text": text, "idxs": idxs, "roles": roles, "prog": per[idxs[0]][1],
               "loops": [tag.vertices[i].loop for i in idxs],
               "outs": per[idxs[0]][2], "ok": all(r[0] != "?" for r in roles)}
        strides = collections.Counter(r[1] for r in roles if r[0] == "V")
        fit["strides"] = sorted(strides)
        base = None
        if strides and len(idxs) == 1 and anchor is not None:
            base, stride = anchor
            fit["stride"] = stride
        elif strides:
            stride = strides.most_common(1)[0][0]
            vs = [r for r in roles if r[0] == "V" and r[1] == stride]
            m = min(r[2] for r in vs)
            buckets = collections.Counter((r[2] - m) // stride for r in vs)
            base = m + stride * buckets.most_common(1)[0][0]
            fit["stride"] = stride
            if anchor is None:
                anchor = (base, stride)
        elif anchor is not None and any(r[0] == "V" for r in roles):
            base, stride = anchor
            fit["stride"] = stride
        else:
            fit["stride"] = None
        fit["base"] = base
        fit["top"] = top
        # the abstract roles (the function's identity) and the uniform list (the batch's header)
        abstract, uniforms = [], []
        for r in roles:
            if r[0] == "U":
                d = r[1]
                if d[0] == "m":
                    abstract.append(("U", "m"))
                else:
                    abstract.append(("U",) + d)
                uniforms.append(d)
            elif r[0] == "V":
                rel = (r[2] - base) % 1024
                if rel >= 512:
                    rel -= 1024           # a neighbour before base (the address wrapped)
                abstract.append(("V", r[1], rel, r[3]))
            elif r[0] == "G4":
                abstract.append(("G4", r[1], ((r[2] - base) % 1024) if base is not None else r[2]))
            elif r[0] == "P":
                b0 = base if base is not None else 0
                abstract.append(("P", r[1], r[2], tuple((None, None) if a is None else ((a - b0) % 1024, l)
                                                        for a, l in r[3])))
            else:
                abstract.append(("?",))
        fit["abstract"] = tuple(abstract)
        fit["uniforms"] = uniforms
        fit["signature"] = (names, text, tuple(abstract))
        b.classes.append((key, idxs))
        b.fits.append(fit)
    return b


def holes(dag, a, b, out=None, memo=None):
    """Anti-unification of two DAGs: the pairs of differing subterms (the first difference down each path)."""
    if out is None:
        out, memo = [], set()
    if a == b or (a, b) in memo:
        return out
    memo.add((a, b))
    na, nb = dag.nodes[a], dag.nodes[b]
    if na[0] == nb[0] and na[2] == nb[2] and len(na[1]) == len(nb[1]) and na[1]:
        for x, y in zip(na[1], nb[1]):
            holes(dag, x, y, out, memo)
    else:
        out.append((a, b))
    return out


def contains(dag, big, small):
    """Whether small is a subterm of big."""
    stack, seen = [big], set()
    while stack:
        x = stack.pop()
        if x == small:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(dag.nodes[x][1])
    return False


def uniform_rows(fit):
    """The uniform memory rows a function reads: {qword: lanes}."""
    rows = collections.defaultdict(set)
    for d in fit["uniforms"]:
        if d[0] == "m":
            rows[d[1]].add("xyzw"[d[2]])
    return {q: "".join(sorted(ls, key="xyzw".index)) for q, ls in sorted(rows.items())}


def leaf_value_for(fit, i, leafval_mem, regval, uvals=None):
    """The leaf values of vertex i under the fit's roles (for checks): a list by placeholder index."""
    out = []
    base = fit["base"]
    for j, r in enumerate(fit["roles"]):
        if r[0] == "U":
            d = r[1]
            out.append(leafval_mem(d[1], d[2]) if d[0] == "m" else regval(d))
        elif r[0] == "V":
            out.append(leafval_mem((r[2] + r[1] * i) & 0x3FF, r[3]))
        elif r[0] == "G4":
            out.append(leafval_mem((r[2] + r[1] * (i // 4)) & 0x3FF, i % 4))
        elif r[0] == "P":
            a, l = r[3][i % r[1]]
            out.append(leafval_mem((a + r[2] * i) & 0x3FF, l))
        else:
            raise ValueError("slot %d does not generalize" % j)
    return out


def run_prog(prog, outs, leafvals):
    """Evaluates a canonical program (post-order) with the leaves' bits."""
    env = {}
    for j, x in enumerate(leafvals):
        env["L%d" % j] = x
    ops = vu1sym.OPS

    def get(r):
        if r[0] == "c":
            return int(r[1:], 16)
        return env[r]
    for t, op, attr, args in prog:
        env[t] = ops[op](attr, *[get(a) for a in args])
    return [get(o) for o in outs]
