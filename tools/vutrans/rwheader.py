"""vutrans milestone 3: RenderWare's native header (engine/host/include/ps2host/rw_native.h, made by Burnout 3's
games/burnout3/port/vu_native_rw.cpp) as expressions of a VU1 run's initial state, and the binding of a derived
per-vertex function (generalize.project's fit) to it: what gs_mesh_rw.vert reads in place of VU1's memory.

A header word is an expression (a tuple): ('c', bits) a constant, ('m', qword, lane) / ('vf', reg, lane) an
initial data memory word / register (vu1sym's leaves), (op, attr, arg, ...) an op of vu1sym.OPS - the port's
arithmetic written as the VU's (rw::Mul a mul, rw::Madd(acc, a, b) add(acc, mul(a, b)), rw::Recip div(1.0, x)).
Constants fold, as vu1sym.Dag folds them.

bind(): the function's uniform part (every op whose leaves are uniform: header rows, carried registers) covered by
header words, top down - a uniform subterm the header holds is read from it, else its op is made per vertex from its
arguments' bindings; a uniform leaf the header does not hold leaves the function unbound. A subterm matches a word
  structurally: the same expression (add and mul's operands in either order: both commute bit for bit), or
  by value: the same leaves and the same bits on every batch the function serves (the port's arithmetic not the
  microcode's in form - e.g. add(0, x) for x - but equal on every value the snapshot holds).
"""
import struct

import vu1sym

OPS = vu1sym.OPS
ONE, ZERO, HALF = ("c", 0x3F800000), ("c", 0), ("c", 0x3F000000)


def C(bits):
    return ("c", bits & 0xFFFFFFFF)


def Cf(f):
    return C(struct.unpack("<I", struct.pack("<f", f))[0])


def E(op, *args, attr=None):
    if all(a[0] == "c" for a in args):
        return C(OPS[op](attr, *[a[1] for a in args]))
    if op == "sub" and args[0] == args[1]:
        return ZERO
    return (op, attr) + tuple(args)


def mul(a, b):
    return E("mul", a, b)


def add(a, b):
    return E("add", a, b)


def sub(a, b):
    return E("sub", a, b)


def madd(acc, a, b):
    return add(acc, mul(a, b))


def m(q, l):
    return ("m", q, l)


def vf(r, l):
    return ("vf", r, l)


LEAVES = ("m", "vf", "vi", "acc", "Q", "I", "P", "top", "itop", "mac", "clipold")


def is_leaf(e):
    return e[0] in LEAVES


def leaves(e, out=None):
    if out is None:
        out = set()
    if e[0] == "c":
        return out
    if is_leaf(e):
        out.add(e)
        return out
    for a in e[2:]:
        leaves(a, out)
    return out


class State:
    """A run's initial state (the snapshot's head and data) for header expressions' values."""

    def __init__(self, head, data):
        self.vf = struct.unpack_from("<128I", head, 28)
        self.vc = struct.unpack_from("<32I", head, 28 + 512)
        self.mem = memoryview(data).cast("I")

    def leaf(self, e):
        if e[0] == "m":
            return self.mem[(e[1] & 0x3FF) * 4 + e[2]]
        if e[0] == "vf":
            return self.vf[e[1] * 4 + e[2]]
        if e[0] == "vi":
            return self.vc[e[1]] & 0xFFFF
        raise KeyError(e)

    def value(self, e, memo=None):
        if memo is None:
            memo = {}
        r = memo.get(e)
        if r is not None:
            return r
        if e[0] == "c":
            r = e[1]
        elif is_leaf(e):
            r = self.leaf(e)
        else:
            r = OPS[e[0]](e[1], *[self.value(a, memo) for a in e[2:]])
        memo[e] = r
        return r

    def lo(self, q, l):
        return self.mem[q * 4 + l] & 0xFFFF


# ---------------------------------------------------------------- the headers (rw_native.h's kRw*)

kRwInstanced, kRwPathB, kRwPathC = 1 << 8, 1, 2
kRwWorld, kRwWCull, kRwWGuard, kRwWLit, kRwWAlphaZero, kRwWFog = 1 << 9, 1, 2, 4, 8, 16
kRwEnvStream, kRwGuardPlace, kRwSkin = 1, 2, 4

# vu_native_rw.cpp's kWorldImages: (vi12 of paths A, B, C, cull, lit, mirror)
WORLD_IMAGES = {
    0x2678EB26: (0x057, 0x0A0, 0x0F3, True, False, False),
    0x1332C99F: (0x0D0, 0x119, 0x16C, True, True, False),
    0x01CE30D5: (0x057, 0x081, 0x0B2, False, False, False),
    0x0835A68B: (0x05F, 0x089, 0x0BA, False, False, True),
}
INSTANCED_IMAGES = (0xD4B61373, 0x34C3CF67)
CARS_IMAGES = (0x7E86891B,)


def program_of(image):
    if image in WORLD_IMAGES:
        return "world"
    if image in INSTANCED_IMAGES:
        return "instanced"
    if image in CARS_IMAGES:
        return "cars"
    return None


def world_header(image, st):
    """DecideWorld's header (None: declined, or path C - its draws the CPU's but for the fast path's fog)."""
    va, vb, vc, cull, lit, mirror = WORLD_IMAGES[image]
    rebuild = st.lo(1013, 3) != 0
    if rebuild:
        x = st.lo(1013, 0)
        if not (x & 2):
            vi12 = 0 if (x & 1) else va
        elif (x & 8) and not (x & 4):
            vi12 = vc
        else:
            vi12 = 0 if (x & 1) else vb
    else:
        vi12 = st.lo(1013, 1)
    path_b, path_c = vi12 == vb, vi12 == vc
    if vi12 != va and not path_b:
        return None
    h = {}
    if rebuild:
        t = [[m(1014 + r, l) for l in range(4)] for r in range(4)]
        for r in range(4):
            t[r][2] = ONE if r == 3 else ZERO
            hh = mul(t[r][3], HALF)
            t[r][0] = sub(t[r][0], hh)
            t[r][1] = sub(t[r][1], hh)
        for r in range(4):
            t[r][0] = mul(t[r][0], m(1018, 0))
            t[r][1] = mul(t[r][1], m(1018, 1))
        t[3][2] = mul(t[3][2], m(1018, 2))
        o = [m(1019, 0), m(1019, 1), m(1019, 2), ZERO]
    else:
        t = [[vf(28 + r, l) for l in range(4)] for r in range(4)]
        o = [vf(2, l) for l in range(4)]
    if mirror:
        k = m(1009, 1)
        for l in (0, 1, 3):
            acc = add(t[3][l], ZERO)
            acc = madd(acc, t[1][l], k)
            t[3][l] = madd(acc, t[1][l], k)
            t[1][l] = sub(ZERO, t[1][l])
    mode = kRwWorld | (kRwWCull if cull else 0) | (kRwWGuard if path_b else 0)
    if lit:
        mode |= kRwWAlphaZero if st.lo(1013, 2) != 0 else kRwWLit
    for r in range(4):
        for l in range(4):
            h[(1 + r, l)] = t[r][l]
    for l in range(4):
        h[(5, l)] = o[l]
        h[(6, l)] = m(1018, l)
        h[(7, l)] = m(1019, l)
        h[(8, l)] = m(1022, l)
        h[(9, l)] = m(1023, l)
        h[(10, l)] = m(1007, l)
        h[(11, l)] = m(1008, l)
        h[(12, l)] = m(1009, l)
    if lit:
        for l in range(3):
            h[(13, l)] = mul(m(1007, l), C(0x39020610))
        h[(13, 3)] = mul(m(1007, 3), C(0x3C23D70A))
        s = m(1008, 3)
        h[(14, 1)] = sub(E("itof", add(s, s), attr=0), mul(m(1007, 3), C(OPS["itof"](0, 0x3F800000))))
    if path_c:
        mode |= kRwWFog
    return [(mode, h)]


def instanced_headers(image, st):
    """DecideInstanced's headers: one an instance."""
    lit = st.lo(932, 1) != 0
    last = st.lo(932, 0)
    last = last - 0x10000 if last & 0x8000 else last
    count = last + 1 if last > 0 else 1
    if count > 15:
        return None
    V = [[m(1014 + r, l) for l in range(4)] for r in range(4)]
    S = [m(1018, l) for l in range(4)]
    far, near = m(1022, 3), m(1023, 3)
    out = []
    for i in range(count):
        block = 934 + 5 * i
        path_b = st.lo(block + 3, 2) != 0
        R = [[m(block + r, l) for l in range(4)] for r in range(4)]
        M = [[None] * 4 for _ in range(4)]
        for n in range(4):
            for l in (0, 1, 3):
                acc = mul(V[0][l], R[n][0])
                acc = madd(acc, V[1][l], R[n][1])
                acc = madd(acc, V[2][l], R[n][3])
                if n == 3:
                    acc = madd(acc, V[3][l], ONE)
                M[n][l] = acc
            M[n][2] = ONE if n == 3 else ZERO
        for n in range(4):
            hh = mul(M[n][3], HALF)
            M[n][0] = sub(M[n][0], hh)
            M[n][1] = sub(M[n][1], hh)
        if not path_b:
            for n in range(4):
                M[n][0] = mul(M[n][0], S[0])
                M[n][1] = mul(M[n][1], S[1])
            M[3][2] = mul(ONE, S[2])
        else:
            sx, sy = mul(S[0], Cf(1.0 / 1024.0)), mul(S[1], Cf(1.0 / 1024.0))
            q = E("div", ONE, add(far, near))
            za = mul(sub(far, near), q)
            zb = mul(mul(mul(far, near), Cf(-2.0)), q)
            for n in range(4):
                M[n][0] = mul(M[n][0], sx)
            for n in range(4):
                M[n][1] = mul(M[n][1], sy)
            for n in range(4):
                M[n][2] = mul(za, M[n][3])
            M[3][2] = add(M[3][2], zb)
        h = {}
        for n in range(4):
            for l in range(4):
                h[(1 + n, l)] = M[n][l]
                h[(5, l)] = S[l]
                h[(6, l)] = m(1019, l)
                h[(7, l)] = m(block + 4, l)
        h[(8, 0)] = m(block, 2)
        h[(8, 1)] = m(block + 1, 2)
        mode = kRwInstanced | (kRwPathB if path_b else 0) | (kRwPathC if lit else 0)
        out.append((mode, h))
    return out


def cars_headers(image, st):
    """Decide's header (cars), the palette aside; both streams' modes - the lit pass's, then path A's environment
    pass's (rwbind: a kick's tag binds to its stream's, by its place among the run's tags)."""
    rebuild = st.lo(965, 3) == 0
    path_a = st.lo(1013, 1) == 0
    if rebuild:
        P = [[m(1014 + r, l) for l in range(4)] for r in range(4)]
        for r in range(4):
            P[r][2] = mul(P[r][2], HALF)
        Cc = []
        for col in range(4):
            wx, wy, wz = m(962 + col, 0), m(962 + col, 1), m(962 + col, 2)
            row = []
            for l in range(4):
                acc = mul(P[0][l], wx)
                acc = madd(acc, P[1][l], wy)
                acc = madd(acc, P[2][l], wz)
                if col == 3:
                    acc = madd(acc, P[3][l], ONE)
                row.append(acc)
            Cc.append(row)
        c = []
        for col in range(4):
            c.append([mul(sub(Cc[col][0], Cc[col][2]), m(1018, 0)), mul(sub(Cc[col][1], Cc[col][2]), m(1018, 1)),
                      mul(ONE, m(1018, 2)) if col == 3 else ZERO, Cc[col][3]])
        o = [m(1019, 0), m(1019, 1), m(1019, 2), ZERO]
        vi12 = st.lo(1013, 0) & 2
    else:
        c = [[vf(28 + col, l) for l in range(4)] for col in range(4)]
        o = [vf(2, l) for l in range(4)]
        vi12 = st.vc[12] & 0xFFFF
    guard = vi12 != 0
    t4 = guard and path_a
    h = {}
    for col in range(4):
        for l in range(4):
            h[(1 + col, l)] = c[col][l]
    for l in range(4):
        h[(5, l)] = o[l]
        for k in range(8):
            h[(6 + k, l)] = m(966 + k, l)
        for k in range(3):
            h[(14 + k, l)] = m(1010 + k, l)
        h[(17, l)] = m(1022, l)
        h[(18, l)] = m(1023, l)
    if t4:
        igx, igy, zA, zB = m(1022, 1), m(1023, 1), m(1022, 2), m(1023, 2)
        for col in range(4):
            h[(19 + col, 0)] = mul(c[col][0], igx)
            h[(19 + col, 1)] = mul(c[col][1], igy)
            h[(19 + col, 2)] = madd(add(ZERO, zA), zB, c[col][3]) if col == 3 else mul(zB, c[col][3])
            h[(19 + col, 3)] = mul(c[col][3], ONE)
    skin = kRwSkin if st.lo(1013, 2) != 0 else 0
    base = (kRwGuardPlace if t4 else 0) | skin
    return [(base, h), (base | kRwEnvStream, h)]


def headers(image, head, data):
    """[(mode, {(row, lane): expression})] - the run's headers (an instance's each, or the streams'), or None."""
    st = State(head, data)
    p = program_of(image)
    if p == "world":
        return world_header(image, st)
    if p == "instanced":
        return instanced_headers(image, st)
    if p == "cars":
        return cars_headers(image, st)
    return None


# ---------------------------------------------------------------- the binding

COMMUTE = {"add", "mul"}


class Interner:
    """Hash-consed expression ids (add and mul's operands sorted: their order not the identity)."""

    def __init__(self):
        self.ids = {}

    def id(self, key):
        n = self.ids.get(key)
        if n is None:
            n = self.ids[key] = len(self.ids)
        return n

    def expr(self, e, memo):
        r = memo.get(e)
        if r is not None:
            return r
        if e[0] == "c" or is_leaf(e):
            r = self.id(e)
        else:
            args = [self.expr(a, memo) for a in e[2:]]
            if e[0] in COMMUTE:
                args.sort()
            r = self.id((e[0], e[1]) + tuple(args))
        memo[e] = r
        return r


def uniform_terms(fit, U):
    """The fit's uniform temporaries (every leaf under them U): {ref: (expression, bits)} with U the batch's uniform
    words (slot order)."""
    roles = fit["roles"]
    ui = {}
    k = 0
    for j, r in enumerate(roles):
        if r[0] == "U":
            ui["L%d" % j] = (r[1], U[k])
            k += 1
    env = {}
    for ref, (desc, bits) in ui.items():
        env[ref] = (desc if desc[0] != "c" else desc, bits)

    def get(a):
        if a[0] == "c":
            v = int(a[1:], 16)
            return ("c", v), v
        return env.get(a)
    for t, op, attr, args in fit["prog"]:
        got = [get(a) for a in args]
        if any(g is None for g in got):
            continue
        e = (op, attr) + tuple(g[0] for g in got)
        env[t] = (e, OPS[op](attr, *[g[1] for g in got]))
    return env


def bind_batch(fit, U, hdrs, intern):
    """Per uniform temporary/leaf of one batch: the header words that hold it - {ref: {(h, row, lane): how}} with h
    the header's index in hdrs, how 's' (structural) or 'v' (by value, the same leaves)."""
    env = uniform_terms(fit, U)
    out = {}
    for h, (mode, slots, values) in enumerate(hdrs):
        sid = {}
        memo = {}
        for key, e in slots.items():
            if e[0] == "c":
                continue
            sid.setdefault(intern.expr(e, memo), []).append(key)
        bylv = {}
        for key, e in slots.items():
            if e[0] == "c":
                continue
            bylv.setdefault((frozenset(leaves(e)), values[key]), []).append(key)
        memo2 = {}
        for ref, (e, bits) in env.items():
            hit = {}
            for key in sid.get(intern.expr(e, memo2), ()):
                hit[(h,) + key] = "s"
            for key in bylv.get((frozenset(leaves(e)), bits), ()):
                hit.setdefault((h,) + key, "v")
            if hit:
                out.setdefault(ref, {}).update(hit)
    return out


def fmt(ref, defs, roles, depth=4):
    """A fit's temporary as text (its first batch's leaves)."""
    if ref[0] == "c":
        return ref[1:]
    if ref[0] == "L":
        r = roles[int(ref[1:])]
        return "%s%s" % (r[0], r[1:]) if r[0] != "U" else "%s%s" % (r[1][0], list(r[1][1:]))
    if depth == 0:
        return "..."
    op, attr, args = defs[ref]
    return "%s%s(%s)" % (op, "" if attr is None else "<%s>" % (attr,), ", ".join(fmt(a, defs, roles, depth - 1)
                                                                             for a in args))


CLEAN_OPS = {"add", "sub", "mul", "max", "mini", "abs", "div", "sqrt", "esqrt", "rsqrt", "ersqrt", "itof"}


def canonical(prog, outs, leaves_, abstract):
    """The program in a canonical order: post-order from the outputs, a commutative op's (add, mul) operands
    visited in the order of their structure (leaves by role, mesh inputs alike - so an unrolled loop's lanes
    line up), temporaries and leaves numbered in that visit. Two programs that differ only in the order of
    commutative operands, or of independent ops, come out the same."""
    defs = {t: (op, attr, args) for t, op, attr, args in prog}
    keyof = {}
    ids = {}

    def key(r):
        if r in keyof:
            return keyof[r]
        if r[0] == "c":
            k = ("c", r)
        elif r[0] == "L":
            ab = abstract[int(r[1:])]
            k = ("M",) if ab[0] in ("V", "G4", "P") else ab
        else:
            op, attr, args = defs[r]
            ks = [key(a) for a in args]
            if op in COMMUTE:
                ks.sort()
            k = (op, str(attr), tuple(ks))
        k = ids.setdefault(k, len(ids))
        keyof[r] = k
        return k

    def kids(r):
        op, attr, args = defs[r]
        if op in COMMUTE:
            return sorted(args, key=lambda a: (structure(a), a))
        return list(args)
    struct_memo = {}

    def structure(r):
        """A sortable structural text of r (leaves by role)."""
        if r in struct_memo:
            return struct_memo[r]
        if r[0] == "c":
            sx = "c" + r[1:]
        elif r[0] == "L":
            ab = abstract[int(r[1:])]
            sx = "M" if ab[0] in ("V", "G4", "P") else repr(ab)
        else:
            op, attr, args = defs[r]
            parts = [structure(a) for a in args]
            if op in COMMUTE:
                parts.sort()
            sx = "%s<%s>(%s)" % (op, attr, ",".join(parts))
        struct_memo[r] = sx
        return sx
    ren, newprog, newleaves, newab = {}, [], [], []

    def visit(r):
        if r[0] == "c" or r in ren:
            return
        if r[0] == "L":
            ren[r] = "L%d" % len(newleaves)
            j = int(r[1:])
            newleaves.append(leaves_[j])
            newab.append(abstract[j])
            return
        stack = [(r, False)]
        while stack:
            x, done = stack.pop()
            if x in ren:
                continue
            if x[0] == "c":
                continue
            if x[0] == "L":
                visit(x)
                continue
            if not done:
                stack.append((x, True))
                for a in reversed(kids(x)):
                    stack.append((a, False))
                continue
            op, attr, args = defs[x]
            a2 = tuple(ren.get(a, a) for a in args)
            if op in COMMUTE:
                a2 = tuple(ren.get(a, a) for a in kids(x))
            nt = "t%d" % len(newprog)
            ren[x] = nt
            newprog.append((nt, op, attr, a2))
    for o in outs:
        visit(o)
    return newprog, tuple(ren.get(o, o) for o in outs), newleaves, newab


def bind(fit, batches, program):
    """fit: a projected function (its first batch's); batches: [(fit_b, U_b, hdrs_b)] with hdrs_b [(mode, slots,
    values)]. Returns (bound fit or None, info). A batch's header: the one of hdrs_b every binding agrees on (an
    instance's); the bound fit's uniform leaves ('U', ('h', row, lane)), abstract ('H', row, lane)."""
    intern = Interner()
    per = []
    for fb, U, hdrs in batches:
        per.append(bind_batch(fb, U, hdrs, intern))
    # the header of each batch: the index most of its bindings name
    choice = []
    for b in per:
        votes = {}
        for ref, hits in b.items():
            for k in hits:
                votes[k[0]] = votes.get(k[0], 0) + 1
        choice.append(max(votes, key=lambda h: votes[h]) if votes else 0)
    # per ref: the (row, lane) every batch agrees on (with its header), structural preferred
    common = {}
    for ref in set().union(*[set(b) for b in per]) if per else ():
        cands = None
        how = {}
        for b, c in zip(per, choice):
            hits = {k[1:]: v for k, v in b.get(ref, {}).items() if k[0] == c}
            cands = set(hits) if cands is None else cands & set(hits)
            for k, v in hits.items():
                if v == "v":
                    how[k] = "v"
                else:
                    how.setdefault(k, "s")
            if not cands:
                break
        if cands:
            best = sorted(cands, key=lambda k: (how.get(k) != "s", k))[0]
            common[ref] = (best, how.get(best, "s"))
    defs = {t: (op, attr, args) for t, op, attr, args in fit["prog"]}
    roles = fit["roles"]
    prog, leaves_, abstract, used = [], [], [], {}
    info = {"structural": 0, "value": 0, "unbound": [], "choice": choice,
            "modes": [hdrs[c][0] for (fb, U, hdrs), c in zip(batches, choice)]}
    newref = {}

    def leaf_of(key, role, ab):
        if key not in used:
            used[key] = "L%d" % len(leaves_)
            leaves_.append(role)
            abstract.append(ab)
        return used[key]

    def mesh_role(j):
        """A uniform leaf the header does not hold that is the batch's mesh (a lone vertex's input: the strip's
        first, 1332's normals): per batch one vertex i and one quadword from TOP and lane - fitted over the batches
        as a mesh role (generalize._fit_slot; one vertex alone: ('V', 3, rel - 3 i, lane))."""
        import generalize
        pts = {}
        for fb, _U, _h in batches:
            d = fb["roles"][j]
            if d[0] != "U" or d[1][0] != "m" or fb.get("top") is None or len(set(fb["loops"])) != 1:
                return None
            if d[1][1] >= 932:            # (the headers' rows: RenderWare's from 932)
                return None
            pt = ("m", (d[1][1] - fb["top"]) % 1024, d[1][2])
            if pts.setdefault(fb["loops"][0], pt) != pt:
                return None
        if len(pts) == 1:
            i, (_m, rel, lane) = pts.popitem()
            return ("V", 3, rel - 3 * i, lane)
        r = generalize._fit_slot(sorted(pts.items()), None)
        sg = lambda a: a - 1024 if a >= 512 else a      # noqa: E731
        if r[0] == "V":
            return ("V", r[1], sg(r[2]), r[3])
        if r[0] == "G4":
            return ("G4", r[1], r[2])
        if r[0] == "P":
            return ("P", r[1], r[2], tuple((None, None) if a is None else (sg(a), lane) for a, lane in r[3]))
        return None

    cover = {}

    def covered(ref):
        """Whether ref's uniform subterm is made of header words matched structurally (and constants): its value
        then made as the microcode makes it, in preference to a word equal by value alone."""
        if ref[0] == "c":
            return True
        if ref in cover:
            return cover[ref]
        if ref in common and common[ref][1] == "s":
            r = True
        elif ref[0] == "L" or ref not in defs:
            r = False
        else:
            r = all(covered(a) for a in defs[ref][2])
        cover[ref] = r
        return r

    def walk(ref):
        if ref[0] == "c":
            return ref
        if ref in newref:
            return newref[ref]
        if ref in common and not (common[ref][1] == "v" and ref in defs and
                                  all(covered(x) for x in defs[ref][2])):
            (row, lane), how = common[ref]
            info["structural" if how == "s" else "value"] += 1
            if how == "v":
                info.setdefault("by_value", []).append((fmt(ref, defs, roles), row, lane))
            r = leaf_of(("H", row, lane), ("U", ("h", row, lane)), ("H", row, lane))
        elif ref[0] == "L":
            j = int(ref[1:])
            role = roles[j]
            mr = mesh_role(j) if role[0] == "U" else None
            if mr is not None:
                info["mesh"] = info.get("mesh", 0) + 1
                r = leaf_of(("L", mr), mr, mr)
            elif role[0] == "U":
                info["unbound"].append(role[1])
                r = leaf_of(("U", role[1]), role, ("?", role[1]))
            else:
                r = leaf_of(("L", fit["abstract"][j]), role, fit["abstract"][j])
        else:
            op, attr, args = defs[ref]
            a2 = tuple(walk(a) for a in args)
            r = "t%d" % len(prog)
            prog.append((r, op, attr, a2))
        newref[ref] = r
        return r
    outs = tuple(walk(o) for o in fit["outs"])
    # hash-cons the bound program (the uniform part made per vertex may repeat)
    # and two canonical rewrites, exact for every value: RSQRT 1.0 / sqrt(x) (FDIV's) is ERSQRT (the EFU's), both
    # _quot(1.0, the root toward zero), KMAX at 0; ADD +0 of an op's result never -0 (`positive`) is it
    # MUL by 1.0 of an op's float result is it (every float op's result is clean: no denormal, no infinity -
    # only a leaf, memory as loaded, may not be)
    seen, ren, prog2, positive, clean = {}, {}, [], set(), set()
    for t, op, attr, args in prog:
        a2 = tuple(ren.get(a, a) for a in args)
        if op == "rsqrt" and a2[0] == "c3F800000":
            op, a2 = "ersqrt", (a2[1],)
            info["rewrites"] = info.get("rewrites", 0) + 1
        if op == "add" and "c00000000" in a2:
            other = a2[1] if a2[0] == "c00000000" else a2[0]
            if other in positive:
                ren[t] = other
                info["rewrites"] = info.get("rewrites", 0) + 1
                continue
        if op == "mul" and "c3F800000" in a2:
            other = a2[1] if a2[0] == "c3F800000" else a2[0]
            if other in clean:
                ren[t] = other
                info["rewrites"] = info.get("rewrites", 0) + 1
                continue
        key = (op, attr, a2)
        if key in seen:
            ren[t] = seen[key]
            continue
        nt = "t%d" % len(prog2)
        seen[key] = nt
        ren[t] = nt
        prog2.append((nt, op, attr, a2))
        # (never -0, and clean: ADD +0 of it is it. ERSQRT, SQRT, ABS and ITOF never make -0; an ADD makes -0
        # only of two -0s, so one with an operand never -0 - a constant without its sign, e.g. ADD +0 - never does)
        if op in ("ersqrt", "sqrt", "esqrt", "abs", "itof") or (op == "add" and any(
                x in positive or (x[0] == "c" and not int(x[1:], 16) >> 31) for x in a2)):
            positive.add(nt)
        if op in CLEAN_OPS:
            clean.add(nt)
    outs = tuple(ren.get(o, o) for o in outs)
    if info["unbound"]:
        return None, info
    prog2, outs, leaves_, abstract = canonical(prog2, outs, leaves_, abstract)
    text = "\n".join("%s=%s%s(%s)" % (t, op, "" if attr is None else "<%s>" % (attr,), ",".join(args))
                     for t, op, attr, args in prog2) + "\n=>" + ",".join(outs)
    b = dict(fit)
    b.update({"prog": prog2, "outs": outs, "roles": leaves_, "abstract": tuple(abstract), "text": text,
              "uniforms": [r[1] for r in leaves_ if r[0] == "U"],
              "signature": (fit["names"], text, tuple(abstract), fit.get("out_masks"))})
    return b, info
