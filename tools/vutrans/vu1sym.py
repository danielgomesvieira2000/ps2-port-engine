"""A VU1 interpreter that runs concretely AND symbolically (vutrans milestone 1).

A port of burnout3-decomp's build/native/vu1emu.py (itself engine/host/src/vu.cpp's Interpret: values and
timing - FMAC results 4 cycles after issue, Q 7 / 13 cycles after DIV / RSQRT, P after the EFU's latency, MAC and
CLIP flags landing 4 cycles after their op, lower writes held back until the upper has read, delay slots, XGKICK,
XTOP / XITOP) in which every float lane, ACC lane, vi register, Q, P, I, the flags and every data memory word
carries (bits, node): the concrete bits (identical to vu1emu's) and a node of a hash-consed expression DAG
(Dag below) saying how those bits were computed from the run's initial state.

Leaves: ('m', qword, lane) an initial data memory word, ('vf', r, lane) / ('acc', lane) / ('vi', r) / ('Q',) /
('I',) / ('P',) / ('top',) / ('itop',) initial registers, ('clipold', slot) the initial CLIP register's
judgements, ('c', bits) constants. Constant operands fold: an op of constants is the constant.

Ops (exact VU semantics, evaluated by OPS below): add sub mul (madd = add(acc, mul), two roundings), max mini
abs ftoi<sh> itof<sh>, div sqrt rsqrt esqrt ersqrt; integer (16-bit vi) iadd isub iand ior, lo16 (ILW / MTIR),
sext16 (MFIR); clipj (a CLIP's 6-bit judgement), fcand / fcor / fceq / fcget over the last four judgements,
mac (an FMAC's MAC flags); g (identity: a value written under a data-dependent branch, see below).

Arithmetic: float32 round toward zero with flushed denormals and clamped infinities, computed with doubles and
their exact errors (TwoSum, fma) instead of vu1emu's Fractions - checked equal to vu1emu on every run tested
(checkemu.py).

Control flow: every conditional branch is recorded (pc, taken, condition node, kind). A condition is
"counter" when its DAG has no float operation or flag (it comes from header words, XTOP/XITOP and loop
counters), "data" otherwise. After a data branch, everything written until the branch's immediate
post-dominator (static CFG of the image, jr -> the BAL return sites) carries the branch's pc as a control tag
(the node's ctl set; vf / ACC / Q / P / memory writes - vi writes are not tagged), so an output decided by a
data-dependent branch says so.
"""
import math
import struct

HEAD = 4 + 6 * 4 + 32 * 16 + 32 * 4 + 16
KMAX = 0x7F7FFFFF
SIGN = 0x80000000

_I = struct.Struct("<I")
_F = struct.Struct("<f")
_D = struct.Struct("<d")
_Q = struct.Struct("<Q")


def fnv(data):
    h = 2166136261
    for x in data:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def read_runs(path, want=None):
    """Yields (index, head, micro, data) for each run of a --vu-snapshot file."""
    with open(path, "rb") as f:
        i = 0
        while True:
            head = f.read(HEAD)
            if len(head) < HEAD:
                return
            m = f.read(16384)
            d = f.read(16384)
            if want is None or i in want:
                yield i, head, m, d
            i += 1


# ---------------------------------------------------------------- arithmetic (bits in, bits out)

def clean(b):
    e = b & 0x7F800000
    if e == 0:
        return b & SIGN
    if e == 0x7F800000:
        return (b & SIGN) | KMAX
    return b


_VAL = {}


def val(b):
    """The cleaned float's value as a double (signed zero kept)."""
    v = _VAL.get(b)
    if v is None:
        e = b & 0x7F800000
        if e == 0:
            v = -0.0 if b & SIGN else 0.0
        else:
            if e == 0x7F800000:
                b2 = (b & SIGN) | KMAX
            else:
                b2 = b
            v = _F.unpack(_I.pack(b2))[0]
        if len(_VAL) < 1 << 21:
            _VAL[b] = v
    return v


def rz32(s, e=0.0):
    """The exact value s + e (s a double, e smaller than half its ulp, only e's sign and being zero matter)
    narrowed toward zero to float32; flushed below 2^-126, saturated to +-0x7F7FFFFF."""
    if s == 0.0:
        return SIGN if math.copysign(1.0, s) < 0 else 0
    db = _Q.unpack(_D.pack(s))[0]
    sign = SIGN if db >> 63 else 0
    exp = (db >> 52) & 0x7FF
    mant = db & 0xFFFFFFFFFFFFF
    if exp == 0x7FF:
        return sign | KMAX
    e32 = exp - 896          # exp - 1023 + 127
    if e32 >= 255:
        return sign | KMAX
    if e32 <= 0:
        return sign
    bits = (e32 << 23) | (mant >> 29)
    if (mant & 0x1FFFFFFF) == 0 and e != 0.0 and ((e < 0) != (s < 0)):
        bits -= 1
        if (bits & 0x7F800000) == 0:
            return sign
    return sign | bits


def f_add(a, b):
    x, y = val(a), val(b)
    s = x + y
    bb = s - x
    return rz32(s, (x - (s - bb)) + (y - bb))


def f_sub(a, b):
    x, y = val(a), -val(b)
    s = x + y
    bb = s - x
    return rz32(s, (x - (s - bb)) + (y - bb))


def f_mul(a, b):
    return rz32(val(a) * val(b))      # exact: 24 x 24 bits in a double


def order(b):
    m = b & 0x7FFFFFFF
    return -m if b & SIGN else m


def f_max(a, b):
    return b if order(a) < order(b) else a


def f_mini(a, b):
    return b if order(b) < order(a) else a


def f_abs(a):
    return clean(a) & 0x7FFFFFFF


def s32(v):
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & SIGN else v


def s16(v):
    v &= 0xFFFF
    return v - 0x10000 if v & 0x8000 else v


def f_itof(b, sh):
    x = s32(b) / float(1 << sh)          # exact in a double
    if x == 0.0:
        return 0
    return rz32(x)


def f_ftoi(b, sh):
    v = val(b) * float(1 << sh)          # exact
    if v >= 2147483647.0:
        return 0x7FFFFFFF
    if v <= -2147483648.0:
        return 0x80000000
    return int(v) & 0xFFFFFFFF


def _quot(n, d):
    """n / d (doubles, d != 0, n != 0) toward zero to float32."""
    q = n / d
    r = math.fma(-q, d, n)               # exact residual n - q*d
    return rz32(q, (r / d) if r != 0.0 else 0.0) if q != 0.0 else rz32(q)


def f_div(num, den):
    d = val(den)
    if d == 0.0:
        return ((num ^ den) & SIGN) | KMAX
    n = val(num)
    if n == 0.0:
        return 0
    return _quot(n, d)


def _sqrt_down(x):
    """sqrt(x) toward zero as a double (x >= 0)."""
    r = math.sqrt(x)
    if r != 0.0 and math.fma(r, r, -x) > 0.0:
        r = math.nextafter(r, 0.0)
    return r


def f_sqrt(den):
    x = abs(val(den))
    if x == 0.0:
        return 0
    r = math.sqrt(x)
    e = math.fma(r, r, -x)
    return rz32(r, -e)


def f_rsqrt(num, den):
    root = _sqrt_down(abs(val(den)))
    if root == 0.0:
        return (num & SIGN) | KMAX
    n = val(num)
    if n == 0.0:
        return 0
    return _quot(n, root)


def f_ersqrt(den):
    root = _sqrt_down(abs(val(den)))
    if root == 0.0:
        return KMAX
    return _quot(1.0, root)


def clip_judge(x, y, z, w):
    aw = abs(val(w))
    j = 0
    for i, b in enumerate((x, y, z)):
        v = val(b)
        if v > aw:
            j |= 1 << (2 * i)
        if v < -aw:
            j |= 2 << (2 * i)
    return j


def mac_bits(lanes, dest):
    """MAC flags of an FMAC's written lanes (lanes: the four result words)."""
    m = 0
    for l in range(4):
        if not dest & (8 >> l):
            continue
        b = clean(lanes[l])
        if b & 0x7FFFFFFF == 0:
            m |= 8 >> l
        if b & SIGN:
            m |= 0x80 >> l
    return m


def _fc_window(js, masks):
    return sum((j & m) << (6 * i) for i, (j, m) in enumerate(zip(js, masks)))


# Every op's evaluator: OPS[op](attr, *argument bits) -> bits. Used by the emitted evaluators and to check DAGs.
OPS = {
    "add": lambda at, a, b: f_add(a, b),
    "sub": lambda at, a, b: f_sub(a, b),
    "mul": lambda at, a, b: f_mul(a, b),
    "max": lambda at, a, b: f_max(a, b),
    "mini": lambda at, a, b: f_mini(a, b),
    "abs": lambda at, a: f_abs(a),
    "ftoi": lambda at, a: f_ftoi(a, at),
    "itof": lambda at, a: f_itof(a, at),
    "div": lambda at, a, b: f_div(a, b),
    "sqrt": lambda at, a: f_sqrt(a),
    "rsqrt": lambda at, a, b: f_rsqrt(a, b),
    "esqrt": lambda at, a: f_sqrt(a),
    "ersqrt": lambda at, a: f_ersqrt(a),
    "iadd": lambda at, a, b: (a + b) & 0xFFFF,
    "isub": lambda at, a, b: (a - b) & 0xFFFF,
    "iand": lambda at, a, b: a & b & 0xFFFF,
    "ior": lambda at, a, b: (a | b) & 0xFFFF,
    "lo16": lambda at, a: a & 0xFFFF,
    "sext16": lambda at, a: (a | 0xFFFF0000) if a & 0x8000 else (a & 0xFFFF),
    "clipj": lambda at, x, y, z, w: clip_judge(x, y, z, w),
    # at: per used slot its 6-bit mask; args: the judgements of those slots (newest first)
    "fcand": lambda at, *js: 1 if _fc_window(js, at) else 0,
    "fcor": lambda at, *js: 1 if (_fc_window(js, at[0]) | at[1]) & 0xFFFFFF == 0xFFFFFF else 0,
    "fceq": lambda at, *js: 1 if _fc_window(js, (63,) * len(js)) == at else 0,
    "fcget": lambda at, *js: _fc_window(js, (63, 63)),
    "mac": lambda at, *ls: mac_bits(_spread(ls, at), at),
    "g": lambda at, a: a,
    "ceq": lambda at, a, b: 1 if s16(a) == s16(b) else 0,
    "cne": lambda at, a, b: 1 if s16(a) != s16(b) else 0,
    "cltz": lambda at, a: 1 if s16(a) < 0 else 0,
    "cgtz": lambda at, a: 1 if s16(a) > 0 else 0,
    "clez": lambda at, a: 1 if s16(a) <= 0 else 0,
    "cgez": lambda at, a: 1 if s16(a) >= 0 else 0,
    "cand": lambda at, a: 1 if a & at else 0,          # (flag tests on a vi: FMAND results etc.)
}

FLOAT_OPS = {"add", "sub", "mul", "max", "mini", "abs", "ftoi", "itof", "div", "sqrt", "rsqrt", "esqrt",
             "ersqrt", "clipj", "fcand", "fcor", "fceq", "fcget", "mac"}


def _spread(ls, dest):
    out = [0, 0, 0, 0]
    i = 0
    for l in range(4):
        if dest & (8 >> l):
            out[l] = ls[i]
            i += 1
    return out


# ---------------------------------------------------------------- the expression DAG

FDEP = 1          # depends on a float operation or a flag


class Dag:
    """Hash-consed nodes: nodes[id] = (op, args, attr); flags[id]; ctl[id] (frozenset of branch pcs or None)."""

    def __init__(self):
        self.nodes = []
        self.flags = []
        self.ctl = []
        self.bits = []          # the bits the node had when made (constants: their value)
        self.index = {}
        self.active = None      # the control tags in force (frozenset) or None

    def _new(self, key, flags, ctl, bits):
        n = len(self.nodes)
        self.nodes.append(key[:3])
        self.flags.append(flags)
        self.ctl.append(ctl)
        self.bits.append(bits)
        self.index[key] = n
        return n

    def leaf(self, *key):
        k = (key[0], (), key[1:], None)
        n = self.index.get(k)
        if n is None:
            n = self._new(k, 0, None, None)
        return n

    def const(self, bits):
        k = ("c", (), bits, None)
        n = self.index.get(k)
        if n is None:
            n = self._new(k, 0, None, bits)
        return n

    def is_const(self, n):
        return self.nodes[n][0] == "c"

    def mk(self, op, args, attr, bits):
        """The node of op(args) (attr: its immediate), whose value on this run is bits."""
        nodes = self.nodes
        ctl = self.active
        allc = True
        fl = FDEP if op in FLOAT_OPS else 0
        for a in args:
            if nodes[a][0] != "c":
                allc = False
            fl |= self.flags[a]
            c = self.ctl[a]
            if c:
                ctl = c if ctl is None else (ctl | c)
        if allc and op not in ("clipj", "mac", "fcand", "fcor", "fceq", "fcget") and not ctl:
            return self.const(bits)
        if op == "sub" and args[0] == args[1] and not ctl:
            return self.const(0)        # a - a: +0 for every VU value (no NaN / inf; -0 - -0 is +0 too)
        key = (op, args, attr, ctl)
        n = self.index.get(key)
        if n is None:
            n = self._new(key, fl, ctl, bits)
        return n

    def tag(self, n):
        """n as written under the control tags in force."""
        act = self.active
        if not act:
            return n
        c = self.ctl[n]
        if c is not None and act <= c:
            return n
        key = ("g", (n,), None, act if c is None else (act | c))
        m = self.index.get(key)
        if m is None:
            m = self._new(key, self.flags[n], key[3], self.bits[n])
        return m

    def fmt(self, n, depth=6, names=None):
        op, args, attr = self.nodes[n]
        if names and n in names:
            return names[n]
        if op == "c":
            return "%08X" % attr
        if not args:
            return op + "".join("[%s]" % (x,) for x in attr)
        if depth <= 0:
            return "#%d" % n
        inner = ", ".join(self.fmt(a, depth - 1, names) for a in args)
        if attr is not None and op not in ("mac",):
            return "%s<%s>(%s)" % (op, attr, inner)
        return "%s(%s)" % (op, inner)

    def evaluate(self, n, leafval, memo=None):
        """The node's bits with the leaves' bits from leafval(node id) (iterative; memo shared across calls)."""
        if memo is None:
            memo = {}
        nodes = self.nodes
        stack = [n]
        while stack:
            x = stack[-1]
            if x in memo:
                stack.pop()
                continue
            op, args, attr = nodes[x]
            if op == "c":
                memo[x] = attr
                stack.pop()
                continue
            if not args:
                memo[x] = leafval(x)
                stack.pop()
                continue
            todo = [a for a in args if a not in memo]
            if todo:
                stack.extend(todo)
                continue
            memo[x] = OPS[op](attr, *[memo[a] for a in args])
            stack.pop()
        return memo[n]

    def leaves(self, n, out=None):
        """The set of leaf nodes under n."""
        if out is None:
            out = set()
        seen = set()
        stack = [n]
        nodes = self.nodes
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            op, args, attr = nodes[x]
            if not args:
                if op != "c":
                    out.add(x)
            else:
                stack.extend(args)
        return out


# ---------------------------------------------------------------- decoding helpers (as vu.cpp / vu1emu)

def lanebit(l):
    return 8 >> l


def upper_reads(word):
    funct = word & 0x3F
    dest = (word >> 21) & 0xF
    fs, ft = (word >> 11) & 31, (word >> 16) & 31
    if funct < 0x1C:
        return [(fs, dest), (ft, lanebit(funct & 3))]
    if funct < 0x28:
        return [(fs, dest)]
    if funct < 0x30:
        ln = 0xE if funct == 0x2E else dest
        return [(fs, ln), (ft, ln)]
    if funct < 0x3C:
        return []
    index = (word & 3) | ((word >> 4) & 0x7C)
    if index == 0x2F:
        return []
    if index == 0x1F:
        return [(fs, 0xE), (ft, 1)]
    if index < 0x10 or (0x18 <= index < 0x1C):
        return [(fs, dest), (ft, lanebit(index & 3))]
    if index < 0x28:
        return [(fs, dest)]
    ln = 0xE if index == 0x2E else dest
    return [(fs, ln), (ft, ln)]


def lower_reads(word):
    op = word >> 25
    dest = (word >> 21) & 0xF
    is_, it = (word >> 11) & 31, (word >> 16) & 31
    fsf, ftf = lanebit((word >> 21) & 3), lanebit((word >> 23) & 3)
    if op == 0x01:
        return [(is_, dest)]
    if op != 0x40 or (word & 0x3F) < 0x3C:
        return []
    index = (word & 3) | ((word >> 4) & 0x7C)
    if index == 0x30:
        return [(is_, dest)]
    if index == 0x31:
        return [(is_, ((dest << 3) | (dest >> 1)) & 0xF)]
    if index in (0x35, 0x37):
        return [(is_, dest)]
    if index in (0x38, 0x3A):
        return [(is_, fsf), (it, ftf)]
    if index == 0x39:
        return [(it, ftf)]
    if index in (0x3C, 0x42, 0x43):
        return [(is_, fsf)]
    if index in (0x78, 0x79, 0x7A, 0x7C, 0x7D, 0x7E):
        return [(is_, fsf)]
    return []


def _lower_index(lw):
    if lw >> 25 == 0x40 and (lw & 0x3F) >= 0x3C:
        return (lw & 3) | ((lw >> 4) & 0x7C)
    return -1


def imm11(word):
    return (word & 0x7FF) - 0x800 if word & 0x400 else word & 0x7FF


EFU_LATENCY = {0x70: 11, 0x71: 18, 0x72: 18, 0x73: 24, 0x74: 54, 0x75: 54, 0x76: 12, 0x78: 12, 0x79: 18,
               0x7A: 12, 0x7C: 29, 0x7D: 54, 0x7E: 44}
COND_OPS = {0x28: "ceq", 0x29: "cne", 0x2C: "cltz", 0x2D: "cgtz", 0x2E: "clez", 0x2F: "cgez"}


class Decoded:
    __slots__ = ("lower", "upper", "imm", "reads", "qwait", "pwait", "ebit")


def decode_image(micro):
    out = {}
    for pc in range(0, 0x4000, 8):
        lw, uw = struct.unpack_from("<II", micro, pc)
        d = Decoded()
        d.lower, d.upper = lw, uw
        d.imm = bool(uw & 0x80000000)
        rd = upper_reads(uw) + ([] if d.imm else lower_reads(lw))
        d.reads = tuple((r, l) for r, ln in rd if r for l in range(4) if ln & (8 >> l))
        ix = -1 if d.imm else _lower_index(lw)
        d.qwait = ix in (0x38, 0x39, 0x3A, 0x3B)
        d.pwait = ix in EFU_LATENCY or ix == 0x7B
        d.ebit = bool(uw & 0x40000000)
        out[pc] = d
    return out


# ---------------------------------------------------------------- static CFG and post-dominators

def build_ipdom(micro):
    """Immediate post-dominator of each pc as a CFG node (EXIT = -1). A branch at P diverges after its delay
    slot: node P+8's successors are the target and P+16. jr / jalr go to every BAL / JALR return site and EXIT."""
    EXIT = -1
    succ = {}
    words = [struct.unpack_from("<II", micro, pc) for pc in range(0, 0x4000, 8)]
    returns = set()
    for i, (lw, uw) in enumerate(words):
        if uw & 0x80000000:
            continue
        op = lw >> 25
        if op in (0x21, 0x25):
            returns.add((i * 8 + 16) & 0x3FFF)
    override = {}
    for i, (lw, uw) in enumerate(words):
        pc = i * 8
        nxt = (pc + 8) & 0x3FFF
        succ[pc] = [nxt]
        if uw & 0x40000000:
            override[nxt] = [EXIT]
        if uw & 0x80000000:
            continue
        op = lw >> 25
        tgt = (pc + 8 + imm11(lw) * 8) & 0x3FFF
        if op in (0x20, 0x21):
            override.setdefault(nxt, [tgt])
        elif op in COND_OPS:
            override.setdefault(nxt, [tgt, (pc + 16) & 0x3FFF])
        elif op in (0x24, 0x25):
            override.setdefault(nxt, sorted(returns) + [EXIT])
    succ.update(override)
    # reverse graph; post-dominators by the iterative algorithm over the reverse post-order from EXIT
    preds = {EXIT: []}
    for p in succ:
        preds.setdefault(p, [])
    for p, ss in succ.items():
        for s in ss:
            preds.setdefault(s, []).append(p)
    order, seen = [], {EXIT}
    stack = [(EXIT, iter(preds[EXIT]))]
    while stack:
        n, it = stack[-1]
        nx = next(it, None)
        if nx is None:
            order.append(n)
            stack.pop()
        elif nx not in seen:
            seen.add(nx)
            stack.append((nx, iter(preds.get(nx, ()))))
    order.reverse()                      # reverse post-order on the reverse graph, EXIT first
    num = {n: i for i, n in enumerate(order)}
    ipdom = {EXIT: EXIT}

    def meet(a, b):
        while a != b:
            while num[a] > num[b]:
                a = ipdom[a]
            while num[b] > num[a]:
                b = ipdom[b]
        return a
    changed = True
    while changed:
        changed = False
        for n in order[1:]:
            new = None
            for s in succ.get(n, ()):
                if s in ipdom and s in num:
                    new = s if new is None else meet(s, new)
            if new is not None and ipdom.get(n) != new:
                ipdom[n] = new
                changed = True
    return ipdom


# ---------------------------------------------------------------- the interpreter

class SymVu:
    """One run. dag: a Dag (one per run, or shared). Fields mirror vu1emu.Vu; *_n are the nodes."""

    def __init__(self, head, micro, data, dag=None, decoded=None, ipdom=None, engine_moves=False):
        self.dag = dag if dag is not None else Dag()
        D = self.dag
        self.pc = struct.unpack_from("<I", head, 0)[0]
        fl = struct.unpack_from("<6I", head, 4)
        self.mac, self.status, self.clip, self.p, self.top, self.itop = fl
        w = struct.unpack_from("<128I", head, 28)
        self.vf = [list(w[4 * r:4 * r + 4]) for r in range(32)]
        self.vf_n = [[D.leaf("vf", r, l) for l in range(4)] for r in range(32)]
        self.vf_n[0] = [D.const(b) for b in self.vf[0]]
        self.vc = list(struct.unpack_from("<32I", head, 28 + 512))
        self.vi_n = [D.leaf("vi", r) for r in range(16)]
        self.vi_n[0] = D.const(0)
        self.q_n = D.leaf("Q")
        self.i_n = D.leaf("I")
        self.p_n = D.leaf("P")
        self.acc = list(struct.unpack_from("<4I", head, 28 + 512 + 128))
        self.acc_n = [D.leaf("acc", l) for l in range(4)]
        self.mac_n = D.leaf("mac")
        # CLIP: the four newest judgements (newest first) as nodes; the initial register's as leaves
        self.clipw = [D.leaf("clipold", s) for s in range(4)]
        self.clipw_latest = list(self.clipw)
        self.micro = micro
        self.mem = list(struct.unpack("<4096I", data))
        self.mem_n = [None] * 4096
        self.dec = decoded if decoded is not None else decode_image(micro)
        self.ipdom = ipdom
        self.engine_moves = engine_moves
        self.cycle = 0
        self.ready = [[0] * 4 for _ in range(32)]
        self.q_busy = False
        self.q_ready = 0
        self.q_next = 0
        self.q_next_n = None
        self.p_busy = False
        self.p_ready = 0
        self.p_next = 0
        self.p_next_n = None
        self.flags = []          # (ready, mac, mac_n, clip, clipw, sets_mac, sets_clip)
        self.mac_latest = self.mac
        self.mac_latest_n = self.mac_n
        self.clip_latest = self.clip
        self.kicks = []          # (pair count, [(addr, [qword: [(bits, node)] x 4] ...)])
        self.branches = []       # (pc, taken, cond node, kind)
        self.count = 0
        self.regions = []        # active data-dependent regions: (ipdom pc, branch pc)

    # memory
    def mnode(self, i):
        n = self.mem_n[i]
        if n is None:
            n = self.dag.leaf("m", i >> 2, i & 3)
            self.mem_n[i] = n
        return n

    def quad(self, a):
        a = (a & 0x3FF) * 4
        return self.mem[a:a + 4], [self.mnode(a + l) for l in range(4)]

    def setquad(self, a, v, vn, dest):
        a = (a & 0x3FF) * 4
        tag = self.dag.tag
        for l in range(4):
            if dest & (8 >> l):
                self.mem[a + l] = v[l] & 0xFFFFFFFF
                self.mem_n[a + l] = tag(vn[l])

    def kick(self, addr):
        out = []
        addr &= 0x3FF
        for _ in range(4096):
            tb, _tn = self.quad(addr)
            low = tb[0] | tb[1] << 32
            nloop = low & 0x7FFF
            eop = (low >> 15) & 1
            flg = (low >> 58) & 3
            nreg = (low >> 60) or 16
            qw = nloop * nreg if flg == 0 else ((nloop * nreg + 1) // 2 if flg == 1 else nloop)
            pk = []
            for i in range(qw + 1):
                b, n = self.quad(addr + i)
                pk.append(list(zip(b, n)))
            out.append((addr, pk))
            addr = (addr + qw + 1) & 0x3FF
            if eop:
                break
        self.kicks.append((self.count, out))

    def land(self):
        while self.flags and self.flags[0][0] <= self.cycle:
            r, m, mn, c, cw, sm, sc = self.flags.pop(0)
            if sm:
                self.mac = m
                self.mac_n = mn
            if sc:
                self.clip = c
                self.clipw = cw
        if self.q_busy and self.q_ready <= self.cycle:
            self.vc[22] = self.q_next
            self.q_n = self.q_next_n
            self.q_busy = False
        if self.p_busy and self.p_ready <= self.cycle:
            self.p = self.p_next
            self.p_n = self.p_next_n
            self.p_busy = False

    # ---- upper
    def upper(self, word):
        """Runs the upper; returns (written reg, lanes, set_flags)."""
        D = self.dag
        mk = D.mk
        funct = word & 0x3F
        dest = (word >> 21) & 0xF
        fd, fs, ft = (word >> 6) & 31, (word >> 11) & 31, (word >> 16) & 31
        vf, vfn = self.vf, self.vf_n
        to_acc = False
        flags = True
        if funct < 0x3C:
            if funct >= 0x30:
                return -1, 0, False
            if funct < 0x1C:
                op = ("add", "sub", "madd", "msub", "max", "mini", "mul")[funct >> 2]
                second = ("bc", funct & 3)
            elif funct < 0x28:
                op, second = {0x1C: ("mul", "Q"), 0x1D: ("max", "I"), 0x1E: ("mul", "I"), 0x1F: ("mini", "I"),
                              0x20: ("add", "Q"), 0x21: ("madd", "Q"), 0x22: ("add", "I"), 0x23: ("madd", "I"),
                              0x24: ("sub", "Q"), 0x25: ("msub", "Q"), 0x26: ("sub", "I"),
                              0x27: ("msub", "I")}[funct]
            else:
                op = {0x28: "add", 0x29: "madd", 0x2A: "mul", 0x2B: "max", 0x2C: "sub", 0x2D: "msub",
                      0x2E: "opmsub", 0x2F: "mini"}[funct]
                second = ("reg",)
            target = fd
        else:
            index = (word & 3) | ((word >> 4) & 0x7C)
            if index == 0x1F:   # CLIP
                v, vn = vf[fs], vfn[fs]
                w, wn = vf[ft][3], vfn[ft][3]
                j = clip_judge(v[0], v[1], v[2], w)
                jn = mk("clipj", (vn[0], vn[1], vn[2], wn), None, j)
                self.clip_latest = ((self.clip_latest << 6) | j) & 0xFFFFFF
                self.clipw_latest = [jn] + self.clipw_latest[:3]
                return -1, 0, "clip"
            if index == 0x2F:
                return -1, 0, False
            if 0x10 <= index <= 0x17 or index == 0x1D:
                out = list(vf[ft])
                outn = list(vfn[ft])
                for l in range(4):
                    if dest & (8 >> l):
                        a, an = vf[fs][l], vfn[fs][l]
                        if index == 0x1D:
                            out[l] = f_abs(a)
                            outn[l] = mk("abs", (an,), None, out[l])
                        elif index < 0x14:
                            sh = (0, 4, 12, 15)[index & 3]
                            out[l] = f_itof(a, sh)
                            outn[l] = mk("itof", (an,), sh, out[l])
                        else:
                            sh = (0, 4, 12, 15)[index & 3]
                            out[l] = f_ftoi(a, sh)
                            outn[l] = mk("ftoi", (an,), sh, out[l])
                if ft:
                    vf[ft] = out
                    vfn[ft] = [D.tag(x) for x in outn]
                    return ft, dest, False
                return -1, 0, False
            to_acc = True
            target = -2
            if index < 0x1C:
                op = {0x00: "add", 0x04: "sub", 0x08: "madd", 0x0C: "msub", 0x18: "mul"}[index & 0x1C]
                second = ("bc", index & 3)
            elif index < 0x28:
                op, second = {0x1C: ("mul", "Q"), 0x1E: ("mul", "I"), 0x20: ("add", "Q"), 0x21: ("madd", "Q"),
                              0x22: ("add", "I"), 0x23: ("madd", "I"), 0x24: ("sub", "Q"), 0x25: ("msub", "Q"),
                              0x26: ("sub", "I"), 0x27: ("msub", "I")}[index]
            else:
                op = {0x28: "add", 0x29: "madd", 0x2A: "mul", 0x2C: "sub", 0x2D: "msub", 0x2E: "opmula"}[index]
                second = ("reg",)
        if op in ("max", "mini"):
            flags = False
        a, an = vf[fs], vfn[fs]
        if second[0] == "bc":
            bv = [vf[ft][second[1]]] * 4
            bn = [vfn[ft][second[1]]] * 4
        elif second[0] == "reg":
            bv, bn = vf[ft], vfn[ft]
        elif second == "Q":
            bv, bn = [self.vc[22]] * 4, [self.q_n] * 4
        else:
            bv, bn = [self.vc[21]] * 4, [self.i_n] * 4
        acc, accn = self.acc, self.acc_n
        res = [0, 0, 0, 0]
        resn = [None] * 4
        if op in ("opmula", "opmsub"):
            pairs = ((1, 2), (2, 0), (0, 1))
            for l in range(3):
                i, j = pairs[l]
                p = f_mul(a[i], bv[j])
                pn = mk("mul", (an[i], bn[j]), None, p)
                if op == "opmula":
                    res[l], resn[l] = p, pn
                else:
                    res[l] = f_sub(acc[l], p)
                    resn[l] = mk("sub", (accn[l], pn), None, res[l])
            res[3] = 0
            resn[3] = D.const(0)
        else:
            for l in range(4):
                if not dest & (8 >> l):
                    continue
                x, y = a[l], bv[l]
                xn, yn = an[l], bn[l]
                if op == "add":
                    r = f_add(x, y); rn = mk("add", (xn, yn), None, r)
                elif op == "sub":
                    r = f_sub(x, y); rn = mk("sub", (xn, yn), None, r)
                elif op == "mul":
                    r = f_mul(x, y); rn = mk("mul", (xn, yn), None, r)
                elif op == "madd" or op == "msub":
                    p = f_mul(x, y)
                    pn = mk("mul", (xn, yn), None, p)
                    if op == "madd":
                        r = f_add(acc[l], p); rn = mk("add", (accn[l], pn), None, r)
                    else:
                        r = f_sub(acc[l], p); rn = mk("sub", (accn[l], pn), None, r)
                elif op == "max":
                    r = f_max(x, y); rn = mk("max", (xn, yn), None, r)
                else:
                    r = f_mini(x, y); rn = mk("mini", (xn, yn), None, r)
                res[l], resn[l] = r, rn
        if flags:
            m = mac_bits(res, dest)
            self.mac_latest = m
            ls = tuple(resn[l] for l in range(4) if dest & (8 >> l))
            self.mac_latest_n = mk("mac", ls, dest, m)
        if to_acc:
            for l in range(4):
                if dest & (8 >> l):
                    acc[l] = res[l]
                    accn[l] = D.tag(resn[l])
            return -1, 0, flags
        if target > 0:
            for l in range(4):
                if dest & (8 >> l):
                    vf[target][l] = res[l]
                    vfn[target][l] = D.tag(resn[l])
            return target, dest, flags
        return -1, 0, flags

    # ---- lower
    def lower(self, word, pc):
        D = self.dag
        mk = D.mk
        P = {"vf": None, "vi": None, "q": None, "p": None, "branch": None}
        op = word >> 25
        dest = (word >> 21) & 0xF
        it, is_ = (word >> 16) & 31, (word >> 11) & 31
        vi, vin = self.vc, self.vi_n
        target = (pc + 8 + imm11(word) * 8) & 0x3FFF

        def setvi(r, v, n):
            # (vi writes are not control-tagged: they are counters and addresses - tagging them taints every
            # later loop test; an integer that reaches an output from a float or flag is FDEP anyway)
            if 0 < r < 16:
                vi[r] = v & 0xFFFF
                vin[r] = n

        def load(addr):
            b, n = self.quad(addr)
            P["vf"] = (it, dest, b, n)

        def first(b, n):
            for l in range(4):
                if dest & (8 >> l):
                    return b[l], n[l]
            return 0, D.const(0)

        def iop(name, x, xn, y, yn, v):
            return mk(name, (xn, yn), None, v & 0xFFFF)

        def addimm(r, imm):
            v = (vi[r] + imm) & 0xFFFF
            return v, mk("iadd", (vin[r], D.const(imm & 0xFFFF)), None, v)

        if op == 0x40:
            funct = word & 0x3F
            if 0x30 <= funct <= 0x35:
                idr = (word >> 6) & 31
                if funct == 0x32:
                    imm = idr - 32 if idr & 0x10 else idr
                    v, n = addimm(is_, imm)
                    setvi(it, v, n)
                    return P
                x, y = vi[is_], vi[it]
                name = {0x30: "iadd", 0x31: "isub", 0x34: "iand", 0x35: "ior"}[funct]
                v = OPS[name](None, x, y)
                setvi(idr, v, mk(name, (vin[is_], vin[it]), None, v))
                return P
            index = (word & 3) | ((word >> 4) & 0x7C)
            fsf, ftf = (word >> 21) & 3, (word >> 23) & 3
            if index == 0x34:
                load(vi[is_]); v, n = addimm(is_, 1); setvi(is_, v, n); return P
            if index == 0x35:
                self.setquad(vi[it], self.vf[is_], self.vf_n[is_], dest)
                v, n = addimm(it, 1); setvi(it, v, n); return P
            if index == 0x36:
                v, n = addimm(is_, -1); setvi(is_, v, n); load(vi[is_]); return P
            if index == 0x37:
                v, n = addimm(it, -1); setvi(it, v, n)
                self.setquad(vi[it], self.vf[is_], self.vf_n[is_], dest); return P
            if index == 0x3E:     # ILWR
                b, n = first(*self.quad(vi[is_]))
                setvi(it, b, mk("lo16", (n,), None, b & 0xFFFF)); return P
            if index == 0x3F:     # ISWR
                self.setquad(vi[is_], [vi[it] & 0xFFFF] * 4, [vin[it]] * 4, dest); return P
            if index == 0x68:
                setvi(it, self.top, D.leaf("top")); return P
            if index == 0x69:
                setvi(it, self.itop, D.leaf("itop")); return P
            if index == 0x6C:
                self.kick(vi[is_]); return P
            if index == 0x64:     # MFP
                P["vf"] = (it, dest, [self.p] * 4, [self.p_n] * 4); return P
            if index == 0x30:     # MOVE
                if it:
                    P["vf"] = (it, dest, list(self.vf[is_]), list(self.vf_n[is_])); P["c2"] = True
                return P
            if index == 0x31:     # MR32
                s, sn = self.vf[is_], self.vf_n[is_]
                if it:
                    P["vf"] = (it, dest, [s[1], s[2], s[3], s[0]], [sn[1], sn[2], sn[3], sn[0]]); P["c2"] = True
                return P
            if index == 0x3C:     # MTIR
                if 0 < it < 16:
                    b = self.vf[is_][fsf] & 0xFFFF
                    P["vi"] = (it, b, mk("lo16", (self.vf_n[is_][fsf],), None, b))
                return P
            if index == 0x3D:     # MFIR
                x = vi[is_] & 0xFFFF
                if x & 0x8000:
                    x |= 0xFFFF0000
                if it:
                    n = mk("sext16", (vin[is_],), None, x)
                    P["vf"] = (it, dest, [x] * 4, [n] * 4); P["c2"] = True
                return P
            if index == 0x38:     # DIV
                num, den = self.vf[is_][fsf], self.vf[it][ftf]
                q = f_div(num, den)
                P["q"] = (q, 7, mk("div", (self.vf_n[is_][fsf], self.vf_n[it][ftf]), None, q))
                return P
            if index == 0x39:     # SQRT
                d = self.vf[it][ftf]
                q = f_sqrt(d)
                P["q"] = (q, 7, mk("sqrt", (self.vf_n[it][ftf],), None, q))
                return P
            if index == 0x3A:     # RSQRT
                num, den = self.vf[is_][fsf], self.vf[it][ftf]
                q = f_rsqrt(num, den)
                P["q"] = (q, 13, mk("rsqrt", (self.vf_n[is_][fsf], self.vf_n[it][ftf]), None, q))
                return P
            if index == 0x78:     # ESQRT
                d = self.vf[is_][fsf]
                p = f_sqrt(d)
                P["p"] = (p, EFU_LATENCY[index], mk("esqrt", (self.vf_n[is_][fsf],), None, p))
                return P
            if index == 0x79:     # ERSQRT
                d = self.vf[is_][fsf]
                p = f_ersqrt(d)
                P["p"] = (p, EFU_LATENCY[index], mk("ersqrt", (self.vf_n[is_][fsf],), None, p))
                return P
            if index in (0x3B, 0x7B):
                return P
            raise NotImplementedError("lower %08X at %04X" % (word, pc))
        if op == 0x00:
            load(vi[is_] + imm11(word)); return P
        if op == 0x01:
            self.setquad(vi[it] + imm11(word), self.vf[is_], self.vf_n[is_], dest); return P
        if op == 0x04:
            b, n = first(*self.quad(vi[is_] + imm11(word)))
            setvi(it, b, mk("lo16", (n,), None, b & 0xFFFF)); return P
        if op == 0x05:
            self.setquad(vi[is_] + imm11(word), [vi[it] & 0xFFFF] * 4, [vin[it]] * 4, dest); return P
        if op == 0x08 or op == 0x09:
            imm = ((word >> 21) & 0xF) << 11 | (word & 0x7FF)
            v, n = addimm(is_, imm if op == 0x08 else -imm)
            setvi(it, v, n); return P
        if op in (0x10, 0x12, 0x13, 0x1C):
            imm = word & 0xFFFFFF
            cw = self.clipw
            if op == 0x12:
                v = 1 if (self.clip & imm) else 0
                masks = [(imm >> (6 * s)) & 63 for s in range(4)]
                used = [s for s in range(4) if masks[s]]
                n = mk("fcand", tuple(cw[s] for s in used), tuple(masks[s] for s in used), v)
            elif op == 0x13:
                v = 1 if ((self.clip | imm) & 0xFFFFFF) == 0xFFFFFF else 0
                n = mk("fcor", tuple(cw), ((63, 63, 63, 63), imm), v)
            elif op == 0x10:
                v = 1 if (self.clip & 0xFFFFFF) == imm else 0
                n = mk("fceq", tuple(cw), imm, v)
            else:
                v = self.clip & 0xFFF
                n = mk("fcget", (cw[0], cw[1]), None, v)
                setvi(it, v, n); return P
            setvi(1, v, n); return P
        if op == 0x1A:   # FMAND
            v = vi[is_] & self.mac
            setvi(it, v, mk("iand", (vin[is_], self.mac_n), None, v & 0xFFFF)); return P
        if op == 0x1B:   # FMOR
            v = vi[is_] | self.mac
            setvi(it, v, mk("ior", (vin[is_], self.mac_n), None, v & 0xFFFF)); return P
        if op == 0x11:   # FCSET
            imm = word & 0xFFFFFF
            self.clip = self.clip_latest = imm
            self.clipw = [D.const((imm >> (6 * s)) & 63) for s in range(4)]
            self.clipw_latest = list(self.clipw)
            return P
        if op == 0x20:
            P["branch"] = target; return P
        if op == 0x21:
            v = (pc + 16) // 8
            setvi(it, v, D.const(v)); P["branch"] = target; return P
        if op == 0x24:
            P["branch"] = (vi[is_] * 8) & 0x3FFF; P["jr"] = vin[is_]; return P
        if op == 0x25:
            to = (vi[is_] * 8) & 0x3FFF
            P["jr"] = vin[is_]
            v = (pc + 16) // 8
            setvi(it, v, D.const(v)); P["branch"] = to; return P
        if op in COND_OPS:
            a, b = s16(vi[is_]), s16(vi[it])
            name = COND_OPS[op]
            taken = {0x28: a == b, 0x29: a != b, 0x2C: a < 0, 0x2D: a > 0, 0x2E: a <= 0, 0x2F: a >= 0}[op]
            args = (vin[is_], vin[it]) if op in (0x28, 0x29) else (vin[is_],)
            cn = mk(name, args, None, 1 if taken else 0)
            P["cond"] = (cn, taken, args)
            if taken:
                P["branch"] = target
            return P
        raise NotImplementedError("lower %08X at %04X" % (word, pc))

    def run(self, limit=10 ** 7):
        D = self.dag
        pc = self.pc
        delayed = None
        stopping = False
        dec = self.dec
        ready = self.ready
        pending_region = None
        ipdom = self.ipdom
        while self.count < limit:
            if self.regions:
                kept = [r for r in self.regions if r[0] != pc]
                if len(kept) != len(self.regions):
                    self.regions = kept
                    D.active = frozenset(r[1] for r in kept) or None
            d = dec[pc]
            lower, upper = d.lower, d.upper
            immediate = d.imm
            issue = self.cycle
            for r, l in d.reads:
                t = ready[r][l]
                if t > issue:
                    issue = t
            if d.qwait and self.q_busy and self.q_ready > issue:
                issue = self.q_ready
            if d.pwait and self.p_busy and self.p_ready > issue:
                issue = self.p_ready
            self.cycle = issue
            self.land()
            P = None
            if not immediate:
                P = self.lower(lower, pc)
            wr, wl, sf = self.upper(upper)
            if immediate:
                self.vc[21] = lower
                self.i_n = D.const(lower)
            if P is not None:
                if P["vf"] is not None:
                    r, ln, val_, vn = P["vf"]
                    if r > 0:
                        changed = True
                        if self.engine_moves and "c2" in P:
                            cur = self.vf[r]
                            changed = any(ln & (8 >> l) and cur[l] != val_[l] for l in range(4))
                        for l in range(4):
                            if ln & (8 >> l) and not (wr == r and wl & (8 >> l)):
                                self.vf[r][l] = val_[l]
                                self.vf_n[r][l] = D.tag(vn[l])
                                if changed:
                                    ready[r][l] = self.cycle + 4
                if P["q"] is not None:
                    self.q_next, lat, self.q_next_n = P["q"]
                    self.q_ready = self.cycle + lat
                    self.q_busy = True
                    self.q_next_n = D.tag(self.q_next_n)
                if P["p"] is not None:
                    self.p_next, lat, self.p_next_n = P["p"]
                    self.p_ready = self.cycle + lat
                    self.p_busy = True
                    self.p_next_n = D.tag(self.p_next_n)
                if P["vi"] is not None:
                    self.vc[P["vi"][0]] = P["vi"][1]
                    self.vi_n[P["vi"][0]] = P["vi"][2]
            if wr > 0:
                for l in range(4):
                    if wl & (8 >> l):
                        ready[wr][l] = self.cycle + 4
            if sf:
                self.flags.append((self.cycle + 4, self.mac_latest, self.mac_latest_n, self.clip_latest,
                                   list(self.clipw_latest), sf != "clip", sf == "clip"))
            self.vc[0] = 0
            if pending_region is not None:
                # the delay slot has run: the data-dependent region starts
                self.regions.append(pending_region)
                D.active = frozenset(r[1] for r in self.regions)
                pending_region = None
            if P is not None and "cond" in P:
                cn, taken, cargs = P["cond"]
                data = bool(D.flags[cn] & FDEP)
                self.branches.append((pc, taken, cn, "data" if data else "counter"))
                if data and ipdom is not None:
                    pd = ipdom.get((pc + 8) & 0x3FFF, -1)
                    pending_region = (pd, pc)
            elif P is not None and "jr" in P:
                jn = P["jr"]
                data = bool(D.flags[jn] & FDEP)
                self.branches.append((pc, P["branch"], jn, "jr-data" if data else "jr"))
            self.cycle += 1
            self.count += 1
            if stopping:
                self.pc = (pc + 8) & 0x3FFF
                return self.count
            if d.ebit:
                stopping = True
            nxt = delayed if delayed is not None else (pc + 8) & 0x3FFF
            delayed = P["branch"] if P is not None else None
            pc = nxt
        self.pc = pc
        return self.count

    def leafval_fn(self, head, data):
        """leafval for Dag.evaluate: a leaf's bits in the run's initial state (head, data as in the snapshot)."""
        vf = struct.unpack_from("<128I", head, 28)
        vc = struct.unpack_from("<32I", head, 28 + 512)
        acc = struct.unpack_from("<4I", head, 28 + 512 + 128)
        fl = struct.unpack_from("<6I", head, 4)
        mem = struct.unpack("<4096I", data)
        nodes = self.dag.nodes

        def leafval(n):
            op, _, at = nodes[n]
            if op == "m":
                return mem[at[0] * 4 + at[1]]
            if op == "vf":
                return vf[at[0] * 4 + at[1]]
            if op == "vi":
                return vc[at[0]] & 0xFFFF
            if op == "acc":
                return acc[at[0]]
            if op == "Q":
                return vc[22]
            if op == "I":
                return vc[21]
            if op == "P":
                return fl[3]
            if op == "top":
                return fl[4]
            if op == "itop":
                return fl[5]
            if op == "mac":
                return fl[0]
            if op == "clipold":
                return (fl[2] >> (6 * at[0])) & 63
            raise KeyError(op)
        return leafval

    def data_bytes(self):
        return struct.pack("<4096I", *self.mem)


def image_runs(path, image, limit=None, skip=0):
    """(snapshot index, head, micro, data) of the runs of one image (FNV-1a of the micro memory)."""
    hashes = {}
    n = 0
    for i, head, m, d in read_runs(path):
        h = hashes.get(m)
        if h is None:
            h = hashes[m] = fnv(m)
        if h != image:
            continue
        if n >= skip:
            yield i, head, m, d
        n += 1
        if limit is not None and n >= skip + limit:
            return
