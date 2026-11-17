"""Splits a run's XGKICKed GIF packets (vu1sym.SymVu.kicks: qwords of (bits, node)) into per-vertex output words.

A vertex is one loop of a PACKED or REGLIST tag that writes an XYZ register; its words are named
'<REG>.<lane>' (PACKED: lanes x y z w of the qword; REGLIST: lo / hi of the 64-bit value). Each word carries a
mask of the bits the GS reads (0: ignored - e.g. ST's w word, often stale memory), so a generalizer can leave
ignored words out. Imported by generalize.py / translate.py as `import slice as gifslice` (not a stdlib name).
"""

REG_NAMES = {0x0: "PRIM", 0x1: "RGBAQ", 0x2: "ST", 0x3: "UV", 0x4: "XYZF2", 0x5: "XYZ2", 0x6: "TEX0_1",
             0x7: "TEX0_2", 0x8: "CLAMP_1", 0x9: "CLAMP_2", 0xA: "FOG", 0xC: "XYZF3", 0xD: "XYZ3", 0xE: "A+D",
             0xF: "NOP"}
XYZ_REGS = {0x4, 0x5, 0xC, 0xD}

# PACKED: the GS-read bits of each lane word
PACKED_MASK = {
    0x0: (0x7FF, 0, 0, 0),
    0x1: (0xFF, 0xFF, 0xFF, 0xFF),
    0x2: (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0),
    0x3: (0x3FFF, 0x3FFF, 0, 0),
    0x4: (0xFFFF, 0xFFFF, 0x0FFFFFF0, 0x8FF0),
    0x5: (0xFFFF, 0xFFFF, 0xFFFFFFFF, 0x8000),
    0xA: (0, 0, 0, 0xFF0),
    0xC: (0xFFFF, 0xFFFF, 0x0FFFFFF0, 0x8FF0),
    0xD: (0xFFFF, 0xFFFF, 0xFFFFFFFF, 0x8000),
    0xF: (0, 0, 0, 0),
}
LANES = "xyzw"


class Tag:
    __slots__ = ("kick", "index", "addr", "nloop", "eop", "pre", "prim", "flg", "nreg", "regs", "words",
                 "vertices", "state")

    def __repr__(self):
        return "Tag(kick %d #%d @%d nloop %d flg %d regs %s prim %s, %d vertices)" % (
            self.kick, self.index, self.addr, self.nloop, self.flg,
            ",".join(REG_NAMES.get(r, "%X" % r) for r in self.regs), "%X" % self.prim if self.prim is not None
            else "-", len(self.vertices))


class Vertex:
    __slots__ = ("tag", "loop", "words", "addr")

    def __init__(self, tag, loop, addr):
        self.tag = tag
        self.loop = loop
        self.words = {}       # name -> (bits, node, mask)
        self.addr = addr      # qword of its first register


def parse_kicks(kicks):
    """kicks: SymVu.kicks. Returns the list of Tags (with .vertices) in order."""
    tags = []
    for ki, (_count, out) in enumerate(kicks):
        for ti, (addr, pk) in enumerate(out):
            head = pk[0]
            w = [b for b, _ in head]
            low = w[0] | w[1] << 32
            t = Tag()
            t.kick, t.index, t.addr = ki, ti, addr
            t.words = head
            t.nloop = low & 0x7FFF
            t.eop = (low >> 15) & 1
            t.pre = (low >> 46) & 1
            t.prim = (low >> 47) & 0x7FF if t.pre else None
            t.flg = (low >> 58) & 3
            t.nreg = (low >> 60) or 16
            regs64 = w[2] | w[3] << 32
            t.regs = [(regs64 >> (4 * j)) & 0xF for j in range(t.nreg)]
            t.vertices = []
            t.state = []
            body = pk[1:]
            if t.flg == 0:
                for lp in range(t.nloop):
                    v = None
                    for j, r in enumerate(t.regs):
                        q = body[lp * t.nreg + j]
                        if r == 0xE:
                            t.state.append(q)
                            continue
                        if v is None:
                            v = Vertex(t, lp, (addr + 1 + lp * t.nreg + j) & 0x3FF)
                        mask = PACKED_MASK.get(r, (0xFFFFFFFF,) * 4)
                        name = REG_NAMES.get(r, "R%X" % r)
                        for l in range(4):
                            b, n = q[l]
                            v.words["%s.%s" % (name, LANES[l])] = (b, n, mask[l])
                    if v is not None and any(r in XYZ_REGS for r in t.regs):
                        t.vertices.append(v)
            elif t.flg == 1:
                flat = [x for q in body for x in q]
                for lp in range(t.nloop):
                    v = Vertex(t, lp, (addr + 1 + (lp * t.nreg) // 2) & 0x3FF)
                    for j, r in enumerate(t.regs):
                        k = 2 * (lp * t.nreg + j)
                        name = REG_NAMES.get(r, "R%X" % r)
                        for h, part in enumerate(("lo", "hi")):
                            b, n = flat[k + h]
                            v.words["%s.%s" % (name, part)] = (b, n, 0xFFFFFFFF if r != 0xF else 0)
                    if any(r in XYZ_REGS for r in t.regs):
                        t.vertices.append(v)
            tags.append(t)
    return tags


def significant(v):
    """The vertex's words the GS reads (mask != 0), in a fixed order: [(name, bits, node, mask)]."""
    return [(k, b, n, m) for k, (b, n, m) in v.words.items() if m]
