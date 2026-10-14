"""Native blocks: vurecomp.py's blocks again, with the pipeline worked out
ahead of time.

vurecomp.py's blocks replay the interpreter's model step by step: each pair
looks up when its operands land (vf_ready_), records when its results will,
queues its flags and keeps every register in the Vu. These blocks end in the
same state - registers, memory, flags, Q and P, every pipeline, the cycle -
but most of that bookkeeping is done here, while generating:

- Time. A block starts at cycle c0 (the Vu's); each pair issues one cycle
  after the one before or later, when it stalls. An issue time is kept here
  as a base and an offset - a base is a cycle the code computes, c0 or one
  after a stall that could not be told ahead - and two times compare here
  whenever their bases are the same, or the later base is known far enough
  ahead (a base's lower bound from c0, and a later base never has less slack
  than an earlier one). Only the rest compare as the block runs: the first
  pairs' operands written before the block (they land by c0 + 3), Q and P
  from before it, and moves that do not change their register (the model
  holds no write for those, so their fields keep the time they had).
  vf_ready_ is written as the block leaves, once a field.
- Registers. VF, ACC, the VI registers, I, Q and P are locals for the block
  (processor registers), loaded as it starts and stored as it leaves. A
  register's Clean is computed once and kept until it is written; fields an
  FMAC wrote are known clean.
- Flags. MAC and CLIP values are computed only where an instruction reads
  them or where they leave the block; STATUS's sticky bits gather in vector
  registers. A reader sees the latest result that has landed by its issue
  (told ahead where it can be, compared as it runs where not, and the Vu's
  own queue below that - what was in flight before the block). As the block
  leaves, the results that have landed for certain are set as landed, and
  only those still on their way go into the Vu's queue.

A block with an operation these do not do (the STATUS instructions, FCSET,
anything through macro mode's COP2, a VI register past 15) is the exact block,
called as it is; so is any block entered with a branch or an E bit pending
(the delay slot or last pair of what came before). The host tests run the
corpus both ways (VuNativeIsTheExact).
"""

from __future__ import annotations

import vurecomp as R


class Unsupported(Exception):
    """An operation the native blocks leave to the exact block."""


def lanes_of(dest: int) -> list:
    return [lane for lane in range(4) if dest & (8 >> lane)]


class Block:
    def __init__(self, name: str, exact: str, block: list, pairs: dict, mask: int):
        self.name, self.exact, self.block, self.pairs, self.mask = name, exact, block, pairs, mask
        self.body = []
        self.lb = [0]                   # each base's lower bound, in cycles from c0
        self.n = 0                      # temporaries
        self.ready = {}                 # (reg, lane) -> ('x', time) | ('d', expr, upper bound)
        self.clean = {}                 # (reg | 'acc', lane) -> written clean
        self.clean_copy = {}            # reg | 'acc' -> a Clean of it, until it is written
        self.vf_used, self.vf_written = set(), set()
        self.acc_used = self.acc_written = False
        self.vi_used, self.vi_written = set(), set()
        self.i_used = self.i_written = False
        self.q = None                   # None untouched; 'unknown'; 'idle'; ('pend', ready)
        self.p = None
        self.mem = False
        self.events = []                # flag results, in order (see flag_event)
        self.zv = self.sv = None        # STATUS's sticky zero and sign lanes so far
        self.mac_cache = {}
        self.cl = None                  # CLIP's latest judgements (a temporary), None: the Vu's
        self.mac_latest_entry = False

    # ------------------------------------------------------------ time
    def tmp(self, prefix: str) -> str:
        self.n += 1
        return "%s%d" % (prefix, self.n)

    def low(self, t) -> int:
        return self.lb[t[0]] + t[1]

    def le(self, a, b) -> bool:
        """a <= b, for certain."""
        if a[0] == b[0]:
            return a[1] <= b[1]
        return a[0] < b[0] and self.low(a) <= self.low(b)

    def lt(self, a, b) -> bool:
        if a[0] == b[0]:
            return a[1] < b[1]
        return a[0] < b[0] and self.low(a) < self.low(b)

    @staticmethod
    def texpr(t) -> str:
        return "c%d" % t[0] + (" + %d" % t[1] if t[1] else "")

    def rexpr(self, ready) -> str:
        return self.texpr(ready[1]) if ready[0] == "x" else ready[1]

    def ready_of(self, r: int, lane: int):
        return self.ready.get((r, lane), ("d", "C::Ready(v, %d)[%d]" % (r, lane), (0, 3)))

    # ------------------------------------------------------------ registers
    def f(self, r: int) -> str:
        if r == 0:
            return "kVf0"
        self.vf_used.add(r)
        return "f%d" % r

    def acc(self) -> str:
        self.acc_used = True
        return "ac"

    def vi(self, r: int) -> str:
        if r >= 16:
            raise Unsupported("VI%d" % r)
        if r == 0:
            return "0u"
        self.vi_used.add(r)
        return "i%d" % r

    def setvi(self, r: int, expr: str) -> list:
        if not 0 < r < 16:
            return []
        self.vi_used.add(r)
        self.vi_written.add(r)
        return ["i%d = (%s) & 0xFFFFu;" % (r, expr)]

    def is_clean(self, r, lane) -> bool:
        return r == 0 or self.clean.get((r, lane), False)

    def operand(self, r, lanes: list) -> str:
        """A register as the FMAC takes it: Cleaned, unless its fields that
        matter are known clean."""
        raw = self.acc() if r == "acc" else self.f(r)
        if all(self.is_clean(r, lane) for lane in lanes):
            return raw
        if r not in self.clean_copy:
            name = self.tmp("k")
            self.body.append("const __m128i %s = C::CleanOf(%s);" % (name, raw))
            self.clean_copy[r] = name
        return self.clean_copy[r]

    def wrote(self, r, lanes: list, clean: bool):
        self.clean_copy.pop(r, None)
        for lane in lanes:
            self.clean[(r, lane)] = clean

    # ------------------------------------------------------------ Q and P
    def touch(self, which: str):
        if getattr(self, which) is None:
            setattr(self, which, "unknown")

    def land(self, which: str, t):
        """Q or P as the pair issuing at t sees it (the model lands it at the issue)."""
        self.touch(which)
        state = getattr(self, which)
        if state == "idle":
            return
        if state == "unknown" or not (self.le(state[1], t) or self.lt(t, state[1])):
            self.body.append("if (%sb && %sr <= %s) { %s = %sn; %sb = false; }" %
                             (which, which, self.texpr(t), which, which, which))
            setattr(self, which, "unknown")
        elif self.le(state[1], t):
            self.body.append("%s = %sn; %sb = false;" % (which, which, which))
            setattr(self, which, "idle")

    def wait_terms(self, which: str, t):
        """A wait on Q or P (DIV, WAITQ; the EFU, WAITP): (terms to compare as
        the block runs, t moved ahead where it can be told)."""
        self.touch(which)
        state = getattr(self, which)
        if state == "idle":
            return [], t
        if state == "unknown":
            return ["(%sb ? %sr : u64(0))" % (which, which)], t
        ready = state[1]
        if self.le(ready, t):
            return [], t
        if ready[0] == t[0]:
            return [], ready
        return ["%sr" % which], t

    # ------------------------------------------------------------ flags
    def mac_of(self, e) -> str:
        if e["kind"] != "mac":
            raise AssertionError
        if "mac" not in e:
            name = self.tmp("m")
            self.body.append("const u32 %s = C::MacOf<%du>(%s);" % (name, e["dest"], e["vec"]))
            e["mac"] = name
        return e["mac"]

    def status_of(self, e) -> str:
        if "status" not in e:
            name = self.tmp("st")
            self.body.append("const u32 %s = (sl0 & ~0xC3u) | (sl0 & 0xC0u) | C::StickyOf(%s, %s) | C::LowOf<%du>(%s);"
                             % (name, e["zv"], e["sv"], e["dest"], e["vec"]))
            e["status"] = name
        return e["status"]

    def snapshot(self, index: int):
        """(mac, status, clip) latest as event `index` set them."""
        mac = status = None
        for e in reversed(self.events[:index + 1]):
            if e["kind"] == "mac":
                mac, status = self.mac_of(e), self.status_of(e)
                break
        if mac is None:
            self.mac_latest_entry = True
            mac, status = "ml0", "sl0"
        clip = self.events[index]["cl"] or "C::ClipLatest(v)"
        return mac, status, clip

    def flag_read(self, kind: str, t) -> str:
        """The landed MAC or CLIP as the pair issuing at t reads it."""
        parts = []
        base = None
        for e in reversed(self.events):
            if e["kind"] != kind:
                continue
            value = self.mac_of(e) if kind == "mac" else e["cl"]
            if self.le(e["ready"], t):
                base = value
                break
            if self.lt(t, e["ready"]):
                continue
            parts.append((e["ready"], value))
        if base is None:
            base = "C::%s(v, %s)" % ("Mac" if kind == "mac" else "Clip", self.texpr(t))
        expr = base
        for ready, value in reversed(parts):
            expr = "(%s <= %s ? %s : %s)" % (self.texpr(ready), self.texpr(t), value, expr)
        name = self.tmp("fl")
        self.body.append("const u32 %s = %s;" % (name, expr))
        return name

    def mac_event(self, t, vec: str, dest: int):
        zv, sv = self.tmp("zv"), self.tmp("sv")
        self.body.append("const __m128i %s = _mm_or_si128(%s, C::ZeroLanes<%du>(%s));" %
                         (zv, self.zv or "_mm_setzero_si128()", dest, vec))
        self.body.append("const __m128i %s = _mm_or_si128(%s, C::SignLanes<%du>(%s));" %
                         (sv, self.sv or "_mm_setzero_si128()", dest, vec))
        self.zv, self.sv = zv, sv
        self.events.append({"kind": "mac", "t": t, "ready": (t[0], t[1] + 4), "vec": vec, "dest": dest,
                            "zv": zv, "sv": sv, "cl": self.cl})

    def clip_event(self, t, judgement: str):
        cl = self.tmp("cl")
        self.body.append("const u32 %s = ((%s << 6) | %s) & 0xFFFFFFu;" % (cl, self.cl or "C::ClipLatest(v)", judgement))
        self.cl = cl
        self.events.append({"kind": "clip", "t": t, "ready": (t[0], t[1] + 4), "cl": cl})

    # ------------------------------------------------------------ the lower
    def lower(self, word: int, pc: int, t):
        """C++ for the lower (into self.body); returns (held write, branch
        expression or None, new Q (value, latency) or None, new P likewise).
        A held write: (reg, lanes, value, 'always' | changed-expression)."""
        op, dest = word >> 25, R.field(word, 21, 4)
        it, is_ = R.field(word, 16, 5), R.field(word, 11, 5)
        b = self.body
        target = (pc + 8 + R.imm11(word) * 8) & self.mask
        imm = R.imm11(word) & 0xFFFFFFFF

        def address(r):
            return "%s + 0x%Xu" % (self.vi(r), imm)

        def held_move(value: str):
            """MOVE, MR32, MFIR: a write held only where it changes the register."""
            before = self.f(it)
            after, changed = self.tmp("h"), self.tmp("ch")
            b.append("const __m128i %s = C::Blend(C::LaneMask(%du), %s, %s);" % (after, dest, value, before))
            b.append("const bool %s = C::Changed(%s, %s);" % (changed, after, before))
            return (it, dest, after, changed)

        if op == 0x40:
            funct = word & 0x3F
            if 0x30 <= funct <= 0x35:
                ident = R.field(word, 6, 5)
                if funct == 0x32:
                    small = ident - 32 if ident & 0x10 else ident
                    b += self.setvi(it, "u32(std::int32_t(%s) + %d)" % (self.vi(is_), small))
                    return None, None, None, None
                x, y = self.vi(is_), self.vi(it)
                expression = {0x30: "%s + %s", 0x31: "%s - %s", 0x34: "(%s & %s)"}.get(funct, "(%s | %s)") % (x, y)
                b += self.setvi(ident, expression)
                return None, None, None, None
            if funct < 0x3C:
                raise Unsupported("COP2 lower 0x%08X" % word)
            index = R.special_index(word)
            fsf, ftf = R.field(word, 21, 2), R.field(word, 23, 2)
            if index in (0x34, 0x36):                                   # LQI, LQD
                self.mem = True
                if index == 0x36:
                    b += self.setvi(is_, "%s - 1" % self.vi(is_))
                held = None
                if it:
                    value = self.tmp("ld")
                    b.append("const __m128i %s = C::LoadQuad(mem, msk, %s);" % (value, self.vi(is_)))
                    held = (it, dest, value, "always")
                if index == 0x34:
                    b += self.setvi(is_, "%s + 1" % self.vi(is_))
                return held, None, None, None
            if index in (0x35, 0x37):                                   # SQI, SQD
                self.mem = True
                if index == 0x37:
                    b += self.setvi(it, "%s - 1" % self.vi(it))
                b.append("C::StoreQuad<%du>(mem, msk, %s, %s);" % (dest, self.vi(it), self.f(is_)))
                if index == 0x35:
                    b += self.setvi(it, "%s + 1" % self.vi(it))
                return None, None, None, None
            if index == 0x3E:                                           # ILWR
                lanes = lanes_of(dest)
                if lanes:
                    self.mem = True
                    b += self.setvi(it, "C::WordOf(mem, msk, %s, %d)" % (self.vi(is_), lanes[0]))
                return None, None, None, None
            if index == 0x3F:                                           # ISWR
                self.mem = True
                b.append("C::StoreWord<%du>(mem, msk, %s, %s);" % (dest, self.vi(is_), self.vi(it)))
                return None, None, None, None
            if index in (0x64, 0x30, 0x31, 0x3D) and it == 0:
                return None, None, None, None
            if index == 0x64:                                           # MFP
                self.land("p", t)
                value = self.tmp("mp")
                b.append("const __m128i %s = _mm_set1_epi32(int(p));" % value)
                return (it, dest, value, "always"), None, None, None
            if index == 0x68:
                b += self.setvi(it, "C::Top(v)")
                return None, None, None, None
            if index == 0x69:
                b += self.setvi(it, "C::Itop(v)")
                return None, None, None, None
            if index == 0x6C:                                           # XGKICK
                b.append("C::Kick(v, %s);" % self.vi(is_))
                return None, None, None, None
            if 0x70 <= index <= 0x7E and index not in (0x77, 0x7B):     # the EFU
                latency = R.EFU_LATENCY[(index - 0x70) & 15]
                fs = self.f(is_)

                def val(lane):
                    return "lifted::Value(C::Lane<%d>(%s))" % (lane, fs)
                if index in (0x70, 0x71, 0x72, 0x73):
                    sum_ = self.tmp("sum")
                    b.append("const double %s = %s * %s + %s * %s + %s * %s;" %
                             (sum_, val(0), val(0), val(1), val(1), val(2), val(2)))
                    value = {0x70: "%s", 0x71: "1.0 / %s", 0x72: "std::sqrt(%s)",
                             0x73: "1.0 / std::sqrt(%s)"}[index] % sum_
                else:
                    value = {0x74: "std::atan2(%s, %s)" % (val(1), val(0)),
                             0x75: "std::atan2(%s, %s)" % (val(2), val(0)),
                             0x76: "%s + %s + %s + %s" % (val(0), val(1), val(2), val(3)),
                             0x78: "std::sqrt(std::fabs(%s))" % val(fsf),
                             0x79: "1.0 / std::sqrt(std::fabs(%s))" % val(fsf),
                             0x7A: "1.0 / %s" % val(fsf),
                             0x7C: "std::sin(%s)" % val(fsf),
                             0x7D: "std::atan(%s)" % val(fsf),
                             0x7E: "std::exp(-%s)" % val(fsf)}[index]
                name = self.tmp("ef")
                b.append("const u32 %s = lifted::Result(%s);" % (name, value))
                return None, None, None, (name, latency if latency else 1)
            if index == 0x38:                                           # DIV
                name = self.tmp("dq")
                b.append("const u32 %s = C::DivOf(C::Lane<%d>(%s), C::Lane<%d>(%s));" %
                         (name, fsf, self.f(is_), ftf, self.f(it)))
                return None, None, (name, 7), None
            if index == 0x39:                                           # SQRT
                name = self.tmp("dq")
                b.append("const u32 %s = C::SqrtOf(C::Lane<%d>(%s));" % (name, ftf, self.f(it)))
                return None, None, (name, 7), None
            if index == 0x3A:                                           # RSQRT
                name = self.tmp("dq")
                b.append("const u32 %s = C::RsqrtOf(C::Lane<%d>(%s), C::Lane<%d>(%s));" %
                         (name, fsf, self.f(is_), ftf, self.f(it)))
                return None, None, (name, 13), None
            if index == 0x30:                                           # MOVE
                return held_move(self.f(is_)), None, None, None
            if index == 0x31:                                           # MR32
                return held_move("_mm_shuffle_epi32(%s, _MM_SHUFFLE(0, 3, 2, 1))" % self.f(is_)), None, None, None
            if index == 0x3D:                                           # MFIR
                return held_move("C::Mfir(%s)" % self.vi(is_)), None, None, None
            if index == 0x3C:                                           # MTIR
                if 0 < it < 16:
                    b += self.setvi(it, "C::Lane<%d>(%s)" % (fsf, self.f(is_)))
                return None, None, None, None
            if index in (0x3B, 0x7B):                                   # WAITQ, WAITP
                return None, None, None, None
            raise Unsupported("special lower 0x%X" % index)
        if op == 0x00:                                                  # LQ
            self.mem = True
            if not it:
                self.vi(is_)
                return None, None, None, None
            value = self.tmp("ld")
            b.append("const __m128i %s = C::LoadQuad(mem, msk, %s);" % (value, address(is_)))
            return (it, dest, value, "always"), None, None, None
        if op == 0x01:                                                  # SQ
            self.mem = True
            b.append("C::StoreQuad<%du>(mem, msk, %s, %s);" % (dest, address(it), self.f(is_)))
            return None, None, None, None
        if op == 0x04:                                                  # ILW
            lanes = lanes_of(dest)
            if lanes:
                self.mem = True
                b += self.setvi(it, "C::WordOf(mem, msk, %s, %d)" % (address(is_), lanes[0]))
            return None, None, None, None
        if op == 0x05:                                                  # ISW
            self.mem = True
            b.append("C::StoreWord<%du>(mem, msk, %s, %s);" % (dest, address(is_), self.vi(it)))
            return None, None, None, None
        unsigned = (R.field(word, 21, 4) << 11) | R.field(word, 0, 11)
        if op == 0x08:
            b += self.setvi(it, "%s + 0x%Xu" % (self.vi(is_), unsigned))
            return None, None, None, None
        if op == 0x09:
            b += self.setvi(it, "%s - 0x%Xu" % (self.vi(is_), unsigned))
            return None, None, None, None
        low24 = word & 0xFFFFFF
        if op in (0x10, 0x12, 0x13, 0x1C):
            clip = self.flag_read("clip", t)
            if op == 0x10:
                b += self.setvi(1, "u32((%s & 0xFFFFFFu) == 0x%Xu)" % (clip, low24))
            elif op == 0x12:
                b += self.setvi(1, "u32((%s & 0x%Xu) != 0)" % (clip, low24))
            elif op == 0x13:
                b += self.setvi(1, "u32(((%s | 0x%Xu) & 0xFFFFFFu) == 0xFFFFFFu)" % (clip, low24))
            else:
                b += self.setvi(it, "%s & 0xFFFu" % clip)
            return None, None, None, None
        if op == 0x11 or 0x14 <= op <= 0x17:
            raise Unsupported("FCSET / STATUS 0x%X" % op)
        if op in (0x18, 0x1A, 0x1B):
            x = self.vi(is_)
            mac = self.flag_read("mac", t)
            expression = {0x18: "u32((%s & 0xFFFFu) == (%s & 0xFFFFu))", 0x1A: "%s & %s", 0x1B: "%s | %s"}[op]
            b += self.setvi(it, expression % (x, mac))
            return None, None, None, None
        if op == 0x20:
            return None, "0x%Xu" % target, None, None
        if op == 0x21:
            b += self.setvi(it, "0x%Xu" % ((pc + 16) // 8))
            return None, "0x%Xu" % target, None, None
        if op in (0x24, 0x25):
            to = self.tmp("to")
            b.append("const u32 %s = (%s * 8) & 0x%Xu;" % (to, self.vi(is_), self.mask))
            if op == 0x25:
                b += self.setvi(it, "0x%Xu" % ((pc + 16) // 8))
            return None, to, None, None
        if op in (0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F):
            a, bb = "std::int16_t(%s)" % self.vi(is_), "std::int16_t(%s)" % self.vi(it)
            condition = {0x28: "%s == %s" % (a, bb), 0x29: "%s != %s" % (a, bb), 0x2C: "%s < 0" % a,
                         0x2D: "%s > 0" % a, 0x2E: "%s <= 0" % a, 0x2F: "%s >= 0" % a}[op]
            to = self.tmp("to")
            b.append("const u32 %s = (%s) ? 0x%Xu : kNone;" % (to, condition, target))
            return None, to, None, None
        return None, None, None, None

    # ------------------------------------------------------------ the upper
    def upper(self, word: int, t):
        """C++ for the upper (into self.body); returns the writes to apply at
        the pair's end: [(reg | 'acc', lanes, value, clean, lanes timed)] -
        the fields whose landing time the model sets: OPMSUB's include w - and the flags'
        vector (with its Dest) or None, and a CLIP judgement or None."""
        funct, dest = word & 0x3F, R.field(word, 21, 4)
        ft, fs, fd = R.field(word, 16, 5), R.field(word, 11, 5), R.field(word, 6, 5)
        b = self.body
        lanes = lanes_of(dest)
        special = funct >= 0x3C
        index = R.special_index(word) if special else None
        if not special and funct >= 0x30:
            return [], None, None
        if special and index == 0x2F:                                   # NOP
            return [], None, None
        if special and index == 0x1F:                                   # CLIP
            name = self.tmp("cj")
            b.append("const u32 %s = C::ClipOf(%s, %s);" % (name, self.f(fs), self.f(ft)))
            return [], None, name
        fmac = R.decode_fmac(word)
        if fmac:
            op, operand, to_acc = fmac
            raw = op in (R.K_MAX, R.K_MINI)
            a = self.f(fs) if raw else self.operand(fs, lanes)
            if operand == 6:
                bv = self.f(ft) if raw else self.operand(ft, lanes)
            elif operand < 4:
                bv = "C::Bc<%d>(%s)" % (operand, self.f(ft) if raw else self.operand(ft, [operand]))
            elif operand == 4:
                self.land("q", t)
                bv = "_mm_set1_epi32(int(q))" if raw else "C::CleanOf(_mm_set1_epi32(int(q)))"
            else:
                self.i_used = True
                bv = "_mm_set1_epi32(int(ir))" if raw else "C::CleanOf(_mm_set1_epi32(int(ir)))"
            accv = self.operand("acc", lanes) if op in (R.K_MADD, R.K_MSUB) else "_mm_setzero_si128()"
            result = self.tmp("r")
            b.append("const __m128i %s = C::Lanes<%s>(%s, %s, %s);" % (result, R.OP_NAMES[op], a, bv, accv))
            flags = not raw
            if to_acc:
                writes = [("acc", dest, result, not raw, dest)]
            elif fd == 0:
                writes = []
            else:
                writes = [(fd, dest, result, not raw, dest)]
            return writes, ((result, dest) if flags else None), None
        if not special and funct == 0x2E:                               # OPMSUB
            result = self.tmp("r")
            b.append("const __m128i %s = C::OpmsubOf(%s, %s, %s);" % (result, self.f(fs), self.f(ft), self.acc()))
            if fd == 0:
                vec = self.tmp("r")
                b.append("const __m128i %s = _mm_and_si128(%s, _mm_set_epi32(0, -1, -1, -1));" % (vec, result))
                return [], (vec, dest), None
            return [(fd, dest & 0xE, result, True, dest)], ("after", fd, dest), None
        if special and index == 0x2E:                                   # OPMULA
            result = self.tmp("r")
            b.append("const __m128i %s = C::OpmulaOf(%s, %s);" % (result, self.f(fs), self.f(ft)))
            return [("acc", dest & 0xE, result, True, dest)], ("after", "acc", dest), None
        if special and 0x10 <= index <= 0x17:                           # ITOF, FTOI
            if ft == 0:
                return [], None, None
            result = self.tmp("r")
            kind = "Itof" if index <= 0x13 else "Ftoi"
            b.append("const __m128i %s = C::%sOf<%d>(%s);" % (result, kind, index & 3, self.f(fs)))
            return [(ft, dest, result, kind == "Itof", dest)], None, None
        if special and index == 0x1D:                                   # ABS
            if ft == 0:
                return [], None, None
            result = self.tmp("r")
            b.append("const __m128i %s = C::AbsOf(%s);" % (result, self.f(fs)))
            return [(ft, dest, result, True, dest)], None, None
        raise Unsupported("upper 0x%08X" % word)

    # ------------------------------------------------------------ a pair
    def pair(self, k: int, pc: int, lower: int, upper: int, t):
        """Emits pair k issuing no earlier than t; returns its issue time and
        its branch expression (or None)."""
        b = self.body
        immediate = bool(upper & 0x80000000)
        reads = R.upper_reads(upper) + ([] if immediate else R.lower_reads(lower))
        terms = []
        for r, lanes in reads:
            for lane in lanes_of(lanes):
                if r == 0:
                    continue
                ready = self.ready_of(r, lane)
                if ready[0] == "x":
                    if self.le(ready[1], t):
                        continue
                    if ready[1][0] == t[0]:
                        t = ready[1]
                        continue
                    terms.append(self.texpr(ready[1]))
                else:
                    if self.le(ready[2], t):
                        continue
                    terms.append(ready[1])
        if not immediate and (R.is_divide(lower) or R.is_wait(lower, 0x3B)):
            more, t = self.wait_terms("q", t)
            terms += more
        if not immediate and (R.is_efu(lower) or R.is_wait(lower, 0x7B)):
            more, t = self.wait_terms("p", t)
            terms += more
        if terms:
            # A stall not known ahead: a new base, at least t.
            uniq = []
            for term in terms:
                if term not in uniq:
                    uniq.append(term)
            self.lb.append(self.low(t))
            base = len(self.lb) - 1
            expr = self.texpr(t)
            for term in uniq:
                expr = "std::max<u64>(%s, %s)" % (expr, term)
            b.append("const u64 c%d = %s;" % (base, expr))
            t = (base, 0)
        b.append("// %04X: %08X %08X" % (pc, lower, upper))
        if not immediate and (R.is_divide(lower) or R.is_wait(lower, 0x3B)):
            self.land("q", t)
        if not immediate and (R.is_efu(lower) or R.is_wait(lower, 0x7B)):
            self.land("p", t)
        held, branch, new_q, new_p = (None, None, None, None) if immediate else self.lower(lower, pc, t)
        writes, flags, judgement = self.upper(upper, t)
        if immediate:
            self.i_used = self.i_written = True
            b.append("ir = 0x%08Xu;" % lower)
        ready = (t[0], t[1] + 4)
        upper_lanes = {}
        for reg, _dest, _value, _clean, timed in writes:
            upper_lanes[reg] = timed
        # The lower's held write, giving way to the upper's fields.
        if held:
            reg, dest, value, how = held
            landing = dest & ~upper_lanes.get(reg, 0)
            name = self.f(reg)
            self.vf_written.add(reg)
            if how == "always":
                b.append("%s = C::Blend(C::LaneMask(%du), %s, %s);" % (name, dest, value, name))
                for lane in lanes_of(landing):
                    self.ready[(reg, lane)] = ("x", ready)
                self.wrote(reg, lanes_of(dest), False)
            else:
                b.append("%s = %s;" % (name, value))
                for lane in lanes_of(landing):
                    before = self.ready_of(reg, lane)
                    var = self.tmp("rd")
                    b.append("const u64 %s = %s ? %s : u64(%s);" % (var, how, self.texpr(ready), self.rexpr(before)))
                    bound = before[1] if before[0] == "x" else before[2]
                    bound = ready if self.le(bound, ready) else bound
                    self.ready[(reg, lane)] = ("d", var, bound)
                source = R.field(lower, 11, 5)
                index = R.special_index(lower)
                if index == 0x30:
                    cleans = {lane: self.is_clean(source, lane) for lane in range(4)}
                elif index == 0x31:
                    cleans = {lane: self.is_clean(source, (lane + 1) & 3) for lane in range(4)}
                else:
                    cleans = {lane: False for lane in range(4)}
                self.clean_copy.pop(reg, None)
                for lane in lanes_of(dest):
                    self.clean[(reg, lane)] = cleans[lane]
        # The upper's.
        for reg, dest, value, clean, timed in writes:
            if reg == "acc":
                self.acc_written = True
                name = self.acc()
            else:
                name = self.f(reg)
                self.vf_written.add(reg)
                for lane in lanes_of(timed):
                    self.ready[(reg, lane)] = ("x", ready)
            if dest == 15:
                b.append("%s = %s;" % (name, value))
            elif dest:
                b.append("%s = C::Blend(C::LaneMask(%du), %s, %s);" % (name, dest, value, name))
            self.wrote(reg, lanes_of(dest), clean)
        if flags:
            if flags[0] == "after":
                vec = self.acc() if flags[1] == "acc" else self.f(flags[1])
                copy = self.tmp("r")
                b.append("const __m128i %s = %s;" % (copy, vec))
                self.mac_event(t, copy, flags[2])
            else:
                self.mac_event(t, flags[0], flags[1])
        if judgement:
            self.clip_event(t, judgement)
        if new_q:
            self.touch("q")
            b.append("qn = %s; qr = %s + %d; qb = true;" % (new_q[0], self.texpr(t), new_q[1]))
            self.q = ("pend", (t[0], t[1] + new_q[1]))
        if new_p:
            self.touch("p")
            b.append("pn = %s; pr = %s + %d; pb = true;" % (new_p[0], self.texpr(t), new_p[1]))
            self.p = ("pend", (t[0], t[1] + new_p[1]))
        return t, branch

    # ------------------------------------------------------------ the block
    def emit(self) -> list:
        n = len(self.block)
        t = (0, 0)
        branches = []
        last_t = None
        for k, pc in enumerate(self.block):
            lower, upper = self.pairs[pc]
            last_t, branch = self.pair(k, pc, lower, upper, t)
            branches.append(branch)
            t = (last_t[0], last_t[1] + 1)
        exit_t = t
        tail = []
        # Flags: those landed for certain as landed, the rest queued.
        landed = None
        for i, e in enumerate(self.events):
            if self.le(e["ready"], exit_t):
                landed = i
        if landed is not None:
            tail.append("C::LandFlags(v, %s);" % self.texpr(exit_t))
            mac = next((e for e in reversed(self.events[:landed + 1]) if e["kind"] == "mac"), None)
            clip = next((e for e in reversed(self.events[:landed + 1]) if e["kind"] == "clip"), None)
            if mac:
                tail.append("C::MacLanded(v) = %s;" % self.mac_of(mac))
                tail.append("C::StatusLanded(v) = (C::StatusLanded(v) & ~3u) | (%s & 0xC3u);" % self.status_of(mac))
            if clip:
                tail.append("C::ClipLanded(v) = %s;" % clip["cl"])
        for i in range((landed + 1) if landed is not None else 0, len(self.events)):
            e = self.events[i]
            mac, status, clip = self.snapshot(i)
            tail.append("C::PushFlags(v, %s, %s, %s, %s, %s, %s, %s);" %
                        (self.texpr(e["t"]), self.texpr(e["ready"]), mac, status, clip,
                         "true" if e["kind"] == "mac" else "false", "true" if e["kind"] == "clip" else "false"))
        last_mac = next((e for e in reversed(self.events) if e["kind"] == "mac"), None)
        if last_mac:
            tail.append("C::MacLatest(v) = %s;" % self.mac_of(last_mac))
            tail.append("C::StatusLatest(v) = %s;" % self.status_of(last_mac))
        if self.cl:
            tail.append("C::ClipLatest(v) = %s;" % self.cl)
        # Registers.
        for r in sorted(self.vf_written):
            tail.append("C::StoreVf(v, %d, f%d);" % (r, r))
        if self.acc_written:
            tail.append("C::StoreAcc(v, ac);")
        for r in sorted(self.vi_written):
            tail.append("C::ViRef(v, %d) = i%d;" % (r, r))
        if self.i_written:
            tail.append("C::ViRef(v, lifted::kRegI) = ir;")
        if self.q is not None:
            tail.append("C::ViRef(v, lifted::kRegQ) = q; C::SetQ(v, qn, qr, qb);")
        if self.p is not None:
            tail.append("C::SetP(v, p, pn, pr, pb);")
        if self.q is not None or self.p is not None:
            tail.append("C::QpDue(v);")
        # When the fields land.
        by_reg = {}
        for (reg, lane), ready in self.ready.items():
            by_reg.setdefault(reg, {})[lane] = self.rexpr(ready)
        for reg in sorted(by_reg):
            lanes = by_reg[reg]
            if len(lanes) == 4 and len(set(lanes.values())) == 1:
                tail.append("C::Ready4(v, %d, %s);" % (reg, lanes[0]))
            else:
                for lane in sorted(lanes):
                    tail.append("C::Ready(v, %d)[%d] = %s;" % (reg, lane, lanes[lane]))
        # What may be unclean now.
        dirty, cleaned = 0, 0
        for r in self.vf_written:
            known = [self.clean.get((r, lane)) for lane in range(4)]
            if all(k is True for k in known):
                cleaned |= 1 << r
            elif any(k is False for k in known):
                dirty |= 1 << r
        if dirty or cleaned:
            tail.append("C::Unclean(v) = (C::Unclean(v) | 0x%Xu) & ~0x%Xu;" % (dirty, cleaned))
        if self.acc_written:
            known = [self.clean.get(("acc", lane)) for lane in range(4)]
            if all(k is True for k in known):
                tail.append("C::AccUnclean(v) = false;")
            elif any(k is False for k in known):
                tail.append("C::AccUnclean(v) = true;")
        tail.append("C::Cycle(v) = %s;" % self.texpr(exit_t))
        # Where the run goes: as the exact block's last two pairs leave it.
        next_pc = (self.block[-1] + 8) & self.mask
        last_lower, last_upper = self.pairs[self.block[-1]]
        stop2 = n >= 2 and bool(self.pairs[self.block[-2]][1] & 0x40000000)
        branch2 = branches[-2] if n >= 2 else None
        branch1 = branches[-1]
        end1 = bool(last_upper & 0x40000000)
        tail.append("s.count += %d;" % n)
        if stop2:
            tail.append("s.pc = 0x%Xu; s.done = true; s.stopping = true; s.delayed = %s;" % (next_pc, branch2 or "kNone"))
        else:
            tail.append("s.stopping = %s;" % ("true" if end1 else "false"))
            if branch2:
                tail.append("if (%s != kNone) { s.pc = %s; s.delayed = %s; }" % (branch2, branch2, branch1 or "kNone"))
                tail.append("else { s.pc = 0x%Xu; s.delayed = %s; }" % (next_pc, branch1 or "kNone"))
            else:
                tail.append("s.pc = 0x%Xu; s.delayed = %s;" % (next_pc, branch1 or "kNone"))
        # The head: the block's state as locals.
        head = ["void %s(Vu& v, VuRunState& s, std::uint64_t limit);" % self.exact,
                "void %s(Vu& v, VuRunState& s, std::uint64_t limit) {" % self.name,
                "    if (s.count + %d > limit) return;" % n,
                "    if (s.stopping || s.delayed != kNone) return %s(v, s, limit);" % self.exact,
                "    const u64 c0 = C::Cycle(v);",
                "    const __m128i kVf0 = _mm_set_epi32(0x3F800000, 0, 0, 0);",
                "    (void)kVf0;"]
        if self.mem:
            head.append("    std::uint8_t* const mem = C::Mem(v); const u32 msk = C::MemMask(v);")
        for r in sorted(self.vf_used):
            head.append("    __m128i f%d = C::LoadVf(v, %d);" % (r, r))
        if self.acc_used:
            head.append("    __m128i ac = C::LoadAcc(v);")
        for r in sorted(self.vi_used):
            head.append("    u32 i%d = C::ViRef(v, %d);" % (r, r))
        if self.i_used:
            head.append("    u32 ir = C::ViRef(v, lifted::kRegI);")
        if self.q is not None:
            head.append("    u32 q = C::ViRef(v, lifted::kRegQ), qn; u64 qr; bool qb; C::GetQ(v, qn, qr, qb);")
        if self.p is not None:
            head.append("    u32 p, pn; u64 pr; bool pb; C::GetP(v, p, pn, pr, pb);")
        if self.events or self.mac_latest_entry:
            head.append("    const u32 sl0 = C::StatusLatest(v), ml0 = C::MacLatest(v); (void)sl0; (void)ml0;")
        return head + ["    " + line for line in self.body + tail] + ["}"]


def emit_native(name: str, exact: str, block: list, pairs: dict, mask: int):
    """The native block's C++ lines, or None where the exact block stands."""
    try:
        return Block(name, exact, block, pairs, mask).emit()
    except Unsupported:
        return None
