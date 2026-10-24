#!/usr/bin/env python3
"""Compile vector unit microprograms to C++ ahead of time.

A run with `--vu-record FILE` writes each microprogram it starts: the unit,
the address it started at and the instruction pairs reachable from there (one
line each; see engine/host/src/vu.cpp, RecordProgram). This turns those into
C++ - each pair a block of straight-line code whose registers, fields and
operations are constants - that works the Vu's state through ps2host::VuCore
(engine/host/include/ps2host/vu_core.h), the interpreter's own steps. The
port links it in; Vu::Run takes a compiled program where the code in
instruction memory is still what it was compiled from, and the interpreter
otherwise.

What each instruction does is the interpreter's (vu.cpp: Run, Lower, Upper),
mirrored here pair by pair; `--corpus` writes random programs compiled the same
way for the host tests (VuCompiledIsTheInterpreter), which run each one both
ways and compare the whole state.

    python engine/tools/vurecomp.py --manifest build/vu_programs.txt --out-dir build/port/generated
    python engine/tools/vurecomp.py --corpus --out engine/host/tests/vu_corpus.cpp
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import struct
import sys

NONE = 0xFFFFFFFF
K_ADD, K_SUB, K_MUL, K_MADD, K_MSUB, K_MAX, K_MINI = range(7)
OP_NAMES = ["lifted::kAdd", "lifted::kSub", "lifted::kMul", "lifted::kMadd", "lifted::kMsub",
            "lifted::kMax_", "lifted::kMini"]
EFU_LATENCY = [11, 18, 18, 24, 54, 54, 12, 0, 12, 18, 12, 0, 29, 54, 44, 0]


def field(word: int, low: int, count: int) -> int:
    return (word >> low) & ((1 << count) - 1)


def imm11(word: int) -> int:
    value = word & 0x7FF
    return value - 0x800 if value & 0x400 else value


def lane_bit(lane: int) -> int:
    return 8 >> lane


def special_index(word: int) -> int:
    return (word & 3) | ((word >> 4) & 0x7C)


# ---------------------------------------------------------------- stalls (vu.cpp)

def upper_reads(word: int) -> list:
    reads = []

    def add(r, lanes):
        if r and lanes:
            reads.append((r, lanes))
    funct, dest = word & 0x3F, field(word, 21, 4)
    fs, ft = field(word, 11, 5), field(word, 16, 5)
    if funct < 0x1C:
        add(fs, dest)
        add(ft, lane_bit(funct & 3))
        return reads
    if funct < 0x28:
        add(fs, dest)
        return reads
    if funct < 0x30:
        lanes = 0xE if funct == 0x2E else dest
        add(fs, lanes)
        add(ft, lanes)
        return reads
    if funct < 0x3C:
        return reads
    index = special_index(word)
    if index == 0x2F:
        return reads
    if index == 0x1F:
        add(fs, 0xE)
        add(ft, 0x1)
        return reads
    if index < 0x10 or 0x18 <= index < 0x1C:
        add(fs, dest)
        add(ft, lane_bit(index & 3))
        return reads
    if index < 0x28:
        add(fs, dest)
        return reads
    lanes = 0xE if index == 0x2E else dest
    add(fs, lanes)
    add(ft, lanes)
    return reads


def lower_reads(word: int) -> list:
    reads = []

    def add(r, lanes):
        if r and lanes:
            reads.append((r, lanes))
    op, dest = word >> 25, field(word, 21, 4)
    is_, it = field(word, 11, 5), field(word, 16, 5)
    fsf, ftf = lane_bit(field(word, 21, 2)), lane_bit(field(word, 23, 2))
    if op == 0x01:
        add(is_, dest)
        return reads
    if op != 0x40 or (word & 0x3F) < 0x3C:
        return reads
    index = special_index(word)
    if index == 0x30:
        add(is_, dest)
    elif index == 0x31:
        add(is_, ((dest << 3) | (dest >> 1)) & 0xF)
    elif index in (0x35, 0x37):
        add(is_, dest)
    elif index in (0x38, 0x3A):
        add(is_, fsf)
        add(it, ftf)
    elif index == 0x39:
        add(it, ftf)
    elif index in (0x3C, 0x42, 0x43):
        add(is_, fsf)
    elif index in (0x70, 0x71, 0x72, 0x73):
        add(is_, 0xE)
    elif index == 0x74:
        add(is_, 0xC)
    elif index == 0x75:
        add(is_, 0xA)
    elif index == 0x76:
        add(is_, 0xF)
    elif index in (0x78, 0x79, 0x7A, 0x7C, 0x7D, 0x7E):
        add(is_, fsf)
    return reads


def is_special(word: int) -> bool:
    return (word >> 25) == 0x40 and (word & 0x3F) >= 0x3C


def is_efu(word: int) -> bool:
    return is_special(word) and 0x70 <= special_index(word) <= 0x7E and special_index(word) not in (0x77, 0x7B)


def is_divide(word: int) -> bool:
    return is_special(word) and 0x38 <= special_index(word) <= 0x3A


def is_wait(word: int, which: int) -> bool:
    return is_special(word) and special_index(word) == which


# ---------------------------------------------------------------- FMAC (runtime vu0.cpp)

def decode_fmac(word: int):
    """(op, operand, to_acc) as lifted::DecodeFmac decodes the upper, or None."""
    funct = word & 0x3F
    if funct < 0x3C:
        broadcast = [K_ADD, K_SUB, K_MADD, K_MSUB, K_MAX, K_MINI, K_MUL]
        if funct <= 0x1B:
            return broadcast[funct >> 2], funct & 3, False
        table = {0x1C: (K_MUL, 4), 0x1D: (K_MAX, 5), 0x1E: (K_MUL, 5), 0x1F: (K_MINI, 5),
                 0x20: (K_ADD, 4), 0x21: (K_MADD, 4), 0x22: (K_ADD, 5), 0x23: (K_MADD, 5),
                 0x24: (K_SUB, 4), 0x25: (K_MSUB, 4), 0x26: (K_SUB, 5), 0x27: (K_MSUB, 5),
                 0x28: (K_ADD, 6), 0x29: (K_MADD, 6), 0x2A: (K_MUL, 6), 0x2B: (K_MAX, 6),
                 0x2C: (K_SUB, 6), 0x2D: (K_MSUB, 6), 0x2F: (K_MINI, 6)}
        if funct in table:
            return table[funct][0], table[funct][1], False
        return None
    index = special_index(word)
    broadcast = [K_ADD, K_SUB, K_MADD, K_MSUB]
    if index <= 0x0F:
        return broadcast[index >> 2], index & 3, True
    if 0x18 <= index <= 0x1B:
        return K_MUL, index & 3, True
    table = {0x1C: (K_MUL, 4), 0x1E: (K_MUL, 5), 0x20: (K_ADD, 4), 0x21: (K_MADD, 4),
             0x22: (K_ADD, 5), 0x23: (K_MADD, 5), 0x24: (K_SUB, 4), 0x25: (K_MSUB, 4),
             0x26: (K_SUB, 5), 0x27: (K_MSUB, 5), 0x28: (K_ADD, 6), 0x29: (K_MADD, 6),
             0x2A: (K_MUL, 6), 0x2C: (K_SUB, 6), 0x2D: (K_MSUB, 6)}
    if index in table:
        return table[index][0], table[index][1], True
    return None


# ---------------------------------------------------------------- one pair

def setvi(r: int, value: str) -> list:
    """vu.cpp's setvi: VI1-VI15 only."""
    return ["C::SetVi(v, %d, %s);" % (r, value)] if 0 < r < 16 else []


def emit_lower(word: int, pc: int, mask: int):
    """C++ for the lower instruction; (lines, uses pending, branch: None | 'static' | 'dynamic')."""
    op, dest = word >> 25, field(word, 21, 4)
    it, is_ = field(word, 16, 5), field(word, 11, 5)
    target = (pc + 8 + imm11(word) * 8) & mask
    vi = lambda r: "C::Vi(v, %d)" % r
    plus_imm = lambda base: "%s + 0x%Xu" % (base, imm11(word) & 0xFFFFFFFF)
    if op == 0x40:
        funct = word & 0x3F
        if 0x30 <= funct <= 0x35:
            ident = field(word, 6, 5)
            if funct == 0x32:
                imm = ident - 32 if ident & 0x10 else ident
                return setvi(it, "u32(std::int32_t(%s) + %d)" % (vi(is_), imm)), False, None
            x, y = vi(is_), vi(it)
            expression = {0x30: "%s + %s", 0x31: "%s - %s", 0x34: "(%s & %s)"}.get(funct, "(%s | %s)") % (x, y)
            return setvi(ident, expression), False, None
        if funct < 0x3C:
            return ["C::ThroughCop2(v, pd, 0x%08Xu, %d, %du);" % (word, it, dest)], True, None
        index = special_index(word)
        fsf = field(word, 21, 2)
        if index == 0x34:                                               # LQI
            return (["C::Load(v, pd, %s, %d, %du);" % (vi(is_), it, dest)] +
                    setvi(is_, "%s + 1" % vi(is_))), True, None
        if index == 0x35:                                               # SQI
            return (["C::Store<%du>(v, %s, %d);" % (dest, vi(it), is_)] +
                    setvi(it, "%s + 1" % vi(it))), False, None
        if index == 0x36:                                               # LQD
            return (setvi(is_, "%s - 1" % vi(is_)) +
                    ["C::Load(v, pd, %s, %d, %du);" % (vi(is_), it, dest)]), True, None
        if index == 0x37:                                               # SQD
            return (setvi(it, "%s - 1" % vi(it)) +
                    ["C::Store<%du>(v, %s, %d);" % (dest, vi(it), is_)]), False, None
        if index == 0x3E:                                               # ILWR
            lanes = [lane for lane in range(4) if dest & (8 >> lane)]
            if not lanes:
                return [], False, None
            return setvi(it, "C::Word(v, %s, %d)" % (vi(is_), lanes[0])), False, None
        if index == 0x3F:                                               # ISWR
            return ["C::StoreVi<%du>(v, %s, %d);" % (dest, vi(is_), it)], False, None
        if index in (0x64, 0x30, 0x31, 0x3D) and it == 0:
            # MFP, MOVE, MR32, MFIR into VF0, which keeps 0, 0, 0, 1: nothing
            # (`move vf0, vf0` is the assemblers' NOP, a third of the lowers
            # that held a write). What they read still stalls: lower_reads.
            return [], False, None
        if index == 0x64:                                               # MFP
            return ["C::Mfp(v, pd, %d, %du);" % (it, dest)], True, None
        if index == 0x68:
            return setvi(it, "C::Top(v)"), False, None
        if index == 0x69:
            return setvi(it, "C::Itop(v)"), False, None
        if index == 0x6C:
            return ["C::Kick(v, %s);" % vi(is_)], False, None
        if 0x70 <= index <= 0x7E and index not in (0x77, 0x7B):
            latency = EFU_LATENCY[(index - 0x70) & 15]
            val = lambda lane: "C::Value(v, %d, %d)" % (is_, lane)
            if index in (0x70, 0x71, 0x72, 0x73):
                body = ["const double x = %s, y = %s, z = %s;" % (val(0), val(1), val(2)),
                        "const double sum = x * x + y * y + z * z;"]
                value = {0x70: "sum", 0x71: "1.0 / sum", 0x72: "std::sqrt(sum)",
                         0x73: "1.0 / std::sqrt(sum)"}[index]
            else:
                body = []
                value = {0x74: "std::atan2(%s, %s)" % (val(1), val(0)),
                         0x75: "std::atan2(%s, %s)" % (val(2), val(0)),
                         0x76: "%s + %s + %s + %s" % (val(0), val(1), val(2), val(3)),
                         0x78: "std::sqrt(std::fabs(%s))" % val(fsf),
                         0x79: "1.0 / std::sqrt(std::fabs(%s))" % val(fsf),
                         0x7A: "1.0 / %s" % val(fsf),
                         0x7C: "std::sin(%s)" % val(fsf),
                         0x7D: "std::atan(%s)" % val(fsf),
                         0x7E: "std::exp(-%s)" % val(fsf)}[index]
            return ["{"] + ["    " + line for line in body] + [
                "    C::Efu(pd, %s, %d);" % (value, latency), "}"], True, None
        ftf = field(word, 23, 2)
        if index == 0x38:                                               # DIV
            return ["C::Divide(pd, C::Div(v, %d, %d, %d, %d), 7);" % (is_, fsf, it, ftf)], True, None
        if index == 0x39:                                               # SQRT
            return ["C::Divide(pd, C::Sqrt(v, %d, %d), 7);" % (it, ftf)], True, None
        if index == 0x3A:                                               # RSQRT
            return ["C::Divide(pd, C::Rsqrt(v, %d, %d, %d, %d), 13);" % (is_, fsf, it, ftf)], True, None
        if index == 0x30:                                               # MOVE
            return ["C::Move(v, pd, %d, %d, %du);" % (it, is_, dest)], True, None
        if index == 0x31:                                               # MR32
            return ["C::Mr32(v, pd, %d, %d, %du);" % (it, is_, dest)], True, None
        if index == 0x3D:                                               # MFIR
            return ["C::Mfir(v, pd, %d, %d, %du);" % (it, is_, dest)], True, None
        if index == 0x3C:                                               # MTIR
            return (["C::Mtir(v, pd, %d, %d, %d);" % (it, is_, fsf)] if 0 < it < 16 else []), True, None
        if index in (0x3B, 0x7B):                                       # WAITQ WAITP: stalls
            return [], False, None
        return ["C::ThroughCop2(v, pd, 0x%08Xu, %d, %du);" % (word, it, dest)], True, None
    if op == 0x00:
        return ["C::Load(v, pd, %s, %d, %du);" % (plus_imm(vi(is_)), it, dest)], True, None
    if op == 0x01:
        return ["C::Store<%du>(v, %s, %d);" % (dest, plus_imm(vi(it)), is_)], False, None
    if op == 0x04:
        lanes = [lane for lane in range(4) if dest & (8 >> lane)]
        if not lanes:
            return [], False, None
        return setvi(it, "C::Word(v, %s, %d)" % (plus_imm(vi(is_)), lanes[0])), False, None
    if op == 0x05:
        return ["C::StoreVi<%du>(v, %s, %d);" % (dest, plus_imm(vi(is_)), it)], False, None
    unsigned = (field(word, 21, 4) << 11) | field(word, 0, 11)
    if op == 0x08:
        return setvi(it, "%s + 0x%Xu" % (vi(is_), unsigned)), False, None
    if op == 0x09:
        return setvi(it, "%s - 0x%Xu" % (vi(is_), unsigned)), False, None
    low24 = word & 0xFFFFFF
    if op == 0x10:
        return setvi(1, "u32((C::Clip(v, cyc) & 0xFFFFFF) == 0x%Xu)" % low24), False, None
    if op == 0x11:
        return ["C::Clip(v, cyc) = C::ClipLatest(v) = 0x%Xu;" % low24], False, None
    if op == 0x12:
        return setvi(1, "u32((C::Clip(v, cyc) & 0x%Xu) != 0)" % low24), False, None
    if op == 0x13:
        return setvi(1, "u32(((C::Clip(v, cyc) | 0x%Xu) & 0xFFFFFF) == 0xFFFFFF)" % low24), False, None
    if 0x14 <= op <= 0x17:
        imm = (field(word, 21, 1) << 11) | field(word, 0, 11)
        if op == 0x14:
            return setvi(it, "u32((C::Status(v, cyc) & 0xFFF) == 0x%Xu)" % imm), False, None
        if op == 0x15:
            return ["C::Status(v, cyc) = (C::Status(v, cyc) & 0x3F) | 0x%Xu;" % (imm & 0xFC0),
                    "C::StatusLatest(v) = (C::StatusLatest(v) & 0x3F) | 0x%Xu;" % (imm & 0xFC0)], False, None
        if op == 0x16:
            return setvi(it, "C::Status(v, cyc) & 0x%Xu" % imm), False, None
        return setvi(it, "u32(((C::Status(v, cyc) | 0x%Xu) & 0xFFF) == 0xFFF)" % imm), False, None
    if op == 0x18:
        return setvi(it, "u32((%s & 0xFFFF) == (C::Mac(v, cyc) & 0xFFFF))" % vi(is_)), False, None
    if op == 0x1A:
        return setvi(it, "%s & C::Mac(v, cyc)" % vi(is_)), False, None
    if op == 0x1B:
        return setvi(it, "%s | C::Mac(v, cyc)" % vi(is_)), False, None
    if op == 0x1C:
        return setvi(it, "C::Clip(v, cyc) & 0xFFF"), False, None
    if op == 0x20:
        return ["branch = 0x%Xu;" % target], False, "static"
    if op == 0x21:
        return setvi(it, "0x%Xu" % ((pc + 16) // 8)) + ["branch = 0x%Xu;" % target], False, "static"
    if op == 0x24:
        return ["branch = (%s * 8) & 0x%Xu;" % (vi(is_), mask)], False, "dynamic"
    if op == 0x25:
        return (["const u32 to = (%s * 8) & 0x%Xu;" % (vi(is_), mask)] +
                setvi(it, "0x%Xu" % ((pc + 16) // 8)) + ["branch = to;"]), False, "dynamic"
    conditions = {0x28: "a == b", 0x29: "a != b", 0x2C: "a < 0", 0x2D: "a > 0", 0x2E: "a <= 0", 0x2F: "a >= 0"}
    if op in conditions:
        return (["const std::int16_t a = std::int16_t(%s), b = std::int16_t(%s);" % (vi(is_), vi(it)),
                 "(void)b;",
                 "if (%s) branch = 0x%Xu;" % (conditions[op], target)]), False, "static"
    return [], False, None


def known_clean(word: int, clean: dict) -> int:
    """Which of an FMAC's operands are clean in every lane it keeps, from
    `clean` ((register or 'acc', lane) -> written clean earlier in the block):
    bit 0 fs, bit 1 ft (or its broadcast lane), bit 2 ACC."""
    fmac = decode_fmac(word)
    if not fmac or clean is None:
        return 0
    op, operand, _to_acc = fmac
    dest, ft, fs = field(word, 21, 4), field(word, 16, 5), field(word, 11, 5)
    lanes = [lane for lane in range(4) if dest & (8 >> lane)]
    is_clean = lambda r, lane: r == 0 or clean.get((r, lane), False)
    known = 0
    if all(is_clean(fs, lane) for lane in lanes):
        known |= 1
    if operand < 4:
        known |= 2 if is_clean(ft, operand) else 0
    elif operand == 6:
        known |= 2 if all(is_clean(ft, lane) for lane in lanes) else 0
    if all(clean.get(("acc", lane), False) for lane in lanes):
        known |= 4
    return known


def clean_writes(word: int) -> list:
    """(register or 'acc', lanes, clean) the upper instruction writes: the
    FMAC, OPMULA, OPMSUB, ITOF and ABS leave their fields clean - but MAX and
    MINI hand an operand back as it is; FTOI leaves integers, anything else is
    not known."""
    funct, dest = word & 0x3F, field(word, 21, 4)
    ft, fs, fd = field(word, 16, 5), field(word, 11, 5), field(word, 6, 5)
    fmac = decode_fmac(word)
    if fmac:
        return [("acc" if fmac[2] else fd, dest, OP_NAMES[fmac[0]] not in ("lifted::kMax_", "lifted::kMini"))]
    if funct < 0x3C:
        if funct == 0x2E:
            return [(fd, dest & 0xE, True)]                             # OPMSUB
        if funct >= 0x30:
            return []
        return [(fd, dest, False), ("acc", 0xF, False)]
    index = special_index(word)
    if index in (0x1F, 0x2F):                                           # CLIP, NOP
        return []
    if index == 0x2E:
        return [("acc", dest & 0xE, True)]                              # OPMULA
    if 0x10 <= index <= 0x13 or index == 0x1D:
        return [(ft, dest, True)]                                       # ITOF, ABS
    if 0x14 <= index <= 0x17:
        return [(ft, dest, False)]                                      # FTOI
    return [("acc", 0xF, False)]


def emit_upper(word: int, clean: dict = None):
    """C++ for the upper instruction; (lines, written, lanes, set_flags)."""
    funct, dest = word & 0x3F, field(word, 21, 4)
    ft, fs, fd = field(word, 16, 5), field(word, 11, 5), field(word, 6, 5)
    if funct < 0x3C:
        if funct >= 0x30:
            return [], -1, 0, False
        target = fd
        flags = not (0x10 <= funct <= 0x17 or funct in (0x1D, 0x1F, 0x2B, 0x2F))
    else:
        index = special_index(word)
        if index == 0x1F:
            return ["C::ClipJudge(v, %d, %d);" % (fs, ft)], -1, 0, True
        if index == 0x2F:
            return [], -1, 0, False
        if 0x10 <= index <= 0x17 or index == 0x1D:
            target, flags = ft, False
        else:
            target, flags = -1, True
    fmac = decode_fmac(word)
    lines = []
    if fmac:
        op, operand, to_acc = fmac
        known = known_clean(word, clean)
        lines.append("C::Fmac<%s, %d, %d, %du%s>(v, %d, %d);" % (OP_NAMES[op], operand, -1 if to_acc else fd,
                                                                dest, ", %d" % known if known else "", fs, ft))
    elif funct == 0x2E:                                                 # OPMSUB
        lines.append("C::Opmsub<%du>(v, %d, %d, %d);" % (dest, fd, fs, ft))
    elif funct >= 0x3C and special_index(word) == 0x2E:                 # OPMULA
        lines.append("C::Opmula<%du>(v, %d, %d);" % (dest, fs, ft))
    elif funct >= 0x3C and 0x10 <= special_index(word) <= 0x13:         # ITOF0 4 12 15
        lines.append("C::Itof<%du, %d>(v, %d, %d);" % (dest, special_index(word) & 3, ft, fs))
    elif funct >= 0x3C and 0x14 <= special_index(word) <= 0x17:         # FTOI0 4 12 15
        lines.append("C::Ftoi<%du, %d>(v, %d, %d);" % (dest, special_index(word) & 3, ft, fs))
    elif funct >= 0x3C and special_index(word) == 0x1D:                 # ABS
        lines.append("C::Abs<%du>(v, %d, %d);" % (dest, ft, fs))
    else:
        lines.append("C::Cop2(v, 0x%08Xu);" % word)
    if flags:
        lines.append("C::Flags<%d, %du>(v);" % (target, dest))
    written = target if target > 0 else -1
    return lines, written, dest if target > 0 else 0, flags


def fuse_flags(lines: list) -> list:
    """An FMAC then its flags, as one: Fmac's SetFlags (1 Flags, 2
    StickyFlags) takes them from the result while it is in a register."""
    if len(lines) != 2 or not lines[0].startswith("C::Fmac<"):
        return lines
    mode = 1 if lines[1].startswith("C::Flags<") else 2 if lines[1].startswith("C::StickyFlags<") else 0
    if not mode:
        return lines
    head, args = lines[0][:-len(lines[0].split(">(")[-1]) - 2], lines[0].split(">(")[-1]
    params = head[len("C::Fmac<"):].split(", ")
    if len(params) == 4:
        params.append("0")
    return ["C::Fmac<%s, %d>(%s" % (", ".join(params), mode, args)]


def mac_only(lines: list) -> list:
    """The upper's flags as a fast block of a program that never reads
    STATUS sets them: MAC alone (Fmac's SetFlags 3, MacFlags), STATUS's
    sticky bits not at all."""
    out = []
    for line in lines:
        if line.startswith("C::StickyFlags<"):
            continue
        if line.startswith("C::Flags<"):
            line = "C::MacFlags<" + line[len("C::Flags<"):]
        elif line.startswith("C::Fmac<"):
            head, args = line.split(">(", 1)
            params = head[len("C::Fmac<"):].split(", ")
            if len(params) == 6:
                if params[5] == "2":
                    params = params[:5]
                elif params[5] == "1":
                    params[5] = "3"
                line = "C::Fmac<%s>(%s" % (", ".join(params), args)
        out.append(line)
    return out


# The lower operations that hold a vector register write back to the pair's
# end, and where their arguments name it: (index of `it`, index of `dest`)
# counted from the end of the argument list.
HELD_WRITES = {"Load": (-2, -1), "Move": (-3, -1), "Mr32": (-3, -1), "Mfir": (-3, -1),
               "Mfp": (-2, -1), "ThroughCop2": (-2, -1)}


def lower_writes(lines: list) -> list:
    """(register, lanes) the lower's held write may land in, from its C++."""
    out = []
    for line in lines:
        line = line.strip()
        for name, (it, dest) in HELD_WRITES.items():
            if line.startswith("C::%s(v, pd, " % name):
                args = line[len("C::%s(" % name):].rstrip(");").split(", ")
                out.append((int(args[it]), int(args[dest].rstrip("u"))))
    return out


def needless_wait(r: int, lane: int, index: int, written: dict, entry: dict = None) -> bool:
    """Whether a stall check on (r, lane) at the block's pair `index` can never
    stall: VF0 is never written; a field written by the block's pair j lands
    at j's issue cycle + 4, and every pair takes a cycle at least, so from
    pair j + 4 on it has landed; a field the block has not written was last
    written before the block, landing by the entry cycle + 3 at the latest
    (only ever set to an issue cycle + 4, and the cycle only grows), so from
    pair 3 on."""
    if r == 0:
        return True
    j = written.get((r, lane))
    if j is not None:
        return index - j >= 4
    if entry is not None:
        # An entry version (Chain.version): the fields written in the three
        # pairs before the block, d pairs before the pair it left after (0 the
        # last), the block before having three pairs at least - one written
        # there lands by that pair's issue + 4 - d, and the block's pair
        # `index` issues at that issue + 1 + index at the earliest; any other
        # field landed before the block.
        d = entry.get((r, lane))
        return d is None or d + index >= 3
    return index >= 3


class Flags:
    """A block's flags as the generator follows them. A pair that sets MAC
    (and STATUS) or CLIP does not queue them (FinishUpper's flag queue, a
    dozen stores and a loop to land them): it keeps them in locals, held -
    in flight - until the generator knows they have landed. A pair takes a
    cycle at least, and exactly one where it cannot stall (no Wait: `stalls`
    has those that can), so a reader at pair i knows pair j's flags landed
    (c_i >= c_j + 4) when i - j >= 4, and not when i - j < 4 with no stall
    in between; then it lands them itself (Land Mac/Clip, after what is
    still queued: older, so landed too) and reads them straight, the queue
    known empty. Where it cannot know, or as the block leaves, the held ones
    are queued as FinishUpper would have (QueueFlags) - those certain to
    have landed by then landed instead, as Issue would land them anyway.
    `held`: [(pair, 'mac' | 'clip')], in order; every one is newer than
    anything queued."""

    def __init__(self):
        self.held = []
        self.empty = False          # the queue known empty
        self.stalls = set()
        self.declared = []

    def landed(self, j: int, i: int):
        """Whether pair j's flags have landed by pair i's issue: True, False
        or None (not known: a stall between)."""
        if i - j >= 4:
            return True
        if any(k in self.stalls for k in range(j + 1, i + 1)):
            return None
        return False

    def land(self, entries: list) -> list:
        """Lines landing `entries` (a prefix of held, in order)."""
        lines = []
        if not entries:
            return lines
        if not self.empty:
            lines.append("C::LandAllFlags(v);")
        macs = [j for j, kind in entries if kind == "mac"]
        clips = [j for j, kind in entries if kind == "clip"]
        if macs:
            sticky = "".join(" | (fs%d & 0xC0u)" % j for j in macs[:-1])
            lines.append("C::LandMac(v, fm%d, fs%d%s);" % (macs[-1], macs[-1], sticky))
        if clips:
            lines.append("C::LandClip(v, fl%d);" % clips[-1])
        return lines

    @staticmethod
    def queue(entries: list) -> list:
        return ["C::QueueFlags(v, fc%d, fm%d, fs%d, 0u, true, false);" % (j, j, j) if kind == "mac" else
                "C::QueueFlags(v, fc%d, 0u, 0u, fl%d, false, true);" % (j, j) for j, kind in entries]

    def before_reader(self, i: int) -> list:
        """At a reader in pair i: the held flags landed by now landed, or all
        of them queued if it cannot tell. Changes what is held."""
        states = [self.landed(j, i) for j, _ in self.held]
        if None in states:
            lines = self.queue(self.held)
            self.held = []
            self.empty = False
            return lines
        count = states.count(True)                  # a prefix: later pairs land later
        lines = self.land(self.held[:count])
        if count:
            self.empty = True
        self.held = self.held[count:]
        return lines

    def leaving(self, i: int, quiet: bool = False) -> str:
        """As the block leaves after pair i (its cycle then c_i + 1): those
        landed by then (i - j >= 3) landed, the rest queued - or all landed
        where it goes on into a block that is `quiet`: one that reads no
        flags in its first three pairs, nor leaves before them, so that
        anything read after has landed (held flags land by c_i + 4, and its
        third pair issues at c_i + 3 at the earliest), as Issue would have
        landed them on the way. Changes nothing: the block goes on where it
        does not leave."""
        certain = [(j, kind) for j, kind in self.held if quiet or i - j >= 3]
        rest = self.held[len(certain):]
        return " ".join(self.land(certain) + self.queue(rest))


# How a block leaves: its locals back where the driver and the Vu keep them.
LEAVE = "s.count = count; s.stopping = stopping; s.delayed = delayed; C::Cycle(v) = cyc; return;"


def leave_to(pc: int, chain, recent: dict = None) -> str:
    """LEAVE where the run goes on at `pc`, a pair the generator knows: into
    that block (PS2HOST_VU_NEXT) if the program has one starting there - its
    entry version for `recent` (the fields written in the last three pairs,
    and how many pairs before the last: Chain.version) where it has one."""
    if chain and pc in chain:
        name = chain.version(pc, recent) if hasattr(chain, "version") else chain[pc]
        return LEAVE[:-len("return;")] + "PS2HOST_VU_NEXT(%s);" % name
    return LEAVE


# The entry versions a block may have (Chain.version), past which the run
# goes on into the block as it is.
MAX_VERSIONS = int(os.environ.get("PS2_VU_MAX_VERSIONS", "6"))
# PS2_VU_NO_VERSIONS=1: none (to compare). Versions that also knew the
# fields left clean at the exit (their clamps left out) were slower: more
# code (fast bench 53.8 -> 56.6-58.7 ms, at 6 or 16 versions a block).
VERSIONS = os.environ.get("PS2_VU_NO_VERSIONS", "") != "1"


class Chain:
    """A program's blocks of one kind by where they start (`names`), and the
    entry versions its exits asked for: a block entered from a known exit -
    a block of three pairs or more before it - with the fields written in its
    last three pairs (`recent`: {(register, lane): pairs before the last})
    known, its stall checks on anything else left out (needless_wait).
    `wanted`: (pc, recent, name) still to be written."""

    def __init__(self, names: dict, blocks: dict = None, pairs: dict = None):
        self.names = names
        self.blocks = blocks or {}          # start -> the block's pcs
        self.pairs = pairs or {}
        self.versions = {}
        self.wanted = []
        self.uses = {}

    def used(self, pc: int) -> set:
        """The fields the first three pairs of the block at `pc` read: the
        only stall checks `recent` can leave out (needless_wait) - so its
        versions differ only where the code would."""
        if pc not in self.uses:
            stalls = set()
            for at in self.blocks.get(pc, [])[:3]:
                lower, upper = self.pairs[at]
                for r, lanes in upper_reads(upper) + ([] if upper & 0x80000000 else lower_reads(lower)):
                    for lane in range(4):
                        if lanes & (8 >> lane):
                            stalls.add((r, lane))
            self.uses[pc] = stalls
        return self.uses[pc]

    def __contains__(self, pc) -> bool:
        return pc in self.names

    def __getitem__(self, pc) -> str:
        return self.names[pc]

    def version(self, pc: int, recent: dict) -> str:
        if recent is None or not VERSIONS:
            return self.names[pc]
        if self.blocks:
            stalls = self.used(pc)
            recent = {k: d for k, d in recent.items() if k in stalls}
        key = (pc, frozenset(recent.items()))
        if key not in self.versions:
            count = sum(1 for k in self.versions if k[0] == pc)
            if count >= MAX_VERSIONS:
                return self.names[pc]
            name = "%sE%d" % (self.names[pc], count)
            self.versions[key] = name
            self.wanted.append((pc, dict(recent), name))
        return self.versions[key]


def emit_pair(pc: int, lower: int, upper: int, mask: int, last: bool, index: int = 0,
              written: dict = None, clean: dict = None, dead_flags: bool = False,
              after: tuple = None, no_flags: bool = False, chain: dict = None, fl: Flags = None,
              quiet: set = frozenset(), no_status: bool = False, entry: dict = None,
              unsafe: frozenset = frozenset()) -> list:
    """One pair, as Run runs it, then where the run goes: on to the next pair
    of the block, or (`last`, a branch taken, the program's end) back to the
    driver with s.pc. `index` is its place in the block and `written` where
    the block's pairs before it wrote each vector field; the pair's own
    writes are added to it. `after` is the pair before it in the block
    (lower, upper), None for the first: from the second pair on, a block
    runs with no E bit or branch pending from before it (emit_block), so
    `stopping` and `delayed` can be set only by the pair before - and are
    checked only after one that may set them. `no_flags`: the fast blocks
    of a program that never reads MAC or STATUS (fast_eligible) - an FMAC
    leaves both alone (CLIP is as ever). `no_status`: the fast blocks of a
    program that never reads STATUS - an FMAC sets MAC alone (mac_only).
    `unsafe`: whether any of the program's blocks starts knowing fields
    clean (entry_clean) - a jump whose target is not known says so
    (s.jumped), and where it lands on such a block the driver hands the run
    to the interpreter (VuCompiled::knows)."""
    if written is None:
        written = {}
    immediate = bool(upper & 0x80000000)
    end = bool(upper & 0x40000000)
    reads = upper_reads(upper) + ([] if immediate else lower_reads(lower))
    out = ["    {"]
    body = []
    waits = []
    for r, lanes in reads:
        for lane in range(4):
            if lanes & (8 >> lane) and (r, lane) not in waits and not needless_wait(r, lane, index, written, entry):
                waits.append((r, lane))
    q_waits = not immediate and (is_divide(lower) or is_wait(lower, 0x3B))
    p_waits = not immediate and (is_efu(lower) or is_wait(lower, 0x7B))
    if (waits or q_waits or p_waits) and fl is not None:
        fl.stalls.add(index)
    if waits or q_waits or p_waits:
        body.append("u64 t = cyc;")
        body += ["C::Wait(t, v, %d, %d);" % w for w in waits]
        if q_waits:
            body.append("C::WaitQ(t, v);")
        if p_waits:
            body.append("C::WaitP(t, v);")
        body.append("C::Issue(v, cyc, t);")
    else:
        body.append("C::Issue(v, cyc, cyc);")
    issue_at = len(body) - 1
    lower_lines, pending, branch = ([], False, None) if immediate else emit_lower(lower, pc, mask)
    if pending:
        body.append("C::Pending pd;")
    if branch:
        body.append("u32 branch = kNone;")
    body += lower_lines
    upper_lines, upper_written, lanes, set_flags = emit_upper(upper, clean)
    if no_flags and set_flags and not upper_lines[0].startswith("C::ClipJudge"):
        upper_lines = [line for line in upper_lines if not line.startswith("C::Flags<")]
        set_flags = False
    if dead_flags:
        upper_lines = [line.replace("C::Flags<", "C::StickyFlags<") for line in upper_lines]
        set_flags = False
    upper_lines = fuse_flags(upper_lines)
    if no_status:
        upper_lines = mac_only(upper_lines)
    if fl is not None and any(name in line for line in lower_lines
                              for name in ("C::Mac(v, cyc)", "C::Clip(v, cyc)", "C::Status(v, cyc)")):
        # A flag reader: what it reads landed first (Flags), and straight
        # from the Vu where the queue is known empty.
        at = len(body) - len(lower_lines)
        body[at:at] = fl.before_reader(index)
        if fl.empty:
            for k in range(len(body) - len(lower_lines), len(body)):
                body[k] = (body[k].replace("C::Mac(v, cyc)", "C::MacNow(v)")
                           .replace("C::Clip(v, cyc)", "C::ClipNow(v)")
                           .replace("C::Status(v, cyc)", "C::StatusNow(v)"))
    body += upper_lines
    # Q and P land at a pair's issue only where it can tell: a pair that
    # reads Q (an FMAC's Q operand) or P (MFP), starts a DIV or an EFU op
    # (which would replace a result due but not landed), copies the VI file
    # (MTIR, macro mode's COP2) or waits for either. Any other pair leaves
    # them to the next that can tell, which lands all that is due by then -
    # the same values in Q and P wherever they are read (IssueAt).
    fmac = decode_fmac(upper)
    lands = q_waits or p_waits or (fmac is not None and fmac[1] == 4) or any(
        name in line for line in lower_lines + upper_lines
        for name in ("Cop2(", "Mtir(", "Mfp(", "Divide(", "Efu("))
    if not lands:
        body[issue_at] = body[issue_at].replace("C::Issue(v, cyc, ", "C::IssueAt(cyc, ")
    for r, dest in lower_writes(lower_lines) + [(upper_written, lanes)]:
        for lane in range(4):
            if r > 0 and dest & (8 >> lane):
                written[(r, lane)] = index
    if clean is not None:
        # The pair's end: the lower's held writes (not known clean), then the
        # upper's, which win over them in the fields both write.
        for r, dest in lower_writes(lower_lines):
            for lane in range(4):
                if dest & (8 >> lane):
                    clean[(r, lane)] = False
        for r, dest, is_clean in clean_writes(upper):
            for lane in range(4):
                if dest & (8 >> lane) and r != 0:
                    clean[(r, lane)] = is_clean
    written_reg = upper_written
    if immediate:
        body.append("v.regs.vc[lifted::kRegI] = 0x%08Xu;" % lower)
    sets_mac = "true" if (upper & 0x7FF) != 0x1FF else "false"
    sets_clip = "true" if (upper & 0x7FF) == 0x1FF else "false"
    if set_flags and fl is not None:
        # Held, not queued (Flags): the pair's flags as FinishUpper takes them.
        kind = "mac" if sets_mac == "true" else "clip"
        if kind == "mac":
            body.append("fm%d = C::MacLatest(v); fs%d = %s; fc%d = cyc;"
                        % (index, index, "0u" if no_status else "C::StatusLatest(v)", index))
            fl.declared.append("u32 fm%d, fs%d; u64 fc%d;" % (index, index, index))
        else:
            body.append("fl%d = C::ClipLatest(v); fc%d = cyc;" % (index, index))
            fl.declared.append("u32 fl%d; u64 fc%d;" % (index, index))
        fl.held.append((index, kind))
        set_flags = False
    flags = "true" if set_flags else "false"
    # VI0 back to 0 (End) after a block's first pair (a run may start with
    # it otherwise) and after macro mode's COP2 operations, which may write
    # it; every other pair leaves it 0 (Next).
    vi0 = index == 0 or any("Cop2(" in line for line in body)
    if pending:
        body.append("C::Finish(v, cyc, pd, %d, %du, %s, %s, %s%s);" % (written_reg, lanes, flags, sets_mac,
                                                                      sets_clip, "" if vi0 else ", false"))
    else:
        if written_reg > 0 or set_flags:
            body.append("C::FinishUpper(v, cyc, %d, %du, %s, %s, %s);" % (written_reg, lanes, flags, sets_mac,
                                                                          sets_clip))
        body.append("C::End(v, cyc);" if vi0 else "C::Next(cyc);")
    next_pc = (pc + 8) & mask
    body.append("++count;")
    # What an entry version of where it goes on may know (Chain.version):
    # the fields written in this pair and the two before, three pairs at
    # least run in the block.
    recent = {k: index - j for k, j in written.items() if index - j <= 2} if index >= 2 else None
    leaving = fl.leaving(index) + " " if fl is not None and fl.held else ""
    # Going on into a quiet block with nothing pending (no E bit here, no
    # branch to follow): the held flags all landed (Flags.leaving).
    calm = not end and not branch
    landing_into = lambda to: (fl.leaving(index, to in quiet and calm and to in (chain or {})) + " "
                               if fl is not None and fl.held else "")
    may_stop = after is None or bool(after[1] & 0x40000000)
    may_branch = after is None or is_branch(*after)[0]
    if may_stop:
        body.append("if (stopping) { s.pc = 0x%Xu; s.done = true; %s%s }" % (next_pc, leaving, LEAVE))
    if end:
        body.append("stopping = true;")
    new_delayed = "branch" if branch else "kNone"
    if may_branch:
        # After a branch of the block with a target of its own, delayed is
        # that target if it is set at all.
        static = after is not None and is_branch(*after)[1] == "static"
        target = (pc - 8 + 8 + imm11(after[0]) * 8) & mask if static else None
        if target is not None:
            body.append("if (delayed != kNone) { s.pc = 0x%Xu; delayed = %s; %s%s }" %
                        (target, new_delayed, landing_into(target), leave_to(target, chain, recent)))
        else:
            guard = "s.jumped = true; " if unsafe else ""
            body.append("if (delayed != kNone) { s.pc = delayed; %sdelayed = %s; %s%s }" %
                        (guard, new_delayed, leaving, LEAVE))
    if branch:
        body.append("delayed = branch;")
    if last:
        body.append("s.pc = 0x%Xu;" % next_pc)
        if leaving:
            body.append(landing_into(next_pc).strip())
        body.append(leave_to(next_pc, chain, recent))
    out += ["        " + line for line in body]
    out.append("    }")
    return out


def is_branch(lower: int, upper: int):
    """(a branch, its static target if it has one, conditional) for the pair's lower."""
    if upper & 0x80000000:
        return False, None, False
    op = lower >> 25
    conditional = op in (0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F)
    if op in (0x20, 0x21) or conditional:
        return True, "static", conditional
    return op in (0x24, 0x25), None, False


def blocks_of(pairs: dict, entries: list, mask: int) -> list:
    """The program as blocks - [[pc, ...], ...] - each entered at its first
    pair only: where runs start, branch targets, the pair after an
    unconditional branch's delay slot (a return's place), the start of each
    run of code. A block ends at an unconditional branch's delay slot, at the
    pair after an E bit and where code or a leader breaks it. A conditional
    branch does not end it: the block goes on where it is not taken, and
    leaves after the delay slot where it is (a side exit) - so what the
    generator knows (the fields written, the clean registers, the flags
    held) goes on with it, where a block of its own started knowing
    nothing: the race's clipping loops were blocks of two and three pairs."""
    leaders = set(e for e in entries if e in pairs)
    for pc, (lower, upper) in pairs.items():
        branch, static, conditional = is_branch(lower, upper)
        if branch:
            if static:
                target = (pc + 8 + imm11(lower) * 8) & mask
                if target in pairs:
                    leaders.add(target)
            if not conditional:
                leaders.add((pc + 16) & mask)
        if upper & 0x40000000:
            leaders.add((pc + 16) & mask)
    blocks = []
    previous = None
    ends_block = False
    for pc in sorted(pairs):
        if not blocks or pc in leaders or ends_block or previous + 8 != pc:
            blocks.append([])
        blocks[-1].append(pc)
        lower, upper = pairs[pc]
        before = pairs.get((pc - 8) & mask)
        ends_block = bool(before and ((is_branch(*before)[0] and not is_branch(*before)[2]) or
                                      before[1] & 0x40000000))
        previous = pc
    return blocks


# PS2_VU_NO_ENTRY_CLEAN=1: blocks start knowing nothing clean (to compare).
ENTRY_CLEAN = os.environ.get("PS2_VU_NO_ENTRY_CLEAN", "") != "1"


def clean_after(state: set, pc: int, lower: int, upper: int, mask: int) -> set:
    """The fields clean after a pair, from those before it - as emit_pair
    keeps `clean`: the lower's held writes not known clean, then the upper's."""
    s = set(state)
    if not upper & 0x80000000:
        for r, dest in lower_writes(emit_lower(lower, pc, mask)[0]):
            for lane in range(4):
                if dest & (8 >> lane):
                    s.discard((r, lane))
    for r, dest, is_clean in clean_writes(upper):
        for lane in range(4):
            if dest & (8 >> lane) and r != 0:
                (s.add if is_clean else s.discard)((r, lane))
    return s


def entry_clean(blocks: list, pairs: dict, entries: list, mask: int) -> dict:
    """The fields each block can start knowing clean - in every lane, from
    every way into it: {start: frozenset of (register or 'acc', lane)}.
    A clean field is one an operand's clamp would leave as it is (an FMAC's
    result, ITOF's, ABS's...: known_clean); a block knew only those written
    in it, so 79% of the clamps the explosion's VU1 ran were of fields from
    before the block. Its predecessors are the blocks that go on into it
    (straight on, a branch, a conditional branch's side exit); a block a run
    may start at (an entry, the pair after an E bit, which a later run goes
    on from), a return's place (after an unconditional branch's delay slot:
    JR's target), or one with no known way in starts knowing nothing. A jump
    to a register's address may land anywhere: where it lands on a block that
    knows something (`unsafe`), the run goes on in the interpreter (emit_pair).
    No versions: the same code, its clamps fewer."""
    by_start = {b[0]: b for b in blocks}
    if not ENTRY_CLEAN:
        return {start: frozenset() for start in by_start}
    unknown = set(e for e in entries if e in by_start)
    for pc, (lower, upper) in pairs.items():
        branch, static, conditional = is_branch(lower, upper)
        after = (pc + 16) & mask
        if after in by_start and ((branch and not conditional) or upper & 0x40000000):
            unknown.add(after)

    def exits(block, state):
        out, s, taken = [], set(state), None
        for pc in block:
            lower, upper = pairs[pc]
            s = clean_after(s, pc, lower, upper, mask)
            if taken is not None:
                out.append((taken, frozenset(s)))
                taken = None
            branch, static, _conditional = is_branch(lower, upper)
            if branch and static:
                target = (pc + 8 + imm11(lower) * 8) & mask
                if target in by_start:
                    taken = target                  # after its delay slot
            if upper & 0x40000000:
                return out
        if taken is not None:
            out.append((taken, frozenset(s)))
        last = block[-1]
        before = pairs.get((last - 8) & mask)
        jumped = before is not None and is_branch(*before)[0] and not is_branch(*before)[2]
        if not jumped and not pairs[last][1] & 0x40000000 and ((last + 8) & mask) in by_start:
            out.append(((last + 8) & mask, frozenset(s)))
        return out

    every = frozenset([(r, lane) for r in range(1, 32) for lane in range(4)] + [("acc", lane) for lane in range(4)])
    state = {start: frozenset() if start in unknown else every for start in by_start}
    changed = True
    while changed:
        changed = False
        met = {start: None for start in by_start}
        for block in blocks:
            for target, s in exits(block, state[block[0]]):
                met[target] = s if met[target] is None else met[target] & s
        for start in by_start:
            now = frozenset() if start in unknown or met[start] is None else met[start]
            if now != state[start]:
                state[start] = now
                changed = True
    return state


def reads_mac(lower: int, upper: int) -> bool:
    """FSEQ, FSSET, FSAND, FSOR, FMEQ, FMAND, FMOR: the lowers that read or
    set the landed MAC or STATUS flags (FCEQ ... FCOR and FCGET touch CLIP
    alone)."""
    return not (upper & 0x80000000) and 0x14 <= (lower >> 25) <= 0x1B


def sets_mac(upper: int) -> bool:
    """Whether the upper pushes MAC and STATUS flags (an FMAC; not CLIP)."""
    return emit_upper(upper)[3] and (upper & 0x7FF) != 0x1FF


def dead_flags(block: list, pairs: dict) -> set:
    """The block's pairs whose MAC/STATUS flags no one can see: the next
    pair j of the block that pushes its own is four or more pairs from the
    block's end, and nothing reads or sets the landed MAC or STATUS from
    after i to three pairs after j. j's flags land four cycles after it
    issues, and a pair takes one cycle at least, so from j + 4 on - and
    after the block, whatever comes next - the landed flags are j's or
    later ones; before that a reader might see i's. The block runs whole
    (it does not start where it could not: emit_block) but for its side
    exits (a conditional branch taken, after its delay slot k): none may
    come from i to three pairs after j - before j, j does not run; after
    j, its flags may not have landed. STATUS's sticky bits are still
    gathered."""
    n = len(block)
    exits = [k for k in range(1, n) if is_branch(*pairs[block[k - 1]])[0]]
    dead = set()
    for i in range(n):
        if not sets_mac(pairs[block[i]][1]):
            continue
        j = next((j for j in range(i + 1, n) if sets_mac(pairs[block[j]][1])), None)
        if j is None or j >= n - 3 or any(i <= k < j + 3 for k in exits):
            continue
        if not any(reads_mac(*pairs[block[k]]) for k in range(i + 1, j + 4)):
            dead.add(i)
    return dead


def reads_status(lower: int, upper: int) -> bool:
    """FSEQ, FSSET, FSAND, FSOR: the lowers that read or set STATUS."""
    return not (upper & 0x80000000) and 0x14 <= (lower >> 25) <= 0x17


def reads_mac_only(lower: int, upper: int) -> bool:
    """FMEQ, FMAND, FMOR: the lowers that read the landed MAC flags."""
    return not (upper & 0x80000000) and (lower >> 25) in (0x18, 0x1A, 0x1B)


class PairFacts:
    """What the timing of a pair depends on, for mac_live: the vector fields
    its stall checks read (as emit_pair has them), the fields it writes (the
    upper's result, the lower's held write), and whether it waits for Q or P."""

    def __init__(self, pairs: dict, mask: int):
        self.pairs, self.mask, self.cache = pairs, mask, {}

    def of(self, pc: int):
        if pc not in self.cache:
            lower, upper = self.pairs[pc]
            immediate = bool(upper & 0x80000000)
            reads = set()
            for r, lanes in upper_reads(upper) + ([] if immediate else lower_reads(lower)):
                for lane in range(4):
                    if r and lanes & (8 >> lane):
                        reads.add((r, lane))
            writes = set()
            _lines, written, lanes, _flags = emit_upper(upper)
            if written > 0:
                writes |= {(written, lane) for lane in range(4) if lanes & (8 >> lane)}
            if not immediate:
                for r, dest in lower_writes(emit_lower(lower, pc, self.mask)[0]):
                    writes |= {(r, lane) for lane in range(4) if r > 0 and dest & (8 >> lane)}
            qp = not immediate and (is_divide(lower) or is_wait(lower, 0x3B) or is_efu(lower) or
                                    is_wait(lower, 0x7B))
            self.cache[pc] = (frozenset(reads), frozenset(writes), qp)
        return self.cache[pc]


# PS2_VU_OLD_MAC_LIVE=1: mac_live without the timing of a path that cannot
# stall (any reader before a later FMAC's flags landed keeps them).
OLD_MAC_LIVE = os.environ.get("PS2_VU_OLD_MAC_LIVE", "") == "1"


def mac_live(pairs: dict, mask: int, start: int, entries: set = None, facts: PairFacts = None) -> bool:
    """Whether a reader of MAC can see the flags of the FMAC at `start`: a
    walk of the program from the pair after it, branches and delay slots as
    they run, until a later FMAC's flags have landed for certain - four
    pairs after it, a pair taking a cycle at least - or the program ends.
    A reader met before then sees them - unless it issues before they land
    (four cycles after `start`'s issue) whatever the stalls on the way: the
    walk keeps a bound on each pair's issue, from `start`'s - a cycle after
    the pair before, or when a field it reads, written in the three pairs
    before it, lands (four after its writer's issue: needless_wait), the
    later; none where it waits for Q or P, or reads a field whose writer is
    not known. Before `start` they are the three pairs before it in the
    program's order where none of them or `start` is a branch target or an
    entry (`entries`: the program's entries and static targets), issued a
    cycle apart at least. Code not recorded, or a jump to a register's
    address (where it goes is not known), sees them too."""
    END = "end"
    seen = set()
    todo = []

    def after(pc, pending):
        """The pair to run after `pc` (pending: its branch's targets, if it
        is a delay slot; END after an E bit): [(pc, pending)] or None."""
        if pending == END:
            return []
        if pending is not None:
            return [(t, None) for t in pending]
        lower, upper = pairs[pc]
        branch, static, conditional = is_branch(lower, upper)
        if branch:
            if not static:
                return None                         # JR, JALR: anywhere
            targets = [(pc + 8 + imm11(lower) * 8) & mask]
            if conditional:
                targets.append((pc + 16) & mask)
            return [((pc + 8) & mask, tuple(targets))]
        if upper & 0x40000000:
            return [((pc + 8) & mask, END)]
        return [((pc + 8) & mask, None)]

    # `start` runs as a pair of its own, or (after a branch) as its delay
    # slot - then what follows is the branch's target.
    first = after(start, None)
    if first is None:
        return True
    before = pairs.get((start - 8) & mask)
    if before and is_branch(*before)[0]:
        slot = after((start - 8) & mask, None)
        if slot is None:
            return True                             # JR, JALR: anywhere
        first += after(start, slot[0][1])
    # A path's state: the pair's place after `start` (depth, capped at 4) and
    # the three pairs before it - (writes, a bound on its issue cycle from
    # `start`'s) each, None not known - or no bound (exact False).
    INF = 99
    exact0 = not OLD_MAC_LIVE and facts is not None
    history = None
    if exact0:
        befores = [(start - 8 * k) & mask for k in (2, 1)]
        straight = entries is not None and all(pc in pairs for pc in befores) and not any(
            pc in entries for pc in [start] + befores[1:])
        history = tuple((facts.of(pc)[1], -k) if straight else None for pc, k in zip(befores, (2, 1)))
        history += ((facts.of(start)[1], 0),)
    todo = [(pc, pending, None, 1, exact0, history, 0) for pc, pending in first]
    while todo:
        pc, pending, since, depth, exact, history, bound = todo.pop()
        if since is not None:
            since += 1
            if since >= 4:
                continue                            # a later FMAC's flags have landed
        if (pc, pending, since, depth, exact, history, bound) in seen:
            continue
        seen.add((pc, pending, since, depth, exact, history, bound))
        if pc not in pairs:
            return True
        lower, upper = pairs[pc]
        if pending is not None and pending != END and is_branch(lower, upper)[0]:
            return True                             # a branch in a delay slot: not followed
        issue = INF
        if exact:
            reads, writes, qp = facts.of(pc)
            issue = bound + 1
            if qp:
                issue = INF
            for h in history:
                if h is None:
                    if reads:
                        issue = INF
                elif reads & h[0]:
                    issue = max(issue, h[1] + 4)
        if reads_mac_only(lower, upper) and issue >= 4:
            return True
        # Counted from the first later FMAC: once its flags land, MAC is
        # never `start`'s again (every one that sets it replaces it whole).
        if sets_mac(upper) and (since is None or OLD_MAC_LIVE):
            since = 0
        nexts = after(pc, pending)
        if nexts is None:
            return True
        if issue >= 4:                              # from here on as without bounds
            todo += [(n, np, since, 4, False, None, 0) for n, np in nexts]
        else:
            history = history[1:] + ((facts.of(pc)[1], issue),)
            todo += [(n, np, since, depth + 1, True, history, issue) for n, np in nexts]
    return False


def static_targets(pairs: dict, mask: int) -> set:
    """Where the program's branches with a target of their own go."""
    out = set()
    for pc, (lower, upper) in pairs.items():
        branch, static, _conditional = is_branch(lower, upper)
        if branch and static:
            out.add((pc + 8 + imm11(lower) * 8) & mask)
    return out


def fast_drops(unit: int, pairs: dict, mask: int, entries: list = ()) -> set:
    """The pairs whose MAC and STATUS flags --vu-fast leaves out, or None
    where it runs the program exact. A VU1 program only (the EE reads VU0's
    flags, CFC2; never VU1's), and one that never reads or sets STATUS
    (FSEQ ... FSOR): then the flags matter only to the MAC readers (FMEQ,
    FMAND, FMOR), and an FMAC's only where one can see them (mac_live).
    What fast mode gives up: a later run reading the flags a run left
    before an FMAC of its own lands."""
    if unit != 1 or any(reads_status(lower, upper) for lower, upper in pairs.values()):
        return None
    targets = set(entries) | static_targets(pairs, mask)
    facts = PairFacts(pairs, mask)
    return {pc for pc, (lower, upper) in pairs.items()
            if sets_mac(upper) and not mac_live(pairs, mask, pc, targets, facts)}


def reads_flags(lower: int, upper: int) -> bool:
    """FCEQ ... FCGET, FSEQ ... FSOR, FMEQ ... FMOR: the lowers that read or
    set the landed flags."""
    return not (upper & 0x80000000) and 0x10 <= (lower >> 25) <= 0x1C


def quiet_blocks(blocks: list, pairs: dict) -> set:
    """The blocks (by start) that read no landed flags in their first three
    pairs and run them all: three pairs or more, no branch first (a block
    leaves after its first pair with an E bit or a branch pending - which
    Flags.leaving's caller rules out - after a branch's delay slot, or
    after its last)."""
    return {b[0] for b in blocks if len(b) >= 3 and not is_branch(*pairs[b[0]])[0] and
            not any(reads_flags(*pairs[pc]) for pc in b[:3])}


def untimed_writes(bodies: list) -> None:
    """Leaves out the landing times (vf_ready_) of the block's writes that
    nothing can read them for - in place, in the pairs' C++ (`bodies`, a list
    of each pair's lines). A write's time is read only by a stall check on
    its field: one of the block's within the three pairs after it (from the
    fourth on, needless_wait leaves them out), with no write of the field
    between; or one after the block leaves - and a pair takes a cycle at
    least, so a write three or more pairs before the pair it leaves after
    has landed by the next pair's issue whatever stalls came between (a
    result lands four cycles after its pair's issue, the next pair issues a
    cycle after that one at the earliest). A time left out stays as the
    write before left it, earlier: a later check passes as it would have.
    (The pipeline's times then differ from the interpreter's only where both
    are past - Vu::SameState compares them so.)"""
    exits = [i for i, lines in enumerate(bodies)
             if any("return;" in line or "PS2HOST_VU_NEXT(" in line for line in lines)]
    waits = []          # per pair: the (register, lane) it waits on
    writes = []         # per pair: [(register, lanes, "lower"/"upper", sure)]
    for lines in bodies:
        w = set()
        for line in lines:
            t = line.strip()
            if t.startswith("C::Wait(t, v, "):
                r, lane = t[len("C::Wait(t, v, "):].rstrip(");").split(", ")
                w.add((int(r), int(lane)))
        waits.append(w)
        # A held lower write lands for certain only from a load; the others
        # (MOVE, MR32, MFIR, macro mode's COP2 - HoldVf) only where the value
        # changes, leaving the time before (no later wait's write: `sure`).
        pair_writes = [(r, d, "lower", any(line.strip().startswith("C::Load(v, pd, ") for line in lines))
                       for r, d in lower_writes(lines) if r > 0]
        for line in lines:
            t = line.strip()
            if t.startswith("C::Finish(v, cyc, pd, ") or t.startswith("C::FinishUpper(v, cyc, "):
                args = t.split("(", 1)[1].rstrip(");").split(", ")
                at = 3 if t.startswith("C::Finish(") else 2
                r, lanes = int(args[at]), int(args[at + 1].rstrip("u"))
                if r > 0 and lanes:
                    pair_writes.append((r, lanes, "upper", True))
        writes.append(pair_writes)

    def needed(j: int, r: int, dest: int) -> bool:
        if any(j <= e <= j + 2 for e in exits):
            return True
        for lane in range(4):
            if not dest & (8 >> lane):
                continue
            for i in range(j + 1, min(j + 4, len(bodies))):
                if (r, lane) in waits[i]:
                    return True
                if any(rr == r and dd & (8 >> lane) and sure for rr, dd, _, sure in writes[i]):
                    break
        return False

    for j, lines in enumerate(bodies):
        lower_keep = any(needed(j, r, d) for r, d, k, _ in writes[j] if k == "lower") or not any(
            k == "lower" for _, _, k, _ in writes[j])
        upper_keep = any(needed(j, r, d) for r, d, k, _ in writes[j] if k == "upper") or not any(
            k == "upper" for _, _, k, _ in writes[j])
        if lower_keep and upper_keep:
            continue
        for n, line in enumerate(lines):
            t = line.strip()
            indent = line[:len(line) - len(line.lstrip())]
            if t.startswith("C::Finish(v, cyc, pd, "):
                args = t[len("C::Finish("):].rstrip(");").split(", ")
                if len(args) == 8:
                    args.append("true")
                args += ["true" if lower_keep else "false", "true" if upper_keep else "false"]
                lines[n] = indent + "C::Finish(%s);" % ", ".join(args)
            elif t.startswith("C::FinishUpper(v, cyc, ") and not upper_keep:
                args = t[len("C::FinishUpper("):].rstrip(");").split(", ")
                lines[n] = indent + "C::FinishUpper(%s, false);" % ", ".join(args)


# PS2_VU_ALL_TIMES=1: every write's landing time kept (untimed_writes off), to compare.
TIMES_PASS = os.environ.get("PS2_VU_ALL_TIMES", "") != "1"


def emit_block(name: str, block: list, pairs: dict, mask: int, drops: set = frozenset(),
               chain: dict = None, quiet: set = frozenset(), no_status: bool = False,
               entry: dict = None, entry_clean: frozenset = frozenset(),
               unsafe: frozenset = frozenset()) -> list:
    # The run's state and the cycle as locals, written back where the block
    # leaves (LEAVE): nothing the block does can reach them through memory.
    # A block starts only with as many pairs left to the limit as it has
    # (else the interpreter takes them: Vu::RunCompiled, no progress), so no
    # pair of it checks the limit. A branch or an E bit pending as it starts
    # (a delay slot or the pair after an E bit that begins a block) ends it
    # after its first pair - so from its second pair on it runs whole, which
    # dead_flags relies on; the first pair's flags are never left out.
    # `chain`: the program's blocks of the same kind by where they start, for
    # going straight on into the next (leave_to).
    out = ["PS2HOST_VU_BLOCK void %s(Vu& __restrict v, VuRunState& __restrict s, std::uint64_t limit) {" % name,
           "    if (s.count + %d > limit) return;" % len(block),
           "    u64 count = s.count, cyc = C::Cycle(v);",
           "    bool stopping = s.stopping;",
           "    u32 delayed = s.delayed;"]
    written, clean = {}, {f: True for f in entry_clean}
    dead = dead_flags(block, pairs) - {0}
    fl = Flags()
    head = len(out)
    bodies = [emit_pair(pc, pairs[pc][0], pairs[pc][1], mask, i == len(block) - 1, i, written, clean,
                        i in dead, pairs[block[i - 1]] if i else None, pc in drops, chain, fl, quiet,
                        no_status, entry, unsafe)
              for i, pc in enumerate(block)]
    if TIMES_PASS:
        untimed_writes(bodies)
    for lines in bodies:
        out += lines
    out[head:head] = ["    " + line for line in fl.declared]
    out.append("}")
    # The blocks it goes on into, declared before it (they may be in another file).
    called = sorted(set(line.split("PS2HOST_VU_NEXT(")[1].split(")")[0]
                        for line in out if "PS2HOST_VU_NEXT(" in line))
    return ["PS2HOST_VU_BLOCK void %s(Vu& __restrict v, VuRunState& __restrict s, std::uint64_t limit);" % c for c in called] + out


def runs_of(pairs: dict) -> list:
    """The code as runs of words: [(byte address, [words...]), ...]."""
    runs = []
    for pc in sorted(pairs):
        words = [pairs[pc][0], pairs[pc][1]]
        if runs and runs[-1][0] + 4 * len(runs[-1][1]) == pc:
            runs[-1][1].extend(words)
        else:
            runs.append((pc, words))
    return runs


def code_array(runs: list) -> str:
    values = [len(runs)]
    for base, words in runs:
        values += [base, len(words)] + words
    lines = []
    for i in range(0, len(values), 8):
        lines.append("    " + ", ".join("0x%08Xu" % x for x in values[i:i + 8]) + ",")
    return "\n".join(lines)


HEADER = """/* Generated by engine/tools/vurecomp.py: %s. Do not edit. */

#include <cmath>
#include <cstdint>

#include "ps2host/vu_core.h"

namespace vu_generated {

using ps2host::Vu;
using ps2host::VuCompiled;
using ps2host::VuRunState;
using C = ps2host::VuCore;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
constexpr u32 kNone = ~0u;
"""

FOOTER = """
}  // namespace vu_generated
"""

# Block functions a file: a few thousand lines each, compiled in parallel.
LINES_PER_FILE = 6000


def generate_files(programs: list, what: str, single: str = None) -> dict:
    """programs: [(unit, [entries], pairs)] -> {file name: C++}. `single`
    names one file for everything (the tests' corpus); otherwise the blocks go
    in vu_blocks_N.cpp and the tables in vu_programs.cpp."""
    files = {}
    chunks = [[]]
    tables = []
    registrations = []
    declarations = []
    for number, (unit, entries, pairs) in enumerate(programs):
        mask = 0x3FFF if unit == 1 else 0xFFF
        name = "Vu%dProgram%d" % (unit, number)
        blocks = blocks_of(pairs, entries, mask)
        names = []
        drops = fast_drops(unit, pairs, mask, entries)
        # A fast variant wherever STATUS is never read: its MAC left out where
        # no one sees it (drops), its STATUS everywhere (no_status).
        fast = drops is not None
        quiet = quiet_blocks(blocks, pairs)
        by_start = {b[0]: b for b in blocks}
        known_clean_at = entry_clean(blocks, pairs, entries, mask)
        unsafe = frozenset(start for start, known in known_clean_at.items() if known)
        for block in blocks:
            names.append((block[0], "%sBlock%04X" % (name, block[0])))
        for variant, dropped in [("", frozenset())] + ([("Fast", drops)] if fast else []):
            chain = Chain({b[0]: "%sBlock%04X%s" % (name, b[0], variant) for b in blocks}, by_start, pairs)
            work = [(b[0], None, chain[b[0]]) for b in blocks]
            while work:
                start, entry, block_name = work.pop(0)
                code = emit_block(block_name, by_start[start], pairs, mask, dropped, chain, quiet,
                                  no_status=variant == "Fast", entry=entry, entry_clean=known_clean_at[start],
                                  unsafe=unsafe)
                work += chain.wanted
                chain.wanted = []
                if single is None and len(chunks[-1]) + len(code) > LINES_PER_FILE and chunks[-1]:
                    chunks.append([])
                chunks[-1] += code + [""]
                declarations.append("PS2HOST_VU_BLOCK void %s(Vu& __restrict v, VuRunState& __restrict s, std::uint64_t limit);" % block_name)
        tables.append("const u32 k%sCode[] = {" % name)
        tables.append(code_array(runs_of(pairs)))
        tables.append("};")
        tables.append("const u32 k%sStarts[] = {%s};" % (name, ", ".join("0x%Xu" % pc for pc, _ in names)))
        tables.append("const ps2host::VuBlock k%sBlocks[] = {%s};" % (name, ", ".join(n for _, n in names)))
        if unsafe:
            tables.append("const std::uint8_t k%sKnows[] = {%s};" %
                          (name, ", ".join("1" if pc in unsafe else "0" for pc, _ in names)))
        if fast:
            tables.append("const ps2host::VuBlock k%sFastBlocks[] = {%s};" %
                          (name, ", ".join(n + "Fast" for _, n in names)))
        tables.append("")
        for entry in entries:
            registrations.append("    {%d, 0x%Xu, k%sCode, k%sStarts, k%sBlocks, %du, %s, %s}," %
                                 (unit, entry, name, name, name, len(names),
                                  "k%sFastBlocks" % name if fast else "nullptr",
                                  "k%sKnows" % name if unsafe else "nullptr"))
    main = [HEADER % what]
    if single is not None:
        main += declarations + [""] + chunks[0]
    else:
        main += declarations + [""]
    main += tables
    main.append("const VuCompiled kPrograms[] = {")
    main += registrations or ["    {0, 0, nullptr, nullptr, nullptr, 0},"]
    main.append("};")
    main.append("")
    main.append("struct Registration {")
    main.append("    Registration() {")
    main.append("        for (const VuCompiled& program : kPrograms)")
    main.append("            if (program.block_count) ps2host::RegisterVuProgram(&program);")
    main.append("    }")
    main.append("} registration;")
    main.append(FOOTER)
    main.append("/* The programs, for the tests (the port has them registered). */")
    main.append("const ps2host::VuCompiled* VuGeneratedPrograms(std::size_t* count) {")
    main.append("    *count = sizeof vu_generated::kPrograms / sizeof vu_generated::kPrograms[0];")
    main.append("    return vu_generated::kPrograms;")
    main.append("}")
    files[single or "vu_programs.cpp"] = "\n".join(main) + "\n"
    if single is None:
        for i, chunk in enumerate(chunks):
            if chunk:
                files["vu_blocks_%d.cpp" % i] = "\n".join([HEADER % what] + chunk + [FOOTER])
    return files


# ---------------------------------------------------------------- the manifest

def read_manifest(path: str) -> list:
    """Each line: `vu<unit> <entry> <base>:<word>,<word>...;<base>:...` (hex).
    Programs with the same code become one function, entered at each entry."""
    by_code = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 3 or not parts[0].startswith("vu"):
                continue
            unit, entry = int(parts[0][2:]), int(parts[1], 16)
            pairs = {}
            for run in parts[2].split(";"):
                base, words = run.split(":")
                words = [int(w, 16) for w in words.split(",")]
                for i in range(0, len(words), 2):
                    pairs[int(base, 16) + 4 * i] = (words[i], words[i + 1])
            key = (unit, tuple(sorted(pairs.items())))
            by_code.setdefault(key, (unit, set(), pairs))[1].add(entry)
    return [(unit, sorted(entries), pairs) for unit, entries, pairs in by_code.values()]


# ---------------------------------------------------------------- the test corpus

def random_upper(rng: random.Random) -> int:
    dest = rng.choice([0xF, 0xF, 0xE, 0x8, 0x4, 0x2, 0x1, 0x3, 0xC, rng.randrange(16)])
    ft, fs, fd = rng.randrange(32), rng.randrange(32), rng.randrange(32)
    if rng.random() < 0.6:
        funct = rng.randrange(0x30)
    else:
        # The special forms (index in fd and funct's low bits); 0x2B is no
        # instruction (macro mode's COP2 stops at it), and no game has it.
        index = rng.choice([i for i in range(0x30) if i != 0x2B])
        funct = 0x3C | (index & 3)
        fd = index >> 2
    word = funct | (fd << 6) | (fs << 11) | (ft << 16) | (dest << 21)
    if rng.random() < 0.05:
        word |= 0x80000000                       # I: the lower is a float
    return word


LOWER_OPS = [0x00, 0x01, 0x04, 0x05, 0x08, 0x09, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
             0x18, 0x1A, 0x1B, 0x1C, 0x20, 0x21, 0x24, 0x25, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F]
SPECIAL_INDEXES = [0x30, 0x31, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
                   0x40, 0x41, 0x42, 0x43, 0x64, 0x68, 0x69, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76,
                   0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E]


def random_lower(rng: random.Random, pc: int, size: int, last: int, flag_reads: bool = True) -> int:
    dest = rng.choice([0xF, 0xF, 0x8, 0x4, 0x2, 0x1, 0xE, rng.randrange(16)])
    it, is_ = rng.randrange(1, 16) if rng.random() < 0.9 else rng.randrange(32), rng.randrange(16)
    roll = rng.random()
    if roll < 0.35:
        index = rng.choice(SPECIAL_INDEXES)
        fd = index >> 2
        return (0x40 << 25) | (dest << 21) | (it << 16) | (is_ << 11) | (fd << 6) | 0x3C | (index & 3)
    if roll < 0.45:                                              # IADD ISUB IADDI IAND IOR (and 0x33)
        return (0x40 << 25) | (it << 16) | (is_ << 11) | (rng.randrange(32) << 6) | rng.randrange(0x30, 0x36)
    if flag_reads is True:
        ops = LOWER_OPS
    elif flag_reads == "mac":
        ops = [o for o in LOWER_OPS if not 0x14 <= o <= 0x17]
    else:
        ops = [o for o in LOWER_OPS if not 0x14 <= o <= 0x1B]
    op = rng.choice(ops)
    imm = rng.randrange(0x800)
    if op in (0x20, 0x21, 0x28, 0x29, 0x2C, 0x2D, 0x2E, 0x2F):
        # Mostly forward, sometimes back: loops end at the run's limit if not before.
        to = rng.randrange(0, last + 1) if rng.random() < 0.2 else rng.randrange(pc // 8, last + 1)
        imm = (to - pc // 8 - 1) & 0x7FF
    return (op << 25) | (dest << 21) | (it << 16) | (is_ << 11) | imm


def random_program(rng: random.Random, unit: int, flag_reads: bool = True) -> dict:
    size = 0x4000 if unit == 1 else 0x1000
    count = rng.randrange(8, 48)
    base = rng.randrange(0, size // 8 - count - 2) * 8
    last = base // 8 + count + 1
    pairs = {}
    for i in range(count):
        pc = base + 8 * i
        upper = random_upper(rng)
        lower = struct.unpack("<I", struct.pack("<f", rng.uniform(-4, 4)))[0] if upper & 0x80000000 \
            else random_lower(rng, pc, size, last, flag_reads)
        pairs[pc] = (lower, upper)
    end = base + 8 * count
    pairs[end] = (0x8000033C, 0x400002FF)                           # NOP with E
    pairs[end + 8] = (0x8000033C, 0x000002FF)                       # the pair after it
    return pairs


def corpus(count: int, seed: int, name: str) -> dict:
    rng = random.Random(seed)
    programs = []
    for _ in range(count):
        unit = 1 if rng.random() < 0.8 else 0
        pairs = random_program(rng, unit)
        programs.append((unit, [min(pairs)], pairs))
    # And VU1 programs for --vu-fast's blocks (VuFastIsTheExactButFlags):
    # none reading STATUS, half reading MAC (FMEQ, FMAND, FMOR) and half
    # neither; a generator of their own, so the ones above stay as they were.
    rng = random.Random(seed + 1000)
    for k in range(count // 3):
        pairs = random_program(rng, 1, flag_reads="mac" if k % 2 else False)
        programs.append((1, [min(pairs)], pairs))
    return generate_files(programs, "%d random programs (seed %d) for VuCompiledIsTheInterpreter"
                          % (count, seed), single=name)


def write_if_changed(path: str, text: str) -> None:
    old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
    if text != old:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="the programs a run recorded (--vu-record)")
    parser.add_argument("--corpus", action="store_true", help="random programs for the host tests")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", help="the corpus: one file")
    parser.add_argument("--out-dir", help="the programs: vu_programs.cpp and vu_blocks_N.cpp here")
    args = parser.parse_args(argv)
    if args.corpus:
        if not args.out:
            parser.error("--corpus needs --out")
        for name, text in corpus(args.count, args.seed, os.path.basename(args.out)).items():
            write_if_changed(os.path.join(os.path.dirname(args.out), name), text)
        return 0
    if not args.out_dir:
        parser.error("--manifest needs --out-dir")
    programs = read_manifest(args.manifest) if args.manifest and os.path.exists(args.manifest) else []
    files = generate_files(programs, "%d programs from %s" % (len(programs), args.manifest or "nothing"))
    for name, text in files.items():
        write_if_changed(os.path.join(args.out_dir, name), text)
    # Blocks files a smaller set of programs no longer needs.
    for name in os.listdir(args.out_dir):
        if name.startswith("vu_blocks_") and name.endswith(".cpp") and name not in files:
            os.remove(os.path.join(args.out_dir, name))
    print("vurecomp: %d programs, %d pairs, %d files" % (len(programs), sum(len(p[2]) for p in programs),
                                                         len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
