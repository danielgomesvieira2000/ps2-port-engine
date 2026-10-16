"""Deriving a switch: a jump through a table of case addresses (`jr $r`).

CodeWarrior's form, as Resident Evil Outbreak's game.bin has it a hundred and
seventy times: bound the index, load the case's address from a table, jump.
Each case is a path whose condition is that the table's word is that case.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine  # noqa: E402
from ps2decomp.paths import Execution, derive_paths  # noqa: E402

BASE = 0x00100000
ZERO, AT, V0, V1, A0, RA = 0, 1, 2, 3, 4, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def r_type(rs, rt, rd, funct, sa=0):
    return (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct


def sltiu(rt, rs, imm): return i_type(0x0B, rs, rt, imm)
def beq(rs, rt, words): return i_type(0x04, rs, rt, words)
def lui(rt, imm): return i_type(0x0F, 0, rt, imm)
def addiu(rt, rs, imm): return i_type(0x09, rs, rt, imm)
def lw(rt, offset, rs): return i_type(0x23, rs, rt, offset)
def sll(rd, rt, sa): return r_type(0, rt, rd, 0x00, sa)
def addu(rd, rs, rt): return r_type(rs, rt, rd, 0x21)
def jr(rs): return r_type(rs, 0, 0, 0x08)


NOP = 0
JR_RA = jr(RA)


def switch_function():
    """f(i): switch (i) { case 0: return 10; case 1: return 20; case 2: return 10; default: return -1; }
    (cases 0 and 2 share a body, as compilers lay them out)."""
    words = [
        sltiu(AT, A0, 3),               # 0x00
        beq(AT, ZERO, 0),               # 0x04: to default (patched below)
        NOP,                            # 0x08
        lui(V0, 0),                     # 0x0C: the table's high half (patched)
        addiu(V1, V0, 0),               # 0x10: ...and low half (patched)
        sll(V0, A0, 2),                 # 0x14
        addu(V0, V0, V1),               # 0x18
        lw(V0, 0, V0),                  # 0x1C
        jr(V0),                         # 0x20
        NOP,                            # 0x24
        JR_RA,                          # 0x28: case 0 and 2
        addiu(V0, ZERO, 10),            # 0x2C
        JR_RA,                          # 0x30: case 1
        addiu(V0, ZERO, 20),            # 0x34
        JR_RA,                          # 0x38: default
        addiu(V0, ZERO, -1),            # 0x3C
    ]
    table = BASE + 4 * len(words)
    words[1] = beq(AT, ZERO, (0x38 - 0x08) // 4)
    words[3] = lui(V0, (table + 0x8000) >> 16)
    words[4] = addiu(V1, V0, table & 0xFFFF)
    words += [BASE + 0x28, BASE + 0x30, BASE + 0x28]
    return words, BASE + 0x40


class TestSwitch(unittest.TestCase):
    def setUp(self):
        words, self.end = switch_function()
        self.image = struct.pack("<%dI" % len(words), *words)

    def test_a_switch_derives_and_agrees_with_the_original(self):
        derivation = derive_paths(self.image, BASE, BASE, self.end, known={})
        machine = Machine(BASE, self.image, 0x02000000 - BASE)
        for index in range(5):
            expected = machine.call(BASE, (index,))
            machine.memory.begin_journal()
            got = derivation.run((index,) + (0,) * 15, Execution(machine))
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual(got, expected, "case %d" % index)
        self.assertEqual([machine.call(BASE, (i,)) for i in range(5)], [10, 20, 10, -1, -1])

    def test_a_jump_through_no_table_is_still_refused(self):
        from ps2decomp.symexec import Unsupported
        words = [jr(A0), NOP]
        image = struct.pack("<2I", *words)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 8, known={})


if __name__ == "__main__":
    unittest.main()
