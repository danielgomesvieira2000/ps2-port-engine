"""How many paths a derivation may enumerate (paths.MAX_PATHS).

Six independent tests of an argument's bits, each storing to its own field: 64
paths. Resident Evil Outbreak's game.bin has hundreds of functions like it -
flag handlers, state updates - that the old cap of 24 paths refused.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine  # noqa: E402
from ps2decomp.paths import Execution  # noqa: E402

import derive  # noqa: E402

BASE = 0x00100000
ZERO, V0, A0, A1, T0, T1, RA = 0, 2, 4, 5, 8, 9, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def andi(rt, rs, imm): return i_type(0x0C, rs, rt, imm)
def beq(rs, rt, words): return i_type(0x04, rs, rt, words)
def addiu(rt, rs, imm): return i_type(0x09, rs, rt, imm)
def sw(rt, offset, rs): return i_type(0x2B, rs, rt, offset)
def lw(rt, offset, rs): return i_type(0x23, rs, rt, offset)


JR_RA = (RA << 21) | 0x08


def six_flags():
    """f(flags, p): for each bit i of flags set, p[i] += i + 1; return p[0]."""
    words = []
    for bit in range(6):
        words += [andi(T0, A0, 1 << bit),
                  beq(T0, ZERO, 4),          # past the update (the slot runs either way)
                  0,
                  lw(T1, 4 * bit, A1),
                  addiu(T1, T1, bit + 1),
                  sw(T1, 4 * bit, A1)]
    words += [lw(V0, 0, A1), JR_RA, 0]
    return words


class TestPathCaps(unittest.TestCase):
    def test_sixty_four_paths_derive_and_agree(self):
        words = six_flags()
        image = struct.pack("<%dI" % len(words), *words)
        func = {"addr": BASE, "end": BASE + 4 * len(words), "size": 4 * len(words)}
        derivation = derive.derive_function(image, BASE, func, {}, None)
        machine = Machine(BASE, image, 0x02000000 - BASE)
        pointer = 0x01800000
        for flags in (0, 1, 0x15, 0x2A, 0x3F):
            machine.memory.write(pointer, bytes(range(24)))
            expected = machine.call(BASE, (flags, pointer))
            after = machine.memory.read(pointer, 24)
            machine.memory.write(pointer, bytes(range(24)))
            machine.memory.begin_journal()
            got = derivation.run((flags, pointer) + (0,) * 14, Execution(machine))
            derived_after = machine.memory.read(pointer, 24)
            machine.memory.rollback()
            machine.memory.end_journal()
            self.assertEqual(got, expected, "flags 0x%X" % flags)
            self.assertEqual(derived_after, after, "flags 0x%X" % flags)


if __name__ == "__main__":
    unittest.main()
