"""Unaligned accesses: a pair of halves is one load or store.

`ldl $t, A+7($b); ldr $t, A($b)` reads the eight bytes at A whatever its
alignment, and sdl/sdr and swl/swr write them; CodeWarrior copies a struct
that may not be aligned this way. Each pair is checked against the
interpreter at every alignment.
"""

import os
import random
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from test_quad import SCRATCH, check  # noqa: E402

from ps2decomp.paths import derive_paths  # noqa: E402
from ps2decomp.symexec import Unsupported  # noqa: E402

BASE = 0x00100000
JR_RA = 0x03E00008
A0, A1, V0, V1 = 4, 5, 2, 3


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


LDL, LDR, SDL, SDR, SWL, SWR, LWL, LWR = 0x1A, 0x1B, 0x2C, 0x2D, 0x2A, 0x2E, 0x22, 0x26


def unaligned(rng):
    return (SCRATCH + rng.randrange(64), SCRATCH + 128 + rng.randrange(64))


SD = 0x3F


def stored(*loads):
    """The loads, then their whole register stored at 16($a1): all 64 bits compared."""
    return list(loads) + [JR_RA, i_type(SD, A1, V0, 0x10)]


class TestUnalignedPairs(unittest.TestCase):
    def test_ldl_ldr_is_an_eight_byte_load(self):
        check(self, stored(i_type(LDL, A0, V0, 0x0F), i_type(LDR, A0, V0, 0x08)), unaligned)

    def test_the_halves_may_come_in_either_order(self):
        check(self, stored(i_type(LDR, A0, V0, 0x00), i_type(LDL, A0, V0, 0x07)), unaligned)

    def test_lwl_lwr_is_a_sign_extended_word(self):
        check(self, stored(i_type(LWL, A0, V0, 0x03), i_type(LWR, A0, V0, 0x00)), unaligned)
        check(self, stored(i_type(LWR, A0, V0, 0x04), i_type(LWL, A0, V0, 0x07)), unaligned)

    def test_the_second_half_of_a_store_in_a_delay_slot(self):
        words = [i_type(LDL, A0, V1, 0x07), i_type(LDR, A0, V1, 0x00),
                 i_type(SDL, A1, V1, 0x07), JR_RA, i_type(SDR, A1, V1, 0x00)]
        check(self, words, unaligned)

    def test_a_branch_to_the_second_half_keeps_the_halves_apart(self):
        # if (a1 == 0) skip the first half: then only ldr runs, which is refused.
        words = [i_type(0x04, A1, 0, 2), 0,                    # beq $a1, $zero, to the ldr
                 i_type(LDL, A0, V0, 0x07), i_type(LDR, A0, V0, 0x00), JR_RA, 0]
        image = struct.pack("<%dI" % len(words), *words)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={})

    def test_a_copy_through_sdl_sdr(self):
        words = [
            i_type(LDL, A0, V1, 0x07), i_type(LDR, A0, V1, 0x00),
            i_type(SDL, A1, V1, 0x07), i_type(SDR, A1, V1, 0x00),
            i_type(SWL, A1, V1, 0x0B), i_type(SWR, A1, V1, 0x08),
            JR_RA, 0,
        ]
        check(self, words, unaligned)

    def test_halves_that_do_not_pair_are_still_refused(self):
        # Offsets six apart: not one access.
        words = [i_type(LDL, A0, V0, 0x06), i_type(LDR, A0, V0, 0x00), JR_RA, 0]
        image = struct.pack("<%dI" % len(words), *words)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={})

    def test_a_load_over_its_own_base_is_not_a_pair(self):
        words = [i_type(LDL, A0, A0, 0x07), i_type(LDR, A0, A0, 0x00), JR_RA, 0]
        image = struct.pack("<%dI" % len(words), *words)
        with self.assertRaises(Unsupported):
            derive_paths(image, BASE, BASE, BASE + 4 * len(words), known={})


if __name__ == "__main__":
    unittest.main()
