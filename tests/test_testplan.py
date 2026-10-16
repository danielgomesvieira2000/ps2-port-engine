"""Test cases the original cannot finish (testplan.run_cases).

Such a case - a loop that runs past the budget on this input, a trap - is left
out rather than failing the function; every case that ran is still compared,
and a disagreement on any of them still fails it.
"""

import os
import struct
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from ps2decomp.emu import Machine  # noqa: E402
from ps2decomp import testplan  # noqa: E402
import derive  # noqa: E402

BASE = 0x00100000
ZERO, V0, A0, T0, RA = 0, 2, 4, 8, 31


def i_type(op, rs, rt, imm):
    return (op << 26) | (rs << 21) | (rt << 16) | (imm & 0xFFFF)


def f_words():
    """f(n): while (n & 1) {} return n + 1  - for ever on an odd n."""
    return [
        i_type(0x0C, A0, T0, 1),         # andi $t0, $a0, 1
        i_type(0x05, T0, ZERO, -1),      # bne $t0, $zero, back to itself
        0,
        (RA << 21) | 0x08,               # jr $ra
        i_type(0x09, A0, V0, 1),         # addiu $v0, $a0, 1
    ]


class TestRunCases(unittest.TestCase):
    def setUp(self):
        words = f_words()
        self.image = struct.pack("<%dI" % len(words), *words)
        self.machine = Machine(BASE, self.image, 0x02000000 - BASE)
        self.func = {"addr": BASE, "end": BASE + 4 * len(words), "size": 4 * len(words)}

    def run_cases(self, arguments, evaluate, record=None, recorded=None):
        arg_sets = [(n,) + (0,) * 15 for n in arguments]
        return testplan.run_cases(self.func, self.machine, arg_sets, "zero", evaluate,
                                  record=record, recorded=recorded)

    def test_cases_the_original_cannot_finish_are_left_out(self):
        record = []
        outcome, tested = self.run_cases([0, 1, 2, 3, 4, 6, 8, 10],
                                         lambda args: (args[0] + 1, []), record)
        self.assertEqual((outcome, tested), ("ok", 6))
        self.assertEqual([r is None for r in record], [False, True, False, True] + [False] * 4)
        # Replayed from the record (the compiled check), the same cases are left out.
        self.assertEqual(self.run_cases([0, 1, 2, 3, 4, 6, 8, 10], lambda args: (args[0] + 1, []),
                                        recorded=record), ("ok", 6))

    def test_a_disagreement_on_a_case_that_ran_still_fails(self):
        outcome, _tested = self.run_cases([1, 3, 0, 2, 4, 6, 8, 10], lambda args: (args[0] + 2, []))
        self.assertEqual(outcome, "mismatch")


class TestCountPlan(unittest.TestCase):
    """memset(p, n): every argument a pointer, n is 25 million and the original
    never finishes; with n a small count it does."""

    def setUp(self):
        A1, ZERO_ = 5, 0
        words = [
            i_type(0x04, A1, ZERO_, 5),      # 0: beq $a1, $zero, to the jr
            0,
            i_type(0x2B, A0, ZERO_, 0),      # 8: sw $zero, 0($a0)
            i_type(0x09, A1, A1, -1),        # 12: addiu $a1, $a1, -1
            i_type(0x04, ZERO_, ZERO_, -5),  # 16: b back to 0
            i_type(0x09, A0, A0, 4),         # 20: addiu $a0, $a0, 4
            (RA << 21) | 0x08,               # 24: jr $ra
            0,
        ]
        self.image = struct.pack("<%dI" % len(words), *words)
        self.machine = Machine(BASE, self.image, 0x02000000 - BASE)
        self.func = {"addr": BASE, "end": BASE + 4 * len(words), "size": 4 * len(words)}

    def test_a_count_argument_makes_it_testable(self):
        plan = testplan.test_plan(self.func, self.machine)
        self.assertIsNotNone(plan)
        arg_sets, _profile = plan
        self.assertTrue(all(args[1] <= testplan.COUNT_LIMIT for args in arg_sets))
        self.assertTrue(all(args[0] > testplan.COUNT_LIMIT for args in arg_sets))

    def test_the_plan_is_the_same_every_time(self):
        self.assertEqual(testplan.test_plan(self.func, self.machine)[0],
                         testplan.test_plan(self.func, self.machine)[0])


class TestFullerProfile(unittest.TestCase):
    """while (p) p = *p: on random links a cycle, for ever; on zeroed memory
    one step."""

    def setUp(self):
        words = [
            i_type(0x23, A0, A0, 0),         # 0: lw $a0, 0($a0)
            i_type(0x05, A0, ZERO, -2),      # 4: bne $a0, $zero, back to 0
            0,
            (RA << 21) | 0x08,               # jr $ra
            0,
        ]
        self.machine = Machine(BASE, struct.pack("<%dI" % len(words), *words),
                               0x02000000 - BASE)
        self.func = {"addr": BASE, "end": BASE + 4 * len(words), "size": 4 * len(words)}
        from ps2decomp import layout, testmem
        import random
        rng = random.Random(1)
        self.arg_sets = [(layout.CURRENT.scratch + 64 * i,) + testmem.arguments(rng)[1:]
                         for i in range(testplan.ARG_SETS)]

    def test_a_profile_completing_every_case_is_found(self):
        from ps2decomp import layout, testmem
        profile = derive.fuller_profile(self.func, self.machine, self.arg_sets, 0)
        self.assertIsNotNone(profile)
        for case, args in enumerate(self.arg_sets):
            self.machine.memory.end_journal()
            self.machine.memory.write(layout.CURRENT.scratch,
                                      testmem.build(profile, BASE + case))
            self.machine.call(BASE, args, max_instructions=testplan.VERIFY_BUDGET)

    def test_none_when_no_profile_does_better(self):
        self.assertIsNone(derive.fuller_profile(self.func, self.machine, self.arg_sets,
                                                  testplan.ARG_SETS))


if __name__ == "__main__":
    unittest.main()
