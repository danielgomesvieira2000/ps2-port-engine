"""Runs of COP2 macro instructions: the VU's rounding set once for each."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ps2decomp import lift  # noqa: E402

ADD = "    lifted::Cop2(c, 0x4A000028u);  /* vadd */"
MUL = "    lifted::Cop2(c, 0x4A00002Au);  /* vmul */"
MOVE = "    lifted::Cop2Move(c, 0x48A72000u, 7);  /* qmtc2 */"


class TestMacroRuns(unittest.TestCase):
    def test_a_run_in_one_scope(self):
        out = lift.macro_runs([MOVE, ADD, MOVE, MUL, MOVE, "    c.r[2] = 0;"])
        self.assertEqual(out, [MOVE, "    { lifted::MacroRun run;", ADD.replace("Cop2(", "Cop2Run("), MOVE,
                               MUL.replace("Cop2(", "Cop2Run("), "    }", MOVE, "    c.r[2] = 0;"])

    def test_one_alone_as_it_was(self):
        lines = [ADD, MOVE, "    c.r[2] = 0;", MUL]
        self.assertEqual(lift.macro_runs(lines), lines)

    def test_a_label_ends_a_run(self):
        lines = [ADD, "L_00100000:", MUL]
        self.assertEqual(lift.macro_runs(lines), lines)

    def test_another_indentation_ends_a_run(self):
        lines = [ADD, "  " + MUL]
        self.assertEqual(lift.macro_runs(lines), lines)

    def test_twice_the_same(self):
        once = lift.macro_runs([ADD, MUL])
        self.assertEqual(lift.macro_runs(once), once)


if __name__ == "__main__":
    unittest.main()
