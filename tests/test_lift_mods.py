"""The mods' lifter: a .pnach's words, and code returning past its call."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lift_mods  # noqa: E402

PNACH = """gametitle=A game
[Mod A]
patch=1,EE,2028CDD0,extended,3C023F80 // a word
patch=1,EE,101A3BA0,extended,000082F0
[Mod B]
patch=0,EE,003954E8,extended,00000001
patch=1,EE,0028CDE8,word,AFA2005C
"""


class TestPnach(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".pnach")
        with os.fdopen(handle, "w") as f:
            f.write(PNACH)

    def tearDown(self):
        os.remove(self.path)

    def test_every_write_with_its_size(self):
        self.assertEqual(lift_mods.words_of_pnach(self.path),
                         [(0x0028CDD0, 0x3C023F80, 4), (0x001A3BA0, 0x82F0, 2),
                          (0x003954E8, 0x01, 1), (0x0028CDE8, 0xAFA2005C, 4)])

    def test_one_section(self):
        self.assertEqual(lift_mods.words_of_pnach(self.path, "Mod B"),
                         [(0x003954E8, 0x01, 1), (0x0028CDE8, 0xAFA2005C, 4)])


class TestReturnsFurther(unittest.TestCase):
    def test_addiu_ra_found(self):
        code = [(0x1000, [0x27FF0034, 0x03E00008, 0]), (0x2000, [0x03E00008, 0]),
                (0x3000, [0x27FFF680, 0x27FF00B8])]
        self.assertEqual(lift_mods.returns_further(code), {0x1000: [0x34], 0x3000: [-0x980, 0xB8]})

    def test_call_carried_on_where_it_returned(self):
        text = "    c.r[31] = 0x00100008ull; lifted::Call(c, 0x00001000u);\n    goto L_00100008;\n"
        out = lift_mods.returning_further(text, {0x1000: [0x10]})
        self.assertIn("case 0x00100018u: goto L_00100018;", out)
        self.assertIn("default: goto L_00100008;", out)


if __name__ == "__main__":
    unittest.main()
