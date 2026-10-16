"""Overlays: the executable an overlay is analysed and verified in (ps2decomp/overlay.py)."""

import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ps2decomp import game, overlay  # noqa: E402
from ps2decomp.elf import ElfFile  # noqa: E402


def fake_base(data=b"\x01\x02\x03\x04" * 4, addr=0x100000, memsz=0x40, entry=0x100008):
    section = types.SimpleNamespace(addr=addr)
    return types.SimpleNamespace(
        exec_sections=[section], section_data=lambda s: data, e_entry=entry, e_flags=0x20924001,
        segments=[types.SimpleNamespace(vaddr=addr, memsz=memsz)])


class Combine(unittest.TestCase):
    def test_lays_images_over_the_executable(self):
        elf = ElfFile(overlay.combine(fake_base(), [(b"\xAA" * 8, 0x100100, 0x20),
                                                    (b"\xBB" * 4, 0x100104, 0)]))
        section = elf.exec_sections[0]
        data = elf.section_data(section)
        self.assertEqual(section.addr, 0x100000)
        self.assertEqual(elf.e_entry, 0x100008)
        self.assertEqual(data[:16], b"\x01\x02\x03\x04" * 4)
        self.assertEqual(data[16:0x100], bytes(0x100 - 16))       # the gap is zero
        self.assertEqual(data[0x100:0x108], b"\xAA" * 4 + b"\xBB" * 4)  # later ones on top
        self.assertEqual(len(data), 0x108)
        # One segment, covering the overlay's BSS (0x100100 + 0x20).
        self.assertEqual([(p.vaddr, p.filesz, p.memsz) for p in elf.segments],
                         [(0x100000, 0x108, 0x120)])
        self.assertEqual(elf.read_u32(0x100100), 0xAAAAAAAA)

    def test_refuses_an_overlay_below_the_executable(self):
        with self.assertRaises(ValueError):
            overlay.combine(fake_base(), [(b"\0" * 4, 0xF0000, 0)])

    def test_in_code(self):
        self.assertTrue(overlay.in_code([], 0x1234))
        self.assertTrue(overlay.in_code([(0x100, 0x200)], 0x100))
        self.assertFalse(overlay.in_code([(0x100, 0x200)], 0x200))


class OverlayGame(unittest.TestCase):
    def test_an_overlay_is_a_game_over_its_base(self):
        with tempfile.TemporaryDirectory() as root:
            base_dir = os.path.join(root, "base")
            os.makedirs(os.path.join(base_dir, "extracted"))
            # The base executable: combine's own output is a valid one.
            with open(os.path.join(base_dir, "extracted", "GAME.ELF"), "wb") as handle:
                handle.write(overlay.combine(fake_base(), []))
            with open(os.path.join(base_dir, "game.json"), "w") as handle:
                json.dump({"name": "Base", "executable": "GAME.ELF",
                           "layout": {"gp": "0x00100800"}}, handle)
            with open(os.path.join(base_dir, "extracted", "ov.bin"), "wb") as handle:
                handle.write(b"\xCC" * 16)
            ov_dir = os.path.join(base_dir, "overlays", "ov")
            os.makedirs(ov_dir)
            with open(os.path.join(ov_dir, "game.json"), "w") as handle:
                json.dump({"name": "Base: ov", "overlay": {
                    "of": "../..", "image": "../../extracted/ov.bin",
                    "address": "0x00100200", "size": "0x40",
                    "code": [["0x00100200", "0x00100210"]]}}, handle)
            loaded = game.load(ov_dir)
            self.assertEqual(loaded.base.name, "Base")
            self.assertEqual(loaded.code_ranges, [(0x100200, 0x100210)])
            self.assertEqual(loaded.layout["gp"], "0x00100800")    # the base's layout
            self.assertTrue(loaded.elf_path.startswith(os.path.normpath(os.path.join(ov_dir, "build"))))
            elf = loaded.load_elf()
            self.assertEqual(elf.read_u32(0x100200), 0xCCCCCCCC)
            self.assertEqual(elf.read_u32(0x100000), 0x04030201)


if __name__ == "__main__":
    unittest.main()
