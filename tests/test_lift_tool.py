"""tools/lift.py where no C++ compiler is installed: what lifted is still
emitted, each function marked NOT VERIFIED, and no verdict is remembered."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import lift as lift_tool  # tools/lift.py, not ps2decomp.lift
from ps2decomp import cppcheck, equivalence, liftcheck


class Verdicts:
    """A verdict store holding only what it is given, and noting what it is told."""

    def __init__(self, known=None):
        self.known, self.told, self.saved = dict(known or {}), [], False

    def get(self, text, abi):
        return self.known.get(text)

    def put(self, text, abi, outcome):
        self.told.append(outcome)

    def save(self):
        self.saved = True


def no_compiler(*_args, **_kwargs):
    raise cppcheck.CompilerUnavailable("none on this machine")


class TestWithoutACompiler(unittest.TestCase):
    def test_every_text_is_emitted_not_verified(self):
        texts = {0x00100000: "void F_00100000(R5900Context* c) {}\n",
                 0x00100010: "void F_00100010(R5900Context* c) {}\n"}
        cache = Verdicts({texts[0x00100010]: "verified"})
        printed = io.StringIO()
        with tempfile.TemporaryDirectory() as work, contextlib.redirect_stdout(printed), \
                mock.patch.object(liftcheck, "build", no_compiler):
            outcomes = lift_tool.verify_texts(texts, work, None, None, 1, cache)
        self.assertEqual(outcomes, {
            0x00100000: "not verified: no C++ compiler (none on this machine)",
            0x00100010: "verified",                     # remembered from an earlier run
        })
        self.assertIn("verification SKIPPED", printed.getvalue())
        self.assertEqual((cache.told, cache.saved), ([], False))

    def test_the_equivalence_check_is_skipped(self):
        cache = Verdicts()
        with tempfile.TemporaryDirectory() as work, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(equivalence, "check", no_compiler):
            passed = lift_tool.equivalent_to_plain([0x00100000], {0x00100000: "structured"},
                                                   {0x00100000: "plain"}, False, cache, work)
        self.assertEqual(passed, [0x00100000])
        self.assertEqual((cache.told, cache.saved), ([], False))


if __name__ == "__main__":
    unittest.main()
