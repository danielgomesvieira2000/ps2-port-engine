"""derive.py's records of the original's test results, kept on disk
(SpilledRecords): what goes in comes back out, as the dictionary they replace
behaves - and nothing is left behind."""

import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import derive  # noqa: E402


class SpilledRecordsTest(unittest.TestCase):
    def test_as_a_dictionary(self):
        records = derive.SpilledRecords()
        self.assertTrue(os.path.isdir(records.folder))
        record = ([(1, 2, 3), (4, 5, 6)], {"profile": [0.5, None]}, [b"\x00\x01", {"pages": [0x100000]}])
        self.assertNotIn(0x00123450, records)
        records[0x00123450] = record
        self.assertIn(0x00123450, records)
        self.assertEqual(records[0x00123450], record)
        self.assertEqual(records.pop(0x00123450), record)
        self.assertNotIn(0x00123450, records)
        self.assertIsNone(records.pop(0x00123450))
        self.assertEqual(records.pop(0x00123450, "absent"), "absent")
        with self.assertRaises(KeyError):
            records[0x00123450]
        self.assertEqual(os.listdir(records.folder), [])

    def test_replaced(self):
        records = derive.SpilledRecords()
        records[0x200] = "first"
        records[0x200] = "second"
        self.assertEqual(records[0x200], "second")


if __name__ == "__main__":
    unittest.main()
