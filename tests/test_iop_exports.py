"""The IOP exports' names (data/iop_exports.csv): numbered as ps2sdk's export
tables number them, by position - a line that opens with a comment too."""
import csv
import os
import unittest

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "iop_exports.csv")


def names() -> dict:
    found = {}
    with open(CSV, encoding="utf-8") as handle:
        for row in csv.reader(line for line in handle if not line.startswith("#")):
            if len(row) >= 3 and row[1].isdigit():
                found.setdefault((row[0], int(row[1])), set()).add(row[2])
    return found


class TestPositions(unittest.TestCase):
    def test_a_commented_line_keeps_its_place(self):
        # ioman's table: `/*05*/ DECLARE_EXPORT(io_close)`. Read without it,
        # read was 5 and every export after named one place early.
        table = names()
        self.assertEqual(table[("ioman", 5)], {"io_close", "iomanX_close"})
        self.assertEqual(table[("ioman", 6)], {"io_read", "iomanX_read"})
        self.assertEqual(table[("ioman", 16)], {"io_getstat", "ioman_getstat"})
        self.assertEqual(table[("iomanx", 6)], {"iomanX_read"})
        self.assertEqual(table[("stdio", 7)], {"puts"})

    def test_iomans_newer_table(self):
        # Tekken 5's NU_File_Manager imports ioman 1.4's 26: iomanX's table
        # for the old library's name.
        self.assertEqual(names()[("ioman", 26)], {"iomanX_chdir"})


if __name__ == "__main__":
    unittest.main()
