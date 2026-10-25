#!/usr/bin/env python3
"""Lift a game's functions as its mods (PCSX2 patches) change them, beside the originals.

    python engine/tools/lift_mods.py --game games/<name> --spec games/<name>/port/tools/mods_spec.py \\
        --out build/port/cmake/mods_lifted.cpp

A mod here is a PCSX2 patch (a .pnach's `patch=` lines): words written over
the game's code, and code of its own written to free memory. A port runs the
game's code translated, not the code in memory, so a word written there would
do nothing: each function a mod changes is lifted again from the executable
with the mod's words written in - once for each combination of the mods that
change it - and a mod's own code is lifted from its words. The output's
choosers ask the port's `bool ModOn(int mod)` (the mod's index in the spec's
MODS) which variant runs; its `kModKeys` are the mods' keys (settings, or
"always": always on) and `RegisterModFunctions()` puts the choosers and the
mods' code in place. The output is the game's code: generated at build time
from the player's disc, never committed.

The spec is a Python file of the port's (its mods are its own):

    MODS = [(key, [(address, value, bytes), ...]), ...]   # in the order words go in
    CODE = [(address, [word, ...]), ...]                  # the mods' own functions (optional)
    NAME = "games/<name>/port/tools/mods_spec.py"         # named in the output (optional)

`words_of_pnach(path, section=None)` reads a .pnach's patch lines (one
[section] of a sectioned file, or all) for MODS.

A mod's function that returns somewhere else than after its call - `addiu
$ra, $ra, k` before its `jr $ra`, a jump back into its caller past (or over)
the caller's own code - is found in CODE and its calls are carried on where
it returned to.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ENGINE]

from ps2decomp import game as game_config, lift  # noqa: E402
from lift import _other_sections  # noqa: E402  (engine/tools/lift.py)


def words_of_pnach(path: str, section: str | None = None):
    """A .pnach's EE patch writes as (address, value, bytes): `patch=1,EE,
    TAAAAAAA,extended,VALUE`, T the size (0 a byte, 1 a halfword, 2 a word).
    `section`: only the lines under `[section]` (a sectioned file's mod)."""
    words, current = [], None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.split("//")[0].strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                continue
            if section is not None and current != section:
                continue
            m = re.match(r"patch\s*=\s*\w+\s*,\s*EE\s*,\s*([0-9A-Fa-f]{8})\s*,\s*(\w+)\s*,\s*([0-9A-Fa-f]+)", line)
            if not m:
                continue
            code, kind, value = int(m.group(1), 16), m.group(2).lower(), int(m.group(3), 16)
            size = {0: 1, 1: 2, 2: 4}.get(code >> 28) if kind == "extended" else \
                {"byte": 1, "short": 2, "word": 4}.get(kind)
            if size is None:
                raise SystemExit("%s: a patch line this tool does not write: %s" % (path, line))
            words.append((code & 0x0FFFFFFF, value & ((1 << (8 * size)) - 1), size))
    return words


def returns_further(code):
    """Each of the mods' functions that returns further on in its caller than
    the call (or back), and by how much: its words adding to $ra (addiu $ra,
    $ra, k)."""
    further = {}
    for start, words in code:
        steps = sorted({(w & 0xFFFF) - (0x10000 if w & 0x8000 else 0) for w in words if w >> 16 == 0x27FF})
        if steps:
            further[start] = steps
    return further


def returning_further(text: str, further) -> str:
    """A lifted function's calls to code that returns further on: carried on
    where the call returned to - $ra as the callee left it - not always after
    the call. Translated code goes on after a call by itself; here, at each
    place it can return to, a label of the caller's own (each instruction has
    one)."""
    def call(match):
        ra, target = int(match.group(2), 16), int(match.group(3), 16)
        steps = further.get(target)
        if not steps:
            return match.group(0)
        cases = " ".join("case 0x%08Xu: goto L_%08X;" % ((ra + k) & 0xFFFFFFFF, (ra + k) & 0xFFFFFFFF)
                         for k in steps)
        # After the call, the lifted code goes to its return address (goto
        # L_ra) or simply carries on into it.
        after = "goto L_%08X;" % ra if match.group(5) else "break;"
        return "%s\n%sswitch (u32(c.r[31])) { %s default: %s }%s" % (
            match.group(1), match.group(4), cases, after, "" if match.group(5) else "\n" + match.group(4))
    return re.sub(r"(c\.r\[31\] = 0x([0-9A-F]{8})ull; lifted::Call\(c, 0x([0-9A-F]{8})u\);)\n(\s*)"
                  r"(goto L_[0-9A-F]{8};)?", call, text)


def containing(functions, address):
    for f in functions:
        if f["addr"] <= address < f["end"]:
            return f
        for start, end in f.get("chunks", ()):
            if start <= address < end:
                return f
    raise SystemExit("no function holds 0x%08X" % address)


def patched(image: bytes, base: int, words) -> bytes:
    data = bytearray(image)
    for address, value, size in words:
        data[address - base:address - base + size] = value.to_bytes(size, "little")
    return bytes(data)


def renamed(text: str, address: int, suffix: str) -> str:
    name = "F_%08X" % address
    head = "void %s(lifted::Context& c)" % name
    if head not in text:
        raise SystemExit("unexpected lifted text for %s" % name)
    return text.replace(head, "void %s_%s(lifted::Context& c)" % (name, suffix), 1)


def load_spec(path: str):
    spec = importlib.util.spec_from_file_location("mods_spec", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mods_spec"] = module
    spec.loader.exec_module(module)
    return module


def generate(game_dir: str, mods, code=(), name: str = "its mods' spec") -> tuple[str, int, int]:
    """The generated source, the functions changed and their variants."""
    game = game_config.load(game_dir)
    database = game.load_database()
    elf = game.load_elf()
    section = elf.exec_sections[0]
    image, base = elf.section_data(section), section.addr
    lift.use_sections(_other_sections(elf, section))
    lift.use_function_starts([f["addr"] for f in database["functions"]] + [a for a, _ in code])
    functions = [f for f in database["functions"] if f["end"] > f["addr"]]
    further = returns_further(code)

    keys = [key for key, _ in mods]
    # Each changed function: the mods changing it, in MODS' order.
    touched = {}
    for index, (_key, words) in enumerate(mods):
        for address, _value, _size in words:
            f = containing(functions, address)
            touched.setdefault(f["addr"], (f, []))
            if index not in touched[f["addr"]][1]:
                touched[f["addr"]][1].append(index)

    out = ["/* The game's functions as its mods change them: generated by",
           " * %s from the player's executable at" % name,
           " * build time (the game's code - never committed). Lifted, not verified. */",
           "",
           '#include "ps2rt/lifted.h"',
           '#include "ps2host/host.h"',
           "",
           "using namespace generated;",
           "using namespace lifted;",
           "",
           "bool ModOn(int mod);     // port/mods.cpp",
           ""]
    for address, words in code:
        data = b"".join(struct.pack("<I", w) for w in words)
        out.append(returning_further(lift.lift_function(data, address, address, address + len(data)), further))
    dispatchers = []
    for address in sorted(touched):
        f, which = touched[address]
        out.append("void F_%08X(lifted::Context& c);" % address)
        variants = ["F_%08X" % address]
        for mask in range(1, 1 << len(which)):
            words = [w for bit, m in enumerate(which) if mask >> bit & 1 for w in mods[m][1]
                     if containing(functions, w[0]) is f]
            text = lift.lift_function(patched(image, base, words), base, address, f["end"],
                                      f.get("chunks", ()))
            out.append(returning_further(renamed(text, address, "m%d" % mask), further))
            variants.append("F_%08X_m%d" % (address, mask))
        test = " | ".join("(ModOn(%d) ? %du : 0u)" % (m, 1 << bit) for bit, m in enumerate(which))
        out += ["void Mods_%08X(lifted::Context& c) {" % address,
                "    static ps2host::GuestFunction const kVariants[] = {%s};" % ", ".join(variants),
                "    kVariants[%s](c);" % test,
                "}",
                ""]
        dispatchers.append((address, ", ".join(keys[m] for m in which)))
    out.append("extern const char* const kModKeys[] = {%s};" % ", ".join('"%s"' % k for k in keys))
    out.append("extern const int kModCount = %d;" % len(keys))
    out.append("void RegisterModFunctions() {")
    for address, _words in code:
        out.append('    ps2host::Override(0x%08Xu, F_%08X, "a mod\'s own code");' % (address, address))
    for address, which in dispatchers:
        out.append('    ps2host::Override(0x%08Xu, Mods_%08X, "MODS: %s");' % (address, address, which))
    out.append("}")
    return "\n".join(out) + "\n", len(touched), sum((1 << len(m)) - 1 for _f, m in touched.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="the game's folder (games/<name>)")
    parser.add_argument("--spec", required=True, help="the port's mods (a Python file: MODS, CODE, NAME)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    text, functions, variants = generate(args.game, spec.MODS, getattr(spec, "CODE", ()),
                                         getattr(spec, "NAME", os.path.basename(args.spec)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("lift_mods: %d functions, %d variants" % (functions, variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
