#!/usr/bin/env python3
"""Lift a game's functions as its mods (PCSX2 patches) change them, beside the originals.

    python engine/tools/lift_mods.py --game games/<name> --spec games/<name>/port/tools/mods_spec.py \\
        --out build/port/cmake/mods_lifted.cpp [--shards N]

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

A mod's words may also carry a fourth item, the pnach's place (0 once at
boot, 1 every frame, 2 both, 3 once as it is turned on): such a mod's words
are also memory - its own code's data, tables, state, the game's data it
changes - and go to `kModWrites` for the port to write while the mod is on
(and to put back once it is off). `words_of_pnach(path, section, places=True)`
reads them so.

A mod's own code need not be named: from the functions its words change, and
from the pointers to its code it writes, its code is followed - its calls,
its jumps into it, the addresses it builds - and each function found is
lifted from the memory as the mods leave it (once for each combination of
the mods that write it). A function's jump into a mod's code and back (a
hook) is lifted as part of the function.

`words_of_pnach(path, section=None)` reads a .pnach's patch lines (one
[section] of a sectioned file, or all) for MODS.

A mod's function that returns somewhere else than after its call - `addiu
$ra, $ra, k` before its `jr $ra`, a jump back into its caller past (or over)
the caller's own code - is found in CODE and its calls are carried on where
it returned to.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.normpath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, ENGINE]

from ps2decomp import game as game_config, lift  # noqa: E402
from ps2decomp.emu import is_supported  # noqa: E402
from ps2decomp.r5900 import DISASM  # noqa: E402
from lift import _other_sections  # noqa: E402  (engine/tools/lift.py)

RAM_SIZE = 32 << 20
# A combination of more of a function's mods than this is not lifted: each
# alone, each pair, and all of them are (mods that share a function are
# rarely all on at once; a combination left out runs the nearest smaller).
FULL_COMBINATIONS = 4


def words_of_pnach(path: str, section: str | None = None, places: bool = False):
    """A .pnach's EE patch writes as (address, value, bytes): `patch=1,EE,
    TAAAAAAA,extended,VALUE`, T the size (0 a byte, 1 a halfword, 2 a word).
    `section`: only the lines under `[section]` (a sectioned file's mod).
    `places`: (address, value, bytes, place) - the line's first number."""
    words, current = [], None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.split("//")[0].strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                continue
            if section is not None and current != section:
                continue
            m = re.match(r"patch\s*=\s*(\d+)\s*,\s*EE\s*,\s*([0-9A-Fa-f]{8})\s*,\s*(\w+)\s*,\s*([0-9A-Fa-f]+)", line)
            if not m:
                continue
            place = int(m.group(1))
            code, kind, value = int(m.group(2), 16), m.group(3).lower(), int(m.group(4), 16)
            size = {0: 1, 1: 2, 2: 4}.get(code >> 28) if kind == "extended" else \
                {"byte": 1, "short": 2, "word": 4}.get(kind)
            if size is None:
                raise SystemExit("%s: a patch line this tool does not write: %s" % (path, line))
            word = (code & 0x0FFFFFFF, value & ((1 << (8 * size)) - 1), size)
            words.append(word + (place,) if places else word)
    if section is not None and not words:
        raise SystemExit("%s: no patch lines under [%s]" % (path, section))
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


# ---------------------------------------------------------------- following code

_DECODED = {}


def decode(word: int, address: int):
    key = (word, address)
    ins = _DECODED.get(key)
    if ins is None:
        ins = _DECODED[key] = DISASM.decode(word, address)
    return ins


def _usable(ins) -> bool:
    return ins.is_valid and (is_supported(ins) or ins.mnemonic in ("vcallms", "vcallmsr"))


class Image:
    """The game's memory with some mods' words written in: the words changed
    kept apart from the memory (shared by every image)."""

    def __init__(self, ram, words):
        self.ram = ram
        self.over = {}
        for word in words:
            address, value, size = word[0], word[1], word[2]
            for k in range(size):
                a = address + k
                aligned = a & ~3
                current = self.over[aligned] if aligned in self.over else struct.unpack_from("<I", ram, aligned)[0]
                shift = 8 * (a & 3)
                byte = (value >> (8 * k)) & 0xFF
                self.over[aligned] = (current & ~(0xFF << shift)) | (byte << shift)

    def word(self, address: int) -> int:
        if address in self.over:
            return self.over[address]
        return struct.unpack_from("<I", self.ram, address)[0]

    def data(self, low: int, high: int) -> bytes:
        out = bytearray(self.ram[low:high])
        for a, v in self.over.items():
            if low <= a < high:
                struct.pack_into("<I", out, a - low, v)
        return bytes(out)


def follow(image, entry: int, stops):
    """The instructions control reaches from `entry` in `image` (its branches,
    its jumps - a jump to one of `stops`, another function, a tail call - its
    calls' returns), the calls it makes and the addresses its code builds
    (lui with addiu/ori). None if a way through it meets what is no
    instruction (data, or code this tool cannot lift)."""
    # (seen: the instructions to lift; visited: those followed from - a
    # delay slot is lifted with its branch, and may be jumped to later, as a
    # hook's way back is)
    seen, visited, calls, work = set(), set(), set(), [entry]
    while work:
        a = work.pop()
        while a not in visited:
            if a & 3 or not 0 <= a < RAM_SIZE - 8:
                return None
            ins = decode(image.word(a), a)
            if not _usable(ins):
                return None
            seen.add(a)
            visited.add(a)
            if not ins.has_delay_slot:
                a += 4
                continue
            slot = decode(image.word(a + 4), a + 4)
            if not _usable(slot) or slot.has_delay_slot:
                return None
            seen.add(a + 4)
            m = ins.mnemonic
            if m == "jr":
                if ins.rs == 0:
                    return None             # (data read as code)
                if ins.rs != 31:
                    work += jump_table(image, a)
                break
            if m in ("j", "b"):
                if ins.target != entry and ins.target in stops:
                    break
                a = ins.target
                continue
            if ins.is_call:
                if ins.target >= 0:
                    calls.add(ins.target)
                a += 8
                continue
            work.append(ins.target)
            a += 8
    if not seen:
        return None
    built, upper = set(), {}
    for a in sorted(seen):
        ins = decode(image.word(a), a)
        raw, m = ins.raw, ins.mnemonic
        if m == "lui":
            upper[ins.rt] = (raw & 0xFFFF) << 16
        elif m in ("addiu", "daddiu", "ori") and ins.rs in upper:
            low = raw & 0xFFFF
            if m != "ori" and low & 0x8000:
                low -= 0x10000
            built.add(((upper[ins.rs] | low) if m == "ori" else upper[ins.rs] + low) & 0xFFFFFFFF)
    return seen, calls, built


def jump_table(image, at: int):
    """The cases of a switch's jump at `at` (jr through a register): the
    table its code builds (lui and addiu) and the bound it tests (sltiu), in
    the instructions before it. None found: no cases."""
    bound, upper, base = None, {}, None
    for a in range(at - 4 * 16, at, 4):
        ins = decode(image.word(a), a)
        raw = ins.raw
        if ins.mnemonic == "sltiu":
            bound = raw & 0xFFFF
        elif ins.mnemonic == "lui":
            upper[ins.rt] = (raw & 0xFFFF) << 16
        elif ins.mnemonic in ("addiu", "daddiu") and ins.rs in upper:
            low = raw & 0xFFFF
            base = (upper[ins.rs] + (low - 0x10000 if low & 0x8000 else low)) & 0xFFFFFFFF
    if bound is None or base is None or not 0 < bound <= 1024 or base + 4 * bound > RAM_SIZE:
        return []
    cases = [image.word(base + 4 * k) for k in range(bound)]
    return [c for c in cases if not c & 3 and 0 < c < RAM_SIZE]


def runs(addresses, entry):
    """`addresses` as lift_function's ranges: the body from `entry`, the rest
    as chunks."""
    spans = []
    for a in sorted(addresses):
        if spans and spans[-1][1] == a:
            spans[-1][1] = a + 4
        else:
            spans.append([a, a + 4])
    body, chunks = None, []
    for s, e in spans:
        if s <= entry < e:
            if s < entry:
                chunks.append((s, entry))
            body = (entry, e)
        else:
            chunks.append((s, e))
    return body, chunks


class Mods:
    """The mods over the game's memory: what each writes, the functions it
    changes and its own code's functions."""

    def __init__(self, game_dir: str, mods, code):
        game = game_config.load(game_dir)
        database = game.load_database()
        elf = game.load_elf()
        self.ram = bytearray(RAM_SIZE)
        for section in elf.sections:
            if section.addr and section.size and section.occupies_file and section.is_alloc:
                data = elf.section_data(section)
                self.ram[section.addr:section.addr + len(data)] = data
        for start, words in code:
            self.ram[start:start + 4 * len(words)] = struct.pack("<%dI" % len(words), *words)
        # Jump tables are read in the executable as it is (the lifted image is
        # only the function's own words).
        exec_section = elf.exec_sections[0]
        lift.use_sections([(elf.section_data(exec_section), exec_section.addr)]
                          + list(_other_sections(elf, exec_section)))
        self.functions = {f["addr"]: f for f in database["functions"] if f["end"] > f["addr"]}
        self.code_starts = {a for a, _ in code}
        # Each instruction of the game's functions, to the function holding it.
        self.owner = {}
        for f in self.functions.values():
            for s, e in self.spans(f):
                for a in range(s, e, 4):
                    self.owner.setdefault(a, f["addr"])
        self.mods = mods
        self.keys = [key for key, _ in mods]
        self.written = []                   # each mod's words' addresses (aligned)
        for _key, words in mods:
            addresses = set()
            for word in words:
                for a in range(word[0] & ~3, word[0] + word[2], 4):
                    addresses.add(a)
            self.written.append(addresses)
        self.entries = {}                   # a mod's function's address -> the mods it belongs to
        self.warnings = []
        self.exits = {}                     # where lifted code calls or jumps out to -> from where
        self._images = {}
        self.ram_image = Image(self.ram, ())

    @staticmethod
    def spans(f):
        return [(f["addr"], f["end"])] + [tuple(c) for c in f.get("chunks", ())]

    def image(self, which) -> Image:
        key = tuple(which)
        if key not in self._images:
            self._images[key] = Image(self.ram, [w for m in which for w in self.mods[m][1]])
        return self._images[key]

    def stops(self):
        return set(self.functions) | self.code_starts | set(self.entries)

    def discover(self):
        """Each mod's own code's functions, followed from what it changes."""
        for m, (key, words) in enumerate(self.mods):
            image = self.image([m])
            mine = self.written[m]
            touched = {self.owner[a] for a in mine if a in self.owner}
            work = {start: "changed" for start in touched}
            for w in words:
                if w[2] == 4 and w[1] in mine and not w[1] & 3 and w[1] not in self.functions:
                    trial = follow(image, w[1], self.stops() - {w[1]})
                    if trial and any(image.word(a) == 0x03E00008 for a in trial[0]):
                        work.setdefault(w[1], "a pointer it writes")
            done = set()
            while work:
                start, why = work.popitem()
                if start in done:
                    continue
                done.add(start)
                found = follow(image, start, self.stops() - {start})
                if found is None:
                    if start not in self.functions:
                        self.warnings.append("%s: no code at 0x%08X (%s)" % (key, start, why))
                    continue
                if start not in self.functions:
                    self.entries.setdefault(start, set()).add(m)
                _seen, calls, built = found
                for a in _seen:
                    if a in mine:
                        ins = decode(image.word(a), a)
                        if ins.mnemonic == "jr" and ins.rs != 31 and not jump_table(image, a):
                            self.warnings.append("%s: a jump through a register at 0x%08X (its targets not followed)"
                                                 % (key, a))
                for target in calls:
                    if target in self.functions or target in self.code_starts:
                        continue
                    if target in mine:
                        work.setdefault(target, "called from 0x%08X" % start)
                    elif target not in self.entries:
                        self.warnings.append("%s: a call to 0x%08X, no function there" % (key, target))
                for value in built:
                    if value in mine and not value & 3 and value not in self.functions and value not in done:
                        trial = follow(image, value, self.stops() - {value})
                        # (an address built of data: kept only if it reads as
                        # a function - one that returns)
                        if trial and any(image.word(a) == 0x03E00008 for a in trial[0]):
                            work.setdefault(value, "built in 0x%08X" % start)

    def touching(self, start):
        """The mods that change the function at `start` (or whose code it is):
        those writing in it, or in the code it reaches with one of them on."""
        span = set()
        if start in self.functions:
            for s, e in self.spans(self.functions[start]):
                span.update(range(s, e, 4))
        first = set(self.entries.get(start, ())) | {m for m in range(len(self.mods)) if span & self.written[m]}
        stops = self.stops() - {start}
        for m in first:
            found = follow(self.image([m]), start, stops)
            if found:
                span |= found[0]
        return sorted(first | {m for m in range(len(self.mods)) if span & self.written[m]})


    def frame_size(self, image, start: int) -> int:
        """The stack a function takes (its `addiu $sp, $sp, -N` among its
        first instructions), 0 for none."""
        for a in range(start, start + 48, 4):
            ins = decode(image.word(a), a)
            if ins.mnemonic in ("addiu", "daddiu") and ins.rs == 29 and ins.rt == 29 and ins.raw & 0x8000:
                return 0x10000 - (ins.raw & 0xFFFF)
        return 0

    def callers(self):
        """Each of the game's functions, to the functions calling it (jal)."""
        if not hasattr(self, "_callers"):
            self._callers = {}
            for f in self.functions.values():
                for s, e in self.spans(f):
                    for a in range(s, e, 4):
                        w = struct.unpack_from("<I", self.ram, a)[0]
                        if w >> 26 == 3:
                            target = (w & 0x03FFFFFF) << 2
                            self._callers.setdefault(target, set()).add(f["addr"])
        return self._callers

    def reads_above(self, m: int, start: int) -> int:
        """How far above the frame of the function at `start` the mod's code
        run in it reads the stack (its callers' frames: their saved $ra and
        registers, which a structured translation never stores) - 0 none.
        The mod's code there: what the function reaches with the mod on, and
        the mod's functions it calls that take no frame of their own."""
        image = self.image([m])
        frame = self.frame_size(image, start)
        mine = self.written[m]
        found = follow(image, start, self.stops() - {start})
        if not found:
            return 0
        addresses = set(found[0])
        for target in found[1]:
            if target in self.entries and m in self.entries[target] and not self.frame_size(image, target):
                more = follow(image, target, self.stops() - {target})
                if more:
                    addresses |= more[0]
        most = 0
        for a in addresses:
            if a not in mine:
                continue
            ins = decode(image.word(a), a)
            if ins.rs == 29 and ins.mnemonic in ("lw", "ld", "lq", "lwu", "lb", "lbu", "lh", "lhu"):
                offset = ins.raw & 0xFFFF
                # (within a caller's frame or two: past that, a frame this
                # tool did not see being made)
                if not offset & 0x8000 and frame <= offset < frame + 0x400:
                    most = max(most, offset - frame + 1)
        return most

    def lift(self, start, which):
        """The function at `start` with the mods `which` on, lifted, or None."""
        image = self.image(which)
        found = follow(image, start, self.stops() - {start})
        attempts = []
        if start in self.functions:
            original = set()
            for s, e in self.spans(self.functions[start]):
                original.update(range(s, e, 4))
            if found:
                attempts.append(original | found[0])
            attempts.append(original)
        if found:
            attempts.append(found[0])
        for addresses in attempts:
            body, chunks = runs(addresses, start)
            if body is None:
                continue
            low, high = min(addresses), max(addresses) + 4
            try:
                text = lift.lift_function(image.data(low, high), low, body[0], body[1], chunks)
            except lift.Refused:
                continue
            # Where it calls and jumps out to (checked once all are known).
            for a in (found[0] if found else ()):
                ins = decode(image.word(a), a)
                if ins.mnemonic == "jal" or (ins.mnemonic in ("j", "b") and ins.target not in addresses):
                    self.exits.setdefault(ins.target, start)
            return text
        return None


def combinations(count):
    """The masks over `count` mods to lift: all of them up to
    FULL_COMBINATIONS mods, else each alone, each pair and all."""
    if count <= FULL_COMBINATIONS:
        return list(range(1, 1 << count))
    masks = {1 << i for i in range(count)} | {(1 << i) | (1 << j) for i in range(count) for j in range(i)}
    masks.add((1 << count) - 1)
    return sorted(masks)


def generate(game_dir: str, mods, code=(), name: str = "its mods' spec", shards: int = 1):
    """The generated sources (the first holds the choosers and tables), the
    functions changed and their variants."""
    state = Mods(game_dir, mods, code)
    state.discover()
    lift.use_function_starts(sorted(state.stops()))
    keys = state.keys

    header = ["/* The game's functions as its mods change them: generated by",
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
    bodies = []                             # (text) of every lifted function
    declarations = []
    further_code = list(code)
    starts = set(state.entries)
    for _key, words in mods:
        for w in words:
            if w[0] in state.owner:
                starts.add(state.owner[w[0]])
    touched = {}
    for start in sorted(starts):
        which = state.touching(start)
        if which:
            touched[start] = which
    # A mod's code that reads its callers' frames (their saved $ra, say, to
    # know where it is called from): those callers translated as plainly as
    # the mod's own (every register stored where the code stores it) while
    # it is on - up the calls as far as it reads.
    for start, which in list(touched.items()):
        if start not in state.functions:
            continue
        for m in which:
            above = state.reads_above(m, start)
            level, reach = {start}, above
            for _depth in range(3):
                if reach <= 0:
                    break
                nxt = set()
                for f in level:
                    for caller in state.callers().get(f, ()):
                        touched.setdefault(caller, [])
                        if m not in touched[caller]:
                            touched[caller] = sorted(set(touched[caller]) | {m})
                            state.warnings.append("0x%08X kept plain for %s (its code reads 0x%X above 0x%08X's frame)"
                                                  % (caller, state.keys[m], above, start))
                        nxt.add(caller)
                reach -= min((state.frame_size(state.ram_image, c) for c in nxt), default=reach) if nxt else reach
                level = nxt

    # Functions whose words add to $ra: their callers carry on where they return.
    for start, which in touched.items():
        image = state.image(which)
        found = follow(image, start, state.stops() - {start})
        if found:
            further_code.append((start, [image.word(a) for a in sorted(found[0])]))
    further = returns_further(further_code)

    for address, words in code:
        data = b"".join(struct.pack("<I", w) for w in words)
        declarations.append("void F_%08X(lifted::Context& c);" % address)
        bodies.append(returning_further(lift.lift_function(data, address, address, address + len(data)), further))
    choosers, registrations, variant_count = [], [], 0
    touched_registered = []
    for start in sorted(touched):
        which = touched[start]
        texts, table = {}, {}
        for mask in combinations(len(which)):
            on = [m for bit, m in enumerate(which) if mask >> bit & 1]
            text = state.lift(start, on)
            if text is None:
                state.warnings.append("0x%08X with %s: not lifted" % (start, ", ".join(keys[m] for m in on)))
                continue
            text = returning_further(text, further)
            digest = hashlib.sha1(text.encode()).hexdigest()
            if digest not in texts:
                texts[digest] = "F_%08X_m%d" % (start, mask)
                bodies.append(renamed(text, start, "m%d" % mask))
                declarations.append("void F_%08X_m%d(lifted::Context& c);" % (start, mask))
                variant_count += 1
            table[mask] = texts[digest]
        if not table:
            continue
        original = "F_%08X" % start if start in state.functions else None
        if original:
            declarations.append("void %s(lifted::Context& c);" % original)
        variants = []
        for mask in range(1 << len(which)):
            # A combination not lifted runs the largest lifted part of it.
            best = max((m for m in table if m & mask == m), key=lambda m: bin(m).count("1"), default=None)
            variants.append(table[best] if best is not None else original or next(iter(table.values())))
        test = " | ".join("(ModOn(%d) ? %du : 0u)" % (m, 1 << bit) for bit, m in enumerate(which))
        choosers += ["void Mods_%08X(lifted::Context& c) {" % start,
                     "    static ps2host::GuestFunction const kVariants[] = {%s};" % ", ".join(variants),
                     "    kVariants[%s](c);" % test,
                     "}",
                     ""]
        touched_registered.append(start)
        what = "a mod's own code" if start not in state.functions else "MODS"
        registrations.append('    ps2host::Override(0x%08Xu, Mods_%08X, "%s: %s");' % (
            start, start, what, ", ".join(keys[m] for m in which)))

    # Every call and jump out of the lifted code lands on a function the port
    # has (the game's, a mod's own code, a chooser): else it would stop the
    # game when it runs - said here.
    known = set(state.functions) | state.code_starts | set(touched_registered)
    for target, where in sorted(state.exits.items()):
        if target not in known:
            state.warnings.append("a call to 0x%08X (from 0x%08X): no function there" % (target, where))

    # The mods' words as memory, for the port to write while each is on.
    writes = []
    for index, (_key, words) in enumerate(mods):
        for w in words:
            if len(w) > 3:
                writes.append("    {0x%08Xu, 0x%08Xu, %d, %d, %d}," % (w[0], w[1], w[2], w[3], index))
    first = header + declarations + [""] + choosers
    first.append("struct ModWrite { u32 address, value; u8 size, place; u16 mod; };")
    first.append("extern const ModWrite kModWrites[] = {")
    first += writes or ["    {0, 0, 0, 0, 0xFFFF},"]
    first.append("};")
    first.append("extern const int kModWriteCount = %d;" % len(writes))
    first.append("extern const char* const kModKeys[] = {%s};" % ", ".join('"%s"' % k for k in keys))
    first.append("extern const int kModCount = %d;" % len(keys))
    # The function at an address as the mods have it: for a port's own
    # override there that calls the game's function itself.
    chosen = sorted(touched_registered)
    first.append("struct ModChooser { u32 address; ps2host::GuestFunction function; };")
    first.append("static const ModChooser kModChoosers[] = {%s};" % (
        ", ".join("{0x%08Xu, Mods_%08X}" % (a, a) for a in chosen) or "{0, nullptr}"))
    first += ["ps2host::GuestFunction ModdedFunction(u32 address, ps2host::GuestFunction original) {",
              "    for (const ModChooser& m : kModChoosers)",
              "        if (m.address == address && m.function) return m.function;",
              "    return original;",
              "}"]
    first.append("void RegisterModFunctions() {")
    for address, _words in code:
        first.append('    ps2host::Override(0x%08Xu, F_%08X, "a mod\'s own code");' % (address, address))
    first += registrations
    first.append("}")
    outputs = ["\n".join(first) + "\n"]
    # The lifted bodies spread over the shards (the first's too), so they compile side by side.
    shard_texts = [[] for _ in range(max(1, shards))]
    sizes = [0] * len(shard_texts)
    for body in bodies:
        k = sizes.index(min(sizes))
        shard_texts[k].append(body)
        sizes[k] += len(body)
    outputs[0] += "\n" + "\n".join(shard_texts[0])
    for texts in shard_texts[1:]:
        outputs.append("\n".join(header + texts) + "\n")
    for warning in state.warnings:
        print("lift_mods: " + warning)
    return outputs, len(touched), variant_count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True, help="the game's folder (games/<name>)")
    parser.add_argument("--spec", required=True, help="the port's mods (a Python file: MODS, CODE, NAME)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=1,
                        help="also write <out>_1.cpp .. <out>_<N-1>.cpp, the lifted code spread over them")
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    outputs, functions, variants = generate(args.game, spec.MODS, getattr(spec, "CODE", ()),
                                            getattr(spec, "NAME", os.path.basename(args.spec)), args.shards)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    stem = os.path.splitext(args.out)[0]
    for k, text in enumerate(outputs):
        path = args.out if k == 0 else "%s_%d.cpp" % (stem, k)
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                old = handle.read()
        if old != text:                     # unchanged shards keep their objects
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
    print("lift_mods: %d functions, %d variants" % (functions, variants))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
