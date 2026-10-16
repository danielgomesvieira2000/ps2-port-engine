"""Code overlays: code a game loads from its disc at run time, over memory the
executable leaves free, and decompiled like the executable itself.

An overlay is described as a game of its own - a directory with a `game.json`
naming the game it belongs to and the overlay's image, already unpacked to how
it lies in memory (unpacking is the game's business, under the game's tools/):

    {
      "name": "Some Game: game.bin",
      "overlay": {
        "of": "../..",                              the base game's directory
        "image": "../../extracted/overlays/2.bin",  relative to this directory
        "address": "0x00570000",                    where the game loads it
        "size": "0x001EA580",                       memory it takes, with its BSS (optional)
        "code": [["0x00570000", "0x00708540"]],     where its functions are (optional)
        "resident": [                               other overlays loaded with it (optional)
          {"image": "../../extracted/overlays/0.bin", "address": "0x00370000", "size": "..."}
        ]
      }
    }

Every tool then works on it through `--game`: `combine` builds an executable of
the base game's image with the overlay at its address, so the interpreter runs
the overlay's calls into the executable's code as they are, and the analysis
keeps only the functions inside `code` (`in_code`). Several overlays may load at
the same address; each is its own game, and its functions are its own. Overlays
resident together (the ones its calls lead into) are laid over memory with it,
but their functions are left to their own descriptions.
"""

from __future__ import annotations

import os
import struct

from .elf import ElfFile

SHF_ALLOC_EXEC_WRITE = 0x7
NUL = bytes(1)
ELF_MAGIC = bytes([0x7F]) + b"ELF"


def combine(base: ElfFile, images) -> bytes:
    """An ELF whose one executable section is the base executable's image with
    each of `images` - (bytes, address, size with BSS) - laid over it in order:
    the memory the game has once those overlays are loaded. The section spans the
    gaps (zero, as the base's BSS is) and one segment covers it all and the BSS."""
    section = base.exec_sections[0]
    data = base.section_data(section)
    start = section.addr
    for _image, address, _size in images:
        if address < start:
            raise ValueError("an overlay below the executable (0x%08x < 0x%08x) is not "
                             "supported" % (address, start))
    end = max([start + len(data)] + [address + len(image) for image, address, _s in images])
    memory = bytearray(end - start)
    memory[:len(data)] = data
    for image, address, _size in images:
        memory[address - start:address - start + len(image)] = image
    memsz = max([end] + [address + max(size, len(image)) for image, address, size in images]
                + [p.vaddr + p.memsz for p in base.segments if p.memsz]) - start

    names = NUL.join([b"", b".text", b".shstrtab", b""])
    header_size, program_size, section_size = 52, 32, 40
    data_offset = 0x1000
    names_offset = data_offset + len(memory)
    sections_offset = (names_offset + len(names) + 3) & ~3
    out = bytearray(sections_offset + 3 * section_size)
    struct.pack_into("<4s5B7x", out, 0, ELF_MAGIC, 1, 1, 1, 0, 0)
    struct.pack_into("<HHIIIIIHHHHHH", out, 16, 2, 8, 1, base.e_entry, header_size,
                     sections_offset, base.e_flags, header_size, program_size, 1,
                     section_size, 3, 2)
    struct.pack_into("<IIIIIIII", out, header_size, 1, data_offset, start, start,
                     len(memory), memsz, SHF_ALLOC_EXEC_WRITE, 0x80)
    out[data_offset:data_offset + len(memory)] = memory
    out[names_offset:names_offset + len(names)] = names
    struct.pack_into("<IIIIIIIIII", out, sections_offset + section_size,
                     1, 1, SHF_ALLOC_EXEC_WRITE, start, data_offset, len(memory), 0, 0, 0x80, 0)
    struct.pack_into("<IIIIIIIIII", out, sections_offset + 2 * section_size,
                     7, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)
    return bytes(out)


def build(base_elf: str, images, target: str) -> None:
    """Write `combine`'s executable to `target` unless it is newer than every
    source and the description (`images`: (path, address, size), resident ones
    first, the overlay itself last) is unchanged."""
    stamp = target + ".sources"
    description = repr([(os.path.normcase(os.path.abspath(p)), a, s) for p, a, s in images])
    sources = [base_elf] + [path for path, _a, _s in images]
    if os.path.exists(target) and os.path.exists(stamp) and os.path.getmtime(target) >= max(
            os.path.getmtime(path) for path in sources):
        with open(stamp, encoding="utf-8") as handle:
            if handle.read() == description:
                return
    loaded = []
    for path, address, size in images:
        with open(path, "rb") as handle:
            loaded.append((handle.read(), address, size))
    combined = combine(ElfFile.load(base_elf), loaded)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".tmp"
    with open(temporary, "wb") as handle:
        handle.write(combined)
    os.replace(temporary, target)
    with open(stamp, "w", encoding="utf-8") as handle:
        handle.write(description)


def namespace(directory: str) -> str:
    """The C++ namespace an overlay's translations are compiled in: overlays
    loaded at the same address have functions of the same names."""
    name = os.path.basename(os.path.normpath(directory))
    return "overlay_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def in_code(ranges, address: int) -> bool:
    """Whether a function at `address` belongs to the overlay (no ranges: all do)."""
    return not ranges or any(low <= address < high for low, high in ranges)
