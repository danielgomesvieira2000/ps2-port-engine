#!/usr/bin/env python3
"""Find a VU1 microprogram's loops and whether their iterations are independent.

Reads one program out of a --vu-snapshot file (the microcode its runs had
loaded), disassembles what is reachable from its entry, and for each loop (a
branch back) reports its size, its loads and stores, the integer registers it
steps (pointers, counters) and its float recurrences: registers whose value at
the end of the body depends on their own value at the start. Software
pipelining - a value loaded at the bottom of one iteration and used at the top
of the next - is not a recurrence. Memory and MAC/CLIP flags are not followed,
nor the divider's latency: Q and P read soon after a DIV/SQRT/RSQRT (or EFU op)
are still the one before's, so a chain through Q or P may not exist. A float
recurrence reported here is a suspect; settle it with PS2HOST_VU_PERTURB
(host/src/vu.cpp), e.g. 14E0's vf12z on 2026-10-29 (no recurrence).

    PS2HOST_VU_PERTURB=14E0,C20,5,12,2,1 <port.exe> <elf> --vu-bench SNAP

    # programs and their image hashes: PS2HOST_VU_BY_PROGRAM=1 <port.exe> <elf> --vu-bench SNAP
    # pair counts: PS2HOST_VU_PC_COUNTS=counts.txt <port.exe> <elf> --vu-bench SNAP
    python engine/tools/vuloops.py SNAP 2A567562 06A0 --counts counts.txt
    python engine/tools/vuloops.py SNAP 7E86891B 14E0 --start 0020 --list   # 14E0 is `jr vi15`, vi15 = 4
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.vu import branch_target, decode_blob

# A snapshot run (host/src/vu.cpp, Snapshot): pc, 6 flag words, vf[32][4],
# vc[32], vacc[4]; then micro and data memory.
HEAD = 4 + 6 * 4 + 32 * 16 + 32 * 4 + 16
RUN = HEAD + 16384 * 2


def fnv(data: bytes) -> int:
    h = 2166136261
    for x in data:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def load_runs(path, image, entry):
    size = os.path.getsize(path)
    if size % RUN:
        raise SystemExit("%s: %d bytes is not whole runs of %d" % (path, size, RUN))
    micro, starts, hashes = None, [], {}
    with open(path, "rb") as f:
        for _ in range(size // RUN):
            head = f.read(HEAD)
            m = f.read(16384)
            f.read(16384)
            if struct.unpack_from("<I", head, 0)[0] != entry:
                continue
            if m not in hashes:
                hashes[m] = fnv(m)
            if hashes[m] != image:
                continue
            micro = m
            vc = struct.unpack_from("<32I", head, 4 + 24 + 512)
            starts.append([v & 0xFFFF for v in vc[:16]])
    return micro, starts


def reachable(by_addr, start):
    reach, work = set(), [start]
    while work:
        a = work.pop()
        while a not in reach and a in by_addr:
            reach.add(a)
            b = by_addr[a]
            target = branch_target(b)
            if b.is_end:
                reach.add(a + 8)                    # the pair after the E bit runs too
                break
            if target is not None:
                work.append(target)
                if b.lower.mnemonic in ("b", "bal"):
                    reach.add(a + 8)
                    break
            if b.lower.mnemonic in ("jr", "jalr"):  # not followed: where it goes is a register
                reach.add(a + 8)
                break
            a += 8
    return reach


def analyse(body):
    written, carried, steps = set(), set(), {}
    dep = {}                                        # register -> the start values it depends on
    for bb in body:
        new = {}
        for ins in (bb.upper, bb.lower):
            src = set()
            for r in ins.reads:
                if r not in written:
                    carried.add(r)
                src |= dep.get(r, {r})
            for w in ins.writes:
                new[w] = src
        written |= set(new)
        dep.update(new)
        low = bb.lower
        if low.mnemonic in ("iaddi", "iaddiu", "isubiu") and len(low.operands) >= 3 \
                and low.operands[0] == low.operands[1]:
            steps[low.operands[0]] = ("-" if low.mnemonic == "isubiu" else "+") + str(low.operands[2])
    recurrences = sorted(r for r in written if r in dep.get(r, set()))
    return {
        "carried": sorted(r for r in carried if r in written),
        "integer": [r for r in recurrences if r.startswith("vi")],
        "float": [r for r in recurrences if not r.startswith("vi")],
        "steps": steps,
        "loads": sum(1 for bb in body if bb.lower.mnemonic.startswith(("lq", "ilw"))),
        "stores": sum(1 for bb in body if bb.lower.mnemonic.startswith(("sq", "isw"))),
        "kicks": sum(1 for bb in body if bb.lower.mnemonic == "xgkick"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot")
    parser.add_argument("image", help="the microcode's hash, as PS2HOST_VU_BY_PROGRAM prints it")
    parser.add_argument("entry", help="the runs' entry address (hex, bytes)")
    parser.add_argument("--start", help="disassemble from here instead (hex), e.g. past a jr")
    parser.add_argument("--counts", help="a PS2HOST_VU_PC_COUNTS file: each loop's share of the pairs run")
    parser.add_argument("--list", action="store_true", help="print the reachable pairs too")
    args = parser.parse_args(argv)

    image, entry = int(args.image, 16), int(args.entry, 16)
    micro, starts = load_runs(args.snapshot, image, entry)
    if micro is None:
        raise SystemExit("no run of %08X:%04X in %s" % (image, entry, args.snapshot))
    print("runs: %d" % len(starts))
    for i in range(16):
        values = sorted({s[i] for s in starts})
        shown = ", ".join("%X" % v for v in values) if len(values) <= 4 else \
            "%d values, %X..%X" % (len(values), values[0], values[-1])
        print("  vi%02d at start: %s" % (i, shown))

    counts = {}
    if args.counts:
        with open(args.counts) as f:
            for line in f:
                img, ent, addr, n = line.split()
                if int(img, 16) == image and int(ent, 16) == entry:
                    counts[int(addr, 16)] = int(n)
    total = sum(counts.values())

    by_addr = {b.addr: b for b in decode_blob(micro, 0)}
    reach = reachable(by_addr, int(args.start, 16) if args.start else entry)
    print("program: %04X..%04X, %d pairs reachable" % (min(reach), max(reach), len(reach)))
    if args.list:
        for a in sorted(reach):
            print(by_addr[a].format())

    print("\nloops (backward branches):")
    for a in sorted(reach):
        target = branch_target(by_addr[a])
        if target is None or target > a:
            continue
        body = [by_addr[x] for x in range(target, a + 16, 8) if x in by_addr]
        r = analyse(body)
        print("  %04X -> %04X: %d pairs, %d loads, %d stores, %d kicks; steps %s" % (
            a, target, len(body), r["loads"], r["stores"], r["kicks"], r["steps"] or "none"))
        if total:
            inside = sum(counts.get(x, 0) for x in range(target, a + 16, 8))
            print("      %.1f%% of the program's pairs; %d iterations, entered %d times" % (
                100.0 * inside / total, counts.get(a, 0), counts.get(target - 8, 0)))
        print("      carried by pipelining: %d registers; integer recurrences: %s" % (
            len(r["carried"]), r["integer"] or "none"))
        print("      float recurrences: %s" % (r["float"] or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
