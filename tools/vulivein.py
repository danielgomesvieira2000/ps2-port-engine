#!/usr/bin/env python3
"""The registers each VU1 program may read before writing them: its live-ins.

For speculative VU1 (PERFORMANCE_PLAN roadmap A): a run started on a
predicted state is right if the prediction matches the true state wherever
the run reads it - data memory as the run goes (tracked), registers from
this list. Conservative by construction:
- every path from the entry is followed (both ways at a branch, the delay
  slot first; after an E bit, the pair after it and no further);
- a jump to a register goes to each address that register held at a run's
  start (x8) and to every return address a BAL/JALR can leave;
- Q, P, MAC, STATUS and CLIP are never taken as written: their writers land
  cycles later (the divider, the EFU, the flag pipeline) and CLIP keeps the
  last results, so a program that reads one anywhere reads the previous
  run's.
VF lanes, VI, ACC lanes, I and R are written by the pair that writes them.

    python engine/tools/vulivein.py SNAPSHOT [-o livein.txt]

Output, a line a program: IMAGE ENTRY reg reg ... (vf lanes as vf12z, then
vi3, ACCx, Q, P, I, R, MAC, STATUS, CLIP), read by --vu-bench's
PS2HOST_VU_SPEC.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.vu import branch_target, decode_blob
from vuloops import HEAD, RUN, fnv

NEVER_KILLED = {"Q", "P", "MAC", "STATUS", "CLIP"}
JUMPS = ("jr", "jalr")
LINKS = ("bal", "jalr")


def programs(path):
    """(image, entry) -> (micro, set of VI start values by register)."""
    size = os.path.getsize(path)
    found, hashes = {}, {}
    with open(path, "rb") as f:
        for _ in range(size // RUN):
            head = f.read(HEAD)
            m = f.read(16384)
            f.read(16384)
            if m not in hashes:
                hashes[m] = fnv(m)
            pc = struct.unpack_from("<I", head, 0)[0]
            key = (hashes[m], pc)
            vc = struct.unpack_from("<32I", head, 4 + 24 + 512)
            entry = found.setdefault(key, (m, [set() for _ in range(16)]))
            for i in range(16):
                entry[1][i].add(vc[i] & 0xFFFF)
    return found


def live_ins(micro, entry, vi_starts):
    mask = len(micro) - 1
    by_addr = {b.addr: b for b in decode_blob(micro, 0)}
    returns = {(b.addr + 16) & mask for b in by_addr.values() if b.lower.mnemonic in LINKS}

    def successors(a, delay_of):
        """Where control goes after pair a (delay_of: the branch pair whose slot a is)."""
        if delay_of is not None:
            br = by_addr[delay_of]
            if br.lower.mnemonic in JUMPS:
                reg = int(str(br.lower.operands[-1]).replace("vi", ""))
                targets = {(v * 8) & mask for v in vi_starts[reg]} | returns
                return [(t, None) for t in sorted(targets)]
            t = branch_target(br) & mask
            out = [(t, None)]
            if br.lower.mnemonic not in ("b", "bal"):
                out.append(((a + 8) & mask, None))
            return out
        b = by_addr[a]
        nxt = (a + 8) & mask
        if b.is_end:
            return [(nxt, "end")]
        if branch_target(b) is not None or b.lower.mnemonic in JUMPS:
            return [(nxt, a)]
        return [(nxt, None)]

    # Nodes: (address, context) - context None, a branch's address (its delay slot) or "end".
    nodes, work = {}, [(entry, None)]
    while work:
        node = work.pop()
        if node in nodes:
            continue
        a, ctx = node
        succ = [] if ctx == "end" else successors(a, ctx if ctx != "end" else None)
        nodes[node] = succ
        work.extend(succ)

    def uses(a):
        b = by_addr[a]
        reads = set(b.upper.reads) | (set() if b.immediate is not None else set(b.lower.reads))
        writes = set(b.upper.writes) | set(b.lower.writes)
        return reads, writes - NEVER_KILLED

    info = {n: uses(n[0]) for n in nodes}
    live = {n: set() for n in nodes}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            out = set()
            for s in nodes[n]:
                out |= live[s]
            reads, kills = info[n]
            new = reads | (out - kills)
            if new != live[n]:
                live[n] = new
                changed = True
    return live[(entry, None)], len(nodes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    lines = []
    for (image, entry), (micro, vi_starts) in sorted(programs(args.snapshot).items()):
        regs, count = live_ins(micro, entry, vi_starts)
        order = sorted(regs, key=lambda r: (not r.startswith("vf"), not r.startswith("vi"), r))
        lines.append("%08X %04X %s" % (image, entry, " ".join(order)))
        print("%08X:%04X %d pairs on paths, %d live-ins: %s" % (
            image, entry, count, len(regs), " ".join(order[:24]) + (" ..." if len(order) > 24 else "")))
    if args.output:
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
