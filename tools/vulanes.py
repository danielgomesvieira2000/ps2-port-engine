#!/usr/bin/env python3
"""How much of VU1's work could run two items in lockstep (SIMD across iterations).

A loop can run iterations i and j side by side, sharing each pair's
bookkeeping (stalls, flags, counts), when both take the same path through it:
- no branch inside its body but the one back (no divergence);
- no float recurrence (vuloops.analyse) - every carried value depends on a
  bounded number of iterations before, rebuilt by a warm-up;
- enough iterations an entry that the warm-up and the setup pay.
This reports, for each program in a PS2HOST_VU_PC_COUNTS file, its loops,
their share of all pairs run, and which pass; then the total share that
passes and a gain estimate (two lanes: each pair's bookkeeping and arithmetic
once for both, its lower half twice; a warm-up of `--warmup` iterations a
lane; `--lower` the lower half's share of a pair's cost).

    python engine/tools/vulanes.py SNAPSHOT COUNTS [--min-iterations 8]
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

from ps2decomp.vu import branch_target, decode_blob
from vulivein import programs
from vuloops import analyse

BRANCHES = ("b", "bal", "ibeq", "ibne", "ibltz", "ibgtz", "iblez", "ibgez", "jr", "jalr")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot")
    parser.add_argument("counts")
    parser.add_argument("--min-iterations", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--lower", type=float, default=0.3)
    args = parser.parse_args(argv)

    counts = {}
    for line in open(args.counts):
        img, ent, addr, n = line.split()
        counts.setdefault((int(img, 16), int(ent, 16)), {})[int(addr, 16)] = int(n)
    grand = sum(sum(c.values()) for c in counts.values())
    progs = programs(args.snapshot)

    passing = 0
    saved = 0.0
    reasons = {}
    rows = []
    for key, c in counts.items():
        if key not in progs:
            continue
        micro = progs[key][0]
        by_addr = {b.addr: b for b in decode_blob(micro, 0)}
        loops = []
        for a, n in c.items():
            b = by_addr.get(a)
            if b is None:
                continue
            t = branch_target(b)
            if t is None or t > a:
                continue
            body_addrs = list(range(t, a + 16, 8))
            body = [by_addr[x] for x in body_addrs if x in by_addr]
            inside = sum(c.get(x, 0) for x in body_addrs)
            loops.append((inside, a, t, body, n))
        # Innermost first: a loop whose body holds another loop's branch is an outer one.
        loops.sort(key=lambda l: l[1] - l[2])
        counted = set()
        for inside, a, t, body, taken in loops:
            addrs = set(range(t, a + 16, 8))
            if addrs & counted:
                continue                      # an outer loop: its inner one is counted
            entered = max(1, c.get(t - 8, 0))
            iterations = taken / entered
            inner = [bb for bb in body if bb.addr != a and bb.lower.mnemonic in BRANCHES]
            r = analyse(body)
            why = None
            if inner:
                why = "branches inside"
            elif r["float"]:
                why = "float recurrence (check by perturbation)"
            elif iterations < args.min_iterations:
                why = "few iterations"
            share = inside / grand
            reasons[why or "passes"] = reasons.get(why or "passes", 0.0) + share
            rows.append((share, key, a, t, len(body), iterations, why or "passes"))
            counted |= addrs
            if not why:
                passing += inside
                # Two lanes over N iterations: N/2 + warmup lockstep iterations,
                # each its bookkeeping and upper once, its lower twice.
                cost_lane = (iterations / 2 + args.warmup) * (1 + args.lower) / iterations
                saved += inside * max(0.0, 1 - cost_lane)
    rows.sort(reverse=True)
    print("loops by share of all VU1 pairs run (innermost; %d pairs in all):" % grand)
    for share, key, a, t, size, iterations, why in rows[:25]:
        print("  %5.1f%%  %08X:%04X  %04X->%04X  %3d pairs  %6.1f iterations an entry  %s" % (
            100 * share, key[0], key[1], a, t, size, iterations, why))
    print("\nshare of all pairs, by verdict:")
    for why, share in sorted(reasons.items(), key=lambda x: -x[1]):
        print("  %5.1f%%  %s" % (100 * share, why))
    print("\npasses: %.1f%% of all pairs; estimated saving %.1f%% of VU1 (warm-up %.0f, lower half %.0f%%)" % (
        100.0 * passing / grand, 100.0 * saved / grand, args.warmup, 100 * args.lower))
    return 0


if __name__ == "__main__":
    sys.exit(main())
