#!/usr/bin/env python3
"""The pictures' judder, thing by thing, from a display-list motion profile
(ps2host::ListMotion; Resident Evil Outbreak's RE_MOTION_TRACE=<file.csv>).

    python engine/tools/list_motion.py trace.csv [--from N] [--to N] [--top 20]

Each line of the profile is a thing (a segment of the scene's list) in a
picture: how far its matrices moved since the picture before. Above 30, a
game at 30 steps shows each step over two pictures: the step's (half way
there, t 0.5) and the one between (all the way). A thing blended moves about
as much in both; a thing left as the game steps it moves all of its step in
the step's picture and stands in the other - it shakes against the rest.

Judged step by step: each step's pictures (the step's and those after it
until the next step's: two at 60, four at 120) where the thing moves (by
more than --moving of its size), "stepped" when it moves in the step's
picture and less than a fifth of that in each other, "uneven" when its
pictures' moves are otherwise unlike (the largest over twice the smallest,
and apart by more than half of it). Steps where
most moving things stepped are the camera's cuts (shown, not blended - as
they should be), counted apart and left out of each thing's share.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--from", dest="first", type=int, default=0)
    p.add_argument("--to", type=int, default=1 << 62)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--moving", type=float, default=2e-4, help="a thing moving in a step: its change, of its size")
    p.add_argument("--cut-share", type=float, default=0.6, help="a step where this share of the moving things stepped: a cut")
    a = p.parse_args(argv)

    pictures = collections.defaultdict(dict)     # frame -> (key, copy) -> (segment, change, size, step, words)
    with open(a.trace, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            if a.first <= frame <= a.to:
                pictures[frame][(row["key"], int(row["copy"]))] = (
                    int(row["segment"]), float(row["change"]), float(row["size"]), row["step"] == "1", int(row["words"]))
    if not pictures:
        print("no pictures in range")
        return 1
    frames = sorted(pictures)
    # Each step's pictures: its own (the step's) and those after it until the
    # next step's - two at 60, four at 120.
    pairs = []
    for i, f in enumerate(frames):
        for ident, (seg, change, size, step, words) in pictures[f].items():
            if not step:
                continue
            seq = [change]
            g = f + 1
            while g in pictures and ident in pictures[g] and not pictures[g][ident][3]:
                seq.append(pictures[g][ident][1])
                g += 1
            if len(seq) < 2 or g not in pictures:
                continue
            pairs.append((f, ident, seg, seq, size, words))
    by_step = collections.defaultdict(list)
    for pr in pairs:
        by_step[pr[0]].append(pr)
    cuts, judged = set(), collections.Counter()
    stepped_at = collections.Counter()
    verdicts = []
    for f, prs in by_step.items():
        moving = [pr for pr in prs if max(pr[3]) > a.moving * max(1e-9, pr[4])]
        # stepped: all of the step's move on its own picture; uneven: its
        # pictures' moves unlike (the largest over twice the smallest)
        st = [pr for pr in moving if max(pr[3][1:]) < 0.2 * pr[3][0]]
        if moving and len(st) >= a.cut_share * len(moving) and len(moving) >= 5:
            cuts.add(f)
            continue
        for pr in moving:
            seq = pr[3]
            kind = ("stepped" if max(seq[1:]) < 0.2 * seq[0]
                    else "uneven" if max(seq) > 2.0 * max(1e-12, min(seq)) and max(seq) - min(seq) > 0.5 * max(seq)
                    else "even")
            verdicts.append((pr[1], pr[2], pr[5], kind, f))
            if kind == "stepped":
                stepped_at[f] += 1
    things = collections.defaultdict(collections.Counter)
    where = {}
    for ident, seg, words, kind, f in verdicts:
        things[ident][kind] += 1
        where[ident] = (seg, words)
    total = collections.Counter(v[3] for v in verdicts)
    n = sum(total.values())
    print("%d pictures (%d to %d), %d steps judged, %d of them the camera's cuts (all stepped)" % (
        len(frames), frames[0], frames[-1], len(by_step), len(cuts)))
    if not n:
        print("nothing moving between steps")
        return 0
    print("things' moving steps: %d - even %.1f%%, uneven %.1f%%, stepped %.1f%%" % (
        n, 100.0 * total["even"] / n, 100.0 * total["uneven"] / n, 100.0 * total["stepped"] / n))
    rank = sorted(things.items(), key=lambda kv: -(kv[1]["stepped"] + 0.5 * kv[1]["uneven"]))
    print("the things most often wrong (stepped + uneven, of their moving steps):")
    for ident, c in rank[:a.top]:
        m = sum(c.values())
        if not c["stepped"] and not c["uneven"]:
            break
        print("  segment ~%d key %s copy %d (%d words): %d moving steps, stepped %d, uneven %d" % (
            where[ident][0], ident[0], ident[1], where[ident][1], m, c["stepped"], c["uneven"]))
    worst = stepped_at.most_common(8)
    if worst:
        print("steps with the most things stepped: " + ", ".join("%d (%d)" % w for w in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
