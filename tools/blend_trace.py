#!/usr/bin/env python3
"""The steps a display-list blend may have placed wrongly, from its trace.

    set RE_BLEND_TRACE=trace.csv      (Resident Evil Outbreak; then play)
    python engine/tools/blend_trace.py trace.csv [--top N] [--list N]

A game whose high frame rate blends its display lists (ps2host/display_list.h)
writes a line for each step's blend of each list: how near its blended moves
came to the jump that would have stepped a thing (`worst`, a share of it: 1
would have stepped), how far the camera's view moved (`view`, a share of the
cut's), what came in (`new`), jumped (`jumps`: stepped, not blended) or
moved out of its place. Listed here, each step that stands out from the
steps around it (the median of the 15 before and after):

  view    the camera's view moved much more than around it but less than a
          cut: blended across what may be a cut (a picture between two
          shots), or a lurch
  near    a thing blended with a move near its jump (worst above --near):
          a vertex or a matrix carried far for a picture
  jumps   many things stepped while the rest blended: layers of one thing
          apart for a picture
  new     many things came in at once (shown where the step has them)
  cut     the camera cut (nothing blended) - listed only in gameplay where
          cuts are rare (more than --cuts a minute is said in the summary)

Each line: the frame, its list, the reason and the figures; then a summary.
Look at the worst with RE_BLEND_LIST / RE_BLEND_WORDS and a capture (F12).
"""

from __future__ import annotations

import argparse
import csv
import statistics


def load(path, list_index):
    rows = []
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            try:
                if int(r["list"]) != list_index:
                    continue
                rows.append({k: (float(v) if k in ("worst", "view") else int(v)) for k, v in r.items()})
            except (KeyError, ValueError):
                continue        # a line cut short (the game closed)
    return rows


def around(rows, i, key, span=15):
    values = [rows[j][key] for j in range(max(0, i - span), min(len(rows), i + span + 1)) if j != i]
    return statistics.median(values) if values else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--list", type=int, default=0, help="the list to look at (0: the scene's)")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--near", type=float, default=0.8, help="a blended move this share of its jump: listed")
    p.add_argument("--all-modes", action="store_true", help="not only gameplay (mode 2)")
    args = p.parse_args(argv)

    rows = load(args.trace, args.list)
    if not args.all_modes:
        rows = [r for r in rows if r["mode"] == 2]
    if not rows:
        print("no steps of list %d in %s" % (args.list, args.trace))
        return 1
    found = []
    for i, r in enumerate(rows):
        if r["cut"]:
            found.append((r["frame"], 3.0, "cut", "view %.2f, %d new, %d jumps" % (r["view"], r["new"], r["jumps"])))
            continue
        view_around = around(rows, i, "view")
        if r["view"] > 0.25 and r["view"] > 4 * max(view_around, 0.02):
            found.append((r["frame"], r["view"] / max(view_around, 0.02), "view",
                          "%.2f of a cut, %.3f around" % (r["view"], view_around)))
        if r["worst"] > args.near:
            found.append((r["frame"], 10 * r["worst"], "near",
                          "%.2f of a jump in segment %d (%d near)" % (r["worst"], r["worst_segment"], r["near"])))
        jumps_around = around(rows, i, "jumps")
        if r["jumps"] >= 5 and r["jumps"] > 3 * max(jumps_around, 1):
            found.append((r["frame"], r["jumps"] / max(jumps_around, 1), "jumps",
                          "%d of %d segments stepped, %.0f around" % (r["jumps"], r["segments"], jumps_around)))
        new_around = around(rows, i, "new")
        if r["new"] >= 8 and r["new"] > 3 * max(new_around, 1):
            found.append((r["frame"], r["new"] / max(new_around, 1) / 2, "new",
                          "%d came in, %.0f around" % (r["new"], new_around)))

    found.sort(key=lambda f: -f[1])
    for frame, _score, reason, text in sorted(found[:args.top], key=lambda f: f[0]):
        print("frame %7d  %-5s  %s" % (frame, reason, text))
    frames = rows[-1]["frame"] - rows[0]["frame"] + 1
    minutes = frames / 60.0 / 60.0
    cuts = sum(1 for r in rows if r["cut"])
    counts = {}
    for f in found:
        counts[f[2]] = counts.get(f[2], 0) + 1
    print("\n%d steps of gameplay over %.1f minutes; cuts %d (%.1f a minute); listed %s%s" % (
        len(rows), minutes, cuts, cuts / minutes if minutes else 0.0,
        ", ".join("%s %d" % kv for kv in sorted(counts.items())),
        " (the top %d shown)" % args.top if len(found) > args.top else ""))
    worst = sorted(r["worst"] for r in rows if not r["cut"])
    view = sorted(r["view"] for r in rows if not r["cut"])
    if worst:
        q = lambda v, x: v[min(len(v) - 1, int(x * len(v)))]
        print("blended moves, of their jump: median %.3f, 99%% %.3f, largest %.3f; the view's, of a cut: "
              "median %.3f, 99%% %.3f, largest %.3f" % (q(worst, 0.5), q(worst, 0.99), worst[-1],
                                                        q(view, 0.5), q(view, 0.99), view[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
