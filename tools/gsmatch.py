#!/usr/bin/env python3
"""How well a picture's GS packets match the picture before's - for pictures
made between two (games/burnout3/docs/PERFORMANCE_PLAN.md, option 2, step 0).

Reads a capture's replay.bin (F12, or PS2HOST_CAPTURE_FRAME=N with
PS2HOST_CAPTURE_FRAMES=16: a picture a frame, apart at each 'V'), cuts each
path's words into GIF packets (a tag and its data), and for each picture
after the first lines its packets up with the last picture's by their shape
(path, FLG, NLOOP, NREG, REGS, PRIM where PRE) in order. A pair is:
  same       every word as before (the sky, the HUD, a texture)
  vertices   only vertex values changed - XYZ, ST/UV, RGBAQ, FOG (blendable)
  state      an A+D register or PRIM changed (another texture, blend...)
and a packet with no pair is new. Counted in packets and in vertices (the
XYZ registers' writes), with each kind's largest screen move.

    python engine/tools/gsmatch.py <capture>/replay.bin [--limit N]
"""

from __future__ import annotations

import argparse
import difflib
import struct
import sys
import time

XYZ = {4, 5, 0xC, 0xD}          # XYZF2, XYZ2, XYZF3, XYZ3
VERTEX = XYZ | {1, 2, 3, 0xA}    # and RGBAQ, ST, UV, FOG
AD = 0xE


class Packet:
    __slots__ = ("path", "shape", "regs", "flg", "words", "vertices")

    def __init__(self, path, shape, regs, flg, words, vertices):
        self.path, self.shape, self.regs, self.flg, self.words, self.vertices = path, shape, regs, flg, words, vertices


def pictures(path: str):
    """[[Packet...] a picture], the paths' packets in the order they ended."""
    data = open(path, "rb").read()
    if data[:5] != b"GSRP1":
        raise SystemExit("not a replay.bin")
    at = 5 + (4 << 20)
    streams = {1: [], 2: [], 3: []}     # each path's words not yet a whole packet
    out, picture = [], []

    def cut(p):
        words = streams[p]
        i = 0
        while len(words) - i >= 4:
            lo = words[i] | words[i + 1] << 32
            hi = words[i + 2] | words[i + 3] << 32
            nloop = lo & 0x7FFF
            pre, prim = (lo >> 46) & 1, (lo >> 47) & 0x7FF
            flg, nreg = (lo >> 58) & 3, (lo >> 60) & 0xF or 16
            regs = tuple((hi >> (4 * r)) & 0xF for r in range(nreg))
            if flg == 0:
                size = nloop * nreg * 4
            elif flg == 1:
                size = ((nloop * nreg + 1) // 2) * 4
            else:
                size = nloop * 4
            if len(words) - i - 4 < size:
                break
            body = tuple(words[i + 4:i + 4 + size])
            vertices = nloop * sum(1 for r in regs if r in XYZ) if flg == 0 else 0
            shape = (p, flg, nloop, nreg, regs, prim if pre else -1)
            picture.append(Packet(p, shape, regs, flg, body, vertices))
            i += 4 + size
        del words[:i]

    while at < len(data):
        kind = data[at]
        at += 1
        if kind in (0x52, 0x50):        # 'R', 'P'
            at += 12
        elif kind == 0x57:              # 'W'
            p = data[at]
            n = struct.unpack_from("<I", data, at + 1)[0]
            streams[p].extend(struct.unpack_from("<%dI" % (n // 4), data, at + 5))
            at += 5 + n
            cut(p)
        elif kind == 0x56:              # 'V'
            if picture:
                out.append(picture)
            picture = []
        else:
            raise SystemExit("bad record %r at %d" % (kind, at - 1))
    if picture:
        out.append(picture)
    return out


MOVE_EDGES = (1, 4, 16, 64, 256)        # pixels: a vertex's move, bucketed


def changed(a: Packet, b: Packet, moves=None):
    """'same', 'vertices' or 'state', and the largest XYZ move (pixels);
    `moves` (a list of len(MOVE_EDGES) + 1) counts each vertex's move."""
    if a.words == b.words:
        if moves is not None:
            moves[0] += b.vertices
        return "same", 0.0
    if a.flg != 0:
        return "state", 0.0             # REGLIST, IMAGE: other data
    n = len(a.regs)
    state = False
    move = 0.0
    for q in range(len(a.words) // 4):
        reg = a.regs[q % n]
        wa, wb = a.words[4 * q:4 * q + 4], b.words[4 * q:4 * q + 4]
        if reg in XYZ:
            d = max(abs((wa[0] & 0xFFFF) - (wb[0] & 0xFFFF)), abs((wa[1] & 0xFFFF) - (wb[1] & 0xFFFF))) / 16.0
            move = max(move, d)
            if moves is not None:
                moves[sum(1 for e in MOVE_EDGES if d >= e)] += 1
        elif wa != wb and reg not in VERTEX:
            state = True
    return ("state" if state else "vertices"), move


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("replay")
    parser.add_argument("--limit", type=int, default=0, help="pictures at most")
    args = parser.parse_args(argv)
    pics = pictures(args.replay)
    if args.limit:
        pics = pics[:args.limit]
    print("%d pictures, %s packets each" % (len(pics), "/".join(str(len(p)) for p in pics)))
    total = {}
    hist = [0] * (len(MOVE_EDGES) + 1)
    for k in range(1, len(pics)):
        a, b = pics[k - 1], pics[k]
        began = time.perf_counter()
        sm = difflib.SequenceMatcher(None, [p.shape for p in a], [p.shape for p in b], autojunk=False)
        blocks = sm.get_matching_blocks()
        took = time.perf_counter() - began
        count = {"same": [0, 0], "vertices": [0, 0], "state": [0, 0], "new": [0, 0]}
        moves = {"vertices": 0.0, "state": 0.0}
        matched = set()
        for i, j, n in blocks:
            for d in range(n):
                kind, move = changed(a[i + d], b[j + d], hist)
                count[kind][0] += 1
                count[kind][1] += b[j + d].vertices
                if kind in moves:
                    moves[kind] = max(moves[kind], move)
                matched.add(j + d)
        for j, p in enumerate(b):
            if j not in matched:
                count["new"][0] += 1
                count["new"][1] += p.vertices
        verts = sum(p.vertices for p in b) or 1
        moving = count["vertices"][1] + count["state"][1] + count["new"][1] or 1
        print("picture %2d: %6d packets, %7d vertices: same %5.1f%%, vertices %5.1f%% (moves up to %.0f px), "
              "state %4.1f%%, new %4.1f%% of vertices; of the moving or new: blendable %5.1f%%; match %.2f s"
              % (k, len(b), verts, 100 * count["same"][1] / verts, 100 * count["vertices"][1] / verts,
                 moves["vertices"], 100 * count["state"][1] / verts, 100 * count["new"][1] / verts,
                 100 * count["vertices"][1] / moving, took))
        for kind, (pk, vx) in count.items():
            t = total.setdefault(kind, [0, 0])
            t[0] += pk
            t[1] += vx
    if total:
        verts = sum(v for _, v in total.values()) or 1
        moving = total["vertices"][1] + total["state"][1] + total["new"][1] or 1
        print("all: " + ", ".join("%s %.1f%% of packets, %.1f%% of vertices" %
                                  (k, 100 * pk / max(1, sum(x for x, _ in total.values())), 100 * vx / verts)
                                  for k, (pk, vx) in total.items()))
        print("of the vertices that moved or are new, blendable: %.1f%%" % (100 * total["vertices"][1] / moving))
        names = ["< %d px" % MOVE_EDGES[0]] + ["%d-%d px" % (lo, hi) for lo, hi in zip(MOVE_EDGES, MOVE_EDGES[1:])] + \
                [">= %d px" % MOVE_EDGES[-1]]
        whole = sum(hist) or 1
        print("matched vertices by move: " + ", ".join("%s %.1f%%" % (n, 100 * c / whole) for n, c in zip(names, hist)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
