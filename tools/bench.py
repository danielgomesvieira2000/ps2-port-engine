#!/usr/bin/env python3
"""A port's deterministic benchmark: the same work each time, timed a few times over.

    python engine/tools/bench.py --game games/<name> --replay pile.bin --replay boom.bin \\
        --vu vu_crash.bin --run race --times 5 --json after.json [--compare before.json]

What a change does to the CPU, without the noise of playing in real time:

  --replay FILE   a GS capture (F12's replay.bin) replayed on the GPU renderer
                  (PS2HOST_REPLAY_GPU, PS2HOST_REPLAY_REPEAT passes): the GS
                  thread's CPU a pass and the renderer thread's.
  --vu FILE       a VU1 snapshot (--vu-bench, the compiled VU: --vu-fast): the
                  best pass.
  --run NAME      a scripted run (port/runs/NAME.args, virtual time, the GPU):
                  its wall-clock time, start to end.

Each is run --times times (default 5); the median is reported with the spread
(the lowest and highest). --json keeps them; --compare prints each against an
earlier --json's, with the change in percent. Captures are made from the
game's disc (its code and data): keep them out of the repositories.

Measure like with like: the same build kind (a PGO build's profile drops the
functions an edit touched), the PC otherwise idle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE)]

from ps2decomp import game as game_config  # noqa: E402

REPLAY = re.compile(r"gs-replay: (\d+) passes, ([\d.]+) ms a pass .*this thread's CPU ([\d.]+) ms a pass, "
                    r"the renderer's ([\d.]+)")
VU = re.compile(r"vu-bench: .*best pass ([\d.]+) ms")


def run(command, env=None):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace",
                            env=dict(os.environ, **(env or {})))
    return result.returncode, result.stdout + result.stderr


def replay(executable, elf, capture, repeat):
    with tempfile.TemporaryDirectory(prefix="bench_") as out:
        code, text = run([executable, elf, "--gs-replay", capture, out],
                         {"PS2HOST_REPLAY_GPU": "1", "PS2HOST_REPLAY_REPEAT": str(repeat)})
    m = REPLAY.search(text)
    if not m:
        raise SystemExit("bench: no replay timing from %s (exit %d):\n%s" % (capture, code, text[-2000:]))
    return {"wall ms": float(m.group(2)), "GS thread ms": float(m.group(3)), "renderer ms": float(m.group(4))}


def vu(executable, elf, snapshot):
    code, text = run([executable, elf, "--vu-fast", "--vu-bench", snapshot])
    m = VU.search(text)
    if not m:
        raise SystemExit("bench: no VU timing from %s (exit %d):\n%s" % (snapshot, code, text[-2000:]))
    return {"best pass ms": float(m.group(1))}


def scripted(executable, elf, disc, game, name):
    with open(game.path(os.path.join("port", "runs", name + ".args")), encoding="utf-8") as handle:
        args = handle.read().split()
    if any("{memcard}" in a for a in args):
        card = tempfile.mkdtemp(prefix="bench_card_")
        args = [a.replace("{memcard}", card) for a in args]
    command = [executable, elf] + (["--disc", disc] if disc and os.path.exists(disc) else []) + args + [
        "--timeout", "3600", "--gpu"]
    started = time.perf_counter()
    code, text = run(command)
    seconds = time.perf_counter() - started
    # A run ends at its --until-frame as a stop (exit 2): that is its end.
    if code != 0 and "reached (--until-frame)" not in text:
        raise SystemExit("bench: run %s failed (exit %d):\n%s" % (name, code, text[-2000:]))
    return {"seconds": seconds}


def summary(samples):
    """Each measure's median, lowest and highest over the samples."""
    out = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        out[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--exe", help="the executable (default: the game's build)")
    parser.add_argument("--replay", action="append", default=[], metavar="FILE")
    parser.add_argument("--repeat", type=int, default=100, help="a replay's passes (default 100)")
    parser.add_argument("--vu", action="append", default=[], metavar="FILE")
    parser.add_argument("--run", action="append", default=[], metavar="NAME")
    parser.add_argument("--times", type=int, default=5)
    parser.add_argument("--json", metavar="FILE", help="the results, kept")
    parser.add_argument("--compare", metavar="FILE", help="an earlier --json, each result against it")
    args = parser.parse_args(argv)

    game = game_config.load(args.game)
    executable = args.exe or game.path(os.path.join(game.build, "port", "cmake", os.path.basename(
        os.path.normpath(game.directory)) + ".exe"))
    elf = game.path(os.path.join(game.extracted, game.executable))
    disc = game.path(game.iso) if game.iso else ""
    jobs = [("replay " + os.path.basename(f), lambda f=f: replay(executable, elf, f, args.repeat)) for f in args.replay]
    jobs += [("vu " + os.path.basename(f), lambda f=f: vu(executable, elf, f)) for f in args.vu]
    jobs += [("run " + r, lambda r=r: scripted(executable, elf, disc, game, r)) for r in args.run]
    if not jobs:
        parser.error("nothing to measure: --replay, --vu or --run")

    before = {}
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            before = json.load(handle)
    results = {}
    for name, job in jobs:
        # Interleaved would share the noise better, but each job warms its own
        # caches: one after another, a first pass thrown away.
        job()
        results[name] = summary([job() for _ in range(args.times)])
        for key, s in results[name].items():
            line = "%-28s %-14s %9.2f  (%.2f-%.2f)" % (name, key, s["median"], s["min"], s["max"])
            was = before.get(name, {}).get(key)
            if was:
                line += "   was %.2f: %+.1f%%" % (was["median"], 100.0 * (s["median"] / was["median"] - 1.0))
            print(line, flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
