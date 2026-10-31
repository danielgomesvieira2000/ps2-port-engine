"""Every GPU backend checked the same way: the renderer's tests, then a
game's GPU regressions, on Vulkan 1.3, Vulkan 1.2 with dynamic rendering,
Vulkan 1.2 with render passes (PS2HOST_VK_LEGACY=1, 2) and Direct3D 11
(PS2HOST_RENDERER=d3d11) - one table at the end.

    python engine/tools/backends_check.py --game games/burnout3 [--runs car_select,race] [--gpu 4070]
        [--backends vk13,vk12,vk12rp,d3d11] [--tests-only]

Build first (port.py build). The regressions are regress.py's (--gpu), each
compared with its recording: Direct3D 11 with a run's own <run>.d3d11.crc
where there is one. A run takes a minute or two; the whole default set on
four backends ten to twenty.
"""

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

BACKENDS = {
    "vk13": ("Vulkan 1.3", {}),
    "vk12": ("Vulkan 1.2 + dynamic rendering", {"PS2HOST_VK_LEGACY": "1"}),
    "vk12rp": ("Vulkan 1.2, render passes", {"PS2HOST_VK_LEGACY": "2"}),
    "d3d11": ("Direct3D 11", {"PS2HOST_RENDERER": "d3d11"}),
}


def run(cmd, env, timeout):
    start = time.time()
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace", timeout=timeout)
        return out.returncode, out.stdout + out.stderr, time.time() - start
    except subprocess.TimeoutExpired:
        return -1, "timed out", time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", required=True)
    parser.add_argument("--runs", default="car_select,race_training,race",
                        help="regress.py runs, by commas (port/runs/<run>.args)")
    parser.add_argument("--backends", default=",".join(BACKENDS), help="of " + ", ".join(BACKENDS))
    parser.add_argument("--gpu", default="", help="PS2HOST_GPU: part of the GPU's name (both APIs' adapter)")
    parser.add_argument("--tests-only", action="store_true", help="the renderer's tests alone")
    args = parser.parse_args()
    game = os.path.abspath(args.game)
    tests = os.path.join(game, "build", "port", "cmake", "ps2gpu", "ps2gpu_tests.exe")
    if not os.path.exists(tests):
        sys.exit("no %s: build first" % tests)
    rows = []
    for key in args.backends.split(","):
        name, extra = BACKENDS[key]
        env = dict(os.environ)
        env.update(extra)
        if args.gpu:
            env["PS2HOST_GPU"] = args.gpu
        code, text, seconds = run([tests], env, 900)
        found = [l for l in text.splitlines() if "renderer tests:" in l]
        line = found[-1] if found else "(no result)"
        ok = code == 0 and "all passed" in line
        rows.append((name, "renderer tests", "pass" if ok else "FAIL", "%.0f s" % seconds, line))
        print("%-34s %-16s %s" % (name, "renderer tests", line), flush=True)
        if args.tests_only:
            continue
        for r in args.runs.split(","):
            code, text, seconds = run([sys.executable, os.path.join(HERE, "regress.py"), "--game", game, r, "--gpu"],
                                      env, 3600)
            m = re.search(r"(\d+) of (\d+) pictures as recorded", text)
            got = "%s/%s" % (m.group(1), m.group(2)) if m else "?"
            changed = re.findall(r"(frame_\d+)\.bmp: changed", text)
            ok = bool(m) and m.group(1) == m.group(2)
            rows.append((name, r, "pass" if ok else "FAIL", "%.0f s" % seconds,
                         got + ("" if ok else " (" + ", ".join(changed[:6]) + ")")))
            print("%-34s %-16s %s %s" % (name, r, got, "" if ok else " ".join(changed[:6])), flush=True)
    print("\n%-34s %-16s %-5s %-7s %s" % ("backend", "check", "", "time", "result"))
    for row in rows:
        print("%-34s %-16s %-5s %-7s %s" % row)
    sys.exit(0 if all(row[2] == "pass" for row in rows) else 1)


if __name__ == "__main__":
    main()
