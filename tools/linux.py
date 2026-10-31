#!/usr/bin/env python3
"""Linux: a port built and tested for Linux (x86-64) in a container, from
Windows or Linux alike - Docker with the image in tools/linux/Dockerfile
(clang 19, CMake, Ninja, SDL's system headers, Mesa's Vulkan: lavapipe draws on
the processor, so the GPU renderer runs its tests without a GPU).

    python engine/tools/linux.py --game games/<name> build          # <game>/build/linux/<target>
    python engine/tools/linux.py --game games/<name> test [--runs car_select,race] [--gpu]
    python engine/tools/linux.py --game games/<name> shell          # a shell in the container

The repository (the folder holding engine/ and games/) is mounted at /src;
the build is kept in a Docker volume (ps2port_linux_<target>: building on the
mounted Windows folder is slow) and the program copied out. The dispatch
table is generated first, here (port.py's), as for every build. `test` runs
the host's tests, the renderer's (on lavapipe), then regress.py's runs with
the software GS (--gpu: with lavapipe's Vulkan - its pictures are its own,
as an Intel GPU's are, so a count, not a pass, against NVIDIA's recordings).
"""

import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from ps2decomp import game as game_config  # noqa: E402
import port  # noqa: E402

IMAGE = "ps2port-linux"


def run(cmd, **kw):
    print("+ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""), flush=True)
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise SystemExit("linux: %s failed (%d)" % (cmd[0], result.returncode))
    return result


def root_of(game) -> str:
    """The repository: the nearest folder above the game holding engine/."""
    here = os.path.abspath(game.directory)
    while True:
        if os.path.isdir(os.path.join(here, "engine", "host")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise SystemExit("linux: no engine/ above %s" % game.directory)
        here = parent


def inside(root: str, path: str) -> str:
    """`path` as the container sees it (under /src)."""
    rel = os.path.relpath(os.path.abspath(path), root)
    if rel.startswith(".."):
        raise SystemExit("linux: %s is not inside %s" % (path, root))
    return "/src/" + rel.replace(os.sep, "/")


def image():
    """The image, made (or brought up to date: Docker's cache) from tools/linux."""
    run(["docker", "build", "-q", "-t", IMAGE, os.path.join(HERE, "linux")], stdout=subprocess.DEVNULL)


def container(game, script: str, interactive: bool = False):
    root = root_of(game)
    target = port_target(game)
    cmd = ["docker", "run", "--rm", "-v", root + ":/src", "-v", "ps2port_linux_%s:/build" % target, "-w", "/src"]
    if interactive:
        cmd += ["-it", IMAGE, "bash"]
    else:
        cmd += [IMAGE, "bash", "-c", script]
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"       # (Git Bash's path rewriting off: these are the container's)
    return run(cmd, env=env)


def port_target(game) -> str:
    return os.path.basename(os.path.normpath(game.directory))


def build(game, jobs: int):
    generated = port.generate(game)
    root = root_of(game)
    target = port_target(game)
    out = game.path(os.path.join(game.build, "linux"))
    os.makedirs(out, exist_ok=True)
    script = " && ".join([
        "cmake -S %s -B /build/%s -G Ninja -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 "
        "-DCMAKE_BUILD_TYPE=Release -DPS2_GENERATED_DIR=%s -DPS2HOST_LTO_JOBS=%d > /build/configure.log"
        % (inside(root, game.path("port")), target, inside(root, generated), jobs),
        "cmake --build /build/%s -j %d" % (target, jobs),
        "cp /build/%s/%s %s/" % (target, target, inside(root, out)),
    ])
    container(game, script)
    print("built %s" % os.path.join(out, target))


def test(game, runs: str, gpu: bool):
    root = root_of(game)
    target = port_target(game)
    steps = ["/build/%s/ps2host/ps2host_tests | tail -1" % target,
             "/build/%s/ps2gpu/ps2gpu_tests | tail -1" % target]
    for r in [r for r in runs.split(",") if r]:
        steps.append("python3 engine/tools/regress.py --game %s %s --exe /build/%s/%s%s | tail -1"
                     % (shlex.quote(inside(root, game.directory)), r, target, target, " --gpu" if gpu else ""))
    container(game, "set -o pipefail; " + " && ".join(steps))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game_config.add_argument(parser)
    parser.add_argument("action", nargs="?", default="build", choices=("build", "test", "shell"))
    parser.add_argument("--runs", default="", help="test: regress.py runs, by commas")
    parser.add_argument("--gpu", action="store_true", help="test: the runs with the GPU renderer (lavapipe)")
    parser.add_argument("--jobs", type=int, default=6, help="build: compile jobs (the container's memory: ~1 GB each)")
    args = parser.parse_args(argv)
    game = game_config.from_args(args)
    image()
    if args.action == "build":
        build(game, args.jobs)
    elif args.action == "test":
        test(game, args.runs, args.gpu)
    else:
        container(game, "", interactive=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
